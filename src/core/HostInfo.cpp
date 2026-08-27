// core/HostInfo.cpp — platform host-description probes (#154).
//
// See HostInfo.h for the contract: empty string / 0 always means "the platform
// would not say", never a fabricated value. Structured exactly like
// core/Rss.cpp — three platform bodies plus an honest-unknown fallback,
// selected at compile time so exactly one is ever compiled and no CI runner
// sees a dead branch to warn about.
//
// Unlike Rss.cpp this file is allowed to allocate and to use <fstream>: it is
// called once per process to label a JSON artifact, not between timed bench
// stages, so readability wins over the allocation-free discipline the RSS
// readers need.
#include "core/HostInfo.h"

#include <thread>

#include "core/SafeMath.h"  // safe_mul: overflow-safe pages*page_size

#if defined(__linux__)
#include <unistd.h>

#include <fstream>
#include <set>
#include <string>
#include <utility>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>

#include <vector>
#elif defined(_WIN32)
// <vector> FIRST, and NOMINMAX before windows.h: windows.h defines min/max as
// macros, which then mangle the std:: templates any header included after it
// uses. WIN32_LEAN_AND_MEAN drops the socket/RPC/OLE headers this file has no
// use for.
#include <vector>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace netvis {

namespace {

// Architecture and OS come from the PREPROCESSOR, not from uname/sysctl. See
// HostInfo.h: the question a benchmark needs answered is "what code ran", and
// a cross-built or emulated binary runs the ISA it was compiled for regardless
// of what the kernel calls the machine underneath it.
#if defined(__x86_64__) || defined(_M_X64)
constexpr const char* kArch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
constexpr const char* kArch = "arm64";
#elif defined(__i386__) || defined(_M_IX86)
constexpr const char* kArch = "x86";
#elif defined(__arm__) || defined(_M_ARM)
constexpr const char* kArch = "arm";
#else
constexpr const char* kArch = "unknown";
#endif

#if defined(__linux__)
constexpr const char* kOs = "linux";
#elif defined(__APPLE__)
constexpr const char* kOs = "macos";
#elif defined(_WIN32)
constexpr const char* kOs = "windows";
#else
constexpr const char* kOs = "unknown";
#endif

}  // namespace

#if defined(__linux__)

namespace {

// Strip leading/trailing spaces and tabs. /proc/cpuinfo pads its values to a
// column ("model name\t: AMD ..."), and a CPU name with a ragged tab in front
// of it would show up in every JSON artifact and every gate message.
std::string trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// "<key>\t: <value>" -> value, or "" if this line is not that key. Matches on
// the KEY side of the colon only, so a value that happens to contain the key's
// text cannot produce a false hit.
std::string field(const std::string& line, const char* key) {
  size_t colon = line.find(':');
  if (colon == std::string::npos) return {};
  if (trim(line.substr(0, colon)) != key) return {};
  return trim(line.substr(colon + 1));
}

// Parsed by hand rather than with std::stol: a non-numeric topology field on
// an exotic kernel is a "this host would not say" case, and std::stol answers
// that with an exception, which is a heavy way to say "unknown" in a probe
// whose entire contract is that it never throws.
long field_as_long(const std::string& line, const char* key, bool* ok) {
  const std::string v = field(line, key);
  if (v.empty()) return 0;
  long parsed = 0;
  for (char c : v) {
    if (c < '0' || c > '9') return 0;
    parsed = parsed * 10 + (c - '0');
  }
  *ok = true;
  return parsed;
}

}  // namespace

HostInfo host_info() {
  HostInfo h;
  h.arch = kArch;
  h.os = kOs;
  h.logical_cores = std::thread::hardware_concurrency();

  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo) {
    // Physical cores are counted as DISTINCT (physical id, core id) pairs
    // rather than read from "cpu cores", because "cpu cores" is per-socket:
    // on a dual-socket box it reports one socket's worth and undercounts the
    // machine by half. The pair set is correct for any socket count.
    std::set<std::pair<long, long>> physical;
    long socket = 0;
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (h.cpu_model.empty()) {
        std::string v = field(line, "model name");
        // "Hardware"/"Model" are the ARM/devicetree spellings; x86 kernels
        // emit "model name" and never reach these.
        if (v.empty()) v = field(line, "Hardware");
        if (v.empty()) v = field(line, "Model");
        if (!v.empty()) h.cpu_model = v;
      }
      bool ok = false;
      long parsed = field_as_long(line, "physical id", &ok);
      if (ok) {
        socket = parsed;
        continue;
      }
      // "physical id" always precedes "core id" within a processor block, so
      // `socket` is already the right one by the time we get here. Kernels
      // that omit "physical id" entirely are single-socket, and 0 is then the
      // correct socket number rather than a guess.
      parsed = field_as_long(line, "core id", &ok);
      if (ok) physical.insert({socket, parsed});
    }
    if (!physical.empty())
      h.physical_cores = static_cast<uint32_t>(physical.size());
    // Left at 0 (unknown) otherwise: ARM /proc/cpuinfo carries no core
    // topology at all, and assuming "no SMT, so physical == logical" there
    // would be a guess dressed up as a measurement.
  }

  const long pages = sysconf(_SC_PHYS_PAGES);
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0)
    h.total_ram_bytes = safe_mul(static_cast<uint64_t>(pages),
                                 static_cast<uint64_t>(page_size));
  return h;
}

#elif defined(__APPLE__)

namespace {

// sysctlbyname into a std::string, sized by the two-call idiom (ask for the
// length, then read). Returns "" if the key does not exist on this kernel —
// notably true of some machdep.* keys on Apple Silicon.
std::string sysctl_string(const char* name) {
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
  std::vector<char> buf(len);
  if (sysctlbyname(name, buf.data(), &len, nullptr, 0) != 0) return {};
  // The kernel includes the NUL in `len`; std::string must not.
  while (!buf.empty() && buf.back() == '\0') buf.pop_back();
  return std::string(buf.begin(), buf.end());
}

// Integer sysctls are int32 (hw.logicalcpu) or int64 (hw.memsize) depending on
// the key, so ask the kernel for the width first and read at exactly that
// width. Reading every key as int64 would leave the high half uninitialized on
// the 32-bit keys and turn a core count into garbage. Returns 0 for a missing
// key, which is the HostInfo unknown value anyway.
uint64_t sysctl_uint(const char* name) {
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0) return 0;
  if (len == sizeof(int32_t)) {
    int32_t v = 0;
    size_t n = sizeof(v);
    if (sysctlbyname(name, &v, &n, nullptr, 0) != 0) return 0;
    return v > 0 ? static_cast<uint64_t>(v) : 0;
  }
  if (len == sizeof(int64_t)) {
    int64_t v = 0;
    size_t n = sizeof(v);
    if (sysctlbyname(name, &v, &n, nullptr, 0) != 0) return 0;
    return v > 0 ? static_cast<uint64_t>(v) : 0;
  }
  return 0;
}

}  // namespace

HostInfo host_info() {
  HostInfo h;
  h.arch = kArch;
  h.os = kOs;
  h.cpu_model = sysctl_string("machdep.cpu.brand_string");
  // hw.model is the model identifier ("MacBookPro18,3"), not a CPU name — a
  // second-best answer, used only when the CPU brand string is absent, which
  // beats reporting an unknown CPU on a machine that will happily name itself.
  if (h.cpu_model.empty()) h.cpu_model = sysctl_string("hw.model");
  h.logical_cores = static_cast<uint32_t>(sysctl_uint("hw.logicalcpu"));
  h.physical_cores = static_cast<uint32_t>(sysctl_uint("hw.physicalcpu"));
  if (h.logical_cores == 0) h.logical_cores = std::thread::hardware_concurrency();
  h.total_ram_bytes = sysctl_uint("hw.memsize");
  return h;
}

#elif defined(_WIN32)

namespace {

// HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0 : ProcessorNameString.
// Chosen over the __cpuid 0x80000002-4 brand-string leaves because those are
// x86-only intrinsics and this file also compiles for windows-arm64, where
// they do not exist. advapi32 is in CMake's default MSVC link set, so reading
// the registry costs no new link dependency (the same "avoid an extra library"
// constraint core/Rss.cpp works under).
std::string registry_cpu_name() {
  char buf[256] = {};
  DWORD size = static_cast<DWORD>(sizeof(buf));
  LSTATUS st = RegGetValueA(
      HKEY_LOCAL_MACHINE,
      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
      "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buf, &size);
  if (st != ERROR_SUCCESS) return {};
  buf[sizeof(buf) - 1] = '\0';
  std::string name(buf);
  while (!name.empty() && (name.back() == ' ' || name.back() == '\0'))
    name.pop_back();
  return name;
}

// Count RelationProcessorCore entries. GetLogicalProcessorInformationEx is the
// EX form on purpose: the non-Ex GetLogicalProcessorInformation cannot describe
// machines with more than 64 logical processors (processor groups), and silently
// reports only the calling group's cores on one.
uint32_t physical_core_count() {
  DWORD bytes = 0;
  if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes))
    return 0;  // a success with a zero-sized buffer would be nonsense
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0) return 0;

  std::vector<unsigned char> buf(bytes);
  auto* first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &bytes))
    return 0;

  // Variable-length records: walk by each entry's own Size. sizeof() is NOT a
  // valid stride or bound here — the struct is a union sized by its largest
  // member (GROUP_RELATIONSHIP), so a PROCESSOR_RELATIONSHIP record is
  // legitimately SMALLER than sizeof(), and bounding the loop by sizeof() would
  // silently drop the last core on machines where the buffer ends short of it.
  // Only the fixed header (Relationship + Size) is guaranteed present, so that
  // is what the loop requires before trusting Size.
  const DWORD kHeader =
      static_cast<DWORD>(sizeof(LOGICAL_PROCESSOR_RELATIONSHIP) + sizeof(DWORD));
  uint32_t cores = 0;
  DWORD offset = 0;
  while (offset + kHeader <= bytes) {
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
        buf.data() + offset);
    // Size 0 would spin forever; a Size past the end means the kernel and this
    // walk disagree, and reading on from there is a buffer overrun.
    if (info->Size == 0 || offset + info->Size > bytes) break;
    if (info->Relationship == RelationProcessorCore) ++cores;
    offset += info->Size;
  }
  return cores;
}

}  // namespace

HostInfo host_info() {
  HostInfo h;
  h.arch = kArch;
  h.os = kOs;
  h.cpu_model = registry_cpu_name();
  h.logical_cores = std::thread::hardware_concurrency();
  h.physical_cores = physical_core_count();

  MEMORYSTATUSEX mem{};
  mem.dwLength = sizeof(mem);
  if (GlobalMemoryStatusEx(&mem))
    h.total_ram_bytes = static_cast<uint64_t>(mem.ullTotalPhys);
  return h;
}

#else

// Unknown platform: report the two things the compiler still knows for certain
// and leave the rest unknown, per the HostInfo.h honesty rule. Guessing a core
// count or a RAM figure here would put a fabricated number into a JSON artifact
// that a human later reads as measured fact.
HostInfo host_info() {
  HostInfo h;
  h.arch = kArch;
  h.os = kOs;
  h.logical_cores = std::thread::hardware_concurrency();
  return h;
}

#endif

}  // namespace netvis
