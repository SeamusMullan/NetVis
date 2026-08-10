// core/Rss.cpp — platform peak/current resident-set-size readers (#101).
//
// See Rss.h for the contract this file implements: 0 always means "the
// platform counter is unavailable", never a fabricated measurement, and
// neither function may throw or allocate more than the platform API itself
// needs. Three platform bodies + an honest-unknown fallback, selected at
// compile time so exactly one of them is ever compiled — no dead branches to
// warn about on any single CI runner (ubuntu/macos/windows-latest).
#include "core/Rss.h"

#include "core/SafeMath.h"  // safe_mul: overflow-safe kB->bytes conversion

#if defined(__linux__)
#include <cstdio>
#include <cstring>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(_WIN32)
// PSAPI_VERSION 2 routes GetProcessMemoryInfo through K32GetProcessMemoryInfo,
// which ships in kernel32.dll (Windows Vista+) rather than psapi.dll. That
// keeps this one counter read from requiring a new psapi.lib link dependency
// on the netvis_core CMake target (see CMakeLists.txt, outside this file's
// scope) — the "avoid the extra library" option Rss.h's caller asked for.
// Must be defined before <psapi.h> is included.
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
#endif

namespace netvis {

#if defined(__linux__)

namespace {

// Read one "VmXXX:    <n> kB" line's value from /proc/self/status, in BYTES.
// `key` must include the trailing colon (e.g. "VmHWM:") so a prefix match
// cannot accidentally hit a differently-named field. Returns 0 if the file
// can't be opened, the key never appears (an exotic kernel build without this
// field — the exact case Rss.h calls out), or the value fails to parse as a
// plain decimal integer. Uses stdio + a fixed-size stack buffer rather than
// <fstream>/std::string so the call stays allocation-free, matching the
// "cheap enough to call between bench stages" promise in the header.
uint64_t read_status_kb_field(const char* key) {
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (!f) return 0;

  char line[256];
  const size_t key_len = std::strlen(key);
  uint64_t kb = 0;
  bool found = false;
  while (std::fgets(line, sizeof(line), f)) {
    if (std::strncmp(line, key, key_len) != 0) continue;
    const char* p = line + key_len;
    while (*p == ' ' || *p == '\t') ++p;
    bool any_digit = false;
    uint64_t v = 0;
    while (*p >= '0' && *p <= '9') {
      any_digit = true;
      v = v * 10 + static_cast<uint64_t>(*p - '0');
      ++p;
    }
    found = any_digit;
    if (found) kb = v;
    break;  // the field appears at most once; stop scanning either way.
  }
  std::fclose(f);
  return found ? safe_mul(kb, 1024ull) : 0;
}

}  // namespace

uint64_t peak_rss_bytes() { return read_status_kb_field("VmHWM:"); }
uint64_t current_rss_bytes() { return read_status_kb_field("VmRSS:"); }

#elif defined(__APPLE__)

namespace {

// Both figures come from a single task_info call, so share the read. Leaves
// *info untouched and returns false on any failure — the caller then reports
// 0 (unknown), never a stale or partially-populated struct.
bool read_task_basic_info(mach_task_basic_info_data_t* info) {
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                               reinterpret_cast<task_info_t>(info), &count);
  return kr == KERN_SUCCESS;
}

}  // namespace

uint64_t peak_rss_bytes() {
  mach_task_basic_info_data_t info{};
  if (!read_task_basic_info(&info)) return 0;
  return static_cast<uint64_t>(info.resident_size_max);
}

uint64_t current_rss_bytes() {
  mach_task_basic_info_data_t info{};
  if (!read_task_basic_info(&info)) return 0;
  return static_cast<uint64_t>(info.resident_size);
}

#elif defined(_WIN32)

namespace {

// Shared by both readers: one GetProcessMemoryInfo call gives peak + current.
// Returns false (leaving *pmc default-initialized) if the API call fails.
bool read_process_memory_counters(PROCESS_MEMORY_COUNTERS* pmc) {
  pmc->cb = sizeof(*pmc);
  return GetProcessMemoryInfo(GetCurrentProcess(), pmc, sizeof(*pmc)) != 0;
}

}  // namespace

uint64_t peak_rss_bytes() {
  PROCESS_MEMORY_COUNTERS pmc{};
  if (!read_process_memory_counters(&pmc)) return 0;
  return static_cast<uint64_t>(pmc.PeakWorkingSetSize);
}

uint64_t current_rss_bytes() {
  PROCESS_MEMORY_COUNTERS pmc{};
  if (!read_process_memory_counters(&pmc)) return 0;
  return static_cast<uint64_t>(pmc.WorkingSetSize);
}

#else

// Unknown platform: report honestly rather than fabricate a figure — 0 always
// means "unknown" per the Rss.h contract, and there is no portable memory
// counter worth guessing at here (spec's honesty rule, same as unresolved
// shapes elsewhere in the engine).
uint64_t peak_rss_bytes() { return 0; }
uint64_t current_rss_bytes() { return 0; }

#endif

}  // namespace netvis
