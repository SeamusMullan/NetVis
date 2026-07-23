# NetVis — speed-first neural network model viewer.
# Single CMake codebase for Windows / Linux / macOS (spec §1, §9).
cmake_minimum_required(VERSION 3.24)

# Version drives CPack/installer metadata. CI passes -DNETVIS_VERSION=X.Y.Z
# extracted from the release tag; the fallback keeps local builds working.
if(NOT DEFINED NETVIS_VERSION)
  set(NETVIS_VERSION "0.0.0")
endif()
project(netvis VERSION ${NETVIS_VERSION} LANGUAGES C CXX)

# --- Global standards --------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# DECISION (spec §9): Release uses -O2 + LTO for the perf budget. LTO only in
# Release so Debug stays fast to build.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _ipo OUTPUT _ipo_msg)
  if(_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()

# Warnings-as-errors on the project's OWN code only (spec §2.6, §9). Applied via
# a helper target, never to third_party.
add_library(netvis_warnings INTERFACE)
if(MSVC)
  target_compile_options(netvis_warnings INTERFACE /W4 /WX /permissive-)
  # MSVC flags standard C funcs (getenv/fopen/...) as "unsafe" C4996 under /WX.
  # They are used correctly and portably; silence the CRT deprecation so /WX
  # still catches real warnings.
  target_compile_definitions(netvis_warnings INTERFACE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(netvis_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Werror
    -Wno-error=unused-parameter)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Some pinned deps (miniz 3.0.2, glfw 3.4) declare cmake_minimum_required below
# what CMake 4.x still supports. Allow their old policy floor rather than forking
# them — we never rely on their deprecated policy behavior.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# --- Options -----------------------------------------------------------------
option(NETVIS_BUILD_TESTS "Build unit tests" ON)
option(NETVIS_BUILD_APP "Build the GUI app (needs OpenGL/GLFW)" ON)

# Sanitizers (CI correctness gate). One of: OFF, address, undefined, thread.
# Applied to our OWN code via netvis_warnings so third_party is not instrumented
# for -Werror purposes, but the flags must also reach the linker — so they go on
# a dedicated INTERFACE target linked by every NetVis target.
set(NETVIS_SANITIZER "OFF" CACHE STRING "Sanitizer: OFF|address|undefined|thread")
set_property(CACHE NETVIS_SANITIZER PROPERTY STRINGS OFF address undefined thread)
add_library(netvis_sanitize INTERFACE)
if(NOT NETVIS_SANITIZER STREQUAL "OFF")
  if(MSVC)
    # MSVC only ships /fsanitize=address; UBSan/TSan are no-ops there.
    if(NETVIS_SANITIZER STREQUAL "address")
      target_compile_options(netvis_sanitize INTERFACE /fsanitize=address)
    endif()
  else()
    # address+undefined can combine; thread must run alone (incompatible).
    if(NETVIS_SANITIZER STREQUAL "address")
      set(_san "address,undefined")
    else()
      set(_san "${NETVIS_SANITIZER}")
    endif()
    target_compile_options(netvis_sanitize INTERFACE
      -fsanitize=${_san} -fno-omit-frame-pointer -g)
    target_link_options(netvis_sanitize INTERFACE -fsanitize=${_san})
  endif()
endif()

# =============================================================================
#  Dependencies (FetchContent, pinned — spec §2.6, §9)
# =============================================================================

# nlohmann/json — SafeTensors header + config files.
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# miniz — PyTorch zip reader. Amalgamated single-file lib.
FetchContent_Declare(miniz
  GIT_REPOSITORY https://github.com/richgel999/miniz.git
  GIT_TAG 3.0.2)
set(BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(miniz)

# stb — stb_image_write for PNG export (header-only, pinned commit).
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 31c1ad37456438565541f4919958214b6e762fb4)
FetchContent_MakeAvailable(stb)
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

# wasm3 — single-file WebAssembly interpreter for the v0.6.0 WASM plugin backend
# (#10). Chosen over wasmtime to keep the single-static-binary invariant (no LLVM/
# codegen dep) and cross-platform portability; the host-API boundary is kept
# runtime-agnostic so wasmtime is a later drop-in. Source-only, pinned tag. The
# gas-metering define enables the fuel/step cap used to kill a hostile module.
option(NETVIS_ENABLE_WASM "Build the WASM (wasm3) plugin backend" ON)
if(NETVIS_ENABLE_WASM)
  FetchContent_Declare(wasm3
    GIT_REPOSITORY https://github.com/wasm3/wasm3.git
    GIT_TAG v0.5.0)
  # POPULATE ONLY — do NOT run wasm3's own CMakeLists (it pulls in uvwasi/WASI,
  # which we explicitly do NOT want: the sandbox must expose zero syscalls). We
  # compile only the interpreter core from source/*.c into our own static target.
  FetchContent_GetProperties(wasm3)
  if(NOT wasm3_POPULATED)
    FetchContent_Populate(wasm3)
  endif()
  file(GLOB WASM3_SRC ${wasm3_SOURCE_DIR}/source/*.c)
  add_library(wasm3 STATIC ${WASM3_SRC})
  target_include_directories(wasm3 PUBLIC ${wasm3_SOURCE_DIR}/source)
  target_compile_definitions(wasm3 PUBLIC d_m3MaxFunctionStackHeight=1024)
  # wasm3's own C sources must NOT inherit our -Werror/-Wpedantic.
  set_target_properties(wasm3 PROPERTIES COMPILE_FLAGS "-w")
endif()

# doctest — unit test harness (tests only).
if(NETVIS_BUILD_TESTS)
  FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/onqtam/doctest.git
    GIT_TAG v2.4.11)
  FetchContent_MakeAvailable(doctest)
endif()

# =============================================================================
#  netvis_core — core + ir + parsers + engine. NO GUI dependencies (spec §9).
# =============================================================================
file(GLOB_RECURSE NETVIS_CORE_SRC CONFIGURE_DEPENDS
  ${CMAKE_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_SOURCE_DIR}/src/ir/*.cpp
  ${CMAKE_SOURCE_DIR}/src/parsers/*.cpp
  ${CMAKE_SOURCE_DIR}/src/engine/*.cpp)

add_library(netvis_core STATIC ${NETVIS_CORE_SRC})
target_include_directories(netvis_core PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(netvis_core
  PUBLIC nlohmann_json::nlohmann_json netvis_sanitize
  PRIVATE netvis_warnings)
# miniz target name varies by version; link if present.
if(TARGET miniz)
  target_link_libraries(netvis_core PUBLIC miniz)
endif()
target_include_directories(netvis_core PUBLIC ${miniz_SOURCE_DIR} ${miniz_BINARY_DIR})

find_package(Threads REQUIRED)
target_link_libraries(netvis_core PUBLIC Threads::Threads)

# wasm3 plugin backend (#10): link + define the guard so the WASM binding code
# compiles in. When NETVIS_ENABLE_WASM is OFF, the backend compiles to no-ops.
if(NETVIS_ENABLE_WASM AND TARGET wasm3)
  target_link_libraries(netvis_core PUBLIC wasm3)
  target_compile_definitions(netvis_core PUBLIC NETVIS_ENABLE_WASM=1)
endif()

# =============================================================================
#  netvis — the GUI application (GLFW + OpenGL + Dear ImGui).
# =============================================================================
if(NETVIS_BUILD_APP)
  # GLFW — windowing + input.
  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4)
  set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
  set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
  set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
  set(GLFW_INSTALL OFF CACHE INTERNAL "")
  # On Linux, GLFW 3.4 builds the Wayland backend by default, which requires
  # wayland-scanner + Wayland dev headers not present on a stock CI image. Build
  # the X11 backend only — it works everywhere (incl. XWayland) and drops the
  # extra build-time dependency. NETVIS_GLFW_WAYLAND=ON re-enables both backends.
  option(NETVIS_GLFW_WAYLAND "Also build the GLFW Wayland backend on Linux" OFF)
  if(UNIX AND NOT APPLE AND NOT NETVIS_GLFW_WAYLAND)
    set(GLFW_BUILD_WAYLAND OFF CACHE INTERNAL "")
    set(GLFW_BUILD_X11 ON CACHE INTERNAL "")
  endif()
  FetchContent_MakeAvailable(glfw)

  # Dear ImGui (docking branch) — panels, dialogs, and the ImDrawList canvas.
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.8-docking)
  FetchContent_MakeAvailable(imgui)

  add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(imgui PUBLIC glfw)
  find_package(OpenGL REQUIRED)
  target_link_libraries(imgui PUBLIC OpenGL::GL)

  # tinyfiledialogs — native open-file dialog (spec §8.7). C source.
  FetchContent_Declare(tinyfiledialogs
    GIT_REPOSITORY https://github.com/native-toolkit/tinyfiledialogs.git
    GIT_TAG 2.9.3)
  FetchContent_MakeAvailable(tinyfiledialogs)
  add_library(tinyfiledialogs STATIC ${tinyfiledialogs_SOURCE_DIR}/tinyfiledialogs.c)
  target_include_directories(tinyfiledialogs PUBLIC ${tinyfiledialogs_SOURCE_DIR})

  file(GLOB_RECURSE NETVIS_VIEW_SRC CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/view/*.cpp)

  add_executable(netvis ${CMAKE_SOURCE_DIR}/src/main.cpp ${NETVIS_VIEW_SRC})
  target_link_libraries(netvis
    PRIVATE netvis_core imgui glfw tinyfiledialogs stb OpenGL::GL netvis_warnings)
  target_include_directories(netvis PRIVATE ${CMAKE_SOURCE_DIR}/src)

  # macOS: build a .app bundle so the .pkg installs a normal double-clickable app.
  if(APPLE)
    set_target_properties(netvis PROPERTIES
      MACOSX_BUNDLE TRUE
      MACOSX_BUNDLE_BUNDLE_NAME "NetVis"
      MACOSX_BUNDLE_GUI_IDENTIFIER "com.seamusmullan.netvis"
      MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
      MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
  endif()

  # --- Install rules (consumed by CPack) -------------------------------------
  # Windows/Linux install the bare executable; macOS installs the .app bundle.
  # Install the app. Specify BOTH BUNDLE and RUNTIME destinations so the correct
  # one fires per platform (macOS = BUNDLE for the .app, Win/Linux = RUNTIME for
  # the bare exe) from a single rule — a BUNDLE-only rule on macOS was staging an
  # empty payload in some CPack/productbuild setups.
  install(TARGETS netvis
          BUNDLE  DESTINATION .
          RUNTIME DESTINATION bin)
  install(FILES ${CMAKE_SOURCE_DIR}/README.md ${CMAKE_SOURCE_DIR}/DECISIONS.md
          DESTINATION share/netvis OPTIONAL)
endif()

# =============================================================================
#  netvis_tests — links only netvis_core (spec §9, headless ctest).
# =============================================================================
if(NETVIS_BUILD_TESTS)
  enable_testing()
  file(GLOB_RECURSE NETVIS_TEST_SRC CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/tests/*.cpp)
  if(NETVIS_TEST_SRC)
    add_executable(netvis_tests ${NETVIS_TEST_SRC})
    target_link_libraries(netvis_tests PRIVATE netvis_core doctest::doctest)
    target_include_directories(netvis_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

    # Generate the tiny fixture models at build time (Python 3 stdlib only, no
    # pip) so `ctest` always has them, and run tests from the source root so the
    # relative "tests/fixtures/..." paths resolve regardless of the build dir.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_custom_command(TARGET netvis_tests PRE_BUILD
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/gen_fixtures.py
                ${CMAKE_SOURCE_DIR}/tests/fixtures
        COMMENT "Generating test fixtures"
        VERBATIM)
    endif()

    add_test(NAME netvis_tests COMMAND netvis_tests
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endif()
endif()

# =============================================================================
#  Packaging (CPack): NSIS installer on Windows, .pkg on macOS, .zip on Linux.
#  CI runs `cpack -G <gen>` after the install; the resulting artifact is what
#  gets attached to a GitHub Release on a vX.Y.Z tag.
# =============================================================================
if(NETVIS_BUILD_APP)
  set(CPACK_PACKAGE_NAME "NetVis")
  set(CPACK_PACKAGE_VENDOR "Seamus Mullan")
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
  # NSIS derives its installer name/version from the split components; set them
  # explicitly so the packaged filename carries the full X.Y.Z (not just "0").
  set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Speed-first neural network model viewer")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "NetVis")
  # Monolithic (non-component) install: package everything from our install()
  # rules as one payload. Without this, component-aware generators (productbuild)
  # can stage an EMPTY payload when no components are declared — which produced a
  # ~1 KB macOS .pkg with no app inside.
  set(CPACK_MONOLITHIC_INSTALL ON)
  set(CPACK_PACKAGE_EXECUTABLES "netvis" "NetVis")
  # Per-arch, per-os artifact file name, e.g. NetVis-1.2.0-Linux-x86_64.
  set(CPACK_PACKAGE_FILE_NAME
      "NetVis-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
  # NOTE: do NOT set CPACK_RESOURCE_FILE_README to a .md — productbuild only
  # accepts .rtf/.rtfd/.html/.txt and errors on other extensions.

  if(WIN32)
    # NSIS: a standard Start-menu + uninstall installer. `makensis` must be on
    # PATH in CI (choco install nsis).
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_PACKAGE_NAME "NetVis")
    set(CPACK_NSIS_DISPLAY_NAME "NetVis")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH OFF)
    # Start-menu shortcut -> bin/netvis.exe
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\NetVis\\\\NetVis.lnk' '$INSTDIR\\\\bin\\\\netvis.exe'")
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\NetVis\\\\NetVis.lnk'")
  elseif(APPLE)
    # DragNDrop -> a .dmg containing NetVis.app (drag into /Applications). This is
    # the reliable macOS distribution format for a GUI .app: the productbuild
    # generator repeatedly produced a ~1 KB empty .pkg in CI (its pkgbuild payload
    # walk did not embed the bundle), whereas DragNDrop just images the staged
    # tree. The .app is installed to a relative "." so the DMG root holds it.
    set(CPACK_GENERATOR "DragNDrop")
    set(CPACK_PACKAGE_VENDOR "seamusmullan")
  else()
    # Linux: a portable .zip (spec-requested). TGZ also produced as a courtesy.
    set(CPACK_GENERATOR "ZIP")
  endif()

  include(CPack)
endif()
