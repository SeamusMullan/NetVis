# NetVis — speed-first neural network model viewer.
# Single CMake codebase for Windows / Linux / macOS (spec §1, §9).
cmake_minimum_required(VERSION 3.24)
project(netvis LANGUAGES C CXX)

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
  PUBLIC nlohmann_json::nlohmann_json
  PRIVATE netvis_warnings)
# miniz target name varies by version; link if present.
if(TARGET miniz)
  target_link_libraries(netvis_core PUBLIC miniz)
endif()
target_include_directories(netvis_core PUBLIC ${miniz_SOURCE_DIR} ${miniz_BINARY_DIR})

find_package(Threads REQUIRED)
target_link_libraries(netvis_core PUBLIC Threads::Threads)

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
