cmake_minimum_required(VERSION 3.20)
project(minillm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Compiler warnings ─────────────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wpedantic)

# ── Build-type flags ──────────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # -march=native on AArch64 (M2) enables NEON and all other ISA extensions.
    add_compile_options(-O3 -march=native)
endif()

# ── Core library ──────────────────────────────────────────────────────────────
# INTERFACE for now (header-only); will become STATIC in Milestone 1.
add_library(minillm INTERFACE)
target_include_directories(minillm INTERFACE ${PROJECT_SOURCE_DIR}/include)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Benchmarks ────────────────────────────────────────────────────────────────
add_subdirectory(benchmarks)
