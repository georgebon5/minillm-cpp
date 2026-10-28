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

# ── Dependencies ──────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(minillm STATIC
    src/tensor.cpp
    src/safetensors.cpp
    src/model_config.cpp
    src/ops.cpp
    src/model.cpp
)
target_include_directories(minillm PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(minillm PUBLIC cxx_std_20)
target_link_libraries(minillm PUBLIC nlohmann_json::nlohmann_json)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)

# ── Benchmarks ────────────────────────────────────────────────────────────────
add_subdirectory(benchmarks)
