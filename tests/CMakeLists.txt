include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# Makes catch_discover_tests() available
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ── Helper to register a test executable ─────────────────────────────────────
function(add_minillm_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE Catch2::Catch2WithMain minillm)
    # Expose the repo root so tests can locate model files and reference data.
    target_compile_definitions(${name} PRIVATE
        MINILLM_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    )
    catch_discover_tests(${name})
endfunction()

# ── Tests (one line per test file added each milestone) ───────────────────────
add_minillm_test(test_smoke)
add_minillm_test(test_tensor)
add_minillm_test(test_safetensors)
add_minillm_test(test_model_config)
add_minillm_test(test_loader_integration)
add_minillm_test(test_ops)
add_minillm_test(test_model_integration)
add_minillm_test(test_tokenizer)
add_minillm_test(test_tokenizer_integration)
add_minillm_test(test_generation)
add_minillm_test(test_quantize)
