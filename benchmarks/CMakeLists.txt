function(add_minillm_bench name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE minillm)
endfunction()

add_minillm_bench(bench_ops)
