# CHARLTON Benchmarks (Google Benchmark C++ calling C API)

# Each benchmark has its own main()
add_executable(charlton_bench_pricer benchmark_pricer.cpp)
target_link_libraries(charlton_bench_pricer PRIVATE charlton::charlton benchmark::benchmark)

add_executable(charlton_bench_abm benchmark_abm.cpp)
target_link_libraries(charlton_bench_abm PRIVATE charlton::charlton benchmark::benchmark)

add_executable(charlton_bench_greeks benchmark_greeks.cpp)
target_link_libraries(charlton_bench_greeks PRIVATE charlton::charlton benchmark::benchmark)

# Convenience target
add_custom_target(charlton_benchmarks
    DEPENDS charlton_bench_pricer charlton_bench_abm charlton_bench_greeks
)
