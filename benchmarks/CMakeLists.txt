# CHARLTON Benchmarks

add_executable(charlton_benchmarks
    benchmark_pricer.cpp
    benchmark_abm.cpp
    benchmark_greeks.cpp
)

target_link_libraries(charlton_benchmarks PRIVATE
    charlton::charlton
    benchmark::benchmark_main
)
