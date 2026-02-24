# CHARLTON Examples (Pure C99)

add_executable(example_basic_pricing basic_pricing.c)
target_link_libraries(example_basic_pricing PRIVATE charlton::charlton)

add_executable(example_greeks greeks.c)
target_link_libraries(example_greeks PRIVATE charlton::charlton)

add_executable(example_calibration calibration.c)
target_link_libraries(example_calibration PRIVATE charlton::charlton)

add_executable(example_conformal_bootstrap conformal_bootstrap.c)
target_link_libraries(example_conformal_bootstrap PRIVATE charlton::charlton)

add_executable(example_american_pricing american_pricing.c)
target_link_libraries(example_american_pricing PRIVATE charlton::charlton)
