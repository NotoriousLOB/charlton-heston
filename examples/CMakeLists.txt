# CHARLTON Examples

add_executable(example_basic_pricing basic_pricing.cpp)
target_link_libraries(example_basic_pricing PRIVATE charlton::charlton)

add_executable(example_greeks greeks.cpp)
target_link_libraries(example_greeks PRIVATE charlton::charlton)

add_executable(example_calibration calibration.cpp)
target_link_libraries(example_calibration PRIVATE charlton::charlton)

add_executable(example_conformal_bootstrap conformal_bootstrap.cpp)
target_link_libraries(example_conformal_bootstrap PRIVATE charlton::charlton)
