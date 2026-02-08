# CHARLTON Tests

# Test executable
add_executable(charlton_tests
    test_main.cpp
    test_abm_solver.cpp
    test_pricer.cpp
    test_greeks.cpp
    test_calibration.cpp
)

target_link_libraries(charlton_tests PRIVATE
    charlton::charlton
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(charlton_tests)

# Add test data directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_data.hpp 
               ${CMAKE_CURRENT_BINARY_DIR}/test_data.hpp COPYONLY)
