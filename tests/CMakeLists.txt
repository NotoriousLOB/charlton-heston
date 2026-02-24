# CHARLTON Tests (GTest C++ calling C API via extern "C")

add_executable(charlton_tests
    test_main.cpp
    test_abm_solver.cpp
    test_pricer.cpp
    test_greeks.cpp
    test_calibration.cpp
    test_american.cpp
)

target_link_libraries(charlton_tests PRIVATE
    charlton::charlton
    GTest::gtest_main
)

target_include_directories(charlton_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(charlton_tests)
