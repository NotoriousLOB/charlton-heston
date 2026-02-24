# CHARLTON - Conformal Hyperbolic Accelerated Rough Levy Transform for Option Numerics
# C99 header-only library with C++17 test/benchmark harness

cmake_minimum_required(VERSION 3.16)
project(charlton
    VERSION 2.0.0
    DESCRIPTION "High-performance C99 Rough Heston pricing and calibration library"
    LANGUAGES C CXX
)

# C99 for library code, C++17 for GTest/Google Benchmark
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(CHARLTON_BUILD_TESTS "Build tests" ON)
option(CHARLTON_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CHARLTON_BUILD_EXAMPLES "Build examples" ON)
option(CHARLTON_BUILD_PYTHON "Build Python shared library for ctypes" ON)
option(CHARLTON_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(CHARLTON_ENABLE_SIMD "Enable SIMD optimizations" ON)

include(FetchContent)

# ============================================================================
# Dependencies
# ============================================================================

# Notorious-FFT (header-only, C API)
FetchContent_Declare(
    notorious_fft
    GIT_REPOSITORY https://github.com/NotoriousLOB/Notorious-FFT.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_Populate(notorious_fft)

# Google Test
if(CHARLTON_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Google Benchmark
if(CHARLTON_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
        GIT_SHALLOW TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    add_compile_options(-Wall -Wextra -O3)
    # -march=native for local builds; remove for portability
    if(NOT CMAKE_CROSSCOMPILING)
        add_compile_options(-march=native)
    endif()

    if(CHARLTON_ENABLE_SIMD)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        check_c_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

        if(COMPILER_SUPPORTS_AVX512 AND COMPILER_SUPPORTS_AVX512DQ)
            add_compile_options(-mavx512f -mavx512dq)
            message(STATUS "AVX-512 enabled")
        elseif(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2 -mfma)
            message(STATUS "AVX2 enabled")
        endif()

        # ARM NEON (auto-detected on aarch64, flag needed on armv7)
        check_c_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON_FLAG)
        if(COMPILER_SUPPORTS_NEON_FLAG)
            add_compile_options(-mfpu=neon)
            message(STATUS "ARM NEON enabled (via flag)")
        endif()
    endif()

    if(CHARLTON_ENABLE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_C_FOUND)
            message(STATUS "OpenMP enabled")
        else()
            message(WARNING "OpenMP not found")
        endif()
    endif()

elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /O2)
    if(CHARLTON_ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
    endif()
    if(CHARLTON_ENABLE_OPENMP)
        add_compile_options(/openmp)
    endif()
endif()

# ============================================================================
# Static Library (C99)
# ============================================================================

add_library(charlton STATIC src/charlton_impl.c)
add_library(charlton::charlton ALIAS charlton)

target_include_directories(charlton PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${notorious_fft_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(UNIX)
    target_link_libraries(charlton PUBLIC m)
endif()

if(CHARLTON_ENABLE_OPENMP AND OpenMP_C_FOUND)
    target_link_libraries(charlton PUBLIC OpenMP::OpenMP_C)
endif()

# ============================================================================
# Shared Library (for Python ctypes)
# ============================================================================

if(CHARLTON_BUILD_PYTHON)
    add_library(charlton_shared SHARED src/charlton_shlib.c)

    target_include_directories(charlton_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${notorious_fft_SOURCE_DIR}/include>
    )

    if(UNIX)
        target_link_libraries(charlton_shared PUBLIC m)
    endif()
    if(CHARLTON_ENABLE_OPENMP AND OpenMP_C_FOUND)
        target_link_libraries(charlton_shared PUBLIC OpenMP::OpenMP_C)
    endif()

    set_target_properties(charlton_shared PROPERTIES
        C_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

# ============================================================================
# Subdirectories
# ============================================================================

if(CHARLTON_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CHARLTON_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CHARLTON_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS charlton
    EXPORT charlton-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT charlton-targets
    FILE charlton-targets.cmake
    NAMESPACE charlton::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/charlton
)

install(FILES include/charlton.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/charlton-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/charlton-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/charlton
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/charlton-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/charlton-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/charlton-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/charlton
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "CHARLTON Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Standard: ${CMAKE_C_STANDARD}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD} (tests/benchmarks only)")
message(STATUS "  Tests: ${CHARLTON_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${CHARLTON_BUILD_BENCHMARKS}")
message(STATUS "  Examples: ${CHARLTON_BUILD_EXAMPLES}")
message(STATUS "  Python ctypes: ${CHARLTON_BUILD_PYTHON}")
message(STATUS "  OpenMP: ${CHARLTON_ENABLE_OPENMP}")
message(STATUS "  SIMD: ${CHARLTON_ENABLE_SIMD}")
message(STATUS "  FFT: Notorious-FFT (auto-fetched)")
message(STATUS "")
