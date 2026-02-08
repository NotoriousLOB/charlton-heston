# CHARLTON - Conformal Hyperbolic Accelerated Rough Lévy Transform for Option Numerics
# CMake Build System

cmake_minimum_required(VERSION 3.16)
project(charlton 
    VERSION 1.0.0
    DESCRIPTION "High-performance Rough Heston pricing and calibration library"
    LANGUAGES CXX C
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(CHARLTON_BUILD_TESTS "Build tests" ON)
option(CHARLTON_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CHARLTON_BUILD_EXAMPLES "Build examples" ON)
option(CHARLTON_BUILD_PYTHON "Build Python bindings" ON)
option(CHARLTON_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(CHARLTON_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(CHARLTON_USE_MKL "Use Intel MKL for FFT" OFF)

# Include FetchContent for dependencies
include(FetchContent)

# ============================================================================
# Dependencies
# ============================================================================

# FFTW3 (or MKL)
if(CHARLTON_USE_MKL)
    find_package(MKL REQUIRED)
    add_compile_definitions(USE_MKL)
else()
    # Use system FFTW3 (fallback to FetchContent can be added later for systems without it)
    find_library(FFTW3_LIB fftw3 PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
    find_path(FFTW3_INCLUDE fftw3.h PATHS /usr/include /usr/local/include)

    if(FFTW3_LIB AND FFTW3_INCLUDE)
        message(STATUS "Found system FFTW3: ${FFTW3_LIB}")
        include_directories(${FFTW3_INCLUDE})
        add_library(fftw3 INTERFACE)
        target_link_libraries(fftw3 INTERFACE ${FFTW3_LIB})
        target_include_directories(fftw3 INTERFACE ${FFTW3_INCLUDE})
    else()
        message(WARNING "System FFTW3 not found. Install libfftw3-dev or enable CHARLTON_USE_MKL")
        message(STATUS "TODO: Add FetchContent fallback for systems without FFTW3")
        message(FATAL_ERROR "FFTW3 required but not found")
    endif()
endif()

# Google Test (for tests)
if(CHARLTON_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
endif()

# Google Benchmark (for benchmarks)
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

# Detect compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
    # Common flags for GCC/Clang/Intel
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O3 -march=native)
    
    if(CHARLTON_ENABLE_SIMD)
        # Enable SIMD intrinsics
        include(CheckCXXCompilerFlag)
        
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
        
        if(COMPILER_SUPPORTS_AVX512 AND COMPILER_SUPPORTS_AVX512DQ)
            add_compile_options(-mavx512f -mavx512dq)
            add_compile_definitions(HAS_AVX512)
            message(STATUS "AVX-512 enabled")
        elseif(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2 -mfma)
            add_compile_definitions(HAS_AVX2)
            message(STATUS "AVX2 enabled")
        endif()
        
        # ARM NEON
        check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
        if(COMPILER_SUPPORTS_NEON)
            add_compile_options(-mfpu=neon)
            add_compile_definitions(HAS_NEON)
            message(STATUS "ARM NEON enabled")
        endif()
    endif()
    
    if(CHARLTON_ENABLE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            add_compile_options(${OpenMP_CXX_FLAGS})
            add_link_options(${OpenMP_CXX_FLAGS})
            add_compile_definitions(_OPENMP)
            message(STATUS "OpenMP enabled")
        else()
            message(WARNING "OpenMP not found, parallelization disabled")
        endif()
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC flags
    add_compile_options(/W4 /O2)
    
    if(CHARLTON_ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(HAS_AVX2)
    endif()
    
    if(CHARLTON_ENABLE_OPENMP)
        add_compile_options(/openmp)
        add_compile_definitions(_OPENMP)
    endif()
endif()

# ============================================================================
# Library Target
# ============================================================================

# Header-only library interface
add_library(charlton INTERFACE)
add_library(charlton::charlton ALIAS charlton)

target_include_directories(charlton INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
if(CHARLTON_USE_MKL)
    target_link_libraries(charlton INTERFACE MKL::MKL)
else()
    # Link FFTW3 directly to avoid export issues
    target_link_libraries(charlton INTERFACE ${FFTW3_LIB})
    target_include_directories(charlton INTERFACE ${FFTW3_INCLUDE})
endif()

if(CHARLTON_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(charlton INTERFACE OpenMP::OpenMP_CXX)
endif()

# Link math library on Unix
if(UNIX)
    target_link_libraries(charlton INTERFACE m)
endif()

# ============================================================================
# Tests
# ============================================================================

if(CHARLTON_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(CHARLTON_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Examples
# ============================================================================

if(CHARLTON_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Python Bindings
# ============================================================================

if(CHARLTON_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS charlton
    EXPORT charlton-targets
)

install(EXPORT charlton-targets
    FILE charlton-targets.cmake
    NAMESPACE charlton::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/charlton
)

install(FILES include/charlton.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Create config file
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
# Print Configuration
# ============================================================================

message(STATUS "")
message(STATUS "CHARLTON Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests: ${CHARLTON_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${CHARLTON_BUILD_BENCHMARKS}")
message(STATUS "  Examples: ${CHARLTON_BUILD_EXAMPLES}")
message(STATUS "  Python Bindings: ${CHARLTON_BUILD_PYTHON}")
message(STATUS "  OpenMP: ${CHARLTON_ENABLE_OPENMP}")
message(STATUS "  SIMD: ${CHARLTON_ENABLE_SIMD}")
message(STATUS "  MKL: ${CHARLTON_USE_MKL}")
message(STATUS "")
