# NGLab HFT Native Loop (Tier 1 / Hot Path) — module-level C++ env file.
cmake_minimum_required(VERSION 3.20)

project(nglab_hft LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(NGLAB_BUILD_TESTS "Build HFT unit tests" ON)
option(NGLAB_BUILD_BENCH "Build HFT micro-benchmarks" OFF)
option(NGLAB_NATIVE_ARCH "Optimise for the build host (-march=native)" ON)

# --- Core DOD + shared-memory library -------------------------------------
add_library(nglab_hft
    src/order_book.cpp
    src/shm_metrics.cpp
)
target_include_directories(nglab_hft PUBLIC include)
target_compile_features(nglab_hft PUBLIC cxx_std_20)

# Hot-path optimisation on Release builds.
target_compile_options(nglab_hft PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<AND:$<CONFIG:Release>,$<BOOL:${NGLAB_NATIVE_ARCH}>>:-march=native>
)

# POSIX shared memory (shm_open) needs librt on Linux.
if(UNIX AND NOT APPLE)
    target_link_libraries(nglab_hft PUBLIC rt)
endif()

# --- Daemon executable ------------------------------------------------------
add_executable(hft_daemon src/main.cpp)
target_link_libraries(hft_daemon PRIVATE nglab_hft)

# --- Tests (self-contained; no external framework required) -----------------
if(NGLAB_BUILD_TESTS)
    enable_testing()
    add_executable(hft_tests tests/test_hft.cpp)
    target_link_libraries(hft_tests PRIVATE nglab_hft)
    add_test(NAME hft_tests COMMAND hft_tests)
endif()

# --- Optional micro-benchmarks ---------------------------------------------
if(NGLAB_BUILD_BENCH)
    add_executable(hft_bench tests/bench_hft.cpp)
    target_link_libraries(hft_bench PRIVATE nglab_hft)
endif()
