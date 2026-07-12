# NGLab — top-level CMake entry point (repo-root C++ env file).
#
# The HFT Native Loop (Tier 1 / Hot Path) lives in cpp/. This top-level file
# exists so the C++ tree can be configured from the repository root; the
# module-level env file is cpp/CMakeLists.txt.
cmake_minimum_required(VERSION 3.20)

project(nglab LANGUAGES CXX)

add_subdirectory(cpp)
