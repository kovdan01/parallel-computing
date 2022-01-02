cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(intrinsics LANGUAGES CXX)

find_package(ntc-cmake REQUIRED)
include(ntc-dev-build)

if(NOT(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"))
    message(WARNING "Compiler is not ICPC or Intel LLVM, you might face problems when compiling")
endif()

find_package(my-benchmark REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE my::benchmark)

ntc_target(${PROJECT_NAME})
