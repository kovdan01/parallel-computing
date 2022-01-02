cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cuda-dot-product LANGUAGES CXX CUDA)

find_package(ntc-cmake REQUIRED)
include(ntc-dev-build)

find_package(my-benchmark REQUIRED)

find_package(cuda-api-wrappers 0.4.4 REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/kernel.cu include/interfaces.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE my::benchmark)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda-api-wrappers::runtime-api)

ntc_target(${PROJECT_NAME})

# Build of .cu file fails with -pedantic-errors, manually disable that
get_target_property(target_options ${PROJECT_NAME} COMPILE_OPTIONS)
list(REMOVE_ITEM target_options "-pedantic-errors")
set_property(TARGET ${PROJECT_NAME} PROPERTY COMPILE_OPTIONS ${target_options})
