cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(openmp LANGUAGES CXX)

find_package(ntc-cmake REQUIRED)
include(ntc-dev-build)

find_package(my-benchmark REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE my::benchmark)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

ntc_target(${PROJECT_NAME})
