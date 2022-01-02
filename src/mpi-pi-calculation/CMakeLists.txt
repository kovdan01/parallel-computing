cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(mpi-pi-calculation LANGUAGES CXX)

find_package(ntc-cmake REQUIRED)
include(ntc-dev-build)

find_package(PkgConfig REQUIRED)

if(PC_MPI_USE_MPICH)
    pkg_check_modules(mpi REQUIRED IMPORTED_TARGET mpich)
else()
    pkg_check_modules(mpi REQUIRED IMPORTED_TARGET ompi-cxx)
endif()

if(PC_MPI_USE_LIBPMI)
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-lpmi")
endif()

pkg_check_modules(gmpxx REQUIRED IMPORTED_TARGET gmpxx)

find_package(my-benchmark REQUIRED)
find_package(my-mpi REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

target_link_libraries(${PROJECT_NAME} PRIVATE my::benchmark)
target_link_libraries(${PROJECT_NAME} PRIVATE my::mpi)
target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::mpi)
target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::gmpxx)
