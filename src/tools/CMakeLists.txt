cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(tools LANGUAGES CXX)

find_package(ntc-cmake REQUIRED)
include(ntc-dev-build)

add_library(my-benchmark INTERFACE include/benchmark.hpp)
target_compile_features(my-benchmark INTERFACE cxx_std_20)

ntc_target(my-benchmark
    ALIAS_NAME my::benchmark
)

if(PC_BUILD_MPI_DOT_PRODUCT OR PC_BUILD_MPI_PI_CALCULATION)
    find_package(PkgConfig REQUIRED)

    if(PC_MPI_USE_MPICH)
        pkg_check_modules(mpi REQUIRED IMPORTED_TARGET mpich)
    else()
        pkg_check_modules(mpi REQUIRED IMPORTED_TARGET ompi-cxx)
    endif()

    if(PC_MPI_USE_LIBPMI)
        list(APPEND CMAKE_SHARED_LINKER_FLAGS "-lpmi")
    endif()

    add_library(my-mpi SHARED include/mpi.hpp src/mpi.cpp)
    target_compile_features(my-mpi PRIVATE cxx_std_20)
    target_link_libraries(my-mpi PUBLIC PkgConfig::mpi)

    ntc_target(my-mpi
        ALIAS_NAME my::mpi
        HEADER_PREFIX my/mpi/
    )
endif()

if(PC_BUILD_MPI_PI_CALCULATION OR PC_BUILD_BOOST_MPI_PI_CALCULATION)
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(gmpxx REQUIRED IMPORTED_TARGET gmpxx)

    add_library(my-pi-helpers SHARED include/pi_helpers.hpp src/pi_helpers.cpp)
    target_compile_features(my-pi-helpers PRIVATE cxx_std_20)
    target_link_libraries(my-pi-helpers PUBLIC PkgConfig::gmpxx)

    ntc_target(my-pi-helpers
        ALIAS_NAME my::pi-helpers
        HEADER_PREFIX my/pi_helpers/
    )
endif()
