cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(parallel-computing)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/ntc-cmake/ntc-cmake")

# Include ntc-dev-build to disable find_package(ntc-cmake)
# in subprojects and establish default for BUILD_SHARED_LIBS.
include(ntc-dev-build)

option(PC_BUILD_INTRINSICS                "Build intrinsics (requires Intel compiler)" OFF)
option(PC_BUILD_OPENMP                    "Build openmp"                                ON)
option(PC_BUILD_MPI_DOT_PRODUCT           "Build mpi-dot-product"                       ON)
option(PC_BUILD_MPI_PI_CALCULATION        "Build mpi-pi-calculation"                    ON)
option(PC_BUILD_BOOST_MPI_PI_CALCULATION  "Build boost-mpi-pi-calculation"              ON)
option(PC_BUILD_CUDA_DOT_PRODUCT          "Build cuda-dot-product"                      ON)
option(PC_MPI_USE_MPICH                   "Use MPICH instead of OpenMPI"                ON)
option(PC_MPI_USE_LIBPMI                  "Link with libpmi"                            ON)

add_subdirectory(src/tools)

if(PC_BUILD_INTRINSICS)
    add_subdirectory(src/intrinsics)
endif()

if(PC_BUILD_OPENMP)
    add_subdirectory(src/openmp)
endif()

if(PC_BUILD_MPI_DOT_PRODUCT)
    add_subdirectory(src/mpi-dot-product)
endif()

if(PC_BUILD_MPI_PI_CALCULATION)
    add_subdirectory(src/mpi-pi-calculation)
endif()

if(PC_BUILD_BOOST_MPI_PI_CALCULATION)
    add_subdirectory(src/boost-mpi-pi-calculation)
endif()

if(PC_BUILD_CUDA_DOT_PRODUCT)
    add_subdirectory(src/cuda-dot-product)
endif()

include(FeatureSummary)
feature_summary(WHAT ALL)
