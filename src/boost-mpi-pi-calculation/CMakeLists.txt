cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(boost-mpi-pi-calculation LANGUAGES CXX)

find_package(ntc-cmake REQUIRED)
include(ntc-dev-build)

find_package(MPI REQUIRED)

if(PC_MPI_USE_LIBPMI)
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-lpmi")
endif()

set(Boost_USE_STATIC_LIBS OFF)

find_package(
    Boost 1.68 REQUIRED
    COMPONENTS
    mpi
)

find_package(PkgConfig REQUIRED)

pkg_check_modules(gmpxx REQUIRED IMPORTED_TARGET gmpxx)

find_package(my-benchmark REQUIRED)
find_package(my-pi-helpers REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

target_link_libraries(${PROJECT_NAME} PRIVATE my::benchmark)
target_link_libraries(${PROJECT_NAME} PRIVATE my::pi-helpers)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::mpi)
target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::gmpxx)
