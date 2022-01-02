#!/bin/bash

self_dir=`dirname "$0"`

module unload INTEL
cmake \
    -S $self_dir \
    -B $self_dir/build-g++ \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=g++ \
    -D CMAKE_CXX_FLAGS="-march=native" \
    -D MPI_C_COMPILER=mpicc \
    -D MPI_CXX_COMPILER=mpic++
cmake --build $self_dir/build-g++ --target all

module load INTEL
cmake \
    -S $self_dir \
    -B $self_dir/build-icpc \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=icpc \
    -D CMAKE_CXX_FLAGS="-march=native" \
    -D MPI_C_COMPILER=mpiicc \
    -D MPI_CXX_COMPILER=mpiicpc \
    -D PC_BUILD_INTRINSICS=ON \
    -D PC_BUILD_BOOST_MPI_PI_CALCULATION=OFF
cmake --build $self_dir/build-icpc --target all

module unload INTEL
