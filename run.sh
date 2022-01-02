#!/bin/bash

self_dir=`dirname "$0"`

run_with_compiler()
{
    echo "$1 openmp"
    srun \
        --cpus-per-task=16 \
        --nodes=1 \
        --ntasks=1 \
        $self_dir/build-$1/openmp/openmp

    echo "$1 mpi-dot-product"
    srun \
        --ntasks=16 \
        --nodes=4 \
        --tasks-per-node=4 \
        --cpus-per-task=1 \
        $self_dir/build-$1/mpi-dot-product/mpi-dot-product

    echo "$1 mpi-pi-calculation"
    srun \
        --ntasks=200 \
        --nodes=20 \
        --ntasks-per-node=10 \
        --cpus-per-task=1 \
        $self_dir/build-$1/mpi-pi-calculation/mpi-pi-calculation

    echo "$1 cuda-dot-product"
    srun \
        --gpus=1 \
        $self_dir/build-$1/cuda-dot-product/cuda-dot-product
}

run_with_compiler g++
echo "g++ boost-mpi-pi-calculation"
srun \
    --ntasks=200 \
    --nodes=20 \
    --ntasks-per-node=10 \
    --cpus-per-task=1 \
    $self_dir/build-g++/boost-mpi-pi-calculation/boost-mpi-pi-calculation

run_with_compiler icpc
echo "icpc intrinsics"
srun $self_dir/build-icpc/intrinsics/intrinsics
