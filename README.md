# Parallel computing

## Contents

* [Parallel computing](#parallel-computing)
   * [Overview](#overview)
   * [Build &amp; run](#build--run)
   * [intrinsics - measuring CPU performance](#intrinsics---measuring-cpu-performance)
      * [Description](#description)
      * [Benchmarks (home)](#benchmarks-home)
      * [Benchmarks (HPC)](#benchmarks-hpc)
      * [Benchmarks (home)](#benchmarks-home-1)
   * [openmp - vectorizing and paralleling numeric integration](#openmp---vectorizing-and-paralleling-numeric-integration)
      * [Description](#description-1)
      * [Benchmarks (home)](#benchmarks-home-2)
      * [Benchmarks (HPC)](#benchmarks-hpc-1)
   * [mpi-dot-product - paralleling dot product calculation using MPI](#mpi-dot-product---paralleling-dot-product-calculation-using-mpi)
      * [Description](#description-2)
      * [Benchmarks (HPC)](#benchmarks-hpc-2)
   * [mpi-pi-calculation - paralleling pi calculation using MPI (via raw API)](#mpi-pi-calculation---paralleling-pi-calculation-using-mpi-via-raw-api)
      * [Description](#description-3)
      * [Benchmarks (HPC)](#benchmarks-hpc-3)
      * [Results (HPC)](#results-hpc)
   * [boost-mpi-pi-calculation - paralleling pi calculation using MPI (via Boost.MPI)](#boost-mpi-pi-calculation---paralleling-pi-calculation-using-mpi-via-boostmpi)
      * [Description](#description-4)
      * [Benchmarks (HPC)](#benchmarks-hpc-4)
   * [cuda-dot-product - paralleling dot product calculation using MPI](#cuda-dot-product---paralleling-dot-product-calculation-using-mpi)
      * [Description](#description-5)
      * [Benchmarks (HPC)](#benchmarks-hpc-5)

## Overview

This repository contains a set of samples that show basic usage of intrinsics, OpenMP, MPI (via both raw API and Boost.MPI) and NVIDIA CUDA.

Benchmarks were performed using two hardware sets.

1. "Home" - my own laptop with Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz. During benchmarks frequency was locked to 2.20GHz and turbo-boost was disabled. Tests were run on Arch Linux virtual machine with 6 CPU cores and 12GB RAM.
2. "HPC" - [cHARISMa](https://hpc.hse.ru/hardware/hpc-cluster) supercomputer cluster of Higher School of Economics. Configuration details are described in each sample description.

## Build & run

If you are using cHARISMa, you can just run `./build.sh` and `./run.sh`. The first script will configure and compile the samples with `g++` and `icpc`.

If you do not use cHARISMa, please use `build.sh` and `run.sh` scripts as a reference to configure builds and run benchmarks manually.

Details that you should take into account:

- For mpi-dot-product and mpi-pi-calculation samples OpenMPI and MPICH are supported as possible MPI implementations. To choose MPI backend, use `PC_MPI_USE_MPICH` cmake option (default `ON`). If you wish to use libpmi as a 3rd-party process manager, use `PC_MPI_USE_LIBPMI` cmake option (default `ON`).
- Note that boost-mpi-pi-calculation sample will use the MPI backend that Boost.MPI was compiled with.

## intrinsics - measuring CPU performance

### Description

Build of this sample is enabled with `PC_BUILD_INTRINSICS` cmake option (default `OFF`). Note that this sample can be built only with Intel ICPC or Intel LLVM-based compiler.

This sample compares scalar and vector division of two 64-bit signed integers.

Vector operation is `_mm256_div_epi64` from AVX extension (not available as intrinsic with regular gcc or clang, defined in libsvml, which is installed among with Intel ICC compiler). Performs four 64-bit signed integer divisions as one operation.

Ticks are counted with `rdtsc` instruction.

### Benchmarks (home)

<details>
<summary><b>Intel ICPC</b></summary>

```
+--------------------+--------------+-----------+
|      operation     | ticks / iter | ns / iter |
+--------------------+--------------+-----------+
| independent scalar |      28.4806 |   12.8986 |
+--------------------+--------------+-----------+
|  dependent scalar  |      42.3657 |   19.1871 |
+--------------------+--------------+-----------+
| independent vector |      22.7078 |   10.2842 |
+--------------------+--------------+-----------+
|  dependent vector  |      32.4164 |   14.6812 |
+--------------------+--------------+-----------+
```
</details>

<details>
<summary><b>Intel LLVM-based compiler</b></summary>

```
+--------------------+--------------+-----------+
|      operation     | ticks / iter | ns / iter |
+--------------------+--------------+-----------+
| independent scalar |      1.08797 |   0.49274 |
+--------------------+--------------+-----------+
|  dependent scalar  |      25.4767 |   11.5383 |
+--------------------+--------------+-----------+
| independent vector |      24.1993 |   10.9597 |
+--------------------+--------------+-----------+
|  dependent vector  |       32.294 |   14.6258 |
+--------------------+--------------+-----------+
```
</details>

<details>
<summary><b>GNU GCC</b></summary>

Unavailable due to lack of `_mm256_div_epi64`
</details>

### Benchmarks (HPC)

### Benchmarks (home)

<details>
<summary><b>Intel ICPC</b></summary>

```
+--------------------+--------------+-----------+
|      operation     | ticks / iter | ns / iter |
+--------------------+--------------+-----------+
| independent scalar |     0.836742 |  0.279568 |
+--------------------+--------------+-----------+
|  dependent scalar  |      32.9652 |   11.0141 |
+--------------------+--------------+-----------+
| independent vector |      17.3629 |   5.80118 |
+--------------------+--------------+-----------+
|  dependent vector  |      25.5591 |   8.53965 |
+--------------------+--------------+-----------+
```
</details>

<details>
<summary><b>Intel LLVM-based compiler</b></summary>

Unavailable due to compilation problems.
</details>

<details>
<summary><b>GNU GCC</b></summary>

Unavailable due to lack of `_mm256_div_epi64`
</details>

## openmp - vectorizing and paralleling numeric integration

### Description

Build of this sample is enabled with `PC_BUILD_OPENMP` cmake option (default `ON`).

This sample compares different variants of numeric integration.

- "Dummy" single-thread calculation.
- Calculation vectorized with OpenMP
- Calculation parallelized with OpenMP using different threads count.

Function to be integrated is defined as a table of its values in points *from*, *from + dx*, *from + 2dx*, ..., *to*. Here *dx = (to - from) / n*, where *n* is number of segments to split *\[from; to\]* segment.

### Benchmarks (home)

<details>
<summary><b>Intel ICPC</b></summary>

```
+----------------------------+----------------+---------------+
|          operation         |  ticks / iter  |   ns / iter   |
+----------------------------+----------------+---------------+
|      integrate dummy       |    10845158.04 |    4932022.54 |
+----------------------------+----------------+---------------+
|     integrate omp simd     |    10828490.51 |    4924846.01 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 1  |    18179023.56 |    8253084.30 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 2  |     9589733.79 |    4361244.32 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 3  |     6985018.73 |    3181948.56 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 4  |     6216703.93 |    2834112.84 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 5  |     5939075.67 |    2708988.63 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 6  |     6061977.57 |    2766919.99 |
+----------------------------+----------------+---------------+
```
</details>

<details>
<summary><b>Intel LLVM-based compiler</b></summary>

Unavailable due to problems with OpenMP.
</details>

<details>
<summary><b>GNU GCC</b></summary>

```
+----------------------------+----------------+---------------+
|          operation         |  ticks / iter  |   ns / iter   |
+----------------------------+----------------+---------------+
|      integrate dummy       |    34339403.18 |   15572950.02 |
+----------------------------+----------------+---------------+
|     integrate omp simd     |    10838969.24 |    4929447.89 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 1  |    34272412.32 |   15540118.80 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 2  |    17236436.64 |    7825217.52 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 3  |    11596463.97 |    5271108.65 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 4  |     8933118.66 |    4065504.48 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 5  |     7384450.33 |    3364648.08 |
+----------------------------+----------------+---------------+
|  integrate omp parallel 6  |     6384912.18 |    2913110.82 |
+----------------------------+----------------+---------------+
```
</details>

### Benchmarks (HPC)

<details>
<summary><b>Intel ICPC</b></summary>

```
+-----------------------------+----------------+---------------+
|           operation         |  ticks / iter  |   ns / iter   |
+-----------------------------+----------------+---------------+
|       integrate dummy       |     8380447.63 |    4000905.42 |
+-----------------------------+----------------+---------------+
|      integrate omp simd     |     8378524.01 |    3999958.85 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  1  |     8082323.28 |    3858333.85 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  2  |     4121121.59 |    1967504.03 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  3  |     2867182.71 |    1368938.72 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  4  |     2197377.26 |    1049168.72 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  5  |     1815192.17 |     866714.07 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  6  |     1556680.23 |     743283.88 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  7  |     1340370.51 |     640024.72 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  8  |     1173647.77 |     560437.74 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  9  |     1074443.22 |     513068.29 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 10  |      985159.07 |     470436.04 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 11  |      938532.49 |     448196.90 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 12  |      843983.29 |     403082.39 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 13  |      798041.16 |     381183.36 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 14  |      780807.82 |     372961.97 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 15  |      791373.72 |     377988.86 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 16  |      753182.68 |     359777.49 |
+-----------------------------+----------------+---------------+
```
</details>

<details>
<summary><b>Intel LLVM-based compiler</b></summary>

Unavailable due to problems with OpenMP.
</details>

<details>
<summary><b>GNU GCC</b></summary>

```
+-----------------------------+----------------+---------------+
|           operation         |  ticks / iter  |   ns / iter   |
+-----------------------------+----------------+---------------+
|       integrate dummy       |    26863360.96 |    8975833.81 |
+-----------------------------+----------------+---------------+
|      integrate omp simd     |    10027036.87 |    3350557.58 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  1  |    26908147.74 |    8990722.14 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  2  |    13592098.33 |    4541687.56 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  3  |     9211627.24 |    3078072.28 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  4  |     6893566.89 |    2303609.22 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  5  |     6572301.63 |    2196263.87 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  6  |     5448113.15 |    1820750.16 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  7  |     4662692.30 |    1558282.33 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  8  |     4111082.73 |    1373994.26 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel  9  |     3719048.50 |    1242970.88 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 10  |     3346800.59 |    1118619.17 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 11  |     3115620.30 |    1041420.57 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 12  |     2836617.38 |     948196.40 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 13  |     2653445.09 |     886985.73 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 14  |     2814391.34 |     940732.60 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 15  |     3687116.86 |    1232270.85 |
+-----------------------------+----------------+---------------+
|  integrate omp parallel 16  |     2232142.90 |     746254.54 |
+-----------------------------+----------------+---------------+
```
</details>

## mpi-dot-product - paralleling dot product calculation using MPI

### Description

Build of this sample is enabled with `PC_BUILD_MPI_DOT_PRODUCT` cmake option (default `ON`).

This sample compares "dummy" single-thread variant of dot product calculation versus an MPI-paralleled one. TLDR: dot product is a bad task to be paralleled with MPI because copying data between workers is more expensive than just calculating everything within one worker.

### Benchmarks (HPC)

Benchmarks were run with 16 MPI workers on 4 nodes (4 workers per node):

```
srun \
    --ntasks=16 \
    --nodes=4 \
    --tasks-per-node=4 \
    --cpus-per-task=1 \
    ./mpi-dot-product
```

**Intel ICPC**

```
Regular time:     10817006.35
    MPI time:    864159048.03
```

**GNU GCC**

```
Regular time:     12274970.40
    MPI time:    863880223.23
```

## mpi-pi-calculation - paralleling pi calculation using MPI (via raw API)

### Description

Build of this sample is enabled with `PC_BUILD_MPI_PI_CALCULATION` cmake option (default `ON`).

This sample compares "dummy" single-thread variant of pi calculation versus an MPI-paralleled one. MPI is used via its native API and tiny self-implemented wrappers.

Two formulas for pi calculation are implemented: Leibniz's and Bellard's series.

GMP library is used to work with high-precision floating point numbers.

### Benchmarks (HPC)

Benchmarks were run with 200 MPI workers on 20 nodes (10 workers per node). Leibniz series was used.

```
srun \
    --ntasks=200 \
    --nodes=20 \
    --ntasks-per-node=10 \
    --cpus-per-task=1 \
    ./mpi-pi-calculation
```

**Intel ICPC**

```
Regular time:   4411034873.16
    MPI time:     23207346.97
```

**GNU GCC**

```
Regular time:   4409928111.07
    MPI time:     23248179.89
```

### Results (HPC)

We tried to compute as many pi digits as possible in a reasonable period of time (several hours). 200 MPI workers were used.

**Leibniz's series:**

- Summands count: 2^45.
- GMP floating point number precision: 2^7 bits.
- Execution time: 3 hours 8 minutes.
- Correct pi digits: 14 (including 3 before decimal dot).

The series converges **extremely** slow.

**Bellard's series:**

- Summands count: 2^22.
- GMP floating point number precision: 2^26 bits.
- Execution time: 9 hours 4 minutes.
- Correct pi digits: 12'626'121 (including 3 before decimal dot).

## boost-mpi-pi-calculation - paralleling pi calculation using MPI (via Boost.MPI)

### Description

Build of this sample is enabled with `PC_BUILD_BOOST_MPI_PI_CALCULATION` cmake option (default `ON`).

This sample is similar to mpi-pi-calculation one, but Boost.MPI is used instead of raw MPI API.

### Benchmarks (HPC)

Benchmarks were run with 200 MPI workers on 20 nodes (10 workers per node):

```
srun \
    --ntasks=200 \
    --nodes=20 \
    --ntasks-per-node=10 \
    --cpus-per-task=1 \
    ./boost-mpi-pi-calculation
```

**Intel ICPC**

Unavailable due to compilation problems.

**GNU GCC**

```
Regular time:   4420494915.14
    MPI time:     34435969.13
```

## cuda-dot-product - paralleling dot product calculation using MPI

### Description

Build of this sample is enabled with `PC_BUILD_CUDA_DOT_PRODUCT` cmake option (default `ON`).

This sample compares the following variants of dot product computation:

- "Dummy" single-thread CPU calculation.
- Calculation paralleled with CUDA (`h`). Source vectors are initially stored in host memory, allocating device memory and copying vectors from host to device is included in time measurement.
- Calculation paralleled with CUDA (`d1`). Source vectors are initially stored in device memory, allocating intermediate device memory regions is included in time measurement.
- Calculation paralleled with CUDA (`d2`). Source vectors are initially stored in device memory, all necessary device memory regions (including intemediate ones) are allocated one time and this allocation is not included in time measurement.

Calculations on CUDA are performed in 2 steps:

1. Compute multiplication results on each vector position in parallel.
2. Sumarize these results using parallel reduction algorithm. In this sample `reduce3` kernel from official cuda-samples-11.5 is used.

### Benchmarks (HPC)

Benchmarks were run with 1 GPU device:

```
srun \
    --gpus=1 \
    ./cuda-dot-product
```

**Intel ICPC**

```
     CPU time:     14661979.86
GPU time  (h):     24036057.18
GPU time (d1):      3962556.10
GPU time (d2):       381491.84
```

**GNU GCC**

```
     CPU time:     12800219.12
GPU time  (h):     19064239.50
GPU time (d1):      3210658.34
GPU time (d2):       391104.56
```
