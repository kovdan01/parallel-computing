#!/bin/bash

#SBATCH --job-name=cuda.sbatch            # Название задачи
#SBATCH --error=cuda.err                  # Файл для вывода ошибок
#SBATCH --output=cuda.log                 # Файл для вывода результатов
#SBATCH --time=00:05:00                   # Максимальное время выполнения
#SBATCH --ntasks=1                        # Количество MPI процессов
#SBATCH --ntasks-per-node=1               # Количество процессов на каждом узле
#SBATCH --nodes=1                         # Требуемое кол-во узлов
#SBATCH --gpus=2                          # Требуемое кол-во GPU
#SBATCH --cpus-per-task=1                 #

srun /home/dyukovalev/Projects/cuda-dot-product/build/cuda-dot-product
