#!/bin/bash

# 35+1 rows, 70 nodes
sbatch spmm_gpu_bench_70.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 2000000 128

sbatch spmm_gpu_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030 $1 1090000 128
# 23 rows, 45 nodes required
sbatch spmm_gpu_bench_45.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 3000000 128
# 18 rows, 35 nodes
sbatch spmm_gpu_bench_35.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 4000000 128
# 27 nodes
sbatch spmm_gpu_bench_23.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 5000000 128
# 12 rows, 23 nodes
sbatch spmm_gpu_bench_23.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 6000000 128


sbatch spmm_gpu_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030 $1 1090000 64

sbatch spmm_gpu_bench_70.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 2000000 64
# 23 rows, 45 nodes required
sbatch spmm_gpu_bench_45.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 3000000 64
# 18 rows, 35 nodes
sbatch spmm_gpu_bench_35.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 4000000 64
# 27 nodes
sbatch spmm_gpu_bench_27.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 5000000 64
# 12 rows, 23 nodes
sbatch spmm_gpu_bench_23.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 6000000 64

sbatch spmm_gpu_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030 $1 1090000 32
sbatch spmm_gpu_bench_70.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 2000000 32
# 23 rows, 45 nodes required
sbatch spmm_gpu_bench_45.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 3000000 32
# 18 rows, 35 nodes
sbatch spmm_gpu_bench_35.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 4000000 32
# 27 nodes
sbatch spmm_gpu_bench_27.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 5000000 32
# 12 rows, 23 nodes
sbatch spmm_gpu_bench_23.sh ../../../datasets/mawi_scale/mawi_201512020030/mawi_201512020030 $1 6000000 32