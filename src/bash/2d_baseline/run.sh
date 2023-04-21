#!/bin/bash

# GPU Experiments

# MAWI strong scaling
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 128
sbatch spmm_gpu_2d_bench_64.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 128
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 128

# Will Probably OOM
sbatch spmm_gpu_2d_bench_16.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 64
# Will Probably OOM
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 64
sbatch spmm_gpu_2d_bench_64.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 64
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 64

sbatch spmm_gpu_2d_bench_8.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32
sbatch spmm_gpu_2d_bench_16.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32
sbatch spmm_gpu_2d_bench_64.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32

sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/webbase-2001/webbase-2001_A.npz $1 32
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/webbase-2001/webbase-2001_A.npz $1 64
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/webbase-2001/webbase-2001_A.npz $1 128

sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/kmer_A2a/kmer_A2a_A.npz $1 32
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/kmer_A2a/kmer_A2a_A.npz $1 64
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/kmer_A2a/kmer_A2a_A.npz $1 128

sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/kmer_V2a/kmer_V2a_A.npz $1 32
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/kmer_V2a/kmer_V2a_A.npz $1 64
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/kmer_V2a/kmer_V2a_A.npz $1 128

# MAWI WEAK SCALING
sbatch spmm_gpu_2d_bench_8.sh ../../../datasets/mawi_201512012345/mawi_201512012345_A.npz $1 32
sbatch spmm_gpu_2d_bench_16.sh ../../../datasets/mawi_201512020000/mawi_201512020000_A.npz $1 32
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32
sbatch spmm_gpu_2d_bench_64.sh ../../../datasets/mawi_201512020130/mawi_201512020130_A.npz $1 32
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32

sbatch spmm_gpu_2d_bench_8.sh ../../../datasets/mawi_201512012345/mawi_201512012345_A.npz $1 64
sbatch spmm_gpu_2d_bench_16.sh ../../../datasets/mawi_201512020000/mawi_201512020000_A.npz $1 64
sbatch spmm_gpu_2d_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 32
sbatch spmm_gpu_2d_bench_64.sh ../../../datasets/mawi_201512020130/mawi_201512020130_A.npz $1 64
sbatch spmm_gpu_2d_bench_128.sh ../../../datasets/mawi_201512020030/mawi_201512020030_A.npz $1 64