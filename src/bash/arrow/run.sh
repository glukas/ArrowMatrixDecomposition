#!/bin/bash


#24 nodes
sbatch spmm_gpu_bench_32.sh ../../../datasets/kmer_V2a/kmer_V2a $1 128
#28 nodes
sbatch spmm_gpu_bench_32.sh ../../../datasets/kmer_U1a/kmer_U1a $1 128
#56 nodes
sbatch spmm_gpu_bench_64.sh ../../../datasets/kmer_P1a/kmer_P1a $1 128

sbatch spmm_gpu_bench_32.sh ../../../datasets/kmer_V2a/kmer_V2a $1 64
sbatch spmm_gpu_bench_64.sh ../../../datasets/webbase-2001/webbase-2001 $1 64
sbatch spmm_gpu_bench_128.sh ../../../datasets/kmer_A2a/kmer_A2a $1 64

sbatch spmm_gpu_bench_32.sh ../../../datasets/kmer_V2a/kmer_V2a $1 32
sbatch spmm_gpu_bench_64.sh ../../../datasets/webbase-2001/webbase-2001 $1 32
sbatch spmm_gpu_bench_128.sh ../../../datasets/kmer_A2a/kmer_A2a $1 32

sbatch spmm_gpu_bench_8.sh ../../../datasets/mawi_201512012345/mawi_201512012345 $1 128
sbatch spmm_gpu_bench_16.sh ../../../datasets/mawi_201512020000/mawi_201512020000 $1 128
sbatch spmm_gpu_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030 $1 128
sbatch spmm_gpu_bench_64.sh ../../../datasets/mawi_201512020130/mawi_201512020130 $1 128
sbatch spmm_gpu_bench_128.sh ../../../datasets/mawi_201512020330/mawi_201512020330 $1 128

sbatch spmm_gpu_bench_8.sh ../../../datasets/mawi_201512012345/mawi_201512012345 $1 32
sbatch spmm_gpu_bench_16.sh ../../../datasets/mawi_201512020000/mawi_201512020000 $1 32
sbatch spmm_gpu_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030 $1 32
sbatch spmm_gpu_bench_64.sh ../../../datasets/mawi_201512020130/mawi_201512020130 $1 32
sbatch spmm_gpu_bench_128.sh ../../../datasets/mawi_201512020330/mawi_201512020330 $1 32

sbatch spmm_gpu_bench_8.sh ../../../datasets/mawi_201512012345/mawi_201512012345 $1 64
sbatch spmm_gpu_bench_16.sh ../../../datasets/mawi_201512020000/mawi_201512020000 $1 64
sbatch spmm_gpu_bench_32.sh ../../../datasets/mawi_201512020030/mawi_201512020030 $1 64
sbatch spmm_gpu_bench_64.sh ../../../datasets/mawi_201512020130/mawi_201512020130 $1 64
sbatch spmm_gpu_bench_128.sh ../../../datasets/mawi_201512020330/mawi_201512020330 $1 64


