# Q2 – CUDA Tiled Matrix Multiplication (SGEMM)

This repository implements a tiled SGEMM kernel using CUDA shared memory and synchronization for data reuse.

## Build
```bash
nvcc -O3 -std=c++14 -arch=sm_70 q2_matmul_cuda.cu -o q2_matmul

## RUn
./q2_matmul                # Default: M=2048, K=1024, N=768
./q2_matmul 1024 1024 1024  # Custom M K N

## Output
GEMM: C[2048 x 768] = A[2048 x 1024] * B[1024 x 768], tile=32x32x32
Time: 0.688 ms | GFLOPS: 4682.89 | checksum(sum(C))=2.411545e+02
GEMM: C[1024 x 1024] = A[1024 x 1024] * B[1024 x 1024], tile=32x32x32
Time: 0.465 ms | GFLOPS: 4618.64 | checksum(sum(C))=4.573211e+03
