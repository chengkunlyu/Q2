// nvcc -O3 -std=c++14 -arch=sm_70 q2_matmul_cuda.cu -o q2_matmul
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

#define CHECK(x) do{ auto err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA ERROR %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} }while(0)

// Each thread computes one C element in a TILE_M x TILE_N tile.
__global__ void sgemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  int row = blockIdx.y * TILE_M + threadIdx.y;
  int col = blockIdx.x * TILE_N + threadIdx.x;

  float acc = 0.f;
  for (int tk=0; tk<K; tk+=TILE_K) {
    // load tiles (masked to avoid OOB and keep divergence low)
    int a_r = row, a_c = tk + threadIdx.x;
    As[threadIdx.y][threadIdx.x] = (a_r<M && a_c<K) ? A[a_r*K + a_c] : 0.f;

    int b_r = tk + threadIdx.y, b_c = col;
    Bs[threadIdx.y][threadIdx.x] = (b_r<K && b_c<N) ? B[b_r*N + b_c] : 0.f;

    __syncthreads();

    #pragma unroll
    for (int kk=0; kk<TILE_K; ++kk) {
      acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < M && col < N) C[row*N + col] = acc;
}

int main(int argc, char** argv) {
  // Default: 2048x1024 * 1024x768 → 2048x768
  int M = (argc>1)? atoi(argv[1]) : 2048;
  int K = (argc>2)? atoi(argv[2]) : 1024;
  int N = (argc>3)? atoi(argv[3]) : 768;
  printf("GEMM: C[%d x %d] = A[%d x %d] * B[%d x %d], tile=%dx%dx%d\n",
         M,N,M,K,K,N,TILE_M,TILE_N,TILE_K);

  std::vector<float> hA(M*1ll*K), hB(K*1ll*N), hC(M*1ll*N);
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  for (auto& x: hA) x = uni(rng);
  for (auto& x: hB) x = uni(rng);

  float *dA=nullptr,*dB=nullptr,*dC=nullptr;
  CHECK(cudaMalloc(&dA, M*1ll*K*sizeof(float)));
  CHECK(cudaMalloc(&dB, K*1ll*N*sizeof(float)));
  CHECK(cudaMalloc(&dC, M*1ll*N*sizeof(float)));
  CHECK(cudaMemcpy(dA, hA.data(), M*1ll*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB.data(), K*1ll*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dC, 0, M*1ll*N*sizeof(float)));

  dim3 block(TILE_N, TILE_M);
  dim3 grid((N + TILE_N - 1)/TILE_N, (M + TILE_M - 1)/TILE_M);

  cudaEvent_t t0,t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));
  sgemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
  CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
  float ms=0; CHECK(cudaEventElapsedTime(&ms,t0,t1));

  CHECK(cudaMemcpy(hC.data(), dC, M*1ll*N*sizeof(float), cudaMemcpyDeviceToHost));

  // Simple checksum (no CPU gemm)
  long double checksum=0.0;
  for (double v: hC) checksum += v;
  double gflops = (2.0 * M * N * K) / (ms/1e3) / 1e9;

  printf("Time: %.3f ms | GFLOPS: %.2f | checksum(sum(C))=%.6Le\n", ms, gflops, checksum);

  CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
  return 0;
}
