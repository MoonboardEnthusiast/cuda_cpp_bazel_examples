#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::chrono;

__global__ void add_vec(int *a, int *b, int *c, int n) {
  int threadId = blockIdx.x * blockDim.x * threadIdx.x;

  //Check bounds
  if(threadId < n){
    c[threadId] = a[threadId] + b[threadId];
  }
}

void vec(int n) {
  vector<int> a(n);
  vector<int> b(n);
  vector<int> sum_cpu(n);
  vector<int> sum_gpu(n);

  int *dev_a, *dev_b, *dev_sum;

  for(int i=0;i<n;i++){
    a[i] = i +1;
    b[i] = i +2;
  }


  int threads_per_block = 256;
  int block_per_grid = (n + threads_per_block -1) / threads_per_block;
  auto start_cpu = high_resolution_clock::now();

  for ( int i = 0; i< n; i++){
    sum_cpu[i] = a[i] + b[i];
  }

  auto end_cpu = high_resolution_clock::now();

  auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);

  cudaMalloc((void**) &dev_a, n*sizeof(int));
  cudaMalloc((void**) &dev_b, n*sizeof(int));
  cudaMalloc((void**) &dev_sum, n*sizeof(int));

  cudaMemcpy(dev_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b.data(), n*sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventRecord(start_gpu);

  add_vec<<<block_per_grid, threads_per_block>>>(dev_a, dev_b, dev_sum, n);

  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);

  float duration_gpu = 0;
  cudaEventElapsedTime(&duration_gpu, start_gpu, stop_gpu);


  cudaMemcpy(sum_gpu.data(), dev_sum, n*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_sum);
  
  cout << "Time taken by CPU: " << duration_cpu.count() << " microseconds" << endl;
  cout << "Time taken by GPU: " << duration_gpu << " microseconds" << endl;

  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);

  
}

int main(){
  vector<int>a = {100, 1000, 10000, 1000000, 10000000};

  for ( int i = 0; i<size(a); i++){
    cout << "Vector Size: " << a[i] << endl;
    vec(a[i]);
    cout << endl;
  }
  return 0;
}
