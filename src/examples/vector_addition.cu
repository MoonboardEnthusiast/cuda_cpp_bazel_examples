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
  int threads_per_block = 256;
  int block_per_grid = (n + threads_per_block -1) / threads_per_block;
  auto start_cpu = high_resolution_clock::now();
  auto end_cpu = high_resolution_clock::now();
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
