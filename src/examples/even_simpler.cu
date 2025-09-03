#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d!\n", idx);
}

int main() {
    // Check for CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Launch kernel
    hello_kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
