#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel function - runs on GPU
__global__ void hello_world_kernel() {
    // Get thread ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread prints its ID
    printf("Hello World from GPU! Thread ID: %d, Block ID: %d, Thread in Block: %d\n", 
           threadId, blockIdx.x, threadIdx.x);
}

// CUDA kernel with array processing
__global__ void vector_add_kernel(int *a, int *b, int *c, int size) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (threadId < size) {
        c[threadId] = a[threadId] + b[threadId];
        printf("Thread %d: %d + %d = %d\n", threadId, a[threadId], b[threadId], c[threadId]);
    }
}

int main() {
    printf("CUDA Hello World Program\n");
    printf("========================\n\n");
    
    // Check CUDA device
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    // Get device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    }
    
    printf("\n--- Part 1: Simple Hello World Kernel ---\n");
    
    // Launch configuration: 2 blocks, 4 threads per block = 8 total threads
    dim3 blockSize(4);  // 4 threads per block
    dim3 gridSize(2);   // 2 blocks
    
    printf("Launching kernel with %d blocks of %d threads each...\n", gridSize.x, blockSize.x);
    
    // Launch the kernel
    hello_world_kernel<<<gridSize, blockSize>>>();
    
    // Wait for GPU to finish
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaerr));
        return 1;
    }
    
    printf("\n--- Part 2: Vector Addition Example ---\n");
    
    const int arraySize = 8;
    const int arrayBytes = arraySize * sizeof(int);
    
    // Host (CPU) arrays
    int h_a[arraySize] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_b[arraySize] = {10, 20, 30, 40, 50, 60, 70, 80};
    int h_c[arraySize] = {0}; // Result array
    
    printf("Input arrays:\n");
    printf("A: ");
    for (int i = 0; i < arraySize; i++) printf("%d ", h_a[i]);
    printf("\nB: ");
    for (int i = 0; i < arraySize; i++) printf("%d ", h_b[i]);
    printf("\n\n");
    
    // Device (GPU) arrays
    int *d_a, *d_b, *d_c;
    
    // Allocate GPU memory
    cudaMalloc((void**)&d_a, arrayBytes);
    cudaMalloc((void**)&d_b, arrayBytes);
    cudaMalloc((void**)&d_c, arrayBytes);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);
    
    // Launch vector addition kernel
    int threadsPerBlock = 4;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching vector addition with %d blocks of %d threads...\n", blocksPerGrid, threadsPerBlock);
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, arraySize);
    
    // Wait for GPU to finish
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("Vector kernel launch failed: %s\n", cudaGetErrorString(cudaerr));
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);
    
    printf("\nResult array C = A + B:\n");
    printf("C: ");
 https://github.com/MoonboardEnthusiast/cuda_cpp_bazel_examples.git   for (int i = 0; i < arraySize; i++) {
        printf("%d ", h_c[i]);
    }
    printf("\n\n");
    
    // Verify results
    bool success = true;
    for (int i = 0; i < arraySize; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("✅ Vector addition completed successfully!\n");
    } else {
        printf("❌ Vector addition failed!\n");
    }
    
    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("\n--- Performance Info ---\n");
    
    // Get some timing info for fun
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Run the kernel again for timing
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, arraySize);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %.3f ms\n", milliseconds);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\n🎉 CUDA Hello World completed successfully!\n");
    printf("You're ready to start GPU programming!\n");
    
    return 0;
}
