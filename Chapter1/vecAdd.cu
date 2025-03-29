__global___ void vecAddKernel(float * A, float * B, float * c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = A[i] + B[i];
    }
}
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <chrono>

void vecAdd(float * A_h, float * B_h, float * c_h, int n){
    int size = n * sizeof(float);
    float * A_d, * B_d, * c_d;
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&c_d, size);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, c_d, n);
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(c_d);
}