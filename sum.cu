#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define N 1000000         
#define TOLERANCE 1e-5     

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_C_gpu, *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    float cpu_time, gpu_time;
    cudaEvent_t start, stop;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_C_gpu = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
    clock_t cpu_end = clock();
    cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %f seconds\n", cpu_time);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time /= 1000;

    printf("GPU Time: %f seconds\n", gpu_time);
    printf("Speedup Factor: %f\n", cpu_time / gpu_time);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    int isValid = 1;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - h_C_gpu[i]) > TOLERANCE) {
            isValid = 0;
            break;
        }
    }

    printf("Verification: %s\n", isValid ? "TRUE" : "FALSE");

    free(h_A); free(h_B); free(h_C); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

