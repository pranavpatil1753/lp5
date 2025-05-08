#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define N 512
#define TOLERANCE 1e-3
#define USE_DOUBLE 0

#if USE_DOUBLE
typedef double real;
#else
typedef float real;
#endif

__global__ void matrixMul(real *A, real *B, real *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        real sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(real);
    real *h_A, *h_B, *h_C, *h_C_gpu, *d_A, *d_B, *d_C;
    float cpu_time, gpu_time;
    cudaEvent_t start, stop;

    h_A = (real*)malloc(size);
    h_B = (real*)malloc(size);
    h_C = (real*)malloc(size);
    h_C_gpu = (real*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (real)RAND_MAX;
        h_B[i] = rand() / (real)RAND_MAX;
    }

    clock_t cpu_start = clock();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            real sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C[i * N + j] = sum;
        }
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

    cudaEventRecord(start);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_time, start, stop);
    gpu_time /= 1000;

    printf("GPU Time: %f seconds\n", gpu_time);
    printf("Speedup Factor: %f\n", cpu_time / gpu_time);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    int isValid = 1;
    for (int i = 0; i < N * N; i++) {
        if (fabs(h_C[i] - h_C_gpu[i]) > TOLERANCE * fabs(h_C[i])) {  
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

