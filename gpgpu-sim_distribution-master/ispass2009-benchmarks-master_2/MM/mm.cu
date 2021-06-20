#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include<iostream>
#include<cstdlib>
#include<time.h> 
#include <math.h>
#define Row  150
#define Col 300

 
__global__ void matrix_mul_gpu(int *M, int* N, int* P, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
                
    int sum = 0;
    for(int k=0;k<width;k++)
    {
        int a = M[j*width+k];
        int b = N[k*width+i];
        sum += a*b;
    }
    P[j*width+i] = sum;
}
 
int main()
{
    struct timeval start, end;
    gettimeofday( &start, NULL );

    int *A = (int *)malloc(sizeof(int) * Row * Col);
    int *B = (int *)malloc(sizeof(int) * Row * Col);
    int *C = (int *)malloc(sizeof(int) * Row * Col);
    //malloc device memory
    int *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(int) *Row*Col);
    //set value
    for (int i = 0; i < Row*Col; i++) {
	srand(time(0));
        A[i] = rand();
	srand(time(0));
        B[i] = rand();
    }
                                                                
    cudaMemcpy(d_dataA, A, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    dim3 threadPerBlock(10,10);
    dim3 blockNumber((Col+threadPerBlock.x-1)/ threadPerBlock.x, (Row+threadPerBlock.y-1)/ threadPerBlock.y );
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    //拷贝计算数据-一级数据指针
    cudaMemcpy(C, d_dataC, sizeof(int) * Row * Col, cudaMemcpyDeviceToHost);
                                                                                             
    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);

    return 0;
}
