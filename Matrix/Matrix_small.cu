#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h> 
#include <stdio.h>
#include<iostream>
#include<cstdlib>
#include<time.h> 
#include <math.h>
#define Row  150
#define Col 150

 
__global__ void matrix_mul_gpu(int *M1, int* N1, int *M2, int* N2, int* P1,int* P2 ,int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int aaa=213;
    int bbb=111;
    int *m=&aaa;
    int *n=&bbb;
                
    int sum = 0;
    for(int k=0;k<width;k++)
    {
        int a = M1[j*width+k];
        int b = N1[k*width+i];
        sum += a*b;
    }
    for(int k=0;k<width;k++)
    {
        int a = M2[j*width+k];
        int b = N2[k*width+i];
        sum += a*b;
    }
    /*for(int k=0;k<width;k++)
    {
        int a = M1[j*width+k];
        int b = N1[k*width+i];
        sum += a*b;
    }
    for(int k=0;k<width;k++)
    {
        int a = M2[j*width+k];
        int b = N2[k*width+i];
        sum += a*b;
    }*/
    P1[j*width+i] = sum;
   /* for(int k=0;k<width;k++)
    {
        int a = M1[j*width+k];
        int b = N2[k*width+i];
        sum += a*b;
    }
    for(int k=0;k<width;k++)
    {
        int a = M2[j*width+k];
        int b = N1[k*width+i];
        sum += a*b;
    }
    for(int k=0;k<width;k++)
    {
        int a = M1[j*width+k];
        int b = N2[k*width+i];
        sum += a*b;
    }
    for(int k=0;k<width;k++)
    {
        int a = M2[j*width+k];
        int b = N1[k*width+i];
        sum += a*b;
    }
    P2[j*width+i] = sum;*/
    asm("addc.s32 %0, %1, %2;" : "=r"(*m) : "r"(*m) , "r"(*n));
}
 
int main()
{
    struct timeval start, end;
    gettimeofday( &start, NULL );

    int *A1 = (int *)malloc(sizeof(int) * Row * Col);
    int *B1 = (int *)malloc(sizeof(int) * Row * Col);
    int *A2 = (int *)malloc(sizeof(int) * Row * Col);
    int *B2 = (int *)malloc(sizeof(int) * Row * Col);

    int *B3 = (int *)malloc(sizeof(int) * Row * Col);
    int *B4 = (int *)malloc(sizeof(int) * Row * Col);
    int *C = (int *)malloc(sizeof(int) * Row * Col);
    int *D = (int *)malloc(sizeof(int) * Row * Col);
    //malloc device memory
    int *d_dataA1, *d_dataB1,*d_dataA2, *d_dataB2,*d_dataB3, *d_dataB4, *d_dataC,*d_dataD;
    cudaMalloc((void**)&d_dataA1, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB1, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataA2, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB2, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB3, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB4, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataD, sizeof(int) *Row*Col);
    //set value
    for (int i = 0; i < Row*Col; i++) {
	srand(time(0));
        A1[i] = rand()%10000;
	srand(time(0));
        B1[i] = rand()%10000;
	srand(time(0));
        A2[i] = rand()%10000;
	srand(time(0));
        B2[i] = rand()%10000;
	srand(time(0));
	B3[i] = rand()%10000;
	srand(time(0));
	B4[i] =  rand()%10000;
    }
                                                                
    cudaMemcpy(d_dataA1, A2, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB1, B1, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataA2, A2, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB2, B2, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB3, B3, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB4, B4, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    dim3 threadPerBlock(16,16);
    dim3 blockNumber((Col+threadPerBlock.x-1)/ threadPerBlock.x, (Row+threadPerBlock.y-1)/threadPerBlock.y );
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA1, d_dataB1,d_dataA2, d_dataB2, d_dataC,d_dataD, Col);
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA1, d_dataB3,d_dataA2, d_dataB4, d_dataC,d_dataD, Col);
//拷贝计算数据-一级数据指针
    cudaMemcpy(C, d_dataC, sizeof(int) * Row * Col, cudaMemcpyDeviceToHost);
                                                                                             
    //释放内存
    free(A1);
    free(B1);
    free(A2);
    free(B2);
    free(C);
    free(D);
    cudaFree(d_dataA1);
    cudaFree(d_dataB1);
    cudaFree(d_dataA2);
    cudaFree(d_dataB2);
    cudaFree(d_dataC);
    cudaFree(d_dataD);
    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);

    return 0;
}
