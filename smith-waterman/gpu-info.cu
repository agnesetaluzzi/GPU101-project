#include <stdio.h>

int main(int argc, char *argv[])
{
    int dev;
    cudaDeviceProp devProp;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&devProp, dev);

    printf("Major revision number: %d\n", devProp.major);
    printf("Minor revision number: %d\n", devProp.minor);
    printf("Name: %s\n", devProp.name);
    printf("Total global memory: %lu\n", devProp.totalGlobalMem);
    printf("Total registers per block: %d\n", devProp.regsPerBlock);
    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block: %d\n", i, devProp.maxThreadsDim[i]);

    return 0;
}