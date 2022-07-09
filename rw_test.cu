/*
 * Author:  Hampton Ford
 * Github:  @shford
 *
 * Description:
 *  Test reading and writing between device.
 *      1.) Write host local non-locked variable to device non-locked global variable.
 *      2. Async) Write device local variable to device non-locked global variable.
 *      2. Aysnc) Write host local non-locked variable from device non-locked global variable.
 */

#include <stdio.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// global device memory
__device__ volatile int terminate_signal = 0;

// attempt to update global memory
__global__ void kernel()
{
    while (terminate_signal == 1)
    {
        // write terminate_signal to 2 in global memory
        ++terminate_signal;
    }
}

int main()
{
    int status = 1;

    // write terminate_signal to 1 in global memory
    gpuErrchk( cudaMemcpyToSymbol(terminate_signal, &status, sizeof(volatile int), 0, cudaMemcpyHostToDevice) );
    kernel<<<1, 1>>>();

    // poll for updates
    status = -1;
    do {
        // read status from global memory terminate_signal
        gpuErrchk( cudaMemcpyFromSymbol(&status, terminate_signal, sizeof(terminate_signal), 0, cudaMemcpyDeviceToHost) );

        // print status explanation
        if (status == -1)
        {
            printf("Host failed to read from global mem. Therefore previous write status unknown.\n");
        }
        if (status == 0)
        {
            printf("Host failed to write 0 to 1 in global mem, host read -1 to 0 from global mem.\n");
        }
        if (status == 1)
        {
            printf("Host wrote 0 to 1 in global mem, host read from -1 to 1 from global mem, but the kernel is not updating writing 1 to 2 in global.\n");
        }
        if (status == 2)
        {
            printf("Status updated as expected. Host wrote 0 to 1 in global mem. Kernel wrote 1 to 2 in global mem (plus or minus a some cpu cycles). Host read -1 to 2 from global mem. (tested for up to 1 thread currently).\n");
            break;
        }
    } while(true);

    cudaDeviceSynchronize();

    return 0;
}
