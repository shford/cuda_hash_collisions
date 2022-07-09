/*
 * Author:  Hampton Ford
 * Github:  @shford
 *
 * Description:
 *  Flag framework to safely transfer data and synchronize device
 *
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <fileapi.h>


#define TARGET_COLLISIONS (5)
#define ARBITRARY_MAX_DATA_SIZE (1024)

// globally accessible device variables (L2 cache on CUDA 8.6)
__device__ volatile char* collision;
__device__ bool terminate_signal;


__global__ void kernel()
{
    while (terminate_signal == false)
    {
        // initialize local data
        char* local_data;
        cudaMalloc(&local_data, ARBITRARY_MAX_DATA_SIZE);

        // copy global data to local
        cudaMemcpyAsync(local_data, (const void*)collision, ARBITRARY_MAX_DATA_SIZE, cudaMemcpyDeviceToDevice, 0);

        // replace this with working code
        bool hashes_match = true;
        if (hashes_match == true)
        {
            // write local data to global for host polling
            cudaMemcpyAsync((void*)collision, local_data, ARBITRARY_MAX_DATA_SIZE, cudaMemcpyDeviceToDevice, 0);

            // free local_data
            cudaFree(local_data);

            // update global flag for host polling flag
            terminate_signal = true;
        }
    }
}

int main()
{
    // sample append data
    char* data = (char*)calloc(1, ARBITRARY_MAX_DATA_SIZE);
    sprintf_s(data, ARBITRARY_MAX_DATA_SIZE, "%s", "test");

    // flag true if collision is found
    bool collision_found = false;

    // allocate mem for collision
    cudaMalloc(&collision, ARBITRARY_MAX_DATA_SIZE);

    // copy data to collision in global device memory
    cudaMemcpyToSymbol(&collision, data, strlen(data)+1, 0, cudaMemcpyHostToDevice);

    // set terminate_signal to false in global device memory
    cudaMemcpyToSymbol(&terminate_signal, &collision_found, sizeof(collision_found), 0, cudaMemcpyHostToDevice);

    // hash until 5 hashes are found
    int collision_count = 0;
    while (collision_count != TARGET_COLLISIONS)
    {
        // execution configuration (sync device)
        kernel<<<1, 32>>>();

        // read collision status from device todo check ptr issues in call
        cudaMemcpyFromSymbol(&collision_found, &terminate_signal, sizeof(collision_found), 0, cudaMemcpyDeviceToHost);

        // poll collision flag
        while (collision_found != true)
        {
            // read the reported value (strlen fails on volatile - either loop or copy the whole thing)
            cudaMemcpyFromSymbol(&data, collision, ARBITRARY_MAX_DATA_SIZE, 0, cudaMemcpyDeviceToHost);

            // check hashes match
            bool hashes_match = true;
            if (hashes_match == true)
            {
                // write file (note this should be multi-threaded. this is a waste of time when the gpu is likely idling)

                // increase collision count
                ++collision_count;
            }
        }

        // ensure kernel processes are finished
        cudaDeviceSynchronize();
    }

    cudaFree((void*)collision);
    free(data);

    printf("Everything worked... /");
}