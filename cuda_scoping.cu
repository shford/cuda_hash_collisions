/*
 * Author:  Hampton Ford
 * Github:  @shford
 *
 * Description:
 *  Flag framework to safely transfer data and synchronize device.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "cuda_consts.cuh"


#define TARGET_COLLISIONS (5)
#define ARBITRARY_MAX_DATA_SIZE (1024)

// globally accessible device variables (L2 cache on CUDA 8.6)
__device__ volatile char* collision;
__device__ bool volatile terminate_signal = false;


__global__ void kernel()
{
    // initialize local data
    char* local_data;
    cudaMalloc(&local_data, ARBITRARY_MAX_DATA_SIZE);

    // copy global data to local
    cudaMemcpyAsync(local_data, (const void*)collision, ARBITRARY_MAX_DATA_SIZE, cudaMemcpyDeviceToDevice, 0);

    while (terminate_signal == false)
    {
        // replace this with working code
        bool hashes_match = true;
        if (hashes_match == true)
        {
            // update global flag for host polling flag
            terminate_signal = true;

            // write local data to global for host polling
            cudaMemcpyAsync((void*)collision, local_data, ARBITRARY_MAX_DATA_SIZE, cudaMemcpyDeviceToDevice, 0);
        }
    }
    cudaFree(local_data);
}

int main()
{
    // allocate host local mem for original data
    char* page_locked_host_data;
    gpuErrchk( cudaMallocHost(&page_locked_host_data, ARBITRARY_MAX_DATA_SIZE) );
    sprintf_s(page_locked_host_data, ARBITRARY_MAX_DATA_SIZE, "test");

    // allocate global mem for collision
    gpuErrchk( cudaMalloc(&collision, ARBITRARY_MAX_DATA_SIZE) );

    // copy data to collision in global device memory
    gpuErrchk( cudaMemcpyToSymbol(collision, page_locked_host_data, strlen(page_locked_host_data)+1, 0, cudaMemcpyHostToDevice) );

    // hash until 5 hashes are found
    int collision_count = 0;
    bool collision_found = false;
    while (collision_count != TARGET_COLLISIONS)
    {
        // execution configuration (sync device)
        kernel<<<1, 1>>>();

//        printf(collision_found ? "true\n" : "false\n");

        // poll collision flag
        do {
            // read collision status from device into host flag
            gpuErrchk( cudaMemcpyFromSymbol(&collision_found, terminate_signal, sizeof(collision_found), 0, cudaMemcpyDeviceToHost) );

            if (collision_found) {
                // read the reported value (strlen fails on volatile - either loop or copy the whole thing)
                gpuErrchk( cudaMemcpyFromSymbol((void*)page_locked_host_data, *collision, ARBITRARY_MAX_DATA_SIZE, 0, cudaMemcpyDeviceToHost) );

                // ensure hashes match in case
                if (true) // todo replace true w/ 20 bit comparisons
                {
                    // write file (note this should be multi-threaded. this is a waste of time when the gpu is likely idling)

                    // increase collision count
                    ++collision_count;

                    // reset collision_found and terminate_flag
                    collision_found = false;
                    gpuErrchk( cudaMemcpyToSymbol(terminate_signal, &collision_found, sizeof(collision_found), 0, cudaMemcpyHostToDevice) );
                }
            }
        } while(!collision_found);

        // ensure kernel processes are finished
        cudaDeviceSynchronize();
    }

    cudaFree(page_locked_host_data);
    cudaFree((void*)collision);

    printf("Everything worked... /");
}