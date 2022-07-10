/*
 * Author:  Hampton Ford
 * Github:  @shford
 *
 * Description:
 *  Flag framework to safely transfer data and synchronize device.
 *
 */

#include <stdio.h>
#include "cuda_consts.cuh"


#define TARGET_COLLISIONS (5)
#define ARBITRARY_MAX_DATA_SIZE (1024)


__device__ volatile char* collision;
__device__ volatile bool terminate_signal = false;


__global__ void kernel()
{
    // initialize local data
    char* local_data;
    cudaMalloc(&local_data, ARBITRARY_MAX_DATA_SIZE);

    // copy global data to local
    cudaMemcpyAsync(local_data, (const void*)collision, ARBITRARY_MAX_DATA_SIZE, cudaMemcpyDeviceToDevice, 0);

    // modify local data
//    local_data[0] = '9';
    collision[0] = '9';
    //    local_data[7] = '8';
    //    local_data[8] = '\0';

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
    // allocate page-locked host variables
    char* page_locked_host_data;
    gpuErrchk( cudaMallocHost(&page_locked_host_data, ARBITRARY_MAX_DATA_SIZE) );
    cudaMemset(&page_locked_host_data, 0x00, sizeof(char)*ARBITRARY_MAX_DATA_SIZE);
    strncpy_s(page_locked_host_data, strlen("1234567")+1, "1234567", ARBITRARY_MAX_DATA_SIZE);

    bool* collision_found;
    gpuErrchk( cudaMallocHost(&collision_found, sizeof(bool)) );
    *collision_found = false;

    // allocate global mem for collision
    gpuErrchk( cudaMalloc(&collision, ARBITRARY_MAX_DATA_SIZE) );
    gpuErrchk( cudaMemset((void*)collision, 0x00, sizeof(char)*ARBITRARY_MAX_DATA_SIZE) );

    // copy data to collision in global device memory
    gpuErrchk( cudaMemcpyToSymbol(collision, (const void*)page_locked_host_data, strlen(page_locked_host_data)+1, 0, cudaMemcpyHostToDevice) );

    // hash until 5 hashes are found
    int collision_count = 0;
    while (collision_count != TARGET_COLLISIONS)
    {
        // execution configuration (sync device)
        kernel<<<1, 1>>>();

        // poll collision flag
        do {
            // read collision status from device into host flag
            cudaMemcpyFromSymbol(collision_found, terminate_signal, sizeof(collision_found));
            printf(collision_found ? "true\n" : "false\n");
            if (collision_found) {
                // todo read the reported value (strlen fails on volatile - either loop or copy the whole thing)
                 cudaMemcpyFromSymbol((void*)page_locked_host_data, collision, strlen(page_locked_host_data)+1, 0, cudaMemcpyDeviceToHost);
//                for (int i = 0; i < 5; ++i) {
//                    char byte[10];
//                    cudaMemcpyFromSymbol(byte, collision, sizeof(char)*9);
//                    int tmp = 2;
//                  page_locked_host_data[i] = byte;
//                }

                // ensure hashes match in case
                if (true) // todo replace true w/ 20 bit comparisons
                {
                    // write file (note this should be multi-threaded. this is a waste of time when the gpu is likely idling)

                    // increase collision count
                    ++collision_count;

                    // reset collision_found and terminate_flag
                    *collision_found = false;
                    cudaMemcpyToSymbol(terminate_signal, collision_found, sizeof(collision_found), 0, cudaMemcpyHostToDevice);
                }
            }
        } while(!collision_found);

        // ensure kernel processes are finished
        cudaDeviceSynchronize();
    }

    printf("Everything worked... /\n");
    printf("%c\n", page_locked_host_data[0]);
    printf("Modified string: %s\n", page_locked_host_data);

    cudaFreeHost(page_locked_host_data);
    cudaFreeHost(collision_found);
    cudaFree((void*)collision);
}
