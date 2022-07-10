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
#include <string.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: Err no: %d, code: %s %s %d\n", code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#define TARGET_COLLISIONS (5)
#define ARBITRARY_MAX_BUFF_SIZE (1024)
#define FALSE (0)
#define TRUE (1)


__device__ char* collision;
__device__ unsigned long long collision_cstring_size;
__device__ int read_collision = FALSE;
__device__ int terminate_signal = FALSE;

__global__ void kernel(unsigned long long page_locked_host_data_cstring_size)
{
    // initialize local data
    char *local_data;
    cudaMalloc(&local_data, ARBITRARY_MAX_BUFF_SIZE);

    // copy global data to local
    collision_cstring_size = page_locked_host_data_cstring_size;
    cudaMemcpyAsync(local_data, collision, collision_cstring_size, cudaMemcpyDeviceToDevice, 0);

    // modify local data to test copying back to global
    if (collision_cstring_size + 1 < ARBITRARY_MAX_BUFF_SIZE) {
        local_data[collision_cstring_size - 1] = '8';
        local_data[collision_cstring_size] = '\0';
        ++collision_cstring_size;
    }

    // replace condition to ensure hashes match
    if (true)
    {
        // write local data to global for host polling
        cudaMemcpyAsync(collision, local_data, collision_cstring_size, cudaMemcpyDeviceToDevice, 0);

        // free local data
        cudaFree(local_data);

        // tell host to read collision
        read_collision = TRUE;
    }

    // unsure if necessary - may be wholely unneeded or may could be deleted by applying volatile attr to collision
    // in theory do not release kernel level (L2) memory until host confirms it's been read
    int idling = 0;
    while (!terminate_signal)
    {
        ++idling; // this line is only here to keep the compiler from optimizing the busy loop away
    }
}

int main()
{
    // existing base data
    char tmp_filler_base_data[] = "1234567";
    unsigned long long page_locked_host_data_cstring_size = strlen(tmp_filler_base_data) + 1; // adjust for null terminator

    // allocate and initialize page-locked host variables
    char* page_locked_host_data;
    gpuErrchk( cudaMallocHost(&page_locked_host_data, ARBITRARY_MAX_BUFF_SIZE) );
    cudaMemset(&page_locked_host_data, 0x00, sizeof(char) * ARBITRARY_MAX_BUFF_SIZE);
    strncpy_s(page_locked_host_data, page_locked_host_data_cstring_size, tmp_filler_base_data, ARBITRARY_MAX_BUFF_SIZE);

    int* collision_found;
    gpuErrchk( cudaMallocHost((void**)&collision_found, sizeof(int)) ); //todo CHECK THIS LINE w/ and w/out &
    *collision_found = FALSE;

    // allocate global mem for collision
    gpuErrchk( cudaMalloc((void**)&collision, ARBITRARY_MAX_BUFF_SIZE) ); //todo CHECK THIS LINE w/ and w/out &
    gpuErrchk( cudaMemset(collision, 0x00, sizeof(char) * ARBITRARY_MAX_BUFF_SIZE) );

    // copy data to collision in global device memory
    gpuErrchk( cudaMemcpyToSymbol(collision, page_locked_host_data, page_locked_host_data_cstring_size, 0, cudaMemcpyHostToDevice) );

    // hash until 5 hashes are found
    int collision_count = 0;
    while (collision_count != TARGET_COLLISIONS)
    {
        printf("Reached before kernel.\n");
        // execution configuration (sync device)
        kernel<<<1, 1>>>(page_locked_host_data_cstring_size);
        printf("Reached after kernel.\n");

        /*
         * ensure we're not trying to read from global memory before
         * the kernel can reserve/allocate it
         */
        int wait_cycles = 0;
        while (wait_cycles < 1000)
        {
            ++wait_cycles;
        }

        // poll collision flag
        do {
            const void* dev_N;
            size_t symbolSize;
            gpuErrchk( cudaGetSymbolSize(&symbolSize, read_collision) );
            gpuErrchk( cudaGetSymbolAddress((void**)&dev_N, read_collision) );
            gpuErrchk( cudaMemcpy((void*)collision_found, dev_N, symbolSize, cudaMemcpyDeviceToHost) );
            // read collision status from device into host flag
//            gpuErrchk( cudaMemcpyFromSymbol(collision_found, read_collision, sizeof(*collision_found), 0, cudaMemcpyDeviceToHost) );
            printf("collision_found = %s after read.", *collision_found ? "TRUE" : "FALSE");

            if (*collision_found) {
                // read updated collision size into page_locked_host_data size
                gpuErrchk(cudaMemcpyFromSymbol(&page_locked_host_data_cstring_size, collision_cstring_size, sizeof(page_locked_host_data_cstring_size), 0, cudaMemcpyDeviceToHost) );

                // read from collision to host mem
                gpuErrchk( cudaMemcpyFromSymbol(page_locked_host_data, collision, page_locked_host_data_cstring_size, 0, cudaMemcpyDeviceToHost) );

                // for (int i = 0; i < 5; ++i) {
                // char byte[10];
                // cudaMemcpyFromSymbol(byte, collision, sizeof(char)*9);
                // int tmp = 2;
                // page_locked_host_data[i] = byte;
                // }

                // tell gpu threads to exit when they next read terminate_signal
                gpuErrchk( cudaMemcpyToSymbol(terminate_signal, collision_found, sizeof(*collision_found), 0, cudaMemcpyHostToDevice) );

                // replace condition with hash check
                if (true) // todo replace true w/ 20 bit comparisons
                {
                    // write file (note this should be multi-threaded. this is a waste of time when the gpu is likely idling)

                    // increase collision count
                    ++collision_count;

                    // reset collision_found, terminate_flag, and read_collision
                    *collision_found = FALSE;
                    gpuErrchk( cudaMemcpyToSymbol(terminate_signal, collision_found, sizeof(*collision_found), 0, cudaMemcpyHostToDevice) );
                    gpuErrchk( cudaMemcpyToSymbol(read_collision, collision_found, sizeof(*collision_found), 0, cudaMemcpyHostToDevice) );
                }
            }
        } while(!(*collision_found));

        // ensure kernel processes are finished
        cudaDeviceSynchronize();
    }

    printf("Success... /\n");
    printf("The kernel changed the following character: %c\n", page_locked_host_data[0]);
    printf("Full modified string: %s\n", page_locked_host_data);

    cudaFreeHost(page_locked_host_data);
    cudaFreeHost(collision_found);
    cudaFree((void*)collision);
}
