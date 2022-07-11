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


__device__ int d_collision_flag = FALSE;
__device__ unsigned long long d_collision_cstring_size;
//__device__ int d_terminate_kernel = FALSE;

__global__ void kernel(char* collision, unsigned long long page_locked_host_data_cstring_size)
{
    // initialize local data
    char local_data[ARBITRARY_MAX_BUFF_SIZE];

    // copy global data to local
    d_collision_cstring_size = page_locked_host_data_cstring_size;
    for (int byte_index = 0; byte_index <= d_collision_cstring_size; ++byte_index) {
        local_data[byte_index] = collision[byte_index];
    }
    //cudaMemcpyAsync(local_data, collision, d_collision_cstring_size, cudaMemcpyDeviceToDevice, 0);

    // modify local data to test copying back to global
//    if (d_collision_cstring_size + 1 < ARBITRARY_MAX_BUFF_SIZE) {
        local_data[d_collision_cstring_size - 2] = '6';
        local_data[d_collision_cstring_size - 1] = '7';
        local_data[d_collision_cstring_size] = '\0';
        ++d_collision_cstring_size;
//    }

    // replace condition to ensure hashes match
    if (true)
    {
        // write local data to global for host polling
        for (int byte_index = 0; byte_index <= d_collision_cstring_size; ++byte_index) {
            collision[byte_index] = local_data[byte_index];
        }
//        cudaMemcpyAsync(collision, local_data, d_collision_cstring_size, cudaMemcpyDeviceToDevice, 0);

        // tell host to read collision
        d_collision_flag = TRUE;
    }

    // unsure if necessary - may be wholly unneeded or may could be deleted by applying volatile attr to collision
    // in theory do not release kernel level (L2) memory until host confirms it's been read
//    int idling = 0;
//    while (!d_terminate_kernel)
//    {
//        ++idling; // this line is only here to keep the compiler from optimizing the busy loop away
//    }
}

int main()
{
    // collision storage
    char* h_collisions[TARGET_COLLISIONS];
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        h_collisions[i] = (char*)calloc(1, ARBITRARY_MAX_BUFF_SIZE);
    }
//    char h_collisions[TARGET_COLLISIONS][ARBITRARY_MAX_BUFF_SIZE];
    unsigned long long h_collision_sizes[TARGET_COLLISIONS];

    // existing base data
    char tmp_filler_base_data[] = "0123456";
    unsigned long long h_page_locked_data_cstring_size = strlen(tmp_filler_base_data) + 1; // adjust for null terminator

    // allocate and initialize page-locked host variables
    char* h_page_locked_data;
    gpuErrchk( cudaMallocHost(&h_page_locked_data, ARBITRARY_MAX_BUFF_SIZE) );
    cudaMemset(&h_page_locked_data, 0x00, sizeof(char) * ARBITRARY_MAX_BUFF_SIZE);
    strncpy_s(h_page_locked_data, h_page_locked_data_cstring_size, tmp_filler_base_data, ARBITRARY_MAX_BUFF_SIZE);

    // allocate global mem for collision
    char* d_collision;
    gpuErrchk( cudaMalloc((void **)&d_collision, ARBITRARY_MAX_BUFF_SIZE) );
    cudaMemcpy(d_collision, h_page_locked_data, h_page_locked_data_cstring_size, cudaMemcpyHostToDevice);

    // hash until 5 hashes are found
    int collision_count = 0;
    int h_collision_found = FALSE;
    while (collision_count != TARGET_COLLISIONS)
    {
        // execution configuration (sync device)
        h_collision_found = FALSE;
        kernel<<<3, 128>>>(d_collision, h_page_locked_data_cstring_size);

        // poll collision flag
        while (!h_collision_found || collision_count != TARGET_COLLISIONS)
        {
            // read collision status from device into host flag
            gpuErrchk( cudaMemcpyFromSymbol(&h_collision_found, d_collision_flag, sizeof(h_collision_found), 0, cudaMemcpyDeviceToHost) );
            printf("h_collision_found = %s after read.\n", h_collision_found ? "TRUE" : "FALSE");

            if (h_collision_found)
            {
                // read updated collision size into h_page_locked_data size
                gpuErrchk(cudaMemcpyFromSymbol(&h_collision_sizes[collision_count], d_collision_cstring_size, sizeof(h_page_locked_data_cstring_size), 0, cudaMemcpyDeviceToHost) );

                // read from collision to host mem
                gpuErrchk( cudaMemcpy(h_collisions[collision_count], d_collision, h_page_locked_data_cstring_size, cudaMemcpyDeviceToHost) );

                // tell gpu threads to exit when they next read d_terminate_kernel
//                gpuErrchk( cudaMemcpyToSymbol(d_terminate_kernel, &h_collision_found, sizeof(h_collision_found), 0, cudaMemcpyHostToDevice) );

                // replace condition with hash check
                if (true) // todo replace true w/ 20 bit comparisons
                {
                    // write file (note this should be multi-threaded. this is a waste of time when the gpu is likely idling)

                    // increase collision count
                    ++collision_count;

                    // ensure kernel processes are finished
                    gpuErrchk( cudaDeviceSynchronize() );
                }
            }
        }
    }

    printf("\nSuccess... /\n");
    printf("Original string: %s\n", h_page_locked_data);
    for (int i = 0; i < collision_count; ++i) {
        printf("Collision %d: %s\n", i, h_collisions[i]);
    }

    cudaFreeHost(h_page_locked_data);
    cudaFree(d_collision);
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        free(h_collisions[i]);
    }
}
