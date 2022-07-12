#pragma once

#include "tasks.cuh"


__constant__ __device__ MD5_HASH d_const_md5_digest; // initialized just once prior to kernel launches
__device__ unsigned long long d_collision_size; // reset before every kernel launch
__device__ int d_collision_flag;
__device__ unsigned long long d_unsafe_hash_attempts;

__global__ void find_collisions(char* global_collision_addr) {}

void task1() {
    //===========================================================================================================
    // SEQUENTIAL TASKS (Initial)
    //===========================================================================================================

    // todo v5 cudaMallocHost - code chunk has been tested
    // char* h_page_locked_data;
    // gpuErrchk( cudaMallocHost(&h_page_locked_data, ARBITRARY_MAX_BUFF_SIZE) );
    // cudaMemset(&h_page_locked_data, 0x00, sizeof(char) * ARBITRARY_MAX_BUFF_SIZE);

    // read file data
    char sampleFile_path[] = "C:/Users/shford/CLionProjects/cuda_hashing/sample.txt";
    char* h_sampleFile_buff;
    uint32_t h_sampleFile_buff_size = 0; // handle files up to ~4GiB (2^32-1 bytes)
    get_file_data((char*)sampleFile_path, &h_sampleFile_buff, &h_sampleFile_buff_size);

    // get hash md5_digest
    MD5_HASH md5_digest;
    Md5Calculate((const void*)h_sampleFile_buff, h_sampleFile_buff_size, &md5_digest);

    // format and print digest as a string of hex characters
    char hash[MD5_HASH_SIZE_B + 1]; //MD5 len is 16B, 1B = 2 chars

    char tiny_hash[TINY_HASH_SIZE_B + 1];
    for (int i = 0; i < MD5_HASH_SIZE / 2; ++i)
    {
        sprintf(hash + i * 2, "%2.2x", md5_digest.bytes[i]);
    }
    hash[sizeof(hash)-1] = '\0';
    strncpy_s(tiny_hash, sizeof(tiny_hash), hash, _TRUNCATE);
    tiny_hash[sizeof(tiny_hash)-1] = '\0';

    printf("Full MD5 md5_digest is: %s\n", hash);
    printf("TinyHash md5_digest is: %s\n\n", tiny_hash);

    //===========================================================================================================
    // BEGIN CUDA PARALLELIZATION
    //===========================================================================================================

    // allocate storage for collisions once found - overflows are simply truncated & result in another kernel run
    char* h_collisions[TARGET_COLLISIONS];
    unsigned long long h_collision_sizes[TARGET_COLLISIONS];
    unsigned long long h_unsafe_collision_attempts[TARGET_COLLISIONS];
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        h_collisions[i] = (char*)calloc(1, ARBITRARY_MAX_BUFF_SIZE);
        h_collision_sizes[i] = 0;
        h_unsafe_collision_attempts[i] = 0;
    }

    // allocate global mem for collision - initialized in loop
    char* d_collision;
    gpuErrchk( cudaMalloc((void **)&d_collision, ARBITRARY_MAX_BUFF_SIZE) );

    // hash until 5 hashes are found
    int collision_count = 0;
    while (collision_count != TARGET_COLLISIONS)
    {
        // reset host variables corresponding that get read into
        int h_collision_flag = FALSE;

        // reset global device variables after each run
        cudaMemcpyToSymbol(d_const_md5_digest, &md5_digest, sizeof(md5_digest), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_collision_size, &h_sampleFile_buff_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_collision_flag, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_unsafe_hash_attempts, &h_unsafe_collision_attempts[collision_count], sizeof(h_unsafe_collision_attempts[collision_count]), 0, cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision, h_sampleFile_buff, h_sampleFile_buff_size, cudaMemcpyHostToDevice);

        // execution configuration (sync device)
        find_collisions<<<MULTIPROCESSORS, CUDA_CORES_PER_MULTIPROCESSOR>>>(d_collision);

        //todo update rw's b/w new globals & symbols. then fill in kernel and happy debugging
        // poll collision flag
        while (!h_collision_flag || collision_count != TARGET_COLLISIONS)
        {
            // read collision status from device into host flag
            gpuErrchk( cudaMemcpyFromSymbol(&h_collision_flag, d_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyDeviceToHost) );
            printf("h_collision_flag = %s after read.\n", h_collision_flag ? "TRUE" : "FALSE");

            if (h_collision_flag)
            {
                // read updated collision size into h_page_locked_data size
                gpuErrchk(cudaMemcpyFromSymbol(&h_collision_sizes[collision_count], d_collision_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyDeviceToHost) );

                // read from collision to host mem
                gpuErrchk( cudaMemcpy(h_collisions[collision_count], d_collision, h_sampleFile_buff_size, cudaMemcpyDeviceToHost) );

                // tell gpu threads to exit when they next read d_terminate_kernel
//                gpuErrchk( cudaMemcpyToSymbol(d_terminate_kernel, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice) );

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

    // free collisions
    cudaFreeHost(h_sampleFile_buff);
    cudaFree(d_collision);
    free((void*)h_sampleFile_buff);

    //===========================================================================================================
    // WRITE COLLISIONS TO DISK
    //===========================================================================================================

    printf("Original string: %s\n", h_sampleFile_buff);
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        printf("Collision %d: %s\n", i, h_collisions[i]);

        // todo write collision

        // free collision once written
        free(h_collisions[i]);
    }
}
