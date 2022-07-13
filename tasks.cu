#pragma once

#include "tasks.cuh"


__constant__ __device__ MD5_HASH d_const_md5_digest;    // store digest on L2 or L1 cache (on v8.6)
__device__ uint8_t d_collisions_found;                  // track number of collisions found by active kernel
__device__ unsigned long long d_collision_size;         // track # of characters in collision
__device__ int d_collision_flag;                        // signal host to read
__device__ unsigned long long d_hash_attempts;          // track total number of attempts per collision


__global__ void find_collisions(char* collision) {
    //===========================================================================================================
    // DECLARATIONS & INITIALIZATION
    //===========================================================================================================
    // allocate local buffer and keep track of size in case of resizing
    char* local_collision;
    unsigned long long local_collision_size = d_collision_size;
    unsigned long long local_buff_size = ARBITRARY_MAX_BUFF_SIZE;
    cudaMalloc(&local_collision, local_buff_size);
    for (int byte_index = 0; byte_index <= local_collision_size; ++byte_index) {
        local_collision[byte_index] = collision[byte_index];
    }

    // allocate room for new hash
    MD5_HASH local_md5_digest;

    // allocate storage for random character
    unsigned long long random_index = 0;
    uint8_t randoms[NUM_8BIT_RANDS];

    //===========================================================================================================
    // COMPUTATIONS - GENERATE RANDS, RESIZE BUFFER, APPEND CHAR, HASH, COMPARE { EXIT }
    //===========================================================================================================
    do
    {
        ++d_hash_attempts;

        // generate a new batch of random numbers as needed
        if (random_index == NUM_8BIT_RANDS) {
            random_index = 0;
            for (int i = 0; i < NUM_32BIT_RANDS; ++i) {
                int id = threadIdx.x + blockIdx.x * blockDim.x;
                curandStatePhilox4_32_10_t state;
                curand_init(i, id, 0, &state);
                // assign 4 bytes at a time
                randoms[i*4] = curand(&state);
            }
        }
        ++random_index;

        // resize local_collision
        if (local_collision_size == ARBITRARY_MAX_BUFF_SIZE) {
            // retain ptr to old buffer
            char* old_buff = local_collision;

            // reassign local_collision ptr to new buffer
            local_buff_size *= 2;
            cudaMalloc(&local_collision, local_buff_size);

            // copy data from old buffer to new buffer
            for (int i = 0; i < ARBITRARY_MAX_BUFF_SIZE; ++i) {
                local_collision[i] = old_buff[i];
            }

            // free original buffer
            cudaFree(old_buff);
        }

        // append random char
        uint8_t character = randoms[random_index];
        local_collision[local_collision_size - 1] = character ? character : 1; // no premature null terminators
        local_collision[local_collision_size] = '\0';
        ++local_collision_size;

        // generate new hash
        Md5Calculate((const void*)local_collision, local_collision_size, &local_md5_digest);

        // terminate all threads if first 20 bits of digest match
        if ( ((uint32_t)*d_const_md5_digest.bytes >> 12) == ((uint32_t)*local_md5_digest.bytes >> 12))
        {
            /* todo:
             *  unlikely but possible device wide deadlock if within the same warp
             *  1 thread sets a mutex causing a divergent instruction path and the
             *  scheduler interrupts said thread to schedule another which will then idle
             *  forever, thus preventing the mutex thread from completing.
             *  May want to utilize capability: " Run time limit on kernels:                     Yes"
             */

            // wait for resources to be released before waiting
            while (d_collision_flag) {
                // idle
            }

            // set synchronization barrier/mutex on d_collision_flag, d_collision_size, collision


            // write local_data, local_data_size to global for host polling
            for (int byte_index = 0; byte_index <= local_collision_size; ++byte_index) {
                collision[byte_index] = local_collision[byte_index];
            }

            // tell host to read collision
            d_collision_flag = TRUE;

            while (d_collision_flag) {
                // release synchronization barrier/mutex once host has reading and resets flag
            }
        }
    } while(d_collisions_found < TARGET_COLLISIONS);

}

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

    // allocate storage for collisions once found
    char* h_collisions[TARGET_COLLISIONS];
    unsigned long long h_collision_sizes[TARGET_COLLISIONS];
    unsigned long long h_collision_attempts = 0;
    for (int i = 0; i < TARGET_COLLISIONS; ++i) {
        h_collisions[i] = (char*)calloc(1, ARBITRARY_MAX_BUFF_SIZE);
        h_collision_sizes[i] = 0;
    }
    uint8_t h_num_collisions_found = 0;
    int h_collision_flag = FALSE;

    // allocate global mem for collision - initialized in loop
    char* d_collision;
    gpuErrchk( cudaMalloc((void **)&d_collision, ARBITRARY_MAX_BUFF_SIZE) );

    // parallelization setup
    gpuErrchk( cudaMemcpyToSymbol(d_collisions_found, &h_num_collisions_found, sizeof(h_num_collisions_found), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_collision_flag, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_const_md5_digest, &md5_digest, sizeof(md5_digest), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_collision_size, &h_sampleFile_buff_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_collision, h_sampleFile_buff, h_sampleFile_buff_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpyToSymbol(d_hash_attempts, &h_collision_attempts, sizeof(h_collision_attempts), 0, cudaMemcpyHostToDevice) );

    // run kernel
    while (cudaMemcpyFromSymbol(&h_num_collisions_found, d_collisions_found, sizeof(h_num_collisions_found), 0, cudaMemcpyDeviceToHost) == cudaSuccess && h_num_collisions_found < TARGET_COLLISIONS)
    {
        // execution configuration (sync device)
        find_collisions<<<MULTIPROCESSORS, CUDA_CORES_PER_MULTIPROCESSOR>>>(d_collision);

        // poll collision flag
        while (!h_collision_flag)
        {
            // read collision status from device into host flag
            gpuErrchk( cudaMemcpyFromSymbol(&h_collision_flag, d_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyDeviceToHost) );

            if (h_collision_flag)
            {
                // read updated collision count, collision size, collision, and hash attempts
                gpuErrchk(cudaMemcpyFromSymbol(&h_num_collisions_found, d_collisions_found, sizeof(h_num_collisions_found), 0, cudaMemcpyDeviceToHost) );
                gpuErrchk(cudaMemcpyFromSymbol(&h_collision_sizes[h_num_collisions_found], d_collision_size, sizeof(h_sampleFile_buff_size), 0, cudaMemcpyDeviceToHost) );
                gpuErrchk(cudaMemcpy(h_collisions[h_num_collisions_found], d_collision, h_sampleFile_buff_size, cudaMemcpyDeviceToHost) );
                gpuErrchk(cudaMemcpyFromSymbol(&h_collision_attempts, h_collision_attempts,
                                               sizeof(h_collision_attempts), 0, cudaMemcpyDeviceToHost) );

                // reset flag to continue
                h_collision_flag = FALSE;
                cudaMemcpyToSymbol(d_collision_flag, &h_collision_flag, sizeof(h_collision_flag), 0, cudaMemcpyHostToDevice);

            }
        }
    }
    gpuErrchk( cudaDeviceSynchronize() );

    // error handling
    if (h_num_collisions_found < TARGET_COLLISIONS)
    {
        printf("Error: Failed to update the number of collisions from h_num_collisions_found.\n");
    }
    else
    {
        printf("\nCalculated %d collisions... Success/\n", TARGET_COLLISIONS);
    }

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
