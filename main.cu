/*
 * Author:  Hampton Ford
 * Github:  @shford
 * Status: Incomplete
 *
 * License:
 *
 *
 * CUDA_Driver_Version:                         11.7
 * CUDA Capability Major/Minor version number:  8.6
 *
 * Performance:
 * ~todo time
 *
 */

#include <iostream>
#include <math.h>

#include "subroutines.h"
#include "WjCryptLib_Md5.h"

#define CUDA_API_PER_THREAD_DEFAULT_STREAM per-thread
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_consts.cuh"


// group initial non-parallel tasks for neatness
__host__ int initial_sequential_tasks(char** sample_file_buff, long* sample_buff_size, MD5_HASH* md5_digest)
{
    // read file data
    char sample_file_path[] = "C:/Users/C22Steven.Ford/Downloads/Fall 2022/CyS 431 - Crypto/PEX 2/samplefile.txt";
    get_file_data(sample_file_path, sample_file_buff, sample_buff_size);

    // get hash md5_digest: 5442f94666075ef8d695109af238a5db
    Md5Calculate_host(*sample_file_buff, (uint32_t)strlen(*sample_file_buff), md5_digest);

    // format and print digest as a string of hex characters
    char hash[MD5_HASH_SIZE_B + 1]; //MD5 len is 16B, 1B = 2 chars

    char tiny_hash[TINY_HASH_SIZE_B + 1];
    for (int i = 0; i < MD5_HASH_SIZE / 2; ++i)
    {
        sprintf(hash + i * 2, "%2.2x", (*md5_digest).bytes[i]);
    }
    hash[sizeof(hash)-1] = '\0';
    strncpy_s(tiny_hash, sizeof(tiny_hash), hash, _TRUNCATE);
    tiny_hash[sizeof(tiny_hash)-1] = '\0';

    printf("Full MD5 md5_digest is: %s\n", hash);
    printf("TinyHash md5_digest is: %s\n\n", tiny_hash);

    return 0;
}

// resize cuda buffer
__device__ void cuda_resize_buff(char** parent_scope_buff_dev, long src_buff_size_dev, long dst_buff_size_dev)
{
    char* tmp_src_buff_dev;

    // save original buffer to free after copying
    tmp_src_buff_dev = *parent_scope_buff_dev;

    // allocate new destination buffer (pass by reference)
    cudaMalloc(parent_scope_buff_dev, dst_buff_size_dev);

    // copy original buffer to dst
    cudaMemcpyAsync(parent_scope_buff_dev, tmp_src_buff_dev, src_buff_size_dev, cudaMemcpyDeviceToDevice, 0);

    // free original buffer
    cudaFree(tmp_src_buff_dev);
}

// kernel
__global__
void find_collisions(__device__ __constant__ MD5_HASH original_md5_hash, const char* original_buff_host, const int original_buff_size_host, int* unsafe_collision_counter_host, unsigned long long* unsafe_hash_attempts_host)
{
    // allocate maximum unified memory per thread (gpu space inefficient, but prevents costly buffer resizing)
    char* new_buff_dev;
    long new_buff_size_dev = maximum_evenly_distributed_mem_per_thread_Mb;
    cudaMalloc(&new_buff_dev, new_buff_size_dev);

    // copy host buff (in RAM or CPU cache) to device buff (GPU shared mem w/in SM - e.g. w/in block)
    cudaMemcpyAsync(&new_buff_dev, original_buff_host, original_buff_size_host, cudaMemcpyHostToDevice, 0);

    // append to new_buff until a collision is reached (e.g. tiny_hashes match)
    MD5_HASH new_md5_digest;
    int append_index = original_buff_size_host - 1; // to overwrite null terminator
    do {
        // double buff as needed
        if (append_index + 1 == new_buff_size_dev) {
            new_buff_size_dev *= 2;
            cuda_resize_buff(&new_buff_dev, append_index + 1, new_buff_size_dev);
        }

        // try every 8-bit ascii character except the null terminator "0x00"
        for (int i = 1; i < 256; ++i) {
            // race conditions are fine here - the attempt count only needs to be approximate
            ++(*unsafe_hash_attempts_host);

            new_buff_dev[append_index] = (char)i;
            new_buff_dev[append_index + 1] = '\0';

            Md5Calculate_device(new_buff_dev, (uint32_t)(append_index + 1), &new_md5_digest);

            // exit if first 20 bits of hashes are equal
            if (((uint32_t)*(original_md5_hash.bytes) >> 12) == ((uint32_t)*(new_md5_digest.bytes) >> 12)) {
                ++(*unsafe_collision_counter_host);
            }

            // exit if first 20 bits of the hashes are equal
//            if (original_md5_hash.bytes[0] == new_md5_digest.bytes[0] &&
//                original_md5_hash.bytes[1] == new_md5_digest.bytes[1] &&
//                original_md5_hash.bytes[2] >> 4 == new_md5_digest.bytes[2] >> 4) {
//                goto write_collision;
//            }
        }

        // pseudo randomly append a character - ignore seeding
        new_buff_dev[append_index] = truncf((256 + 0.999999) * curand_uniform() );
        new_buff_dev[append_index + 1] = '\0';

        // increment new_buff index
        ++append_index;
    } while(true);

    // write file todo append traditional shared variable collision_id
    write_collision:
    char output_filename[MAX_FILENAME];
    printf("Saving collision as file: %s.\n", output_filename);
    write_file_data(output_filename, new_buff_dev, strlen(new_buff_dev));

    // free unified memory - leave original buffer RO b/c it's shared w/ other blocks
    cudaFree(new_buff_dev);
}


int main()
{
    printf("Hash Collider - Starting Task 1...");

    // execute sequential tasks
    char* sample_buff;
    long sample_buff_size;
    MD5_HASH md5_digest;
    initial_sequential_tasks(&sample_buff, &sample_buff_size, &md5_digest);

    // 4864 cores/kernels/threads - max number of concurrent kernels possible at time
    int unsafe_collision_counter_host;
    unsigned long long unsafe_hash_attempts_host;
    find_collisions<<<multiprocessors, cuda_cores_per_multiprocessor>>>(md5_digest, sample_buff, sample_buff_size, &unsafe_collision_counter_host, &unsafe_hash_attempts_host);

    // wait for GPU to finish before accessing on host
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // free host memory
    free(sample_buff);

    return 0;
}
