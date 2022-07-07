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
 * This version of the collision is organized into a separate because it's a useful
 * future reference for CUDA programming.
 *
 * Performance:
 * ~ time
 *
 */

#include <iostream>
#include <math.h>
#include "cuda_consts.cuh"
#include "subroutines.h"
#include "WjCryptLib_Md5.h"

int initial_sequential_tasks(char** sample_file_buff, long* sample_buff_size, MD5_HASH* md5_digest);

__global__
void find_collisions(MD5_HASH* original_md5_hash, const char* buff, int buff_size, MD5_HASH* md5_digest);


int main()
{
    printf("Hash Collider - Starting Task 1...");

    // sequential tasks
    char* sample_buff;
    long sample_buff_size;
    MD5_HASH md5_digest;
    initial_sequential_tasks(&sample_buff, &sample_buff_size, &md5_digest);

    // 4864 cores/kernels/threads - max number of concurrent kernels possible at time on the 3060-Ti
    find_collisions<<<multiprocessors, cuda_cores_per_multiprocessor>>>(md5_digest, sample_buff, sample_buff_size);

    // Wait for GPU to finish before accessing on host
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Free memory
    free(sample_buff);

    printf("Hash Collider - Starting Task 2...");

    return 0;
}

int initial_sequential_tasks(char** sample_file_buff, long* sample_buff_size, MD5_HASH* md5_digest)
{
    // read file data
    char sample_file_path[] = "C:/Users/C22Steven.Ford/Downloads/Fall 2022/CyS 431 - Crypto/PEX 2/samplefile.txt";
    get_file_data(sample_file_path, sample_file_buff, sample_buff_size);

    // get hash md5_digest: 5442f94666075ef8d695109af238a5db
    Md5Calculate(*sample_file_buff, (uint32_t)strlen(*sample_file_buff), md5_digest);

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