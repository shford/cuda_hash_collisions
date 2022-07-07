#include <iostream>
#include <math.h>
#include "cuda_consts.cuh"

char to_append[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()-=_+[]{};<>/?";
int collision_id = 0;

// Kernel function to add the elements of two arrays
void find_collisions(const MD5_HASH original_md5_hash, const char* original_traditional_buff, const int current_buff_size)
{
    // allocate maximum unified memory per thread (gpu space inefficient, but prevents costly buffer resizing)
    char* new_unified_buffer;
    gpuErrchk( cudaMallocManaged(&new_unified_buffer, current_buff_size) );


    // append to new_buff until a collision is reached (e.g. tiny_hashes match) todo left off here
    MD5_HASH new_md5_digest;
    int hash_attempts = 0;
    int new_buff_append_index = buff_size - 1;
    do {
        // double buff as needed
        if (new_buff_append_index+1 >= new_buff_size) {
            printf("Reallocating buff.\n");
            new_buff_size *= 2;
            new_buff = resize_buff(new_buff, new_buff_size);
        }

        // try every character for the base buffer
        for (int i=0; i < strlen(to_append); ++i) {
            ++hash_attempts;

            new_buff[new_buff_append_index] = to_append[i];
            new_buff[new_buff_append_index + 1] = '\0';

            Md5Calculate(new_buff, (uint32_t)strlen(new_buff), &new_md5_digest);

            // exit if first 20 bits of the hashes are equal
            if (original_md5_hash.bytes[0] == new_md5_digest.bytes[0] &&
                original_md5_hash.bytes[1] == new_md5_digest.bytes[1] &&
                original_md5_hash.bytes[2] >> 4 == new_md5_digest.bytes[2] >> 4) {
                goto write_collision;
            }
        }

        // pseudo randomly append a character - ignore seeding
        new_buff[new_buff_append_index] = to_append[rand() % strlen(to_append)];
        new_buff[new_buff_append_index + 1] = '\0';

        // increment new_buff index
        ++new_buff_append_index;
        if (hash_attempts % 1000 == 0) {
            printf("Hash attempt: %d.\n", hash_attempts);
        }
    } while(1);

    // write file todo append traditional shared variable collision_id
write_collision:
    printf("Saving collision as file: %s.\n", output_filename);
    char output_filename[MAX_FILENAME];
    sprintf(output_filename, "C:/Users/C22Steven.Ford/Downloads/Fall 2022/CyS 431 - Crypto/PEX 2/collision%d.txt", collision_id);
    write_file_data(output_filename, new_unified_buffer, new_buff_size);

    // free unified memory - leave original buffer RO b/c it's shared w/ other blocks
    cudaFree(new_unified_buffer);
}