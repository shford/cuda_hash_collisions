/*
 * Author:      Hampton Ford
 * Github:      @shford
 * Status:      Incomplete
 *
 * Notes:       Max file size set at 4GiB b/c of Crypto Library limitations
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

#include "tasks.cuh"

//#define CUDA_API_PER_THREAD_DEFAULT_STREAM

/*
 * todo
 * set shared memory capacity to 0 - cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
 *
 * V5:
 *  Multi-Thread/Multi-Process Initial & Final File I/O
 *  Multi-Thread/Multi-Process Task 2
 */


// resize cuda buffer
/*
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
*/

// kernel
/*
__global__ void find_collisions(__device__ __constant__ MD5_HASH original_md5_hash, const char* original_buff_host, const int original_buff_size_host, int* unsafe_collision_counter_host, unsigned long long* unsafe_hash_attempts_host)
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
*/


int main()
{
    printf("Hash Collider - Starting Task 1...");
    task1();

    return 0;
}
