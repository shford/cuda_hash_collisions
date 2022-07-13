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

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

/*
 * set shared memory capacity to 0 - cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
 *
 * todo V5:
 *  Multi-Thread/Multi-Process - Initial & Final File I/O
 *  Multi-Thread/Multi-Process - Task 2
 *  Replace calloc inside read_file() w/ cudaMallocHost() to pin memory
 *  Replace size w/ strlen calls in host b/c no need to transfer that
 *  In kernel change from byte-by-byte copying to memcpyasync or at least by multiples of largest primitive
 */


int main()
{
    printf("Hash Collider - Starting Task 1...");
    task1();

    return 0;
}
