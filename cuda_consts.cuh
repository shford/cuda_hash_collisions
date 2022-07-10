/*
 * Constants for the RTX 3060-Ti obtained from Device Query (sample program)
 */

#ifndef CUDA_HASHING_CUDA_CONSTS_CUH
#define CUDA_HASHING_CUDA_CONSTS_CUH

#define global_memory_Mb 8192
#define multiprocessors 38
#define cuda_cores 4864
#define cuda_cores_per_multiprocessor 128
#define memory_bus_width 256
#define L2_Cache_Size 3145728
#define maximum_texture_dimension_size_1D 131072
#define maximum_texture_dimension_size_2D (131072, 65536)
#define maximum_texture_dimension_size_3D  (16384, 16384, 16384)
#define maximum_layered_texture_size_1D (32768)
#define maximum_num_layers_1D 2048
#define maximum_layered_texture_size_2D (32768, 32768)
#define maximum_num_layers_2D 2048
#define total_number_of_registers_available_per_block 65536
#define warp_size 32
#define maximum_number_of_threads_per_multiprocessor 1536
#define maximum_number_of_threads_per_block 1024
#define max_dimension_size_of_a_thread_block (1024, 1024, 64)
#define max_dimension_size_of_a_grid_size (2147483647, 65535, 65535)

#define maximum_evenly_distributed_mem_per_thread_Mb (global_memory_Mb / cuda_cores)

/*
 * NOTE: ONLY RUN THIS BEFORE <<<>>> || AFTER FINAL SYNCHRONIZATION (IMPLICIT OR EXPLICIT)
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //CUDA_HASHING_CUDA_CONSTS_CUH
