/*
 * Constants for the RTX 3060-Ti obtained from Device Query (sample program)
 */
#pragma once

#ifndef CUDA_HASHING_CUDA_CONSTS_CUH
#define CUDA_HASHING_CUDA_CONSTS_CUH

#include <stdio.h>


//===========================================================================================================
// DEVICE DATA (from DeviceQuery.exe)
//===========================================================================================================
#define GLOBAL_MEMORY_MB 8192
#define CUDA_CORES 4864
#define MULTIPROCESSORS 38
#define CUDA_CORES_PER_MULTIPROCESSOR 128
#define MAX_MEM_PER_ACTIVE_THREAD (GLOBAL_MEMORY_MB*1024*1024 / CUDA_CORES)

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

//===========================================================================================================
// USER DEFINED CONSTANTS
//===========================================================================================================
#define TARGET_COLLISIONS (5)
#define ARBITRARY_MAX_BUFF_SIZE (600)
#define NUM_8BIT_RANDS (48)
#define NUM_32BIT_RANDS (NUM_8BIT_RANDS / 4)
#define ENCODING_SIZE (256)
#define FALSE (0)
#define TRUE (1)

//===========================================================================================================
// CUDA ERROR CATCHING MACRO
// NOTE: ONLY RUN THIS BEFORE <<<>>> || AFTER FINAL SYNCHRONIZATION (IMPLICIT OR EXPLICIT)
//===========================================================================================================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: Err no: %d, code: %s %s %d\n", code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif //CUDA_HASHING_CUDA_CONSTS_CUH
