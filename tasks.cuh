#pragma once

#ifndef CUDA_HASHING_TASKS_CUH
#define CUDA_HASHING_TASKS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "cuda_consts.cuh"
#include "kernels.cuh"
#include "subroutines.cuh"
#include "WjCryptLib_Md5.cuh"

void task1();

#endif //CUDA_HASHING_TASKS_CUH
