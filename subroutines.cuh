#pragma once

#ifndef CUDA_HASHING_SUBROUTINES_CUH
#define CUDA_HASHING_SUBROUTINES_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define MD5_HASH_SIZE_B (128 / 8)
#define TINY_HASH_SIZE_B 5
#define MAX_FILENAME 260
#define SEEK_SUCCESS 0


void get_file_data(char file_path[], char** file_buff, uint32_t* buff_size);

void write_file_data(char file_path[], char* buff, int buff_size);

#endif //CUDA_HASHING_SUBROUTINES_CUH
