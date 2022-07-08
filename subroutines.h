//
// Created by shford on 7/6/2022.
//

#ifndef HELLO_CUDA_SUBROUTINES_H
#define HELLO_CUDA_SUBROUTINES_H

#include "WjCryptLib_Md5.h"

#define MD5_HASH_SIZE_B (128 / 8)
#define TINY_HASH_SIZE_B 5
#define MAX_FILENAME 260
#define SEEK_SUCCESS 0


void get_file_data(char file_path[], char** file_buff, long* buff_size);

void write_file_data(char file_path[], char* buff, int buff_size);

#endif //HELLO_CUDA_SUBROUTINES_H
