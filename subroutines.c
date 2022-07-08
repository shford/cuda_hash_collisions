/*
 * Subtasks for main.c
 *      - reads bin data from a file
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "subroutines.h"


// pass by reference propagated file buffer and file size
void get_file_data(char file_path[], char** file_buff, long* buff_size) {
    // open file
    FILE* file_handle;
    fopen_s(&file_handle, file_path, "rb");
    if (file_handle == NULL) {
        printf_s("Failed to open %s", file_path);
        exit(-1);
    }

    // get file size
    if (fseek(file_handle, 0, SEEK_END) != SEEK_SUCCESS) {
        printf_s("Failed to seek EOF for %s", file_path);
        fclose(file_handle);
        exit(-1);
    }
    *buff_size = ftell(file_handle) + 1;
    rewind(file_handle);

    // allocate buffer
    *file_buff = calloc(*buff_size, 1);
    if (*file_buff == NULL) {
        printf_s("Failed to allocate adequate space for %s", file_path);
        fclose(file_handle);
        exit(-1);
    }

    // read data into buffer
    while (!feof(file_handle))
    {
        fread(*file_buff, *buff_size - 1, 1, file_handle);
    }
    (*file_buff)[*buff_size] = '\0';

    // close file
    fclose(file_handle);
}

// different args since pass by reference is unnecessary
__device__ void write_file_data(char file_path[], char* buff, int buff_size) {
    FILE* file_handle;
    fopen_s(&file_handle, file_path, "wb");
    if (file_handle == NULL) {
        printf_s("Failed to open %s", file_path);
        exit(-1);
    }

    fwrite(buff, sizeof(typeof(buff)), buff_size, file_handle);

    // fwrite error handling done by file IO stream
    if (fclose(file_handle) != 0) {
        printf("Failed to write to path: %s.\n", file_path);
        exit(-1);
    }

}
