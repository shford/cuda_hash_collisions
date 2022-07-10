cmake_minimum_required(VERSION 3.17)

project(cuda_hashing CUDA)

set(CMAKE_CUDA_STANDARD 14)

#add_executable(cuda_hashing main.cu WjCryptLib_Md5.c WjCryptLib_Md5.h subroutines.c subroutines.h cuda_consts.cuh)
add_executable(cuda_hashing cuda_scoping.cu)

set_target_properties(
        cuda_hashing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)