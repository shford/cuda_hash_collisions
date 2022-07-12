cmake_minimum_required(VERSION 3.17)
#set(CMAKE_VERBOSE_MAKEFILE ON)


set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 14)

project(cuda_hashing CUDA)


add_executable(cuda_hashing main.cu tasks.cu tasks.cuh subroutines.cu subroutines.cuh cuda_consts.cuh WjCryptLib_Md5.cu WjCryptLib_Md5.cuh)

set_target_properties(
        cuda_hashing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

