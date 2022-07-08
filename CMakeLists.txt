cmake_minimum_required(VERSION 3.17)

project(hello_cuda CUDA)


set(CMAKE_CUDA_STANDARD 14)

#add_executable(hello_cuda main.cu WjCryptLib_Md5.c WjCryptLib_Md5.h subroutines.c subroutines.h cuda_consts.cuh)
add_executable(hello_cuda cuda_scoping.cu)

set_target_properties(
        hello_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)