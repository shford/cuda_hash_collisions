cmake_minimum_required(VERSION 3.17)

project(hello_cuda CUDA)


set(CMAKE_CUDA_STANDARD 14)

add_executable(hello_cuda main.cu subroutines.h kernel.c cuda_consts.cuh)

set_target_properties(
        hello_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)