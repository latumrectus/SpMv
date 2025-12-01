cmake_minimum_required(VERSION 3.25)
project(sparse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(sparse src/main.cpp
        src/core/csr_matrix.h
        src/io/mtx_reader.h)

target_link_libraries(sparse PRIVATE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cusparse
)