#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

inline void __checkCudaErrors(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

#ifndef checkCudaErrors
#define checkCudaErrors(val) __checkCudaErrors((val), __FILE__, __LINE__)
#endif
