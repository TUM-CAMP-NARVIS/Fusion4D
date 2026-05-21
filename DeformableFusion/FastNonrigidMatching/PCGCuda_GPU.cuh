#ifndef __PCGCUDA_GPU_CUH__
#define __PCGCUDA_GPU_CUH__
#include "EDMatchingHelperCudaImpl_GPU.cuh"
#include "matrix_types_cuda.h"
#include "../Common/cuda/PinnedMemory.h"

namespace Fusion4D_GPU
{
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
    namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "PCGCuda.cuh"
}
#endif