#ifndef __EDMATCHINGHELPERCUDAIMPL_GPU_CUH__
#define __EDMATCHINGHELPERCUDAIMPL_GPU_CUH__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaGlobalMemory.h"
#include "DeformGraphCudaImpl_GPU.cuh"
#include "matrix_types_cuda.h"
#include "utility.h"
#include "../Common/cuda/PinnedMemory.h"

namespace Fusion4D_GPU
{
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
    namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "EDMatchingHelperCudaImpl.cuh"
}

#endif