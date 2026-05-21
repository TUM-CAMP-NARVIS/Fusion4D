#ifndef __NORMALEQUATIONSOLVERPCGCUDA_GPU_H__
#define __NORMALEQUATIONSOLVERPCGCUDA_GPU_H__
#include "lmsolver_gpu.h"
#include "PCGCuda_GPU.cuh"

#include "utility.h"

namespace Fusion4D_GPU
{
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
    namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "NormalEquationSolverPCGCuda.h"
}

#endif