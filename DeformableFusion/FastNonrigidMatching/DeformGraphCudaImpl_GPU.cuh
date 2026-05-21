// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef __DEFORMGRAPHCUDAIMPL_GPU_CUH__
#define __DEFORMGRAPHCUDAIMPL_GPU_CUH__
#include "cuda_vector_fixed.cuh"
#include "cuda_matrix_fixed.cuh"
#include <helper_cuda.h>
#include "CudaGlobalMemory.h"
#include "geometry_types_cuda.h"

#include "utility.h"
#include "../Common/cuda/PinnedMemory.h"

#include "DeformGraphCudaImpl.cuh"

namespace Fusion4D_GPU
{
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
    namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
    using ::DeformGraphCudaImpl;
}

#endif