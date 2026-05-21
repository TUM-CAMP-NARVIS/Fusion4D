// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef __DEFORMGRAPH_CUDA_GPU_H__
#define __DEFORMGRAPH_CUDA_GPU_H__

#include "DeformGraphCore.h"
#include "DeformGraphCudaImpl_GPU.cuh"
#include "VoxelMatrix.h"
#include "utility.h"

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "DeformGraphCuda.h"
}  // namespace Fusion4D_GPU

#endif