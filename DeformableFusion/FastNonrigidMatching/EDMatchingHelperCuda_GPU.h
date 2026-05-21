// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef __EDMATCHINGHELPERCUDA_GPU_CUH__
#define __EDMATCHINGHELPERCUDA_GPU_CUH__

#include <opencv2/opencv.hpp>

#include "../Common/cuda/PinnedMemory.h"
#include "BlockedHessianMatrix.h"
#include "CameraView.h"
#include "DeformGraphCuda_GPU.h"
#include "EDMatchingHelperCudaImpl_GPU.cuh"
#include "UtilVnlMatrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utility.h"
#include "voxel_data_io.h"

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "EDMatchingHelperCuda.h"
}  // namespace Fusion4D_GPU

#endif
