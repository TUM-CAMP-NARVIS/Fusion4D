#ifndef __VOLUMETRIC_FUSION_GPU_CUH__
#define __VOLUMETRIC_FUSION_GPU_CUH__
#include <opencv2/opencv.hpp>
#include <vector>

#include "../Common/cuda/PinnedMemory.h"
#include "BoundingBox3D.h"
#include "CPointCloud.h"
#include "CameraView.h"
#include "TSDF.h"
#include "UtilMatrix.h"
#include "UtilVnlMatrix.h"
#include "color_map.h"
#include "utility.h"
#include "volumetric_fusion_impl_GPU.cuh"

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "volumetric_fusion.h"
}  // namespace Fusion4D_GPU

#endif
