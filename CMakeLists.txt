cmake_minimum_required(VERSION 3.24)

project(DeformableFusionDemoBuild LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_COMPILE_FEATURES_WORKAROUND OFF)

add_compile_options(-Wno-return-type -Wno-permissive)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=1650>)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures; 75 matches RTX 2080 Ti")
endif()

set(PEABODY_VOLUME_CUBES_DIM_MAX 50 CACHE STRING "Maximum two-level TSDF volume dimension in coarse cubes")

set(PEABODY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DEFORMABLE_FUSION_DIR "${PEABODY_ROOT}/DeformableFusion")
set(PEABODY_COMPAT_INCLUDE_DIR "${PEABODY_ROOT}/include/compat")

set(OpenCV_DIR "$ENV{HOME}/opencv/build" CACHE PATH "OpenCV CUDA build directory")
set(VXL_DIR "${PEABODY_ROOT}/Dependencies/vxl-install/share/vxl/cmake" CACHE PATH "VXLConfig.cmake directory")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VXL REQUIRED)
find_package(Ceres REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

find_library(CHOLMOD_LIBRARY cholmod REQUIRED)
find_library(AMD_LIBRARY amd REQUIRED)
find_library(CAMD_LIBRARY camd REQUIRED)
find_library(COLAMD_LIBRARY colamd REQUIRED)
find_library(CCOLAMD_LIBRARY ccolamd REQUIRED)
find_library(SUITESPARSECONFIG_LIBRARY suitesparseconfig REQUIRED)
find_library(METIS_LIBRARY NAMES metis libmetis.so.5 PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_path(SUITESPARSE_INCLUDE_DIR
  NAMES cholmod.h
  PATH_SUFFIXES suitesparse
  REQUIRED
)


set(VXL_INCLUDE_DIRS
  "${VXL_VCL_INCLUDE_DIRS}"
  "${VXL_CORE_INCLUDE_DIRS}"
  "${PEABODY_ROOT}/Dependencies/vxl-install/include/vxl/contrib"
  "${PEABODY_ROOT}/Dependencies/vxl-install/include/vxl/contrib/cul"
  "${PEABODY_ROOT}/Dependencies/vxl-install/include/vxl/contrib/brl/bbas"
  "${PEABODY_ROOT}/Dependencies/vxl-install/include/vxl/contrib/brl/bseg"
  "${VXL_OXL_INCLUDE_DIR}"
)

set(VXL_CORE_LIBS
  bundler
  bapl
  brip
  mvl
  vil_algo
  mbl
  #mbl_io
  GL
  GLU
  vidl
  vpgl_algo
  vpgl
  vnl_algo
  vnl
  vgl_algo
  vgl
  vil
  vil_io
  vsl
  vbl
  vul
  vpl
  vcl
  v3p_netlib
)

link_directories("${VXL_LIBRARY_DIR}")

add_library(project_options INTERFACE)
target_include_directories(project_options INTERFACE
  "${PEABODY_COMPAT_INCLUDE_DIR}"
  "${PEABODY_ROOT}/McLib2/LibUtility/include"
  "${DEFORMABLE_FUSION_DIR}/Common/cuda"
  "${DEFORMABLE_FUSION_DIR}/Utility"
  "${DEFORMABLE_FUSION_DIR}/CameraView"
  "${DEFORMABLE_FUSION_DIR}/CSurface"
  "${DEFORMABLE_FUSION_DIR}/Basics"
  "${DEFORMABLE_FUSION_DIR}/NonRigidMatch"
  "${DEFORMABLE_FUSION_DIR}/FastNonrigidMatching"
  "${DEFORMABLE_FUSION_DIR}/FusionDemo-MultiView"
  "${VXL_INCLUDE_DIRS}"
  "${OpenCV_INCLUDE_DIRS}"
  "${JSONCPP_INCLUDE_DIRS}"
  "${SUITESPARSE_INCLUDE_DIR}"
)
target_compile_definitions(project_options INTERFACE
  MCLIB2_CONFIG_INC=<Config.h>
  TWO_LEVEL_VOLUME_CUBES_DIM_MAX=${PEABODY_VOLUME_CUBES_DIM_MAX}
)
target_compile_options(project_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-fpermissive>
  $<$<COMPILE_LANGUAGE:CXX>:-include${PEABODY_COMPAT_INCLUDE_DIR}/LinuxCompat.h>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

function(peabody_static_library target)
  add_library(${target} STATIC ${ARGN})
  target_link_libraries(${target} PUBLIC project_options)
  set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

peabody_static_library(LibUtility
  McLib2/LibUtility/src/Config.cpp
  McLib2/LibUtility/src/Logger.cpp
  McLib2/LibUtility/src/TimeUtil.cpp
)

peabody_static_library(Utility
  DeformableFusion/Utility/CConnectedComponent.cpp
  DeformableFusion/Utility/CustomSVD.cpp
  DeformableFusion/Utility/iniFileParser.cpp
  DeformableFusion/Utility/MorphologicalOperation.cpp
  DeformableFusion/Utility/UtilMatrix.cpp
  DeformableFusion/Utility/UtilVnlMatrix.cpp
  DeformableFusion/Utility/stdafx.cpp
  DeformableFusion/Utility/utility.cpp
  DeformableFusion/Utility/zlib_compression.cpp
)
target_link_libraries(Utility PUBLIC LibUtility ZLIB::ZLIB ${OpenCV_LIBS} ${VXL_CORE_LIBS})

peabody_static_library(CameraView
  DeformableFusion/CameraView/CameraView.cpp
)
target_link_libraries(CameraView PUBLIC Utility ${OpenCV_LIBS} ${JSONCPP_LIBRARIES})

peabody_static_library(CSurface
  DeformableFusion/CSurface/CIsoSurface.cpp
  DeformableFusion/CSurface/CSurface.hpp
  DeformableFusion/CSurface/ply.cpp
  DeformableFusion/CSurface/Vectors.cpp
)
target_link_libraries(CSurface PUBLIC Utility CameraView OpenGL::GL GLUT::GLUT)

peabody_static_library(Basics
  DeformableFusion/Basics/basic_geometry.cpp
  DeformableFusion/Basics/basic_structure.cpp
  DeformableFusion/Basics/DDF.cpp
  DeformableFusion/Basics/depth_map_filter.cpp
  DeformableFusion/Basics/distortion_correction.cpp
  DeformableFusion/Basics/KL_features.cpp
  DeformableFusion/Basics/normal_estimation.cpp
  DeformableFusion/Basics/voxel_data_io.cpp
  DeformableFusion/Basics/S3DPointMatchSet.cpp
  DeformableFusion/Basics/stdafx.cpp
  DeformableFusion/Basics/track_extraction_basic.cpp
  DeformableFusion/Basics/TSDF.cpp
  DeformableFusion/Basics/visual_feature_matching.cpp
)
target_link_libraries(Basics PUBLIC Utility CameraView CSurface ZLIB::ZLIB ${OpenCV_LIBS} ${VXL_CORE_LIBS})

peabody_static_library(NonRigidMatch
  DeformableFusion/NonRigidMatch/affines_nonrigid_pw_opt.cpp
  DeformableFusion/NonRigidMatch/DeformationGraphOptBasic_ceres.cpp
  DeformableFusion/NonRigidMatch/DeformationGraphOptPlus_ceres.cpp
  DeformableFusion/NonRigidMatch/DeformGraph.cpp
  DeformableFusion/NonRigidMatch/DeformGraphOptDensePtsAndClr_ceres.cpp
  DeformableFusion/NonRigidMatch/LMICP.cpp
  DeformableFusion/NonRigidMatch/nonrigid_matching_surface.cpp
  DeformableFusion/NonRigidMatch/surface_geodesic_distance.cpp
)
target_link_libraries(NonRigidMatch PUBLIC Utility CameraView CSurface Basics Ceres::ceres ${OpenCV_LIBS} ${VXL_CORE_LIBS})

peabody_static_library(FastNonrigidMatching
  DeformableFusion/FastNonrigidMatching/CMotionFieldEstimationF2F.cpp
  DeformableFusion/FastNonrigidMatching/CudaGlobalMemory.cpp
  DeformableFusion/FastNonrigidMatching/DeformGraphCuda_GPU.cpp
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCuda_GPU.cpp
  DeformableFusion/FastNonrigidMatching/EDMatchingJacobianHelper.cpp
  DeformableFusion/FastNonrigidMatching/EDMatchingLSQProblemCuda.cpp
  DeformableFusion/FastNonrigidMatching/Fusion4DGPUKeyFrames.cpp
  DeformableFusion/FastNonrigidMatching/LinearSolverCuSolver.cpp
  DeformableFusion/FastNonrigidMatching/volumetric_fusion_GPU.cpp

  DeformableFusion/FastNonrigidMatching/DeformGraphCudaImpl_GPU.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImpl.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImpl_visualhull.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImpl_rigid.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImpl_residual.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImpl_keypts.cu
  DeformableFusion/FastNonrigidMatching/lmsolver_gpu_impl.cu
  DeformableFusion/FastNonrigidMatching/mc_mesh_processing.cu
  DeformableFusion/FastNonrigidMatching/NormalEquationSolverPCGCuda_GPU.cu
  DeformableFusion/FastNonrigidMatching/PCGCuda_GPU.cu
  DeformableFusion/FastNonrigidMatching/volumetric_fusion_impl.cu
  DeformableFusion/FastNonrigidMatching/volumetric_fusion_impl_GPU.cu
  DeformableFusion/FastNonrigidMatching/cuda_common_kernels.cu
  DeformableFusion/FastNonrigidMatching/fast_pcg.cu
  DeformableFusion/FastNonrigidMatching/CudaTextureHandles.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImplInit.cu
  DeformableFusion/FastNonrigidMatching/EDMatchingHelperCudaImplRegTerm.cu

  DeformableFusion/FastNonrigidMatching/lmsolver.cpp
  DeformableFusion/FastNonrigidMatching/EDMatchingLSQProblem.cpp
  DeformableFusion/FastNonrigidMatching/lmsolver_gpu.cpp
  DeformableFusion/FastNonrigidMatching/matrix_types_cuda_io.cpp
  DeformableFusion/FastNonrigidMatching/stdafx.cpp
  DeformableFusion/FastNonrigidMatching/suitesparse.cpp
)

set_target_properties(FastNonrigidMatching PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(FastNonrigidMatching PUBLIC
  NonRigidMatch Utility CameraView CSurface Basics
  CUDA::cudart CUDA::cusolver CUDA::cublas
  ${CHOLMOD_LIBRARY} ${AMD_LIBRARY} ${CAMD_LIBRARY} ${COLAMD_LIBRARY} ${CCOLAMD_LIBRARY}
  ${SUITESPARSECONFIG_LIBRARY} ${METIS_LIBRARY}
  ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
  ${OpenCV_LIBS} ${VXL_CORE_LIBS}
)
get_target_property(FAST_MATCH_OBJECTS FastNonrigidMatching SOURCES)

add_executable(DeformableFusionDemo
  DeformableFusion/DeformableFusionDemo/DeformableFusionDemo.cpp
  DeformableFusion/FusionDemo-MultiView/FusionConfig.cpp
  DeformableFusion/FusionDemo-MultiView/GlobalDataStatic.cpp
  ${FAST_MATCH_OBJECTS}
)

# Cruciale: Diciamo a CMake che il linker finale deve essere CUDA (NVCC) 
# e abilitiamo la compilazione separata sui sorgenti aggregati.
set_target_properties(DeformableFusionDemo PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(DeformableFusionDemo PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)

# Garantisce che l'eseguibile trovi gli header interni di FastNonrigidMatching
target_include_directories(DeformableFusionDemo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/DeformableFusion/FastNonrigidMatching
)

# Configurazione del Linker con gestione dei cicli di dipendenze C++
target_link_libraries(DeformableFusionDemo PRIVATE
  # Il gruppo risolve ricorsivamente i link incrociati (es. cameraSpace2worldSpace in CameraView/Basics)
  -Wl,--start-group
  NonRigidMatch
  CameraView
  Basics
  CSurface
  Utility
  LibUtility
  -Wl,--end-group

  # Librerie di terze parti e di sistema
  ${CHOLMOD_LIBRARY}
  ${AMD_LIBRARY}
  ${CAMD_LIBRARY}
  ${COLAMD_LIBRARY}
  ${CCOLAMD_LIBRARY}
  ${SUITESPARSECONFIG_LIBRARY}
  ${METIS_LIBRARY}
  Ceres::ceres
  CUDA::cudart CUDA::cusolver CUDA::cublas
  OpenGL::GL
  OpenGL::GLU
  GLUT::GLUT
  ZLIB::ZLIB
  ${OpenCV_LIBS}
  ${JSONCPP_LIBRARIES}
  ${VXL_CORE_LIBS}
)
install(TARGETS DeformableFusionDemo RUNTIME DESTINATION bin)
