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
option(FUSION4D_ENABLE_CONAN "Resolve third-party dependencies via Conan/CMakeDeps" OFF)
option(FUSION4D_WITH_CORE "Build the headless Fusion4D core library" ON)
option(FUSION4D_WITH_DEMO "Build the DeformableFusion demo executable" ON)
option(FUSION4D_WITH_VISUAL_FEATURE_MATCHING "Build the legacy VXL visual feature matching module" ON)

set(PEABODY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DEFORMABLE_FUSION_DIR "${PEABODY_ROOT}/DeformableFusion")
set(PEABODY_COMPAT_INCLUDE_DIR "${PEABODY_ROOT}/include/compat")
set(FUSION4D_VXL_DIR "" CACHE PATH "Path to the directory containing VXLConfig.cmake")
if(NOT FUSION4D_VXL_DIR AND DEFINED ENV{VXL_DIR})
  set(FUSION4D_VXL_DIR "$ENV{VXL_DIR}" CACHE PATH "Path to the directory containing VXLConfig.cmake" FORCE)
endif()
if(NOT FUSION4D_VXL_DIR AND EXISTS "${PEABODY_ROOT}/Dependencies/vxl-install/share/vxl/cmake")
  set(FUSION4D_VXL_DIR "${PEABODY_ROOT}/Dependencies/vxl-install/share/vxl/cmake" CACHE PATH "Path to the directory containing VXLConfig.cmake" FORCE)
endif()
if(FUSION4D_VXL_DIR)
  list(PREPEND CMAKE_PREFIX_PATH "${FUSION4D_VXL_DIR}")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(VXL REQUIRED)
if(NOT FUSION4D_VXL_DIR AND DEFINED VXL_DIR AND EXISTS "${VXL_DIR}")
  set(FUSION4D_VXL_DIR "${VXL_DIR}")
endif()
if(FUSION4D_ENABLE_CONAN)
  find_package(Ceres REQUIRED CONFIG)
  find_package(jsoncpp REQUIRED CONFIG)
  find_package(SuiteSparse REQUIRED CONFIG COMPONENTS CHOLMOD SPQR)
  set(FUSION4D_JSONCPP_TARGET JsonCpp::JsonCpp)
  set(FUSION4D_SUITESPARSE_LINK_LIBS SuiteSparse::CHOLMOD SuiteSparse::SPQR)
else()
  find_package(Ceres REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JSONCPP REQUIRED jsoncpp)
  add_library(fusion4d_jsoncpp INTERFACE)
  target_include_directories(fusion4d_jsoncpp INTERFACE ${JSONCPP_INCLUDE_DIRS})
  target_link_libraries(fusion4d_jsoncpp INTERFACE ${JSONCPP_LINK_LIBRARIES})
  set(FUSION4D_JSONCPP_TARGET fusion4d_jsoncpp)

  find_library(CHOLMOD_LIBRARY cholmod REQUIRED)
  find_library(AMD_LIBRARY amd REQUIRED)
  find_library(CAMD_LIBRARY camd REQUIRED)
  find_library(COLAMD_LIBRARY colamd REQUIRED)
  find_library(CCOLAMD_LIBRARY ccolamd REQUIRED)
  find_library(SUITESPARSECONFIG_LIBRARY suitesparseconfig REQUIRED)
  find_library(SPQR_LIBRARY spqr REQUIRED)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  set(FUSION4D_SUITESPARSE_LINK_LIBS
    ${CHOLMOD_LIBRARY}
    ${AMD_LIBRARY}
    ${CAMD_LIBRARY}
    ${COLAMD_LIBRARY}
    ${CCOLAMD_LIBRARY}
    ${SUITESPARSECONFIG_LIBRARY}
    ${SPQR_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
  )
endif()
find_package(ZLIB REQUIRED)

set(FUSION4D_VXL_ROOT "")
if(FUSION4D_VXL_DIR)
  get_filename_component(_fusion4d_vxl_root "${FUSION4D_VXL_DIR}" DIRECTORY)
  get_filename_component(_fusion4d_vxl_root "${_fusion4d_vxl_root}" DIRECTORY)
  get_filename_component(_fusion4d_vxl_root "${_fusion4d_vxl_root}" DIRECTORY)
  set(FUSION4D_VXL_ROOT "${_fusion4d_vxl_root}")
endif()
if(NOT FUSION4D_VXL_ROOT)
  set(FUSION4D_VXL_ROOT "${PEABODY_ROOT}/Dependencies/vxl-install")
endif()


set(VXL_INCLUDE_DIRS
  "${VXL_VCL_INCLUDE_DIRS}"
  "${VXL_CORE_INCLUDE_DIRS}"
  "${FUSION4D_VXL_ROOT}/include/vxl/contrib"
  "${FUSION4D_VXL_ROOT}/include/vxl/contrib/cul"
  "${FUSION4D_VXL_ROOT}/include/vxl/contrib/brl/bbas"
  "${FUSION4D_VXL_ROOT}/include/vxl/contrib/brl/bseg"
  "${VXL_OXL_INCLUDE_DIR}"
)

set(VXL_BASE_LIBS
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

set(VXL_FEATURE_MATCHING_LIBS
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
target_link_libraries(Utility PUBLIC LibUtility ZLIB::ZLIB ${OpenCV_LIBS} ${VXL_BASE_LIBS})

peabody_static_library(CameraView
  DeformableFusion/CameraView/CameraView.cpp
)
target_link_libraries(CameraView PUBLIC Utility ${OpenCV_LIBS} ${FUSION4D_JSONCPP_TARGET})

peabody_static_library(CSurface
  DeformableFusion/CSurface/CIsoSurface.cpp
  DeformableFusion/CSurface/CSurface.hpp
  DeformableFusion/CSurface/ply.cpp
  DeformableFusion/CSurface/Vectors.cpp
)
target_link_libraries(CSurface PUBLIC Utility CameraView)

set(FUSION4D_BASICS_SOURCES
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
)
if(FUSION4D_WITH_VISUAL_FEATURE_MATCHING)
  list(APPEND FUSION4D_BASICS_SOURCES
    DeformableFusion/Basics/visual_feature_matching.cpp
  )
endif()

peabody_static_library(Basics ${FUSION4D_BASICS_SOURCES})
target_link_libraries(Basics PUBLIC Utility CameraView CSurface ZLIB::ZLIB ${OpenCV_LIBS} ${VXL_BASE_LIBS})
if(FUSION4D_WITH_VISUAL_FEATURE_MATCHING)
  target_link_libraries(Basics PUBLIC ${VXL_FEATURE_MATCHING_LIBS})
endif()

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
target_link_libraries(NonRigidMatch PUBLIC Utility CameraView CSurface Basics Ceres::ceres ${OpenCV_LIBS} ${VXL_BASE_LIBS})

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
  ${FUSION4D_SUITESPARSE_LINK_LIBS}
  ${OpenCV_LIBS} ${VXL_BASE_LIBS}
)
get_target_property(FAST_MATCH_OBJECTS FastNonrigidMatching SOURCES)

if(FUSION4D_WITH_CORE)
  add_library(Fusion4DCore STATIC
    DeformableFusion/FusionDemo-MultiView/FusionConfig.cpp
    DeformableFusion/FusionDemo-MultiView/FusionGlobals.cpp
    DeformableFusion/FusionDemo-MultiView/GlobalDataStatic.cpp
    src/fusion4d_core.cpp
  )

  target_link_libraries(Fusion4DCore PUBLIC
    FastNonrigidMatching
    ${FUSION4D_JSONCPP_TARGET}
  )

  target_include_directories(Fusion4DCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

  set_target_properties(Fusion4DCore PROPERTIES
    POSITION_INDEPENDENT_CODE ON
  )

  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/fusion4d_core" DESTINATION include)
endif()

set(FUSION4D_PACKAGE_TARGETS
  LibUtility
  Utility
  CameraView
  CSurface
  Basics
  NonRigidMatch
  FastNonrigidMatching
)
if(FUSION4D_WITH_CORE)
  list(APPEND FUSION4D_PACKAGE_TARGETS Fusion4DCore)
endif()
install(TARGETS ${FUSION4D_PACKAGE_TARGETS}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(FUSION4D_WITH_DEMO)
  find_package(OpenGL REQUIRED)
  find_package(GLUT REQUIRED)

  add_executable(DeformableFusionDemo
    DeformableFusion/DeformableFusionDemo/DeformableFusionDemo.cpp
    DeformableFusion/DeformableFusionDemo/Live3DViewer.cpp
    DeformableFusion/DeformableFusionDemo/SurfacePreview.cpp
    DeformableFusion/FusionDemo-MultiView/FusionConfig.cpp
    DeformableFusion/FusionDemo-MultiView/FusionGlobals.cpp
    DeformableFusion/FusionDemo-MultiView/GlobalDataStatic.cpp
    ${FAST_MATCH_OBJECTS}
  )

  set_target_properties(DeformableFusionDemo PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(DeformableFusionDemo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
  )

  target_include_directories(DeformableFusionDemo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/DeformableFusion/FastNonrigidMatching
  )

  target_link_libraries(DeformableFusionDemo PRIVATE
    -Wl,--start-group
    NonRigidMatch
    CameraView
    Basics
    CSurface
    Utility
    LibUtility
    -Wl,--end-group
    ${FUSION4D_SUITESPARSE_LINK_LIBS}
    Ceres::ceres
    CUDA::cudart CUDA::cusolver CUDA::cublas
    OpenGL::GL
    OpenGL::GLU
    GLUT::GLUT
    ZLIB::ZLIB
    ${OpenCV_LIBS}
    ${FUSION4D_JSONCPP_TARGET}
    ${VXL_BASE_LIBS}
  )
  if(FUSION4D_WITH_VISUAL_FEATURE_MATCHING)
    target_link_libraries(DeformableFusionDemo PRIVATE ${VXL_FEATURE_MATCHING_LIBS})
  endif()
  install(TARGETS DeformableFusionDemo RUNTIME DESTINATION bin)
endif()
