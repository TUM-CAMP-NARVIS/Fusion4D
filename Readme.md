# Fusion4D Linux Port
This is an unofficial Linux port of the Fusion4D paper implementation you can find [here](https://github.com/microsoft/3DTelecommunications). 

This builds the Linux library and produces a demo executable that allows you to do non-rigid reconstruction from RGBD sequences on Linux.:

## 1. Install System Packages

- Install CUDA Toolkit (recommended v.12.6)
- Make sure `nvcc` is on `PATH` before
configuring CMake.

- Install the compiler, CMake, CUDA toolkit, OpenGL/GLUT, SuiteSparse, Ceres, JSONCPP, BLAS/LAPACK, and METIS:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config ccache \
  libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
  libsuitesparse-dev libceres-dev libjsoncpp-dev \
  libblas-dev liblapack-dev libmetis-dev \
```
- Install VxL from source
```bash
cd Dependencies
git clone --depth 1 git@github.com:vxl/vxl.git
cd vxl
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../vxl-install
make -j
make install
```

You need OpenCV with CUDA support. For more details refer to my gist guide: 
https://gist.github.com/Hydran00/a5b8890a01127c222d40102e28e7fc7e



## 3. Configure

From the repository root:

```bash
cmake -S . -B build-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \ # or adjust for your GPU
  -DPEABODY_VOLUME_CUBES_DIM_MAX=50 # or adjust based on your VRAM
```
If your OpenCV or VXL installs are somewhere else, pass their paths when
configuring:

```bash
-DOpenCV_DIR=/path/to/opencv/build
-DVXL_DIR=/path/to/vxl-install/share/vxl/cmake
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU if needed. For example, use
`86` for many RTX 30-series cards.

## 4. Build

```bash
cmake --build build-linux --target DeformableFusionDemo -j
```

If `ccache` cannot write to its default cache directory, use temporary paths:

```bash
cmake -E env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp \
  cmake --build build-linux --target DeformableFusionDemo -j
```

## 5. Run Upperbody Demo
- Download VolumeDeform Dataset from [here](https://www.lgdv.tf.fau.de/publications/volumedeform-real-time-volumetric-non-rigid-reconstruction/) and put it in `Dataset/`

From `build-linux`:

```bash
./DeformableFusionDemo \
  --input ../Dataset/upperbody/data \ # or another dataset path
  --depth-token depth \
  --rgb-token color \
  --output ../Dataset/outputs \
  --camera-count 1 \
  --gpu 0 \
  --config ../ConfigFileExamples/OfflineFolderFusion_upperbody.cfg \
  --fx 570.342 \
  --fy 570.342 \
  --cx 320.0 \
  --cy 240.0 \
  --bbox -300,300,-300,300,-300,300 \
  --viewer3d \
  --dump-surface-every 10 \
  --dump-surface-format ply \
  --preview
```

The application writes F2F outputs to `../Dataset/outputs/f2f` and Fusion4D
outputs to `../Dataset/outputs/fusion4d`.

## 6. Notes

- Use `--viewer3d` only on a machine/session with OpenGL display access.
- Use `--preview` only when OpenCV can open UI windows.
- The default `PEABODY_VOLUME_CUBES_DIM_MAX=50` is sized for the upperbody
  demo and avoids the large fixed TSDF allocation that can cause CUDA
  out-of-memory errors.


## References
- [Original Fusion4D implementation](https://github.com/microsoft/3DTelecommunications)
- [Fusion4D paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/a114-dou.pdf)
