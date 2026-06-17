# Fusion4D Linux Port
This is an unofficial Linux port of the Fusion4D paper implementation you can find [here](https://github.com/microsoft/3DTelecommunications). 

This builds the Linux library and produces a demo executable that allows you to do non-rigid reconstruction from RGBD sequences on Linux.

The tree now also exposes a headless `Fusion4DCore` library target intended for
embedding into external applications such as Artekmed. The core target wraps
the existing `CMotionFieldEstimationF2F` and `Fusion4DKeyFrames` pipeline behind
an explicit session API in [`include/fusion4d_core/fusion4d_core.h`](include/fusion4d_core/fusion4d_core.h).

## Results on VolumeDeform dataset

<table>
<tr>
<td align="center">Hoodie</td>
<td align="center">Upperbody</td>
<td align="center">Umbrella</td>
</tr>
<tr>
<td align="center">
  <img src="https://github.com/user-attachments/assets/4b899681-6612-4e7e-bd62-b946d18733ce" width="300">
</td>

<td align="center">
  <img src="https://github.com/user-attachments/assets/85cba804-ac68-4b36-98e2-d8ce5c3ef3e6" width="300">
</td>

<td align="center">
  <img src="https://github.com/user-attachments/assets/4575fc28-c764-4f57-b877-421f53e14f5a" width="300">
</td>
</tr>

</table>

## 1. Local Conan setup

Create a local Conan tool env if your host does not already ship Conan 2:

```bash
python3 -m venv $HOME/.venvs/conan2
$HOME/.venvs/conan2/bin/pip install "conan>=2.3,<3"
export PATH="$HOME/.venvs/conan2/bin:$PATH"
```

Detect a profile and export the three local recipes Fusion4D needs beyond Conan Center:

```bash
conan profile detect --force
```

The conan recipes are hosted at https://github.com/TUM-CONAN

The bundled VXL recipe pins `v1.18.0` from upstream and packages the original
`VXLConfig.cmake`, so `find_package(VXL)` continues to work without a manual
`VXL_DIR` override in the Conan path.

## 2. Conan-driven configure

From the repository root:

```bash
conan install . \
  --output-folder=build/Release \
  --build=missing \
  -s build_type=Release \
  -s compiler.cppstd=17

cmake --preset conan-release
```

The Conan path disables the legacy VXL `bapl`-based visual feature matching
module by default. Re-enable it only if you also provide a VXL build that
exports the `bapl` / `bundler` / `mvl` contrib stack.

If you prefer to skip the demo executable during packaging experiments:

```bash
conan install . \
  --output-folder=build/Release \
  --build=missing \
  -s build_type=Release \
  -s compiler.cppstd=17 \
  -o '&:with_demo=False'
```

This configure path still builds `Fusion4DCore`. To disable the legacy viewer
while keeping the headless API enabled in raw CMake mode:

```bash
cmake -S . -B build-core \
  -DCMAKE_BUILD_TYPE=Release \
  -DFUSION4D_WITH_DEMO=OFF \
  -DFUSION4D_WITH_CORE=ON \
  -DFUSION4D_VXL_DIR=$HOME/opt/vxl-1.18.0/share/vxl/cmake
```

## 3. Build

```bash
cmake --build build/Release --parallel
```

## 4. Legacy/manual path

The old host-managed path is still available for debugging by configuring CMake directly, but it no longer requires hardcoded `OpenCV_DIR` or source-local SuiteSparse/METIS lookups:

```bash
cmake -S . -B build-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DFUSION4D_VXL_DIR=$HOME/opt/vxl-1.18.0/share/vxl/cmake
```

## 5. Historical manual package list

The pre-Conan Linux instructions are retained here for reference when debugging a non-Conan host:

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



## 6. Configure

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
-DFUSION4D_VXL_DIR=/path/to/vxl-install/share/vxl/cmake
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU if needed. For example, use
`86` for many RTX 30-series cards.

## 7. Build

```bash
cmake --build build-linux --target DeformableFusionDemo -j
```

If `ccache` cannot write to its default cache directory, use temporary paths:

```bash
cmake -E env CCACHE_DIR=/tmp/ccache CCACHE_TEMPDIR=/tmp \
  cmake --build build-linux --target DeformableFusionDemo -j
```

## 8. Run Upperbody Demo
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

## 9. Notes

- Use `--viewer3d` only on a machine/session with OpenGL display access.
- Use `--preview` only when OpenCV can open UI windows.
- The default `PEABODY_VOLUME_CUBES_DIM_MAX=50` is sized for the upperbody
  demo and avoids the large fixed TSDF allocation that can cause CUDA
  out-of-memory errors.
- `Fusion4DCore` is the preferred integration surface for headless callers.
  Its frame inputs are non-owning views and its mesh readback APIs return owned
  CPU snapshots so transport/application code can keep ownership explicit.


## References
- [Original Fusion4D implementation](https://github.com/microsoft/3DTelecommunications)
- [Fusion4D paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/a114-dou.pdf)
  [VolumeDeform paper](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_22)
