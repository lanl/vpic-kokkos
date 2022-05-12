Compiling VPIC
==============

Compiling VPIC consists of two steps: compiling or finding a Kokkos install, and building VPIC against that Kokkos install.

Quickstart
**********


1) Do a *recursive* clone of this repo, this will pull down a copy of Kokkos
for you.
```
git clone --recursive git@github.com:lanl/vpic-kokkos.git
```
If you switch branches, you might need to update the Kokkos submodule.
```
git submodule update --init
```
2) Load modules for CMake, your compiler, MPI, and any platform specific packages like CUDA.
3) Find a file in `arch/` that is close to your intended system and modify as necessary.
Pay particular attention to lines like::
    
    -DENABLE_KOKKOS_CUDA=ON
    -DKokkos_ARCH_VOLTA70=ON
    -DKokkos_ARCH_POWER9=ON

4) Make a build directory and run the arch file.
5) Type `make`.

This should give you a simple working of the code, but be aware it does come
with caveats. Most notably, one is expected to build and maintain a version of
Kokkos (and optionally VPIC) per target device (one per GPU device, one per CPU
platform, etc), where as the above builds for the specific platform that you're
currently on.

A reminder to NVIDIA GPU users, CUDA 10.2 does not support GCC 8 or newer. We
recommend you use GCC 6 or 7.


Manual Kokkos Install (more powerful, more effort)
**************************************************

It is typical to maintain many different installs of Kokkos (CPU, older
GPU, new GPU, Debug, etc), so it's worth while learning how to install Kokkos
manually. To do this we will use cmake. On can achieve this by:

CPU:
```
cmake -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_KNL=ON ..
```

GPU:
```
cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ..
```

Legacy non-cmake build for old Kokkos versions
**********************************************

1) Clone Kokkos (or use ./kokkos in the recursive clone) from
https://github.com/kokkos/kokkos
2) Make a build folder, and execute `../generate_makefile.bash`, passing the
appropriate options for platform and device architecture. These look something
like:

  - CPU: `../generate_makefile.bash --with-serial --with-openmp --prefix=$KOKKOS_INSTALL_DIR`
  - GPU: `../generate_makefile.bash --with-serial --with-openmp --with-cuda --arch=Kepler30 --with-cuda-options=enable_lambda --compiler=$KOKKOS_SRC_DIR/bin/nvcc_wrapper --prefix=$KOKKOS_INSTALL_DIR`

Further Reading
***************

One can cherry pick the Kokkos specific details from
[here](https://github.com/ECP-copa/Cabana/wiki/Build-Instructions) to get
detailed build instructions for Kokkos (ignore the things about Cabana)

The advanced user should review `CMakeLists.txt` for the Kokkos specific
options that are available. These include:

1. `ENABLE_KOKKOS_OPENMP`
2. `ENABLE_KOKKOS_CUDA`
3. `BUILD_INTERNAL_KOKKOS`
4. `VPIC_KOKKOS_DEBUG`
5. `KOKKOS_ARCH`

Building VPIC + Kokkos
**********************

Then when we build VPIC we need to make sure we using the GPU, you need to
specify the Kokkos `nvcc_wrapper` to be the compiler. This typically looks
something like:

`CXX=$HOME/tools/kokkos_gpu/bin/nvcc_wrapper cmake -DENABLE_KOKKOS_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..`

Note: As with any CMake application, it's critical that you set the
CMAKE_BUILD_TYPE (and other optimization flags) appropriately for your target
system

Optimization Options
********************

VPIC has compilation flags for enabling/disabling various optimizations. VPIC will automatically selct optimization settings depending on the architecture settings from Kokkos. Users can supply their own settings for potentially better performance. The optimization options are as follows:

1. `VPIC_ENABLE_AUTO_TUNING=ON`
  - Control whether to use the automatically determined optimization settings or user supplied compile time flags.
2. `VPIC_ENABLE_HIERARCHICAL=OFF` 
  - Allow finer control over how work is distributed amoung threads. Automatically enabled by certain optimizations (Team reduction, Vectorization) that require explicit control over threads and vector lanes. Performance is highly dependent on how work is distributed. See kokkos_tuning.hpp for setting the number of leagues (thread teams) and team size (threads per team).
3. `VPIC_ENABLE_TEAM_REDUCTION=OFF` 
  - Reduce number of atomic writes in the particle push. Checks if all the particles being processed by active threads / vector lanes belong to the same cell. If so, use fast register based methods to reduce current so that only 1 thread/lane needs to update the fields.
4. `VPIC_ENABLE_VECTORIZATION=OFF` 
  - Enables vectorization with OpenMP SIMD for greater performance on the CPU
5. `VPIC_ENABLE_ACCUMULATORS=OFF`
  - Use an explicit accumulator for collecting current in advance_p. The accumulator results in better memory access patterns when writing current. This is useful on CPUs but not necessary on GPUs which have better random access characteristics.

