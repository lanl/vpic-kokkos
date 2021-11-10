# GPU Specific Instructions

## Obtaining and Using Kokkos

This project relies on Kokkos. There are a few options for a user to obtain
kokkos, documented below.

### Quickstart

1) Do a *recursive* clone of this repo, this will pull down a copy of Kokkos
for you.
```
git clone --recursive git@github.com:lanl/vpic-kokkos.git
```
If you switch branches, you might need to update the Kokkos submodule.
```
git submodule update --init
```
2) Load modules for a) Cuda, and b) MPI
3) Build the project by passing the CMake option `-DBUILD_INTERNAL_KOKKOS=ON`.
This will request VPIC to build and handle Kokkos for you.
4) If you want NVIDIA GPU functionally, also pass `-DENABLE_KOKKOS_CUDA=ON`. One
should manually set the Kokkos target architecture, some example CMake
variables to do this include:
    
```
Kokkos_ARCH_VOLTA70=ON
Kokkos_ARCH_POWER9=ON
```

This should give you a simple working of the code, but be aware it does come
with caveats. Most notably, one is expected to build and maintain a version of
Kokkos (and optionally VPIC) per target device (one per GPU device, one per CPU
platform, etc), where as the above builds for the specific platform that you're
currently on. Additionally, the above approach can be quite brittle when
changing compile flags between builds because Kokkos doesn't treat CMake as a
first class citizen (set to change in early 2020)

A reminder to NVIDIA GPU users, CUDA 10.2 does not support GCC 8 or newer. We
recommend you use GCC 6 or 7.

AMD GPUs are now technically supported, however no guidance is given at this
point

### Manual Kokkos Install (more powerful, more effort)

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

#### Legacy non-cmake build for old Kokkos versions

1) Clone Kokkos (or use ./kokkos in the recursive clone) from
https://github.com/kokkos/kokkos
2) Make a build folder, and execute `../generate_makefile.bash`, passing the
appropriate options for platform and device architecture. These look something
like:
  - CPU: `../generate_makefile.bash --with-serial --with-openmp
  --prefix=$KOKKOS_INSTALL_DIR`
  - GPU: `../generate_makefile.bash --with-serial --with-openmp --with-cuda
  --arch=Kepler30 --with-cuda-options=enable_lambda
  --compiler=$KOKKOS_SRC_DIR/bin/nvcc_wrapper --prefix=$KOKKOS_INSTALL_DIR`

### Further Reading

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

## Building VPIC + Kokkos

Then when we build VPIC we need to make sure we using the GPU, you need to
specify the Kokkos `nvcc_wrapper` to be the compiler. This typically looks
something like:

`CXX=$HOME/tools/kokkos_gpu/bin/nvcc_wrapper cmake -DENABLE_KOKKOS_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..`

Note: As with any CMake application, it's critical that you set the
CMAKE_BUILD_TYPE (and other optimization flags) appropriately for your target
system

## Running on multiple GPUs

To run on multiple GPU's, you can pass the flag: `--kokkos-num-devices=N` (which replaced `--kokkos-ndevices`), where
`N` specifies the number of GPUs (per node). This works by VPIC passing through
options it doesn't understand to Kokkos, and thus VPIC will generate a warning
as it thinks you may have tried to tell it something it doesn't understand...

## How to port user kernels

Most user code/loops can be easily converted to run on the GPU, as long as we follow
some simple rules:

1) The code must be safe to run in parallel (there is no [sane] way to run serial on the GPU). A kokkos scatter view can be used to generate atomics and handle data races. See `advance_p` for an example.
2) The ported loop *cannot* call code that dereferences pointers. This is because the lambdas capture all variables by values, meaning the code will capture the pointer value from CPU space, and try use it in GPU space. This is often the source of pretty tricky bugs, as it may work on CPU but not GPU. An example of #2 above is something like calling the function `voxel()` in a field loop to convert from 2/3D. The function voxel calls `grid->sz` internally. We need to capture dereferenced pointer variables in a local variable above the lambdas

An example of porting a user defined field injection is included below:

Original:
```
    for ( int iz=1; iz<=grid->nz+1; ++iz ) 
      for ( int iy=1; iy<=grid->ny; ++iy )  
        field(1,iy,iz).ey += prefactor 
                             * cos(PHASE) 
//                           * exp(-R2/(global->width*global->width))  // 3D
                             * exp(-R2Z/(global->width*global->width))
                             * MASK * pulse_shape_factor;
```

Modified
```
    int _sy = grid->sy; // safe to dereference grid outside of the loop
    int _sz = grid->sz;
    // Complex, fast, multiple dimension loop
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> left_edge({1, 1}, {nz+2, ny+1});
    Kokkos::parallel_for("Field injection", left_edge, KOKKOS_LAMBDA(const int iz, const int iy) {
        auto DY   =( y0 + (iy-0.5)*dy - ycenter );
        auto DZ   =( z0 + (iz-1  )*dz - zcenter );
        auto R2   =( DY*DY + DZ*DZ );
        auto R2Z  = ( DZ*DZ );                                   
        auto PHASE=( -omega_0*t + h*R2Z/(width*width) );
        auto MASK =( R2Z<=pow(mask*width,2) ? 1 : 0 );
        // We want to call the function voxel(1,iy,iz) (from vpic.h, not the
        //   macro) but that would cause an illegal defeference of grid!
        //   Instead we can use local variables, or use static functions. 
        int vox = ix + _sy*iy + _sz*iz; // locally captured above
        kfield(vox, field_var::ey) += prefactor 
                                     * cos(PHASE) 
                                     * exp(-R2Z/(width*width))
                                     * MASK * pulse_shape_factor;
    });
```

## Known Issues

1. CMake has a bug where MPI detection was changed, and is faulty. So far we've
seen:
  - Does *NOT* work: 3.12.1
  - Does work: 3.12.4, 3.11.1, 3.7.1
2. During a GPU compile, nvcc warns about multiply defined c-standard flags.
   This warning can be safely ignored
3. If you see `../src/util/pipelines/pipelines_thread.cc:239: undefined
   reference to omp_get_num_threads'` when building a deck, it's because
   `-fopenmp` (or equivelent) didn't get added to `./bin/vpic`. This happens
   because of a cmake bug where it fails to detect OpenMP. To fix it, you can
   manually add `-fopenmp` to the build flags in the file.
