How to port user kernels to Kokkos
==================================

Most user code/loops can be easily converted to run on the GPU, as long as we follow
some simple rules:

1) The code must be safe to run in parallel (there is no [sane] way to run serial on the GPU). A kokkos scatter view can be used to generate atomics and handle data races. See `advance_p` for an example.
2) The ported loop *cannot* call code that dereferences pointers. This is because the lambdas capture all variables by values, meaning the code will capture the pointer value from CPU space, and try use it in GPU space. This is often the source of pretty tricky bugs, as it may work on CPU but not GPU. An example of #2 above is something like calling the function `voxel()` in a field loop to convert from 2/3D. The function voxel calls `grid->sz` internally. We need to capture dereferenced pointer variables in a local variable above the lambdas

An example of porting a user defined field injection is included below:

Original:::

      for ( int iz=1; iz<=grid->nz+1; ++iz ) 
        for ( int iy=1; iy<=grid->ny; ++iy )  
          field(1,iy,iz).ey += prefactor 
                               * cos(PHASE) 
  //                           * exp(-R2/(global->width*global->width))  // 3D
                               * exp(-R2Z/(global->width*global->width))
                               * MASK * pulse_shape_factor;

Modified:::

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
