How to port user decks to Kokkos
================================

The amount of work required to convert legacy decks to VPIC 2.0 varies greatly depending on the deck.  Some decks will just work or require only a few lines to change.  Others, with custom boundary conditions and user-added physics, will require extensive modifications.  Most will be in between.  With a willingness to learn a little Kokkos and search the source code for examples, conversion will typically be a manageable task.

Initialization and output
*************************

Initialization and the built-in output functions (particles, fields, and hydro) mostly function as is and require no changes from legacy decks.  VPIC is smart enough to move data to the proper place after initialization and during output while avoiding unnecessary data movement.  Basic decks may work without modification.

Understanding and managing data movement
****************************************

The key to a good port is understanding where data is, whether it is up to date, and eliminating unnecessary data movement.  The same data can exist in up to three places: the legacy arrays that play well with legacy decks, the Kokkos host view, and the Kokkos device view.  To quickly get something that works, you can set:::

    kokkos_field_injection = false;
    kokkos_particle_injection = false;
    kokkos_current_injection = false;
  
in the initialization in your deck.  (This is the default behavior.)  This will make most legacy user injection functions work as is.  However, this will copy the entire field or particle array from the device to the host to the legacy array and back every time the injection is called, which can be extremely inefficient.  If your deck does not use any of the injections, set their intervals to -1 (default) to avoid this excess movement:::

    field_injection_interval = -1;
    particle_injection_interval = -1;
    current_injection_interval = -1;

The best option is to convert any injection function to run on the device and operate on the Kokkos views directly, though this may take some work.  Once completed, you can set ``kokkos_*_injection = true;`` and be assured that you are not moving data unnecessarily.

Custom diagnostics operate similarly.  You can call ``field_array->copy_to_host();`` and ``sp->copy_to_host();`` and run your diagnostics normally.  (You should not be calling any ``copy_to_device()`` function in the diagnostics; that's what the injection functions are for.)  If your diagnostics run every few hundred steps, this is a pretty reasonable data movement cost, but if you run them often or every step, it's best to port these to operate on the device.


How to port user kernels to Kokkos
**********************************

Most user code/loops can be easily converted to run on the GPU, as long as we follow
some simple rules:

1) The code must be safe to run in parallel (there is no [sane] way to run serial on the GPU). A kokkos scatter view can be used to generate atomics and handle data races. See `hydro_p` for an example.
2) The ported loop *cannot* call code that dereferences pointers. This is because the lambdas capture all variables by values, meaning the code will capture the pointer value from CPU space, and try use it in GPU space. This is often the source of pretty tricky bugs, as it may work on CPU but not GPU. An example of this is calling the function `voxel()` in a field loop to convert from 3D. The function voxel calls `grid->sz` internally. We need to capture dereferenced pointer variables in a local variable above the lambdas

An example of porting a user defined field injection is included below (see `sample/short_pulse.cxx` for full context):

Original:::

    # define DY    ( grid->y0 + (iy-0.5)*grid->dy - global->ycenter )
    # define DZ    ( grid->z0 +  (iz-1) *grid->dz - global->zcenter )
    # define R2    ( DY*DY+DZ*DZ )
    # define PHASE ( global->omega_0*t + h*R2/(global->width*global->width) )
    # define MASK  ( R2<=pow(global->mask*global->width,2) ? 1 : 0 )

      for ( int iz=1; iz<=grid->nz+1; ++iz ) 
        for ( int iy=1; iy<=grid->ny; ++iy )  
          field(1,iy,iz).ey += prefactor 
                               * cos(PHASE) 
                               * exp(-R2/(global->width*global->width))
                               * MASK * pulse_shape_factor;

Modified:::

    int _sy = grid->sy; // safe to dereference grid outside of the loop
    int _sz = grid->sz;
    float width = global->width;
    float width2 = width*width;

    k_field_t& kfield = field_array->k_f_d; // Need local reference

    // Complex, fast, multiple dimension loop
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> left_edge({1, 1}, {nz+2, ny+1}); // Top of the range is <, not <=.
    Kokkos::parallel_for("Field injection", left_edge, KOKKOS_LAMBDA(const int iz, const int iy) {
        auto DY = y0 + (iy-0.5)*dy - ycenter; // y0, dy, t, width, etc. all calculated locally above the loop
        auto DZ = z0 + (iz-1  )*dz - zcenter;
        auto R2 = DY*DY + DZ*DZ;
        auto phase = -omega_0*t + h*R2/width2;
        auto mask = R2<=pow(mask*width,2) ? 1 : 0;
        // We want to call the function voxel(1,iy,iz) (from vpic.h, not the
        // macro) but that would cause an illegal defeference of grid!
        // Instead we can use local variables, or use static functions. 
        int vox = ix + _sy*iy + _sz*iz; // locally captured above
        kfield(vox, field_var::ey) += prefactor 
                                     * cos(phase) 
    				     * exp(-R2/width2)
                                     * mask * pulse_shape_factor;
    });
