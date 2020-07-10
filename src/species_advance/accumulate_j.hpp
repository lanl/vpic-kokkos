/* *****************************************************************
    
    This "header" file is meant to be included directly into the 
    move_p_kokkos(...) function in species_advance.h as 
    #include "accumulate_j_OLD.mpp"

    This method was originally developed by Villasenor and Buneman
    in 1992 and can be obtained at 
        
        https://doi.org/10.1016/0010-4655(92)90169-Y

   ****************************************************************/
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;


    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    //printf("\nParticle %d: pre axis %d x %e y %e z %e", pi, axis, p_dx, p_dy, p_dz);

    //printf("\nParticle %d: disp x %e y %e z %e", pi, s_dispx, s_dispy, s_dispz);

    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      v3=2,  axis=3;
    if(v0<v3) v3=v0, axis=0;
    if(v1<v3) v3=v1, axis=1;
    if(v2<v3) v3=v2, axis=2;
    v3 *= 0.5;
    
    printf("\nParticle %d: axis, v0, v1, v2, v3 = %d, %e, %e, %e, %e",
            pi, axis, v0, v1, v2, 2.*v3);
    printf("\nParticle %d: s_midx, s_midy, s_midz = %e, %e, %e",
            pi, s_midx, s_midy, s_midz);
    printf("\nParticle %d: s_dispx, s_dispy, s_dispz = %e, %e, %e\n",
            pi, s_dispx, s_dispy, s_dispz);

    float joe_midx = s_midx, joe_midy = s_midy, joe_midz = s_midz;
    float joe_dispx = s_dispx, joe_dispy = s_dispy, joe_dispz = s_dispz;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;
    
    printf("\nAfter rescaling...");
    printf("\nParticle %d: s_midx, s_midy, s_midz = %e, %e, %e",
            pi, s_midx, s_midy, s_midz);
    printf("\nParticle %d: s_dispx, s_dispy, s_dispz = %e, %e, %e\n",
            pi, s_dispx, s_dispy, s_dispz);

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);

    //a = (float *)(&d_accumulators[ci]);

#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
    //Kokkos::atomic_add(&a[0], v0); \
    //Kokkos::atomic_add(&a[1], v1); \
    //Kokkos::atomic_add(&a[2], v2); \
    //Kokkos::atomic_add(&a[3], v3);

    accumulate_j(x,y,z);
    printf("\nParticle %d depositing (x,y,z) v0, v1, v2, v4 = %e, %e, %e, %e", pi, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;

    accumulate_j(y,z,x);
    printf("\nParticle %d depositing (y,z,x) v0, v1, v2, v4 = %e, %e, %e, %e", pi, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

    accumulate_j(z,x,y);
    printf("\nParticle %d depositing (z,x,y) v0, v1, v2, v4 = %e, %e, %e, %e\n\n", pi, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;

#   undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);

    if( axis==3 ) 
    {
        printf("\n*****************************\nParticle %d is done moving at p_dx, p_dy, p_dz = %e, %e, %e\nIt is supposed to stop at x2, y2, z2 = %e, %e, %e\n****************************\n",
                pi, p_dx, p_dy, p_dz, joe_midx + joe_dispx, joe_midy + joe_dispy, joe_midz + joe_dispz);
        break;
    }

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
    neighbor = d_neighbor( 6*ii + face );

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back


    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      printf("\n*************************\nParticle %d was reflected.\n************************\n", pi);
      k_particles(pi, particle_var::ux + axis) = -k_particles(pi, particle_var::ux + axis);

      // TODO: make this safer
      //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      //k_local_particle_movers(0, particle_mover_var::dispx + axis) = -k_local_particle_movers(0, particle_mover_var::dispx + axis);
      // TODO: replace this, it's horrible
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];


      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      pii = 8*pii + face;
      return 1; // Return "mover still in use"
      }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    pii = neighbor - rangel;
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    k_particles(pi, particle_var::dx + axis) = -v0;      // Convert coordinate system
