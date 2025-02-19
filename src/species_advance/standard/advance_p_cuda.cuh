#include "../../vpic/kokkos_helpers.h"

__device__ int move_p_cuda(
                            pos_t* particles_dx,
                            pos_t* particles_dy,
                            pos_t* particles_dz,
                            mom_t* particles_ux,
                            mom_t* particles_uy,
                            mom_t* particles_uz,
                            float* particles_w,
                            int* particles_i,
                            k_particle_mover_t* mover,
                            float* accumulators,
                            int64_t* neighbors,
                            const int64_t rangel,
                            const int64_t rangeh,
                            const int nv,
                            const float qsp
                          )
{
  pos_t s_midx, s_midy, s_midz;
  pos_t s_dispx, s_dispy, s_dispz;
  pos_t s_dir[3];
  pos_t v0, v1, v2, v3, v4, v5, q;

  int axis, face;
  int64_t neighbor;
  int pi = mover->i;
  const pos_t one_third = 1.0/3.0;

  q = qsp*particles_w[pi];

  for(;;) {
    int ii = particles_i[pi];
    s_midx = particles_dx[pi];
    s_midy = particles_dy[pi];
    s_midz = particles_dz[pi];


    s_dispx = mover->dispx;
    s_dispy = mover->dispy;
    s_dispz = mover->dispz;

    //printf("pre axis %d x %e y %e z %e \n", axis, p_dx, p_dy, p_dz);

    //printf("disp x %e y %e z %e \n", s_dispx, s_dispy, s_dispz);

    s_dir[0] = (s_dispx>static_cast<pos_t>(0)) ? 1 : -1;
    s_dir[1] = (s_dispy>static_cast<pos_t>(0)) ? 1 : -1;
    s_dir[2] = (s_dispz>static_cast<pos_t>(0)) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==static_cast<pos_t>(0)) ? static_cast<pos_t>(3.4e38f) : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==static_cast<pos_t>(0)) ? static_cast<pos_t>(3.4e38f) : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==static_cast<pos_t>(0)) ? static_cast<pos_t>(3.4e38f) : (s_dir[2]-s_midz)/s_dispz;

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

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
//    v5 = q*s_dispx*s_dispy*s_dispz*half(one_third);
    v5 = q*s_dispx*s_dispy*s_dispz*one_third;

    //a = (float *)(&d_accumulators[ci]);

#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = static_cast<pos_t>(1)+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = static_cast<pos_t>(1)-s_mid##Z;     /* v4 = 1-dz                            */  \
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
    atomicAdd(&(accumulators[0*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v0));
    atomicAdd(&(accumulators[1*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v1));
    atomicAdd(&(accumulators[2*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v2));
    atomicAdd(&(accumulators[3*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v3));

    accumulate_j(y,z,x);
    atomicAdd(&(accumulators[0*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v0));
    atomicAdd(&(accumulators[1*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v1));
    atomicAdd(&(accumulators[2*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v2));
    atomicAdd(&(accumulators[3*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v3));

    accumulate_j(z,x,y);
    atomicAdd(&(accumulators[0*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v0));
    atomicAdd(&(accumulators[1*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v1));
    atomicAdd(&(accumulators[2*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v2));
    atomicAdd(&(accumulators[3*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v3));

//#ifdef __CUDA_ARCH__
//if(pi < 4096) {
////  printf("move_p: blockIdx.x: %d,\tthreadIdx.y: %d,\tp_index: %d,\tii: %d\n", blockIdx.x, threadIdx.y,  pi, ii);
//  printf("p_index: %4d, pii: %6d, blockIdx.x: %6d, threadIdx.y: %4d, move_p\n", pi, pii, blockIdx.x, threadIdx.y);
//}
//#endif

#   undef accumulate_j

    // Compute the remaining particle displacment
    mover->dispx -= s_dispx;
    mover->dispy -= s_dispy;
    mover->dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    particles_dx[pi] += s_dispx+s_dispx;
    particles_dy[pi] += s_dispy+s_dispy;
    particles_dz[pi] += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    if(axis == 0) {
        particles_dx[pi] = v0; // Avoid roundoff fiascos--put the particle
    } else if (axis == 1) {
        particles_dy[pi] = v0; // Avoid roundoff fiascos--put the particle
    } else {
        particles_dz[pi] = v0; // Avoid roundoff fiascos--put the particle
    }
                           // _exactly_ on the boundary.
    face = axis; if( v0>static_cast<pos_t>(0) ) face += 3;

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
//    neighbor = d_neighbor( 6*ii + face );
    neighbor = neighbors[6*ii + face];

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back


    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      if(axis == 0) {
          particles_ux[pi] = -particles_ux[pi]; // Avoid roundoff fiascos--put the particle
      } else if (axis == 1) {
          particles_uy[pi] = -particles_uy[pi]; // Avoid roundoff fiascos--put the particle
      } else {
          particles_uz[pi] = -particles_uz[pi]; // Avoid roundoff fiascos--put the particle
      }

      // TODO: make this safer
      //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      //k_local_particle_movers(0, particle_mover_var::dispx + axis) = -k_local_particle_movers(0, particle_mover_var::dispx + axis);
      // TODO: replace this, it's horrible
      (&(mover->dispx))[axis] = -(&(mover->dispx))[axis];


      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
//      pii = 8*pii + face;
      particles_i[pi] = 8*particles_i[pi] + face;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

//    pii = neighbor - rangel;
    particles_i[pi] = neighbor - rangel;
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    if(axis == 0) {
//        k_part.dx(pi) = -v0; // Avoid roundoff fiascos--put the particle
        particles_dx[pi] = -v0;
    } else if (axis == 1) {
        particles_dy[pi] = -v0; // Avoid roundoff fiascos--put the particle
    } else {
        particles_dz[pi] = -v0; // Avoid roundoff fiascos--put the particle
    }
  }
  
  return 0;
}

__global__ void advance_p_cuda( 
                                pos_t* particles_dx,
                                pos_t* particles_dy,
                                pos_t* particles_dz,
                                mom_t* particles_ux,
                                mom_t* particles_uy,
                                mom_t* particles_uz,
                                float* particles_w,
                                int* particles_i,
                                pos_t* particles_copy_dx,
                                pos_t* particles_copy_dy,
                                pos_t* particles_copy_dz,
                                mom_t* particles_copy_ux,
                                mom_t* particles_copy_uy,
                                mom_t* particles_copy_uz,
                                float* particles_copy_w,
                                int* particles_copy_i,
                                float* particle_movers,
                                int* particle_movers_i,
                                float* accumulators,
                                float* interpolators,
                                int* k_nm,
                                int64_t* neighbors,
                                const int64_t rangel,
                                const int64_t rangeh,
                                const float qdt_2mc,
                                const float cdt_dx,
                                const float cdt_dy,
                                const float cdt_dz,
                                const float qsp,
                                const int na,
                                const int nv,
                                const int np,
                                const int max_nm,
                                const int nx,
                                const int ny,
                                const int nz
                              )
{
  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  int per_block = np/gridDim.x;
  if(per_block*gridDim.x < np)
    per_block++;
  int per_thread = per_block/blockDim.x;
  if(per_thread*blockDim.x < per_block)
    per_thread++;

  for(int i=0; i<per_thread; i++) {
    int p_index = per_block*blockIdx.x + per_thread*i + threadIdx.x;
    if(p_index < np) {
      mixed_t v0, v1, v2, v3, v4, v5;

      mixed_t dx = particles_dx[p_index]; // Load position
      mixed_t dy = particles_dy[p_index];
      mixed_t dz = particles_dz[p_index];
      mixed_t ux = particles_ux[p_index]; // Load momentum
      mixed_t uy = particles_uy[p_index];
      mixed_t uz = particles_uz[p_index];
      mixed_t q = static_cast<mixed_t>(particles_w[p_index]);
      int ii = particles_i[p_index];

      const mixed_t f_ex        = interpolators[nv*interpolator_var::ex + ii];
      const mixed_t f_dexdy     = interpolators[nv*interpolator_var::dexdy + ii];
      const mixed_t f_dexdz     = interpolators[nv*interpolator_var::dexdz + ii];
      const mixed_t f_d2exdydz  = interpolators[nv*interpolator_var::d2exdydz + ii];
      const mixed_t f_ey        = interpolators[nv*interpolator_var::ey + ii];
      const mixed_t f_deydz     = interpolators[nv*interpolator_var::deydz + ii];
      const mixed_t f_deydx     = interpolators[nv*interpolator_var::deydx + ii];
      const mixed_t f_d2eydzdx  = interpolators[nv*interpolator_var::d2eydzdx + ii];
      const mixed_t f_ez        = interpolators[nv*interpolator_var::ez + ii];
      const mixed_t f_dezdx     = interpolators[nv*interpolator_var::dezdx + ii];
      const mixed_t f_dezdy     = interpolators[nv*interpolator_var::dezdy + ii];
      const mixed_t f_d2ezdxdy  = interpolators[nv*interpolator_var::d2ezdxdy + ii];
      const mixed_t f_cbx       = interpolators[nv*interpolator_var::cbx + ii];
      const mixed_t f_dcbxdx    = interpolators[nv*interpolator_var::dcbxdx + ii];
      const mixed_t f_cby       = interpolators[nv*interpolator_var::cby + ii];
      const mixed_t f_dcbydy    = interpolators[nv*interpolator_var::dcbydy + ii];
      const mixed_t f_cbz       = interpolators[nv*interpolator_var::cbz + ii];
      const mixed_t f_dcbzdz    = interpolators[nv*interpolator_var::dcbzdz + ii];
      

      mixed_t hax  = qdt_2mc*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
                            dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
      mixed_t hay  = qdt_2mc*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
                            dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
      mixed_t haz  = qdt_2mc*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
                            dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );

      mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
      mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
      mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);

      ux  += hax;                               // Half advance E
      uy  += hay;
      uz  += haz;
      v0   = qdt_2mc/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
      v1   = cbx*cbx + (cby*cby + cbz*cbz);
      v2   = ( v0*v0 ) * v1;
      v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
      v4   = v3/(mixed_t(one)+v1*(v3*v3));
      v4  += v4;
    
      v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
      v1   = uy + v3*( uz*cbx - ux*cbz );
      v2   = uz + v3*( ux*cby - uy*cbx );
    
      ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
      uy  += v4*( v2*cbx - v0*cbz );
      uz  += v4*( v0*cby - v1*cbx );
    
      ux  += hax;                               // Half advance E
      uy  += hay;
      uz  += haz;
    
      particles_ux[p_index] = ux;                               // Store momentum
      particles_uy[p_index] = uy;
      particles_uz[p_index] = uz;

      v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
      ux  *= cdt_dx;
      uy  *= cdt_dy;
      uz  *= cdt_dz;
    
      /**/                                      // Get norm displacement
      ux  *= v0;
      uy  *= v0;
      uz  *= v0;
    
      v0   = dx + ux;                           // Streak midpoint (inbnds)
      v1   = dy + uy;
      v2   = dz + uz;
    
      v3   = v0 + ux;                           // New position
      v4   = v1 + uy;
      v5   = v2 + uz;

      if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
          -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {
  
        // Common case (inbnds).  Note: accumulator values are 4 times
        // the total physical charge that passed through the appropriate
        // current quadrant in a time-step
  
        q *= qsp;
        particles_dx[p_index] = v3;                             // Store new position
        particles_dy[p_index] = v4;
        particles_dz[p_index] = v5;
        dx = v0;                                // Streak midpoint
        dy = v1;
        dz = v2;
        v5 = q*ux*uy*uz*one_third;              // Compute correction
  
       #define ACCUMULATE_J(X,Y,Z)                                        \
        v4  = q*u##X;   /* v2 = q ux                            */        \
        v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
        v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
        v1 += v4;       /* v1 = q ux (1+dy)                     */        \
        v4  = one+d##Z; /* v4 = 1+dz                            */        \
        v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
        v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
        v4  = one-d##Z; /* v4 = 1-dz                            */        \
        v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
        v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
        v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
        v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
        v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
        v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

//        Atomic Accumulation
        ACCUMULATE_J( x,y,z );
        atomicAdd(&(accumulators[0*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v0));
        atomicAdd(&(accumulators[1*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v1));
        atomicAdd(&(accumulators[2*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v2));
        atomicAdd(&(accumulators[3*3*nv + accumulator_var::jx*nv + ii]), static_cast<float>(v3));
        ACCUMULATE_J( y,z,x );
        atomicAdd(&(accumulators[0*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v0));
        atomicAdd(&(accumulators[1*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v1));
        atomicAdd(&(accumulators[2*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v2));
        atomicAdd(&(accumulators[3*3*nv + accumulator_var::jy*nv + ii]), static_cast<float>(v3));
        ACCUMULATE_J( z,x,y );
        atomicAdd(&(accumulators[0*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v0));
        atomicAdd(&(accumulators[1*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v1));
        atomicAdd(&(accumulators[2*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v2));
        atomicAdd(&(accumulators[3*3*nv + accumulator_var::jz*nv + ii]), static_cast<float>(v3));
      } 
//      else
//      {                                    // Unlikely
//        k_particle_mover_t local_pm[1];
//        local_pm->dispx = ux;
//        local_pm->dispy = uy;
//        local_pm->dispz = uz;
//        local_pm->i     = p_index;
//  
//        if(move_p_cuda( particles_dx, particles_dy, particles_dz,  // Unlikely
//                        particles_ux, particles_uy, particles_uz, 
//                        particles_w, particles_i, 
//                        local_pm, accumulators, neighbors, rangel, rangeh, nv, qsp) ){
//          if( *k_nm<max_nm ) {
//            const unsigned int nm = Kokkos::atomic_fetch_add( k_nm, 1 );
//            if (nm >= max_nm) Kokkos::abort("overran max_nm");
//  
//            particle_movers[particle_mover_var::dispx*max_nm + nm] = local_pm->dispx;
//            particle_movers[particle_mover_var::dispy*max_nm + nm] = local_pm->dispy;
//            particle_movers[particle_mover_var::dispz*max_nm + nm] = local_pm->dispz;
//            particle_movers_i[nm] = local_pm->i;
//  
//            // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//            particles_copy_dx[nm] = particles_dx[p_index];
//            particles_copy_dy[nm] = particles_dy[p_index];
//            particles_copy_dz[nm] = particles_dz[p_index];
//            particles_copy_ux[nm] = particles_ux[p_index];
//            particles_copy_uy[nm] = particles_uy[p_index];
//            particles_copy_uz[nm] = particles_uz[p_index];
//            particles_copy_w[nm] = particles_w[p_index];
//            particles_copy_i[nm] = particles_i[p_index];
//          }
//        }
//      }
    }
  }
}
