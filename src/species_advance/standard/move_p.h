#ifndef _move_p_kokkos_h_
#define _move_p_kokkos_h_

template<
  class particle_view_t,
  class particle_i_view_t,
  class neighbor_view_t,
  class accum_t
>
int
KOKKOS_INLINE_FUNCTION
move_p_kokkos(
  const particle_view_t& k_particles,
  const particle_i_view_t& k_particles_i,
  particle_mover_t* ALIGNED(16)  pm,
  const accum_t& accumulate,
  neighbor_view_t& d_neighbor,
  int64_t rangel,
  int64_t rangeh,
  const float qsp
)
{

  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  int pi = pm->i;

  q = qsp*k_particles(pi, particle_var::w);

  for(;;) {

    int ii = k_particles_i(pi);
    s_midx = k_particles(pi, particle_var::dx);
    s_midy = k_particles(pi, particle_var::dy);
    s_midz = k_particles(pi, particle_var::dz);

    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

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

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.
    accumulate(ii, q, s_midx, s_midy, s_midz, s_dispx, s_dispy, s_dispz);

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    // Compute the new particle offset
    k_particles(pi, particle_var::dx) += s_dispx+s_dispx;
    k_particles(pi, particle_var::dy) += s_dispy+s_dispy;
    k_particles(pi, particle_var::dz) += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)

    if( axis==3 ) break;

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
      k_particles_i(pi) = 8*k_particles_i(pi) + face;
      return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    k_particles_i(pi) = neighbor - rangel;               // Note: neighbor - rangel < 2^31 / 6
    k_particles(pi, particle_var::dx + axis) = -v0;      // Convert coordinate system

  }

  return 0; // Return "mover not in use"

}



#endif
