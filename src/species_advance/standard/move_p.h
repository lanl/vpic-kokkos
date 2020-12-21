#ifndef _move_p_kokkos_h_
#define _move_p_kokkos_h_

template<
  class geo_t,
  class particle_view_t,
  class particle_i_view_t,
  class neighbor_view_t,
  class accum_t
>
int
KOKKOS_INLINE_FUNCTION
move_p_kokkos(
  const geo_t& geometry,
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

  float age;
  int axis, dir, face;
  int64_t neighbor;
  int pi = pm->i;

  const float q = qsp*k_particles(pi, particle_var::w);

  for(;;) {

    int ii = k_particles_i(pi);

    // Load initial position (logical)
    float dx = k_particles(pi, particle_var::dx);
    float dy = k_particles(pi, particle_var::dy);
    float dz = k_particles(pi, particle_var::dz);

    // Load initial momentum (logical)
    float ux = k_particles(pi, particle_var::ux);
    float uy = k_particles(pi, particle_var::uy);
    float uz = k_particles(pi, particle_var::uz);

    // Load remaining displacement (Cartesian)
    float dispx = pm->dispx;
    float dispy = pm->dispy;
    float dispz = pm->dispz;

    // Move to a boundary.
    geometry.age_to_boundary(
      ii,
      dx, dy, dz,
      dispx, dispy, dispz,
      axis, dir, age
    );

    // Compute fractional displacement (Cartesian)
    dispx *= age;
    dispy *= age;
    dispz *= age;

    // Update remaining displacement (Cartesian)
    pm->dispx -= dispx;
    pm->dispy -= dispy;
    pm->dispz -= dispz;

    // Compute momentum in the displaced frame
    // Double precision for better momentum conservation.
    geometry.template realign_cartesian_vector<double>(
      ii,
      dx, dy, dz,
      dispx, dispy, dispz,
      ux, uy, uz
    );

    // Transform remaining displacement
    geometry.template realign_cartesian_vector<float>(
      ii,
      dx, dy, dz,
      dispx, dispy, dispz,
      pm->dispx, pm->dispy, pm->dispz
    );

    // Convert displacement from Cartesian to logical
    geometry.displacement_to_half_logical(
      ii,
      dx, dy, dz,
      dispx, dispy, dispz
    );

    // Compute new position in logical space
    float dxmid = dx + dispx;                  // Streak midpoint (inbnds)
    float dymid = dy + dispy;
    float dzmid = dz + dispz;

    dx = dxmid + dispx;                        // New position
    dy = dymid + dispy;
    dz = dzmid + dispz;

    // Accumulate the streak.
    accumulate(ii, q, dxmid, dymid, dzmid, dispx, dispy, dispz);

    // Store new position (logical)
    k_particles(pi, particle_var::dx) = dx;
    k_particles(pi, particle_var::dy) = dy;
    k_particles(pi, particle_var::dz) = dz;

    // Store momentum (logical)
    k_particles(pi, particle_var::ux) = ux;
    k_particles(pi, particle_var::uy) = uy;
    k_particles(pi, particle_var::uz) = uz;

    // If an end streak, return success (should be ~50% of the time)

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    k_particles(pi, particle_var::dx + axis) = dir; // Avoid roundoff fiascos--put the particle
                                                    // _exactly_ on the boundary.
    face = axis; if( dir>0 ) face += 3;

    // TODO: clean this fixed index to an enum
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
    k_particles(pi, particle_var::dx + axis) = -dir;     // Convert coordinate system

  }

  return 0; // Return "mover not in use"

}



#endif
