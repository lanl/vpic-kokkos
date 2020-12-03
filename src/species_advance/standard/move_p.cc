#define IN_spa
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"

// move_p moves the particle m->p by m->dispx, m->dispy, m->dispz
// depositing particle current as it goes. If the particle was moved
// sucessfully (particle mover is no longer in use) returns 0. If the
// particle interacted with something this routine could not handle,
// this routine returns 1 (particle mover is still in use). On a
// successful move, the particle position is updated and m->dispx,
// m->dispy and m->dispz are zerod. On a partial move, the particle
// position is updated to the point where the particle interacted and
// m->dispx, m->dispy, m->dispz contains the remaining particle
// displacement. The displacements are the physical displacments
// normalized current cell size.
//
// Because move_p is frequently called, it does not check its input
// arguments. Higher level routines are responsible for insuring valid
// arguments.
//
// Note: changes here likely need to be reflected in SPE accelerated
// version as well.

//#if defined(V4_ACCELERATION)
#if 0

// High performance variant based on SPE accelerated version

using namespace v4;

int
move_p( particle_t       * RESTRICT ALIGNED(128) p,
        particle_mover_t * RESTRICT ALIGNED(16)  pm,
        accumulator_t    * RESTRICT ALIGNED(128) a,
        const grid_t     *                       g,
        const float                              qsp ) {

  /*const*/ v4float one( 1.f );
  /*const*/ v4float tiny( 1e-37f );
  /*const*/ v4int   sign_bits( 1<<31 );

  v4float dr, r, u, q, q3;
  v4float sgn_dr, s, sdr;
  v4float v0, v1, v2, v3, v4, v5, _stack_vf;
  v4int bits, _stack_vi;

  float * RESTRICT ALIGNED(16) stack_vf = (float *)&_stack_vf;
  int   * RESTRICT ALIGNED(16) stack_vi =   (int *)&_stack_vi;
  float f0, f1;
  int32_t n, voxel;
  int64_t neighbor;
  int type;

  load_4x1( &pm->dispx, dr );  n     = pm->i;
  load_4x1( &p[n].dx,   r  );  voxel = p[n].i;
  load_4x1( &p[n].ux,   u  );

  q  = v4float(qsp)*splat<3>(u); // q  = p_q,   p_q,   p_q,   D/C
  q3 = v4float(1.f/3.f)*q;      // q3 = p_q/3, p_q/3, p_q/3, D/C
  dr = shuffle<0,1,2,2>( dr );  // dr = p_ddx, p_ddy, p_ddz, D/C 
  r  = shuffle<0,1,2,2>( r );  // r  = p_dx,  p_dy,  p_dz,  D/C
  
  for(;;) {

    // At this point:
    //   r     = current particle position in local voxel coordinates
    //           (note the current voxel is on [-1,1]^3.
    //   dr    = remaining particle displacment
    //           (note: this is in voxel edge lengths!)
    //   voxel = local voxel of particle
    // Thus, in the local coordinate system, it is desired to move the
    // particle through all points in the local coordinate system:
    //   streak_r(s) = r + 2 disp s for s in [0,1]
    //
    // Determine the fractional length and type of current
    // streak made by the particle through this voxel.  The streak
    // ends on either the first voxel face intersected by the
    // particle track or at the end of the particle track.
    //
    // Note: a divide by zero cannot occur below due to the shift of
    // the denominator by tiny.  Also, the shift by tiny is large
    // enough that the divide will never overflow when dr is tiny
    // (|sgn_dr-r|<=2 => 2/tiny = 2e+37 < FLT_MAX = 3.4e38).
    // Likewise, due to speed of light limitations, generally dr
    // cannot get much larger than 1 or so and the numerator, if not
    // zero, can generally never be smaller than FLT_EPS/2.  Thus,
    // likewise, the divide will never underflow either. 

    // FIXME: THIS COULD PROBABLY BE DONE EVEN FASTER 
    sgn_dr = copysign( one,  dr );
    v0     = copysign( tiny, dr );
    store_4x1( (sgn_dr-r) / ((dr+dr)+v0), stack_vf );
    /**/                          type = 3;             f0 = 1;
    f1 = stack_vf[0]; if( f1<f0 ) type = 0; if( f1<f0 ) f0 = f1; // Branchless cmov 
    f1 = stack_vf[1]; if( f1<f0 ) type = 1; if( f1<f0 ) f0 = f1;
    f1 = stack_vf[2]; if( f1<f0 ) type = 2; if( f1<f0 ) f0 = f1;
    s = v4float( f0 );

    // At this point:
    //   type = 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //          3        ... streak ends at end of the particle track
    //   s    = SPLAT( normalized length of the current streak )
    //   sgn_dr indicates the sign streak displacements.  This is
    //     useful to determine whether or not the streak hit a face.
    //
    // Compute the streak midpoint the normalized displacement,
    // update the particle position and remaining displacment,
    // compute accumulator coefficients, finish up fetching the
    // voxel needed for this streak and accumule the streak.  Note:
    // accumulator values are 4 times the total physical charge that
    // passed through the appropriate current quadrant in a
    // timestep.

    sdr = s*dr;      // sdr = ux,          uy,          uz,          D/C
    v5  = r + sdr;   // v5  = dx,          dy,          dz,          D/C

    dr -= sdr;       // dr  = p_ddx',      p_ddy',      p_ddz',      D/C
    r  += sdr + sdr; // r   = p_dx',       p_dy',       p_dz',       D/C

    v4  = q*sdr;     // v4  = q ux,        q uy,        q uz,        D/C
    v1  = v4*shuffle<1,2,0,3>( v5 );
    /**/             // v1  = q ux dy,     q uy dz,     q uz dx,     D/C
    v0  = v4 - v1;   // v0  = q ux(1-dy),  q uy(1-dz),  q uz(1-dx),  D/C
    v1 += v4;        // v1  = q ux(1+dy),  q uy(1+dz),  q uz(1+dx),  D/C

    v5  = shuffle<2,0,1,3>( v5 ); // v5 = dz, dx, dy, D/C
    v4  = one + v5;  // v4  = 1+dz,        1+dx,        1+dy,        D/C
    v2  = v0*v4;     // v2  = q ux(1-dy)(1+dz), ...,                 D/C
    v3  = v1*v4;     // v3  = q ux(1+dy)(1+dz), ...,                 D/C
    v4  = one - v5;  // v4  = 1-dz,        1-dx,        1-dy,        D/C
    v0 *= v4;        // v0  = q ux(1-dy)(1-dz), ...,                 D/C
    v1 *= v4;        // v1  = q ux(1+dy)(1-dz), ...,                 D/C

    //v4  = ((q3*splat(sdr,0))*splat(sdr,1))*splat(sdr,2);
    v4  = ((q3*splat<0>(sdr))*splat<1>(sdr))*splat<2>(sdr);
    // FIXME: splat ambiguity in v4 prevents flattening
    /**/             // v4  = q ux uy uz/3,q ux uy uz/3,q ux uy uz/3,D/C
    v0 += v4;        // v0  = q ux[(1-dy)(1-dz)+uy uz/3], ...,       D/C
    v1 -= v4;        // v1  = q ux[(1+dy)(1-dz)-uy uz/3], ...,       D/C
    v2 -= v4;        // v2  = q ux[(1-dy)(1+dz)-uy uz/3], ...,       D/C
    v3 += v4;        // v3  = q ux[(1+dy)(1+dz)+uy uz/3], ...,       D/C

    transpose( v0, v1, v2, v3 );

    increment_4x1( a[voxel].jx, v0 );
    increment_4x1( a[voxel].jy, v1 );
    increment_4x1( a[voxel].jz, v2 );

    // If streak ended at the end of the particle track, this mover
    // was succesfully processed.  Should be just under ~50% of the
    // time.
       
    if( type==3 ) { store_4x1( r, &p[n].dx ); p[n].i = voxel; break; }

    // Streak terminated on a voxel face.  Determine if the particle
    // crossed into a local voxel or if it hit a boundary.  Convert
    // the particle coordinates accordingly.  Note: Crossing into a
    // local voxel should happen the most of the time once we get to
    // this point; hitting a structure or parallel domain boundary
    // should usually be a rare event. */

    clear_4x1( stack_vi );
    stack_vi[type] = -1;
    load_4x1( stack_vi, bits );
    r = merge( bits, sgn_dr, r ); // Avoid roundoff fiascos--put the
                                  // particle _exactly_ on the
                                  // boundary.
    bits &= sign_bits; // bits(type)==(-0.f) and 0 elsewhere

    // Determine if the particle crossed into a local voxel or if it
    // hit a boundary.  Convert the particle coordinates accordingly.
    // Note: Crossing into a local voxel should happen the other ~50%
    // of time; hitting a structure and parallel domain boundary
    // should usually be a rare event.  Note: the entry / exit
    // coordinate for the particle is guaranteed to be +/-1 _exactly_
    // for the particle.

    store_4x1( sgn_dr, stack_vf ); if( stack_vf[type]>0 ) type += 3;
    neighbor = g->neighbor[ 6*voxel + type ];

    if( UNLIKELY( neighbor==reflect_particles ) ) {

      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.

      dr = toggle_bits( bits, dr );
      u  = toggle_bits( bits, u  );
      store_4x1( u, &p[n].ux );
      continue;
    }

    if( UNLIKELY( neighbor<g->rangel || neighbor>g->rangeh ) ) {

      // Cannot handle the boundary condition here.  Save the updated
      // particle position and update the remaining displacement in
      // the particle mover.

      store_4x1( r, &p[n].dx );    p[n].i = 8*voxel + type;
      store_4x1( dr, &pm->dispx ); pm->i  = n;
      return 1; // Mover still in use
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.
      
    voxel = (int32_t)( neighbor - g->rangel );
    r = toggle_bits( bits, r );
  }

  return 0; // Mover not in use
}

#else

#endif

