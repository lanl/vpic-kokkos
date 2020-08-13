/* *****************************************************************
    
    This "header" file is meant to be included directly into the 
    move_p_kokkos(...) function in species_advance.h as 
    #include "accumulate_j_zigzag.hpp"
    
    This method was published by Umeda et al. in 2003 and can be 
    found at

        https://doi.org/10.1016/S0010-4655(03)00437-5

    THIS FILE IS THE ZIGZAG ONE!!!!!!!!!!!!!!!!!!!

   ****************************************************************/

// Create a single accumulate_j macro.
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

    // printf("\nParticle %d in Voxel %d Velocity before accumulation:", pi, pii);
    // printf("\nux, uy, uz = %e, %e, %e", p_ux, p_uy, p_uz);

    // In this approach, we keep the current accumulation the same,
    // and move the particles along the zigzag path. The position 
    // of the zig of the zigzag will be performed only if the 
    // particle leaves the cell. If the particle does not leave the
    // cell, then the zag is performed. 
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;

    // So these displacements are not the actual displacements.
    // Such a thing would be too obvious. Instead this displacement
    // is really half of the true particle displacement, and so the 
    // point 
    //
    //                  s_mid + s_disp,
    //
    // is the MIDPOINT of the full particle motion, not the full
    // particle displacement. The final particle position is then 
    //
    //                  s_mid + 2 * s_disp.
    // 
    // This will be used moving forward.
    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    zig_finalx = s_midx + s_dispx;
    zig_finaly = s_midy + s_dispy;
    zig_finalz = s_midz + s_dispz;

    // printf("\nParticle %d: zig_finalx, zig_finaly, zig_finalz = %e, %e, %e", pi, zig_finalx, zig_finaly, zig_finalz);
   
    // Set the reference points to the midpoint 
    // of the motion by default.
    xr = zig_finalx;
    yr = zig_finaly;
    zr = zig_finalz;
    
    // printf("\nParticle %d: xr, yr, zr = %e, %e, %e", pi, xr, yr, zr);

#if VPIC_DUMP_NEIGHBORS
    print_neighbor.write_final_cell( s_midx + 2. * s_dispx, s_midy + 2. * s_dispy, s_midz + 2. * s_dispz );
#endif

    // TODO: Change this for zigzag.
    // Here is the progression of the displacements moving
    // forward, with x1 = p_dx = s_midx as the initial position
    // and x2 the final position, after the move_p_kokkos(...)
    // function is used:
    //
    //      * Start with s_dispx = pm->dispx = 0.5 * (x2 - x1).
    //
    //      * Find the next endpoint xf (either the boundary
    //        or x2). Then adjust to the new 
    //        midpoint
    //
    //          -- Change s_dispx = xf - x1.
    //
    //          -- Scale s_dispx = 0.5 * s_dispx if the particle
    //             does leave the cell.
    //
    //          -- Set the new midpoint 
    //             s_midx = s_midx + s_dispx = 0.5 * ( xf + x1 )
    //
    //      * *** Accumulate Currents ***
    //
    //      * Change the displacement of the particle as
    //        pm->dispx = pm->dispx - s_dispx
    //                  = 0.5 * ( x2 - x1 ) - 0.5 * ( xf - x1 )
    //                  = 0.5 * ( x2 - xf )
    //
    //      * Change the position of the particle as
    //        p_dx = p_dx + s_dispx + s_dispx
    //             = x1 + 0.5 * ( xf - x1 ) + 0.5 * ( xf - x1 )
    //             = xf
    //
    //      * Determine if the particle leaves the cell and
    //        move it if it does  
    //       
    // Hopefully fully elaborating these steps and how the 
    // displacement and midpoint variables change will help in 
    // future development.
    
    // printf("\n\nMOVE P ENTERED...");
    // printf("\nParticle %d: pre axis %d x %e y %e z %e", pi, axis, p_dx, p_dy, p_dz);

    // printf("\nParticle %d: s_disp x %e y %e z %e", pi, s_dispx, s_dispy, s_dispz);

    // Compute the direction that the particle moves.
    // This value is the also the boundary of the cell 
    // the particle will intercept.
    s_dir[Axis_Label::x] = (s_dispx>0) ? 1 : -1;
    s_dir[Axis_Label::y] = (s_dispy>0) ? 1 : -1;
    s_dir[Axis_Label::z] = (s_dispz>0) ? 1 : -1;

    //printf("\ns_dir = %d, %d, %d", (int)s_dir[Axis_Label::x], (int)s_dir[Axis_Label::y], (int)s_dir[Axis_Label::z]);

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection. This number is the amount of
    // for-loop steps that would need to be taken to reach the 
    // boundary of the cell along each axis. A maximum of 2 for-
    // loop steps are allowed within a single time-step though, so
    // if the value is greater than 2 in a particle direction, then
    // it does not leave the cell in that direction in this
    // function call.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[Axis_Label::x]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[Axis_Label::y]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[Axis_Label::z]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      v3=two,  axis=3;
    if(v0 < two) 
    {
        v3 = v0;
        axis = Axis_Label::x;
        // Set the zizag point along the x-axis
        xr = s_dir[Axis_Label::x];
        s_dispx = ( xr - s_midx );
        zig_finalx = xr;
    }
    if(v1 < two)
    {
        v3 = v1;
        axis = Axis_Label::y;
        // Set the zizag point along the y-axis
        yr = s_dir[Axis_Label::y];
        s_dispy = ( yr - s_midy );
        zig_finaly = yr;
    }
    if(v2 < two)
    {
        v3 = v2;
        axis = Axis_Label::z;
        // Set the zizag point along the z-axis 
        zr = s_dir[Axis_Label::z];
        s_dispz = ( zr - s_midz );
        zig_finalz = zr;
    }
    // Multiply v3 by 1/2 because the particle first moves to the 
    // midpoint if axis != 3, or it stops if axis == 3.
    v3 *= 0.5;

    // printf("\n\nns_disp may have changed!");
    // printf("\nParticle %d: axis, v0, v1, v2, v3 = %d, %e, %e, %e, %e",
    //         pi, axis, v0, v1, v2, 2.*v3);
    // printf("\nParticle %d: s_midx, s_midy, s_midz = %e, %e, %e",
    //         pi, s_midx, s_midy, s_midz);
    // printf("\nParticle %d: s_dispx, s_dispy, s_dispz = %e, %e, %e",
    //         pi, s_dispx, s_dispy, s_dispz);
    /*printf("\nParticle %d: s_midx + s_dispx, s_midy + s_dispy, s_midz + s_dispz = %e, %e, %e",
            pi, s_midx + s_dispx, s_midy + s_dispy, s_midz + s_dispz);
    */
    //printf("\nParticle %d: s_midx + 2*s_dispx, s_midy + 2*s_dispy, s_midz + 2*s_dispz = %e, %e, %e",
    //        pi, s_midx + 2*s_dispx, s_midy + 2*s_dispy, s_midz + 2*s_dispz);
    
    // printf("\nParticle %d: zig_finalx, zig_finaly, zig_finalz = %e, %e, %e", pi, zig_finalx, zig_finaly, zig_finalz);
    // printf("\nParticle %d: xr, yr, zr = %e, %e, %e", pi, xr, yr, zr);
   
    /*
    // If the particle stays in-cell, adjust
    // the final position to be the final point
    // of the motion. 
    if ( axis == 3 )
    {
        zig_finalx += s_dispx;
        zig_finaly += s_dispy;
        zig_finalz += s_dispz;

        xr = zig_finalx;
        yr = zig_finaly;
        zr = zig_finalz;
    }
    
    else
    {
    */
    // If axis != 3 then scale down the s_disp
    // values by a factor of two to account 
    // for the "quarter-point".
    // Note that this else will scale the IF 
    // statements above (e.g. if (v0 < 2){ ...  })
    s_dispx *= 0.5;
    s_dispy *= 0.5;
    s_dispz *= 0.5;
    //}
     
    // printf("\n");
    // printf("\nParticle %ld: TEST REFERENCE POINT = %e, %e, %e", pi, xr, yr, zr);
    // printf("\n");
    
    // With xr, yr, and zr known, we can treat them as the final 
    // location on either the zig or the zag. Now we just need 
    // the new midpoint along this new linear zig or zag.
    
    s_midx = 0.5 * ( s_midx + xr );
    s_midy = 0.5 * ( s_midy + yr );
    s_midz = 0.5 * ( s_midz + zr );
    

    // printf("\n");
    // printf("\nParticle %ld: POST IF STATEMENTS s_midx, s_midy, s_midz = %e, %e, %e", pi, s_midx, s_midy, s_midz);
    // printf("\n");

    // printf("\n");
    // printf("\nParticle %ld: POST IF STATEMENTS s_dispx, s_dispy, s_dispz = %e, %e, %e", pi, s_dispx, s_dispy, s_dispz);
    // printf("\n");

    
    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step. 
    // v5 is used in accumulate_j! DO NOT DELETE!
    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);
 
    // Now accumulate the currents
    accumulate_j(x,y,z);
    // printf("\nParticle %d depositing (x,y,z) v0, v1, v2, v3 = %e, %e, %e, %e", pi, v0, v1, v2, v3);
#if CURRENT_TEST
    printf("\nParticle %d currents-xyz %e %e %e %e %e", pi, v0, v1, v2, v3, v0+v1+v2+v3);
#endif
    k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;

    accumulate_j(y,z,x);
    // printf("\nParticle %d depositing (y,z,x) v0, v1, v2, v3 = %e, %e, %e, %e", pi, v0, v1, v2, v3);
#if CURRENT_TEST
    printf("\nParticle %d currents-yzx %e %e %e %e %e", pi, v0, v1, v2, v3, v0+v1+v2+v3);
#endif
    k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

    accumulate_j(z,x,y);
    // printf("\nParticle %d depositing (z,x,y) v0, v1, v2, v3 = %e, %e, %e, %e\n\n", pi, v0, v1, v2, v3);
#if CURRENT_TEST
    printf("\nParticle %d currents-zxy %e %e %e %e %e", pi, v0, v1, v2, v3, v0+v1+v2+v3);
#endif
    k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;


    //printf("\nParticle mover before updating...\npm->dispx, pm->dispy, pm->dispz = %e, %e, %e", pm->dispx, pm->dispy, pm->dispz);
    
    // Subtract off what has been travelled from the full displacement.
    // This approach is equivalent to:
    //   
    //   starting_x = 2 * s_midx - xr;
    //   s_dispx = 2 * pm->dispx - ( zig_finalx - starting_x );
    //   s_dispx *= 0.5;
    //   pm->dispx = s_dispx;
    //
    // along each direction.
    pm->dispx -= 0.5 * ( zig_finalx - (2 * s_midx - xr) );
    pm->dispy -= 0.5 * ( zig_finaly - (2 * s_midy - yr) );
    pm->dispz -= 0.5 * ( zig_finalz - (2 * s_midz - zr) );
/*
    printf("\nCurrents deposited...\naxis %d x %e y %e z %e s_disp x %e y %e z %e", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    printf("\nParticle mover updated...\npm->dispx, pm->dispy, pm->dispz = %e, %e, %e", pm->dispx, pm->dispy, pm->dispz);
*/
    // Set the new particle position post-accumulation.
    p_dx = zig_finalx;
    p_dy = zig_finaly; 
    p_dz = zig_finalz;

    // If an end streak, return success (should be ~50% of the time)
   // printf("\nStreak ended...\naxis %d x %e y %e z %e s_disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
   
  //  printf("\nParticle %d Velocity after accumulation:", pi);
  //  printf("\nux, uy, uz = %e, %e, %e", p_ux, p_uy, p_uz);

    // TODO: Change this break based on neighbor_index == 13.
    // This should help reduce thread divergence.
    //if( axis == 3 ) 
    //{
   //     printf("\n*****************************\nParticle %d is done moving at p_dx, p_dy, p_dz = %e, %e, %e\nIt is supposed to stop at x2, y2, z2 = %e, %e, %e\n****************************\n",
    //            pi, p_dx, p_dy, p_dz, old_fx, old_fy, old_fz);
        //break;
    //}

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    // Change the value of s_dir to be -1, 0, 1. The zero case 
    // corresponds to when the particle does not leave the cell in a
    // particular direction and is moved to the midpoint in that
    // direction.
    s_dir[Axis_Label::x] = ( xr == s_dir[Axis_Label::x] ? s_dir[Axis_Label::x] : 0 );
    s_dir[Axis_Label::y] = ( yr == s_dir[Axis_Label::y] ? s_dir[Axis_Label::y] : 0 );
    s_dir[Axis_Label::z] = ( zr == s_dir[Axis_Label::z] ? s_dir[Axis_Label::z] : 0 );

#if FINAL_POSITION_TEST 
    printf("\nParticle %d: s_dir = %d, %d, %d", pi, (int)s_dir[Axis_Label::x], (int)s_dir[Axis_Label::y], (int)s_dir[Axis_Label::z]);
#endif
    
    // Compute the neighbor cell index the particle moves to. 
    // Note that 0,0,0 => 13 will return the particle to the
    // same cell. 
    // TODO: neighbor_index should replace the face variable
    neighbor_index = get_neighbor_index(s_dir[Axis_Label::x], s_dir[Axis_Label::y], s_dir[Axis_Label::z], planes_per_axis);
    //printf("\nParticle %ld: neighbor_index = %d", pi, neighbor_index);

#if VPIC_DUMP_NEIGHBORS
    print_neighbor(neighbor_index);
    print_neighbor.write_planes(s_dir[Axis_Label::x], s_dir[Axis_Label::y], s_dir[Axis_Label::z]);
#endif

    /* Old stuffs ...
    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;
    */

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
    // Throw neighbor through this function to get the cell
    // index the particle moves into.
    neighbor = d_neighbor( num_neighbors * pii + neighbor_index );
    //printf("\nParticle %ld: neighbor value, reflect_particles = %d, %d", pi, (int)neighbor, (int)reflect_particles);
    
    /*
    if ( abs(s_dir[Axis_Label::x]) + abs(s_dir[Axis_Label::y]) + abs(s_dir[Axis_Label::z]) > 1)
    { 
        printf("\nParticle %d: s_dir = %d, %d, %d", pi, (int)s_dir[Axis_Label::x], (int)s_dir[Axis_Label::y], (int)s_dir[Axis_Label::z]);
        printf("\nneighbor_index = %d", neighbor_index);
        printf("\nneighbor value, reflect_particles = %d, %d", (int)neighbor, (int)reflect_particles);
    }
    */

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back
#define AXIS_LABELER(AX) Axis_Label:: AX

#define SET_1D_REFLECTION(AX)                                                         \
    k_particles(pi, particle_var::u##AX ) = -k_particles(pi, particle_var::u##AX );   \
    pm->disp##AX *= -1.;                                                              \
    s_dir[AXIS_LABELER(AX)] = 0;

#define SET_2D_REFLECTION(AX_1, AX_2)                                                     \
    float minus_v##AX_1 = -k_particles(pi, particle_var::u##AX_1);                        \
    float minus_v##AX_2 = -k_particles(pi, particle_var::u##AX_2);                        \
    k_particles(pi, particle_var::u##AX_1 ) = minus_v##AX_2;                              \
    k_particles(pi, particle_var::u##AX_2 ) = minus_v##AX_1;                              \
    minus_v##AX_1 = -pm->disp##AX_1;                                                      \
    minus_v##AX_2 = -pm->disp##AX_2;                                                      \
    pm->disp##AX_1 = minus_v##AX_2;                                                       \
    pm->disp##AX_2 = minus_v##AX_1;                                                       \
    s_dir[AXIS_LABELER(AX_1)] = 0;                                                        \
    s_dir[AXIS_LABELER(AX_2)] = 0;                                                        

    
    if( neighbor==reflect_particles ) {
        // Hit a reflecting boundary condition.  Reflect the particle
        // momentum and remaining displacement and keep moving the
        // particle.
        
        // printf("\nI, particle %d, was reflected!\nBefore reflection...", pi);
        // printf("\npii, rangel = %d, %d", pii, rangel);
        // printf("\nneighbor, rangel = %d, %d", neighbor - rangel, rangel);
        // printf("\nux, uy, uz = %e, %e, %e", k_particles(pi, particle_var::ux),
        //                                     k_particles(pi, particle_var::uy),
        //                                     k_particles(pi, particle_var::uz));
      
        // Face: which_case == 1, Edge: which_case == 2, Corner: which_case == 3
        // TODO: Use an enum for readability. This stuff is hard enough.
        int which_case =  s_dir[Axis_Label::x] * s_dir[Axis_Label::x]
                        + s_dir[Axis_Label::y] * s_dir[Axis_Label::y]
                        + s_dir[Axis_Label::z] * s_dir[Axis_Label::z];

        // Do the easy Face case.
        if (which_case == 1)
        {
            if      ( s_dir[Axis_Label::x] != 0 )
            {
                SET_1D_REFLECTION(x);
            }
            else if ( s_dir[Axis_Label::y] != 0 )
            {
                SET_1D_REFLECTION(y);
            }
            else   
            {
                SET_1D_REFLECTION(z);
            }
        }
        // Now do the harder edge cases
        else if (which_case == 1)
        {
            // Check how many axes the edge is reflective along
            // (it can be reflective on a maximum of two meaning
            // the edge is a global edge of the simulation).
            int neighbor_face_1 = 0;
            int neighbor_face_2 = 0;

            if      ( s_dir[Axis_Label::x] == 0 )
            {
                // Intercepts a yz edge.
                neighbor_face_1 = d_neighbor( num_neighbors * ii + get_neighbor_index(0, s_dir[Axis_Label::y], 0) );
                neighbor_face_2 = d_neighbor( num_neighbors * ii + get_neighbor_index(0, 0, s_dir[Axis_Label::z]) );

                if      ( neighbor_face_1 == reflect_particles && neighbor_face_2 != reflect_particles )
                {
                    // Only reflect along y as the z axis is open.
                    SET_1D_REFLECTION(y);
                    neighbor = neighbor_face_2;
                }
                else if ( neighbor_face_1 != reflect_particles && neighbor_face_2 == reflect_particles )
                {
                    // Only reflect along z as the y axis is open.
                    SET_1D_REFLECTION(z);
                    neighbor = neighbor_face_1;
                }
                // TODO: There are no other cases, right?
                else if ( neighbor_face_1 != reflect_particles && neighbor_face_2 != reflect_particles )
                {
                    // Hit a global edge. Need to reflect both using a 2d reflection matrix. 
                    SET_2D_REFLECTION(y, z);
                    neighbor = ii;
                } 
            }
            else if ( s_dir[Axis_Label::y] == 0 )
            {
                // Intercepts a zx edge.
                neighbor_face_1 = d_neighbor( num_neighbors * ii + get_neighbor_index(0, 0, s_dir[Axis_Label::z]) );
                neighbor_face_2 = d_neighbor( num_neighbors * ii + get_neighbor_index(s_dir[Axis_Label::x], 0, 0) );
                
                if      ( neighbor_face_1 == reflect_particles && neighbor_face_2 != reflect_particles )
                {
                    // Only reflect along z as the x axis is open.
                    SET_1D_REFLECTION(z);
                    neighbor = neighbor_face_2;
                }
                else if ( neighbor_face_1 != reflect_particles && neighbor_face_2 == reflect_particles )
                {
                    // Only reflect along x as the z axis is open.
                    SET_1D_REFLECTION(x);
                    neighbor = neighbor_face_1;
                }
                // TODO: There are no other cases, right?
                else if ( neighbor_face_1 != reflect_particles && neighbor_face_2 != reflect_particles )
                {
                    // Hit a global edge. Need to reflect both using a 2d reflection matrix. 
                    SET_2D_REFLECTION(z, x);
                    neighbor = ii;
                }
            }
            else
            {
                // Intercepts an xy edge.
                neighbor_face_index_1 = d_neighbor( num_neighbors * ii + get_neighbor_index(s_dir[Axis_Label::x], 0, 0) );
                neighbor_face_index_2 = d_neighbor( num_neighbors * ii + get_neighbor_index(0, s_dir[Axis_Label::y], 0) );

                if      ( neighbor_face_1 == reflect_particles && neighbor_face_2 != reflect_particles )
                {
                    // Only reflect along x as the y axis is open.
                    SET_1D_REFLECTION(x);
                    neighbor = neighbor_face_2;
                }
                else if ( neighbor_face_1 != reflect_particles && neighbor_face_2 == reflect_particles )
                {
                    // Only reflect along y as the x axis is open.
                    SET_1D_REFLECTION(y);
                    neighbor = neighbor_face_1;
                }
                // TODO: There are no other cases, right?
                else if ( neighbor_face_1 != reflect_particles && neighbor_face_2 != reflect_particles )
                {
                    // Hit a global edge. Need to reflect both using a 2d reflection matrix. 
                    SET_2D_REFLECTION(x, y);
                    neighbor = ii;
                }
            }

        }

        // printf("\nAfter reflection...");
        // printf("\nux, uy, uz = %e, %e, %e", k_particles(pi, particle_var::ux),
        //                                     k_particles(pi, particle_var::uy),
        //                                     k_particles(pi, particle_var::uz));
        
        // Now shift the particles appropriately. Only axes where s_dir != 0 
        // must be shifted.
        k_particles( pi, particle_var::dx ) -= 2. * s_dir[Axis_Label::x];
        k_particles( pi, particle_var::dy ) -= 2. * s_dir[Axis_Label::y];
        k_particles( pi, particle_var::dz ) -= 2. * s_dir[Axis_Label::z];

        //printf("\n##########################################\n");
        return 0;
    }
#undef AXIS_LABELER
#undef SET_1D_REFLECTION
#undef SET_2D_REFLECTION

    if ( neighbor<rangel || neighbor>rangeh ) {
        // Cannot handle the boundary condition here.  Save the updated
        // particle position, face it hit and update the remaining
        // displacement in the particle mover as a bitshift.
        //
        // 26 == 11010 in binary and so multiplying by 32 should be
        // sufficient for now. 
        //
        // TODO: Change this to something better to not reduce the 
        // cell number by too much. 
        //pii = 8*pii + face;
        // TODO: Fix boundary_p
        pii = upper_bound_of_neighbors * pii + neighbor_index;
        return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    // TODO: How is rangel affected by 26 neighbors?
    /*
    if ( abs(s_dir[Axis_Label::x]) + abs(s_dir[Axis_Label::y]) + abs(s_dir[Axis_Label::z]) > 1){
    printf("\npii, rangel = %d, %d", pii, rangel);
    //pii = neighbor - rangel;
    printf("\npii, rangel = %d, %d", neighbor - rangel, rangel);
    }
    */
    //printf("\nParticle %ld: pii, rangel = %d, %d", pi, pii, rangel);
    pii = neighbor - rangel;
    //printf("\nParticle %ld: pii, rangel = %d, %d", pi, pii, rangel);
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    //k_particles(pi, particle_var::dx + axis) = -v0;      // Convert coordinate system
   
    // Convert the coordinate system when the particle changes cells.
    // The coordinate change is x -> x - 2 * s_dir to move the origin
    // of coordinates along the s_dir direction by 2. 
    //
    // In the case where s_dir == 0, then this approach holds and there
    // is no coordinate conversion.
    //
    // TODO: Make an enumeration for x,y,z not being 0,1,2.
    /*
    if ( abs(s_dir[Axis_Label::x]) + abs(s_dir[Axis_Label::y]) + abs(s_dir[Axis_Label::z]) > 1){
    printf("\nParticle %d before coordinate shift", pi);
    printf("\ndx, dy, dz = %e, %e, %e", k_particles( pi, particle_var::dx ),
                                        k_particles( pi, particle_var::dy ),
                                        k_particles( pi, particle_var::dz ));
    }
    */
    // TODO: Use the macros since we have them...
    k_particles( pi, particle_var::dx ) -= 2. * s_dir[Axis_Label::x];
    k_particles( pi, particle_var::dy ) -= 2. * s_dir[Axis_Label::y];
    k_particles( pi, particle_var::dz ) -= 2. * s_dir[Axis_Label::z];

    /*
    if ( abs(s_dir[Axis_Label::x]) + abs(s_dir[Axis_Label::y]) + abs(s_dir[Axis_Label::z]) > 1){
    printf("\nParticle %d after coordinate shift", pi);
    printf("\ndx, dy, dz = %e, %e, %e", k_particles( pi, particle_var::dx ),
                                        k_particles( pi, particle_var::dy ),
                                        k_particles( pi, particle_var::dz ));
    printf("\n##########################################\n");
    }
    */

#   undef accumulate_j

