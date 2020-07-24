/* *****************************************************************
    
    This "header" file is meant to be included directly into the 
    move_p_kokkos(...) function in species_advance.h as 
    #include "accumulate_j_zigzag.hpp"
    
    This method was published by Umeda et al. in 2003 and can be 
    found at

        https://doi.org/10.1016/S0010-4655(03)00437-5

    THIS FILE IS THE ZIGZAG ONE!!!!!!!!!!!!!!!!!!!

   ****************************************************************/
    
    printf("\nParticle %d in Voxel %d Velocity before accumulation:", pi, pii);
    printf("\nux, uy, uz = %e, %e, %e", p_ux, p_uy, p_uz);

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

    printf("\nParticle %d: pre axis %d x %e y %e z %e", pi, axis, p_dx, p_dy, p_dz);

    printf("\nParticle %d: disp x %e y %e z %e", pi, s_dispx, s_dispy, s_dispz);

    // Compute the direction that the particle moves.
    // This value is the also the boundary of the cell 
    // the particle will intercept.
    s_dir[Axis_Label::x] = (s_dispx>0) ? 1 : -1;
    s_dir[Axis_Label::y] = (s_dispy>0) ? 1 : -1;
    s_dir[Axis_Label::z] = (s_dispz>0) ? 1 : -1;

    printf("\ns_dir = %d, %d, %d", (int)s_dir[Axis_Label::x], (int)s_dir[Axis_Label::y], (int)s_dir[Axis_Label::z]);

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
    /**/      v3=2,  axis=3;
    if(v0<v3) v3=v0, axis=0;
    if(v1<v3) v3=v1, axis=1;
    if(v2<v3) v3=v2, axis=2;
    // Multiply v3 by 1/2 because the particle first moves to the 
    // midpoint if axis != 3, or it stops if axis == 3.
    v3 *= 0.5;

    
    printf("\nParticle %d: axis, v0, v1, v2, v3 = %d, %e, %e, %e, %e",
            pi, axis, v0, v1, v2, 2.*v3);
    printf("\nParticle %d: s_midx, s_midy, s_midz = %e, %e, %e",
            pi, s_midx, s_midy, s_midz);
    printf("\nParticle %d: s_dispx, s_dispy, s_dispz = %e, %e, %e",
            pi, s_dispx, s_dispy, s_dispz);
    /*printf("\nParticle %d: s_midx + s_dispx, s_midy + s_dispy, s_midz + s_dispz = %e, %e, %e",
            pi, s_midx + s_dispx, s_midy + s_dispy, s_midz + s_dispz);
    printf("\nParticle %d: s_midx + 2*s_dispx, s_midy + 2*s_dispy, s_midz + 2*s_dispz = %e, %e, %e",
            pi, s_midx + 2*s_dispx, s_midy + 2*s_dispy, s_midz + 2*s_dispz);
    if( axis != 3 )
    {
        printf("\nParticle %d: s_midx + 2*s_dispx -/+ 2, s_midy + 2*s_dispy -/+ 2, s_midz + 2*s_dispz -/+ 2 = %e, %e, %e",
                pi, s_midx + 2*s_dispx - 2*s_dir[0], s_midy + 2*s_dispy - 2*s_dir[1], s_midz + 2*s_dispz - 2*s_dir[2]);
    }
    */

    // Store the old values of s_mid and s_disp before I do crazy
    // things.
    float old_midx = s_midx;
    float old_midy = s_midy;
    float old_midz = s_midz;
    float old_dispx = s_dispx;
    float old_dispy = s_dispy;
    float old_dispz = s_dispz;

    // Umeda algorithm: assume axis == 3 and set xr, yr, and zr
    // to be the end of the the zag (so the final destination 
    // of the particle).
    float xr = s_midx + 2. * s_dispx;
    float yr = s_midy + 2. * s_dispy;
    float zr = s_midz + 2. * s_dispz;
   
    // If the particle crosses the x-boundary change xr
    // to the boundary it hits.
    if ( v0 < 2. )
    {
        xr = s_dir[Axis_Label::x];
        s_dispx *= v0;
    }
    // If the particle crosses the y-boundary change yr
    // to the boundary it hits.
    if ( v1 < 2. )
    {
        yr = s_dir[Axis_Label::y];
        s_dispy *= v1;
    }
    // If the particle crosses the z-boundary change zr
    // to the boundary it hits.
    if ( v2 < 2. )
    {
        zr = s_dir[Axis_Label::z];
        s_dispz *= v2;
    } 
    // With xr, yr, and zr known, we can treat them as the final 
    // location on either the zig or the zag. Now we just need 
    // the new midpoint along this new linear zig or zag.
    s_midx = 0.5 * ( s_midx + xr );
    s_midy = 0.5 * ( s_midy + yr );
    s_midz = 0.5 * ( s_midz + zr );

    // Change the displacement to the midpoint along the zig
    // or zag.
    if ( axis != 3 )
    {
        s_dispx *= 0.5;
        s_dispy *= 0.5;
        s_dispz *= 0.5;
    }

    // Compute the midpoint and the normalized displacement of the 
    // streak. By scaling the displacments by v3, if axis == 3, then
    // nothing is done (v3 == 1.0 at this point), but if axis != 3,
    // then s_disp is set to s_dir - s_mid along the appropriate
    // axis, while the other displacements are scaled to keep the
    // particle on the same linear trajectory.
    /*
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;
    */
    
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
    printf("\nParticle %d depositing (x,y,z) v0, v1, v2, v3 = %e, %e, %e, %e", pi, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;

    accumulate_j(y,z,x);
    printf("\nParticle %d depositing (y,z,x) v0, v1, v2, v3 = %e, %e, %e, %e", pi, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

    accumulate_j(z,x,y);
    printf("\nParticle %d depositing (z,x,y) v0, v1, v2, v3 = %e, %e, %e, %e\n\n", pi, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;

#   undef accumulate_j


    printf("\nParticle mover before updating...\npm->dispx, pm->dispy, pm->dispz = %e, %e, %e", pm->dispx, pm->dispy, pm->dispz);
    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    printf("\nCurrents deposited...\naxis %d x %e y %e z %e disp x %e y %e z %e", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    printf("\nParticle mover updated...\npm->dispx, pm->dispy, pm->dispz = %e, %e, %e", pm->dispx, pm->dispy, pm->dispz);

    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    printf("\nStreak ended...\naxis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
   
    printf("\nParticle %d Velocity after accumulation:", pi);
    printf("\nux, uy, uz = %e, %e, %e", p_ux, p_uy, p_uz);

    if( axis == 3 ) 
    {
        printf("\n*****************************\nParticle %d is done moving at p_dx, p_dy, p_dz = %e, %e, %e\nIt is supposed to stop at x2, y2, z2 = %e, %e, %e\n****************************\n",
                pi, p_dx, p_dy, p_dz, xr, yr, zr);
        break;
    }

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
    
    printf("\ns_dir = %d, %d, %d", (int)s_dir[Axis_Label::x], (int)s_dir[Axis_Label::y], (int)s_dir[Axis_Label::z]);

    // Compute the neighbor cell index the particle moves to. 
    // Note that 0,0,0 => 13 will return the particle to the
    // same cell. 
    // TODO: neighbor_index should replace the face variable
    neighbor_index = get_neighbor_index(s_dir[Axis_Label::x], s_dir[Axis_Label::y], s_dir[Axis_Label::z], planes_per_axis);
    printf("\nneighbor_index = %d", neighbor_index);

#if VPIC_DUMP_NEIGHBORS
    print_neighbor(neighbor_index);
    print_neighbor.write_planes(s_dir[Axis_Label::x], s_dir[Axis_Label::y], s_dir[Axis_Label::z]);
#endif

    /* Old stuffs ...
    // TODO: Change this to allow for corner crossing.
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
    printf("\nneighbor value, reflect_particles = %d, %d", (int)neighbor, (int)reflect_particles);

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back

    
    if( neighbor==reflect_particles ) {
        // Hit a reflecting boundary condition.  Reflect the particle
        // momentum and remaining displacement and keep moving the
        // particle.
        printf("\nI, particle %d, was reflected!\nBefore reflection...", pi);
        printf("\nux, uy, uz = %e, %e, %e", k_particles(pi, particle_var::ux),
                                            k_particles(pi, particle_var::uy),
                                            k_particles(pi, particle_var::uz));
        if ( s_dir[Axis_Label::x] != 0 )
        {
            k_particles(pi, particle_var::ux ) = -k_particles(pi, particle_var::ux );
            pm->dispx *= -1.;
        }

        if ( s_dir[Axis_Label::y] != 0 )
        {
            k_particles(pi, particle_var::uy ) = -k_particles(pi, particle_var::uy );
            pm->dispy *= -1.;
        }

        if ( s_dir[Axis_Label::z] != 0 )
        {
            k_particles(pi, particle_var::uz ) = -k_particles(pi, particle_var::uz );
            pm->dispz *= -1.;
        }

        printf("\nAfter reflection...");
        printf("\nux, uy, uz = %e, %e, %e", k_particles(pi, particle_var::ux),
                                            k_particles(pi, particle_var::uy),
                                            k_particles(pi, particle_var::uz));
        
        /* Old stuffs ... 
        k_particles(pi, particle_var::ux + axis ) = -k_particles(pi, particle_var::ux + axis );

        // TODO: make this safer
        //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
        //k_local_particle_movers(0, particle_mover_var::dispx + axis) = -k_local_particle_movers(0, particle_mover_var::dispx + axis);
        // TODO: replace this, it's horrible
        (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
        */

        continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
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
        printf("\nWeird if statement...\npii = %d", pii);
        // TODO: Fix boundary_p
        pii = 32 * pii + neighbor_index;
        printf("\npii = %d", pii);
        return 1; // Return "mover still in use"
    }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    // TODO: How is rangel affected by 26 neighbors?
    printf("\npii, rangel = %d, %d", pii, rangel);
    pii = neighbor - rangel;
    printf("\npii, rangel = %d, %d", pii, rangel);
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
    printf("\nParticle %d before coordinate shift", pi);
    printf("\ndx, dy, dz = %e, %e, %e", k_particles( pi, particle_var::dx ),
                                        k_particles( pi, particle_var::dy ),
                                        k_particles( pi, particle_var::dz ));
    k_particles( pi, particle_var::dx ) -= 2. * s_dir[Axis_Label::x];
    k_particles( pi, particle_var::dy ) -= 2. * s_dir[Axis_Label::y];
    k_particles( pi, particle_var::dz ) -= 2. * s_dir[Axis_Label::z];


    printf("\nParticle %d after coordinate shift", pi);
    printf("\ndx, dy, dz = %e, %e, %e", k_particles( pi, particle_var::dx ),
                                        k_particles( pi, particle_var::dy ),
                                        k_particles( pi, particle_var::dz ));

    printf("\n##########################################\n");

    

