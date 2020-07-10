/* *****************************************************************
    
    This "header" file is meant to be included directly into the 
    move_p_kokkos(...) function in species_advance.h as 
    #include "accumulate_j_zigzag.hpp"
    
    This method was published by Umeda et al. in 2003 and can be 
    found at

        https://doi.org/10.1016/S0010-4655(03)00437-5

    THIS FILE IS THE ZIGZAG ONE!!!!!!!!!!!!!!!!!!!

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


    // Obtain the cell indices for the particles.
    /*
    float i1 = floor( 0.5 * ( 1. + joe_midx ) ), i2 = floor( 0.5 * ( 1. + joe_midx + joe_dispx ) );
    float j1 = floor( 0.5 * ( 1. + joe_midy ) ), j2 = floor( 0.5 * ( 1. + joe_midy + joe_dispy ) );
    float k1 = floor( 0.5 * ( 1. + joe_midz ) ), k2 = floor( 0.5 * ( 1. + joe_midz + joe_dispz ) );
    */
    
    /* 
    // Destroy the elegance of Umeda's algorithm for funsies.
    // By default i1,j1,k1 should all be the left-side of the 
    // cell, i.e. should be 0 unless the particle is exactly
    // on the boundary.
    float i1 = 0.f, j1 = 0.f, k1 = 0.f;
    if ( joe_midx == 1.f ) i1 = 1.f;
    if ( joe_midy == 1.f ) j1 = 1.f;
    if ( joe_midz == 1.f ) k1 = 1.f;

    // Assume that the particle stays in-cell so that its 
    // Umeda indices are identical. If the particle leaves the cell,
    // increment the index appropriately. Right now, we assume that
    // a particle moving at light speed can travel one cell in one
    // time step.
    float i2 = i1, j2 = j1, k2 = k1;
    if ( joe_midx + joe_dispx >= 1.f ) i2 = 1.f;
    if ( joe_midx + joe_dispx == 3.f ) i2 = 2.f;    // This should never happen.
    if ( joe_midx + joe_dispx <= -1.f ) i2 = -1.f;
    if ( joe_midx + joe_dispx == -3.f ) i2 = -2.f;  // This should never happen.
    
    if ( joe_midy + joe_dispy >= 1.f ) j2 = 1.f;
    if ( joe_midy + joe_dispy == 3.f ) j2 = 2.f;    // This should never happen.
    if ( joe_midy + joe_dispy <= -1.f ) j2 = -1.f;
    if ( joe_midy + joe_dispy == -3.f ) j2 = -2.f;  // This should never happen.
    
    if ( joe_midz + joe_dispz >= 1.f ) k2 = 1.f;
    if ( joe_midz + joe_dispz == 3.f ) k2 = 2.f;    // This should never happen.
    if ( joe_midz + joe_dispz <= -1.f ) k2 = -1.f;
    if ( joe_midz + joe_dispz == -3.f ) k2 = -2.f;  // This should never happen.
    */

#define cell2umeda_pt(val) 0.5 * ( 1. + val )       // Convert a point from [-1,1] to [0,1]
#define cell2umeda_diff(val) 0.5 * val              // Convert a difference from [-1,1] to [0,1]
#define umeda2cell_pt(val) -1. + 2. * val           // Convert a point from [0,1] to [-1,1]
#define umeda2cell_diff(val) 2. * val               // Convert a difference from [0,1] to [-1,1]
   
    joe_midx = cell2umeda_pt(joe_midx), joe_midy = cell2umeda_pt(joe_midy), joe_midz = cell2umeda_pt(joe_midz);
    joe_dispx = cell2umeda_diff(joe_dispx), joe_dispy = cell2umeda_diff(joe_dispy), joe_dispz = cell2umeda_diff(joe_dispz);
     
    // Obtain the cell indices for the particles.
    float i1 = floor(joe_midx), i2 = floor(joe_midx + joe_dispx);
    float j1 = floor(joe_midy), j2 = floor(joe_midy + joe_dispy);
    float k1 = floor(joe_midz), k2 = floor(joe_midz + joe_dispz);

    printf("\nParticle %d: i1, j1, k1 = %d, %d, %d",
            pi, (int)i1, (int)j1, (int)k1);
    printf("\nParticle %d: i2, j2, k2 = %d, %d, %d",
            pi, (int)i2, (int)j2, (int)k2);

    // Obtain the midpoints for the trajectory over one timestep
    float xmid = joe_midx + 0.5 * joe_dispx;
    float ymid = joe_midy + 0.5 * joe_dispy;
    float zmid = joe_midz + 0.5 * joe_dispz;

    // Obtain the reference points for the particles
    // Here the constant 1.f comes from cell coordinates.
    float xr = fmin( fmin(i1, i2) + 1.f, fmax( fmax(i1, i2), xmid ) );
    float yr = fmin( fmin(j1, j2) + 1.f, fmax( fmax(j1, j2), ymid ) );
    float zr = fmin( fmin(k1, k2) + 1.f, fmax( fmax(k1, k2), zmid ) );

    int lval = ( axis == 3 ? 2 : 1 );
    if ( axis == 3 )
    {
        xr = joe_midx, yr = joe_midy, zr = joe_midz;
    }

    printf("\nParticle %d: xmid, ymid, zmid = %e, %e, %e\n",
            pi, xmid, ymid, zmid);
    printf("\nParticle %d: xr, yr, zr = %e, %e, %e\n",
            pi, xr, yr, zr);

    // Get the fluxes in the Umeda coordinates
    float Fx1 = q * ( xr - joe_midx  );
    float Fy1 = q * ( yr - joe_midy  );
    float Fz1 = q * ( zr - joe_midz  );
    
    float Fx2 = q * p_ux - Fx1;
    float Fy2 = q * p_uy - Fy1;
    float Fz2 = q * p_uz - Fz1;
    
    // Finally, get the weights in the Umeda coordinates
    float Wx1 = 0.5 * ( joe_midx + xr ) - i1;
    float Wx2 = 0.5 * ( joe_midx + joe_dispx + xr ) - i2;
    
    float Wy1 = 0.5 * ( joe_midy + yr ) - j1;
    float Wy2 = 0.5 * ( joe_midy + joe_dispy + yr ) - j2;
    
    float Wz1 = 0.5 * ( joe_midz + zr ) - k1;
    float Wz2 = 0.5 * ( joe_midz + joe_dispz + zr ) - k2;
    
    joe_midx = umeda2cell_pt(joe_midx), joe_midy = umeda2cell_pt(joe_midy), joe_midz = umeda2cell_pt(joe_midz);
    joe_dispx = umeda2cell_diff(joe_dispx), joe_dispy = umeda2cell_diff(joe_dispy), joe_dispz = umeda2cell_diff(joe_dispz);
    
#undef cell2umeda_pt
#undef cell2umeda_diff
#undef umeda2cell_pt
#undef umeda2cell_diff

    // This function is defined for the four edges of Jx. Then
    // cyclically permute X,Y,Z to get Jy and Jz weights.
    // I believe all units are already normalized by cell volume...
    // TODO: Verify that l = 1,2 is not actually necessary when
    // particles move.
#   define accumulate_j_zigzag(X,Y,Z,l)                                                      \
    v0 = F##X##l * (1 - W##Y##l) * (1 - W##Z##l); /*   v0 = Fux * (1 - Wuy) * (1 - Wuz)   */ \
    printf("\nUmeda v0 = %e", v0);                                                           \
    v1 = F##X##l * W##Y##l * (1 - W##Z##l);       /*   v1 = Fux * Wuy * (1 - Wuz)         */ \
    v2 = F##X##l * (1 - W##Y##l) * W##Z##l;       /*   v2 = Fux * (1 - Wuy) * Wuz         */ \
    v3 = F##X##l * W##Y##l * W##Z##l;             /*   v2 = Fux * Wuy * Wuz               */ \
    /*   Now convert from the Umeda coordinates back into VPIC cell cooridinates          */ \
    /*   These equations guarantee that the total flux summed around the edges            */ \
    /*   are Fx = 2Fux as they should be (both the fluxes and the weights                 */ \
    /*   transform as differences between coordinate systems.)                            */ \
    printf("\nFux, Wuy, Wuz = %e, %e, %e", F##X##l, W##Y##l, W##Z##l);                       \
    v0 = 8 * v0; /*- 4 * F##X##l * ( 1.5 - W##Y##l - W##Z##l );*/                            \
    printf("\nVPIC v0 = %e\n", v0);                                                          \
    v1 = 8 * v1; /*- 4 * F##X##l * W##Y##l;      */                                          \
    v2 = 8 * v2; /* - 4 * F##X##l * W##Z##l;  */                                             \
    v3 = 8 * v3;

    if ( lval == 1 )
    {   
        accumulate_j_zigzag(x,y,z,1);
    }
    if ( lval == 2 )
    {
        accumulate_j_zigzag(x,y,z,2); 
    }
    printf("\nParticle %d depositing (x,y,z,l) v0, v1, v2, v4 = %e, %e, %e, %e, %d", pi, v0, v1, v2, v3, lval);
    k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;


    if ( lval == 1 )
    {
        accumulate_j_zigzag(y,z,x,1);
    }
    if ( lval == 2 )
    {
        accumulate_j_zigzag(y,z,x,2); 
    }
    printf("\nParticle %d depositing (y,z,x,l) v0, v1, v2, v4 = %e, %e, %e, %e, %d", pi, v0, v1, v2, v3, lval);
    k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

    if ( lval == 1 )
    {
        accumulate_j_zigzag(z,x,y,1);
    }
    if ( lval == 2 )
    {
        accumulate_j_zigzag(z,x,y,2);  
    }
    printf("\nParticle %d depositing (z,x,y,l) v0, v1, v2, v4 = %e, %e, %e, %e, %d\n\n", pi, v0, v1, v2, v3, lval);
    k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;

#   undef accumulate_j_zigzag
    
    /*
    // Compute the remaining particle displacment
    pm->dispx = joe_dispx - ( xr - joe_midx );
    pm->dispy = joe_dispy - ( yr - joe_midy );
    pm->dispz = joe_dispz - ( zr - joe_midz );

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    p_dx = xr;
    p_dy = yr;
    p_dz = zr;
    */

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;



