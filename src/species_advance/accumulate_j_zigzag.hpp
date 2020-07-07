/* *****************************************************************
    
    This "header" file is meant to be included directly into the 
    move_p_kokkos(...) function in species_advance.h as 
    #include "accumulate_j_zigzag.hpp"
    
    This method was published by Umeda et al. in 2003 and can be 
    found at

        https://doi.org/10.1016/S0010-4655(03)00437-5

    THIS FILE IS THE ZIGZAG ONE!!!!!!!!!!!!!!!!!!!

   ****************************************************************/

    // TODO: Change the equations below to reflect that s_mid and 
    // s_disp variables are normalized by cdt_dx.

    // Obtain the cell indices for the particles.
    float i1 = floor( joe_midx ), i2 = floor( joe_midx + joe_dispx );
    float j1 = floor( joe_midy ), j2 = floor( joe_midy + joe_dispy );
    float k1 = floor( joe_midz ), k2 = floor( joe_midz + joe_dispz );

    printf("\nParticle %d: i1, j1, k1 = %d, %d, %d",
            pii, (int)i1, (int)j1, (int)k1);
    printf("\nParticle %d: i2, j2, k2 = %d, %d, %d",
            pii, (int)i2, (int)j2, (int)k2);

    // Obtain the midpoints for the trajectory over one timestep
    float xmid = joe_midx + 0.5 * joe_dispx;
    float ymid = joe_midy + 0.5 * joe_dispy;
    float zmid = joe_midz + 0.5 * joe_dispz;

    // Obtain the reference points for the particles
    // Here the constant 1.f comes from cell coordinates.
    float xr = fmin( fmin(i1, i2) + 1.f, fmax( fmax(i1, i2), xmid ) );
    float yr = fmin( fmin(j1, j2) + 1.f, fmax( fmax(j1, j2), ymid ) );
    float zr = fmin( fmin(k1, k2) + 1.f, fmax( fmax(k1, k2), zmid ) );

    printf("\nParticle %d: xmid, ymid, zmid = %e, %e, %e\n",
            pii, xmid, ymid, zmid);
    printf("\nParticle %d: xr, yr, zr = %e, %e, %e\n",
            pii, xr, yr, zr);
    printf("\n\n*************************************************************************************************************\n\n");

    // Get the fluxes
    // TODO: Verify that the formula below is correct.
    float Fx1 = q * ( xr - joe_midx  );
    float Fy1 = q * ( yr - joe_midy  );
    float Fz1 = q * ( zr - joe_midz  );
    
    float Fx2 = q * p_ux - Fx1;
    float Fy2 = q * p_uy - Fy1;
    float Fz2 = q * p_uz - Fz1;
    
    // Finally, get the weights
    float Wx1 = 0.5 * ( joe_midx + xr ) - i1;
    float Wx2 = 0.5 * ( joe_midx + joe_dispx + xr ) - i2;
    
    float Wy1 = 0.5 * ( joe_midy + yr ) - j1;
    float Wy2 = 0.5 * ( joe_midy + joe_dispy + yr ) - j2;
    
    float Wz1 = 0.5 * ( joe_midz + zr ) - k1;
    float Wz2 = 0.5 * ( joe_midz + joe_dispz + zr ) - k2;

    // This function is defined for the four edges of Jx. Then
    // cyclically permute X,Y,Z to get Jy and Jz weights.
    // I believe all units are already normalized by cell volume...
    // TODO: Verify that l = 1,2 is not actually necessary when
    // particles move.
#   define accumulate_j_zigzag(X,Y,Z,l)                                                   \
    v0 = F##X##l * (1 - W##Y##l) * (1 - W##Z##l); /*   v0 = Fx * (1 - Wy) * (1 - Wz)   */ \
    v1 = F##X##l * W##Y##l * (1 - W##Z##l);       /*   v1 = Fx * Wy * (1 - Wz)         */ \
    v2 = F##X##l * (1 - W##Y##l) * W##Z##l;       /*   v2 = Fx * (1 - Wy) * Wz         */ \
    v3 = F##X##l * W##Y##l * W##Z##l;             /*   v2 = Fx * Wy * Wz               */


    accumulate_j_zigzag(x,y,z,1);
    printf("\nParticle %d depositing (x,y,z) v0, v1, v2, v4 = %e, %e, %e, %e", pii, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;

    accumulate_j_zigzag(y,z,x,1);
    printf("\nParticle %d depositing (y,z,x) v0, v1, v2, v4 = %e, %e, %e, %e", pii, v0, v1, v2, v3);
    k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
    k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

    accumulate_j_zigzag(z,x,y,1);
    printf("\nParticle %d depositing (z,x,y) v0, v1, v2, v4 = %e, %e, %e, %e", pii, v0, v1, v2, v3);
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



