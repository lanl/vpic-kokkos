#define IN_collision

/* #define HAS_V4_PIPELINE */

#include "collision_pipeline.h"

#include "../takizuka_abe.h"

#include "../../util/pipelines/pipelines.h"

#define CMOV(a,b) if(t0<t1) a=b

/*
void scatter_particles(
        int I,
        int J,
        particle_t* P1,
        particle_t* P2,
        float M1,
        float M2,
        float SQRT_VAR
)
{
    float dux, duy, duz;

    float mratio1 = M2/(M1+M2);
    float mratio2 = M1/(M1+M2);

    float ux = (P1)[I].ux-(P2)[J].ux;
    float uy = (P1)[I].uy-(P2)[J].uy;
    float uz = (P1)[I].uz-(P2)[J].uz;

    float uperp = sqrt(ux*ux+uy*uy);
    float u = sqrt(uperp*uperp+uz*uz);

    //float delta           = maxwellian_rand(SQRT_VAR)*pow(u,-1.5);
    float delta           = normal( rng(0), 0, SQRT_VAR)*pow(u,-1.5);
    float sin_theta       = 2*delta/(1+delta*delta);
    float one_m_cos_theta = sin_theta*delta;
    float phi             = 2*M_PI*uniform_rand(0,1);
    float sin_phi         = sin(phi);
    float cos_phi         = cos(phi);

    // General case
    if ( uperp>0 )
    {
        dux = (( ux*uz * sin_theta * cos_phi) - uy * u * sin_theta * sin_phi)
                / uperp - ux * one_m_cos_theta;

        duy = (( uy*uz * sin_theta * cos_phi) + ux * u * sin_theta * sin_phi)
                / uperp - uy * one_m_cos_theta;

        duz =- uperp *  sin_theta * cos_phi - uz * one_m_cos_theta;
    }
    else { // Handle purely z-directed difference vectors separately
        dux =  u * sin_theta * cos_phi;
        duy =  u * sin_theta * sin_phi;
        duz = -u * one_m_cos_theta;
    }

    (P1)[I].ux += mratio1*dux;
    (P1)[I].uy += mratio1*duy;
    (P1)[I].uz += mratio1*duz;
    (P2)[J].ux -= mratio2*dux;
    (P2)[J].uy -= mratio2*duy;
    (P2)[J].uz -= mratio2*duz;
}
*/

// Branchless and direction-agnositc method for computing momentum transfer.
void takizuka_abe_collision(
        particle_t* pi,
        particle_t* pj,
        float  mu_i,
        float  mu_j,
        float std,
        rng_t* rng // TODO: remove?
)
{
    float urx = pi->ux - pj->ux;
    float ury = pi->uy - pj->uy;
    float urz = pi->uz - pj->uz;
    float wi  = pi->w;
    float wj  = pj->w;

    /* There are lots of ways to formulate T vector formation    */
    /* This has no branches (but uses L1 heavily)                */

    float t0 = urx*urx;
    int d0=0;
    int d1=1;
    int d2=2;
    float t1=t0;

    float ur = t0;
    t0 = ury*ury;

    //CMOV(d0,1);
    //CMOV(d1,2);
    //CMOV(d2,0);
    //CMOV(t1,t0);
    // if (t0 < t1)
    if (ury*ury < urx*urx)
    {
        d0 = 1;
        d1 = 2;
        d2 = 0;
        t1 = t0;
    }

    ur += t0;

    t0 = urz*urz;
    //CMOV(d0,2);
    //CMOV(d1,0);
    //CMOV(d2,1);
    // if (t0 < t1)
    if (urz*urz < urx*urx)
    {
        d0 = 2;
        d1 = 0;
        d2 = 1;
    }

    ur += t0;
    ur = sqrtf( ur );

    float stack[3] = { urx, ury, urz };

    t1  = stack[d1];
    float t2  = stack[d2];
    t0  = 1.0 / sqrtf( t1*t1 + t2*t2 + FLT_MIN );
    stack[d0] =  0;
    stack[d1] =  t0*t2;
    stack[d2] = -1*t0*t1;
    float tx = stack[0];
    float ty = stack[1];
    float tz = stack[2];

    t0 = 1;
    t2 = 1/ur;
    t1 = std*sqrtf(t2)*t2;
    CMOV(t1,t0);
    float dd = t1*frandn(rng);

    t0 = 2*dd/(1+dd*dd);
    t1 = 2*M_PI*frand_c0(rng);
    t2 = t0*sin(t1);
    t1 = t0*ur*cos(t1);
    t0 *= -dd;

    /* stack = (1 - cos theta) u + |u| sin theta Tperp */
    stack[0] = (t0*urx + t1*tx) + t2*( ury*tz - urz*ty );
    stack[1] = (t0*ury + t1*ty) + t2*( urz*tx - urx*tz );
    stack[2] = (t0*urz + t1*tz) + t2*( urx*ty - ury*tx );

    /* Handle unequal particle weights. */
    t0 = frand_c0(rng);
    t1 = mu_i;
    t2 = mu_j;
    if(wj < wi && wi*t0 > wj) t1 = 0;
    if(wi < wj && wj*t0 > wi) t2 = 0;

    pi->ux += t1*stack[0];
    pi->uy += t1*stack[1];
    pi->uz += t1*stack[2];
    pj->ux -= t2*stack[0];
    pj->uy -= t2*stack[1];
    pj->uz -= t2*stack[2];
}

void
takizuka_abe_rfb(
    takizuka_abe_t * RESTRICT cm,
    int pipeline_rank,
    int n_pipeline
)
{
    // Self-scattering among species
    /*
    for (auto& species :  species_list)
    {
        for (auto c : cells)
        {
            //int p1 = 0;
            //int p2 = 0;

            takizuka_abe_collision(
                spi_p + k0,
                spj_p + l0,
                mu_i, mu_j,
                std,
                rng
            );
        }
    }
    */
}


void
takizuka_abe_pipeline_scalar_kokkos(
    takizuka_abe_t * RESTRICT cm
)
{
    // TODO: change rng to use kokkos
    rng_t* RESTRICT rng      = cm->rp->rng[ 0 ];

    species_t* RESTRICT spi  = cm->spi;
    species_t* RESTRICT spj  = cm->spj;
    const grid_t* RESTRICT g = spi->g;

    particle_t* RESTRICT ALIGNED(128) spi_p        = spi->p;
    const int* RESTRICT ALIGNED(128) spi_partition = spi->partition;

    particle_t* RESTRICT ALIGNED(128) spj_p        = spj->p;
    const int* RESTRICT ALIGNED(128) spj_partition = spj->partition;

    const float dtinterval_dV = ( g->dt * (float)cm->interval ) / g->dV;
    const float mu    = (spi->m*spj->m)/(spi->m+spj->m);
    const float mu_i  = spj->m/(spi->m+spj->m);
    const float mu_j  = spi->m/(spi->m+spj->m);
    const double cvar = cm->cvar0 * (spi->q*spi->q*spj->q*spj->q) / (mu*mu);

    particle_t ptemp;
    float std, density_k, density_l;
    int i, j, rn;
    int k0, k1, nk, l0, nl; //, size_k, size_l;

    /* Stripe the (mostly non-ghost) voxels over threads for load balance */
    int v = VOXEL( 0,0,0, g->nx,g->ny,g->nz );

    //v1 = VOXEL( g->nx,g->ny,g->nz, g->nx,g->ny,g->nz ) + 1;

    /* Find the species i computational particles, k, and the species j
       computational particles, l, in this voxel, determine the number
       of computational particle pairs, np and the number of candidate
       pairs, nc, to test for collisions within this voxel.  Also,
       split the range of the fastest integer rng into intervals
       suitable for sampling pairs. */

    k0 = spi_partition[v  ];
    nk = spi_partition[v+1] - k0;
    if( !nk ) return; /* Nothing to do */

    // Compute the species density for this cell while doing a Fisher-Yates
    // shuffle. NOTE: shuffling here instead of computing random indicies allows
    // for better optimization of the collision loop and an overall speedup.
    density_k = 0;
    k1 = k0+nk;
    for(i=k0 ; i < k1-1 ; ++i)
    {
        rn = UINT32_MAX / (uint32_t)(k1-i);
        do { j = i + (int)(uirand(rng)/rn); } while( j>=k1 );
        ptemp = spi_p[j], spi_p[j] = spi_p[i], spi_p[i] = ptemp;
        density_k += spi_p[i].w;
    }

    density_k += spi_p[i].w;

    if( spi==spj ) {
        if( nk%2 && nk >= 3 )
        {
            std = sqrtf(0.5*density_k*cvar*dtinterval_dV);

            takizuka_abe_collision(
                spi_p + k0,
                spi_p + k0 + 1,
                mu_i,
                mu_j,
                std,
                rng
            );

            takizuka_abe_collision(
                spi_p + k0,
                spi_p + k0 + 2,
                mu_i,
                mu_j,
                std,
                rng
            );

            takizuka_abe_collision(
                spi_p + k0 + 1,
                spi_p + k0 + 2,
                mu_i,
                mu_j,
                std,
                rng
            );

            nk -= 3;
            k0 += 3;
        }

        nl = nk = nk/2;
        l0 = k0 + nk;
        if( !nk ) return; /* We had exactly 1 or 3 paticles. */
        std = sqrtf(cvar*density_k*dtinterval_dV);
    }
    else
    {
        l0 = spj_partition[v  ];
        nl = spj_partition[v+1] - l0;
        if( !nl ) return; /* Nothing to do */

        // Since spi_p is already randomized, setting spi_j to any specific
        // permutataion will still give a perfectly valid and random set of
        // particle pairs. Which means we can just leave it as is.

        // Compute the species density for this cell.
        density_l = 0;
        for(i=0 ; i < nl ; ++i)
        {
            density_l += spj_p[l0+i].w;
        }

        // Compute the standard deviation of the collision angle.
        std = sqrtf( cvar*(density_l > density_k ? density_k : density_l)*dtinterval_dV );
    }

    if (nk > nl) {
        int ii = nk/nl;
        int rn = nk - ii*nl;

        for( i=0 ; i < rn ; ++i, ++l0 ) {
            for( j=0 ; j <= ii ; ++j, ++k0 ) {
                takizuka_abe_collision( spi_p + k0,
                        spj_p + l0,
                        mu_i, mu_j, std, rng);
            }
        }

        for( ; i < nl ; ++i, ++l0 )  {
            for( j=0 ; j < ii ; ++j, ++k0 ) {
                takizuka_abe_collision( spi_p + k0,
                        spj_p + l0,
                        mu_i, mu_j, std, rng);
            }
        }

    } else {
        int ii = nl/nk;
        int rn = nl - ii*nk;

        for( i=0 ; i < rn ; ++i, ++k0 ) {
            for( j=0 ; j <= ii ; ++j, ++l0 ) {
                takizuka_abe_collision( spi_p + k0,
                        spj_p + l0,
                        mu_i, mu_j, std, rng);
            }
        }

        for( ; i < nk ; ++i, ++k0 ) {
            for( j=0 ; j < ii ; ++j, ++l0 ) {
                takizuka_abe_collision( spi_p + k0,
                        spj_p + l0,
                        mu_i, mu_j, std, rng);
            }
        }
    }
} // end function

#undef CMOV
#undef takizuka_abe_collision

void apply_takizuka_abe_pipeline( takizuka_abe_t * cm )
{
    takizuka_abe_pipeline_scalar_kokkos(cm);
    //EXEC_PIPELINES( takizuka_abe, cm, 0 );
    //WAIT_PIPELINES();
}
