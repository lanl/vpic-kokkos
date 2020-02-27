#define IN_collision

/* #define HAS_V4_PIPELINE */

#include <chrono>
using namespace std::chrono;

#include "src/util/rng_policy.h"
#include "collision_pipeline.h"

#include "../takizuka_abe.h"

#include "../../util/pipelines/pipelines.h"

//#define CMOV(a,b) if(t0<t1) a=b

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
template <class _RNG_t>
KOKKOS_INLINE_FUNCTION
void takizuka_abe_collision(
        const k_particles_t& pi,
        const k_particles_t& pj,
        int i,
        int j,
        float  mu_i,
        float  mu_j,
        float std,
        rng_t* rng, // TODO: remove?
        _RNG_t& rg
)
{
    //particle_t * const RESTRICT pi = (PI);
    //particle_t * const RESTRICT pj = (PJ);
    float dd, ur, tx, ty, tz, t0, t1, t2, stack[3];
    int d0, d1, d2;

    // TODO: do I always want to be reloading these?
    float urx = pi(i, particle_var::ux) - pj(j, particle_var::ux);
    float ury = pi(i, particle_var::uy) - pj(j, particle_var::uy);
    float urz = pi(i, particle_var::uz) - pj(j, particle_var::uz);
    float wi  = pi(i, particle_var::w);
    float wj  = pj(j, particle_var::w);

    /* There are lots of ways to formulate T vector formation    */
    /* This has no branches (but uses L1 heavily)                */

    t0 = urx*urx;
    d0=0;
    d1=1;
    d2=2;
    t1=t0;
    ur  = t0;

    t0 = ury*ury;
    //CMOV(d0,1);
    //CMOV(d1,2);
    //CMOV(d2,0);
    //CMOV(t1,t0);
    if (t0 < t1)
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
    if (t0 < t1)
    {
        d0 = 2;
        d1 = 0;
        d2 = 1;
    }
    ur += t0;

    ur = sqrtf( ur );

    stack[0] = urx;
    stack[1] = ury;
    stack[2] = urz;
    t1  = stack[d1];
    t2  = stack[d2];
    t0  = 1 / sqrtf( t1*t1 + t2*t2 + FLT_MIN );
    stack[d0] =  0;
    stack[d1] =  t0*t2;
    stack[d2] = -t0*t1;
    tx = stack[0];
    ty = stack[1];
    tz = stack[2];

    t0 = 1;
    t2 = 1/ur;
    t1 = std*sqrtf(t2)*t2;

    //CMOV(t1,t0);
    if (t0 < t1)
    {
        t1 = t0;
    }

    //auto _r = frandn(rng);
    auto _r = rg.normal(0, 1);
    //std::cout << "frandn " << _r << " vs " << _r2 << std::endl;
    dd = t1*_r;
    //dd = t1*normal(rng);

    t0 = 2*dd/(1+dd*dd);

    //auto _r2 = frand_c0(rng); // uniform in the interval [0,1)
    auto _r2 = rg.drand(0, 1);
    //t1 = 2*M_PI*frand_c0(rng);
    //std::cout << _r2 << " vs " << _r3 << std::endl;
    t1 = 2*M_PI*_r2;

    t2 = t0*sin(t1);
    t1 = t0*ur*cos(t1);
    t0 *= -dd;

    /* stack = (1 - cos theta) u + |u| sin theta Tperp */
    stack[0] = (t0*urx + t1*tx) + t2*( ury*tz - urz*ty );
    stack[1] = (t0*ury + t1*ty) + t2*( urz*tx - urx*tz );
    stack[2] = (t0*urz + t1*tz) + t2*( urx*ty - ury*tx );

    /* Handle unequal particle weights. */
    //t0 = frand_c0(rng);  // uniform in the interval [0,1)
    t0 = rg.drand(0, 1);
    t1 = mu_i;
    t2 = mu_j;
    if(wj < wi && wi*t0 > wj) t1 = 0 ;
    if(wi < wj && wj*t0 > wi) t2 = 0 ;

    pi(i, particle_var::ux) += t1*stack[0];
    pi(i, particle_var::uy) += t1*stack[1];
    pi(i, particle_var::uz) += t1*stack[2];
    pj(j, particle_var::ux) -= t2*stack[0];
    pj(j, particle_var::uy) -= t2*stack[1];
    pj(j, particle_var::uz) -= t2*stack[2];
}

void
takizuka_abe_pipeline_scalar_kokkos(
        takizuka_abe_t * RESTRICT cm
)
{

  species_t* RESTRICT spi = cm->spi;
  species_t* RESTRICT spj = cm->spj;

  // TODO: move to kokkos rng
  auto& _rng = cm->rp->rng;

  const grid_t* RESTRICT g = spi->g;

  auto& spi_p = spi->k_p_d;
  auto& spi_p_i = spi->k_p_i_d;
  auto& spj_p = spj->k_p_d;

  auto& spj_partition = spj->k_partition_d;
  auto& spi_partition = spi->k_partition_d;

  const float dtinterval_dV = ( g->dt * (float)cm->interval ) / g->dV;
  const float mu    = (spi->m*spj->m)/(spi->m+spj->m);
  const float mu_i  = spj->m/(spi->m+spj->m);
  const float mu_j  = spi->m/(spi->m+spj->m);
  const double cvar = cm->cvar0 * (spi->q*spi->q*spj->q*spj->q) / (mu*mu);

  //int nv = g->nv;

  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;

  //for (int v = 0; v < nv; v++)
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  int v_min = VOXEL(1,1,1,nx,ny,nz);
  int v_max = VOXEL(nx,ny,nz,nx,ny,nz) + 1;
  //Kokkos::RangePolicy< Kokkos::Schedule<Kokkos::Dynamic>, Kokkos::OpenMP> nv_policy_dynamic(v_min, v_max);
  Kokkos::RangePolicy< Kokkos::DefaultExecutionSpace > nv_policy(v_min, v_max);
  Kokkos::parallel_for("collisions cell v loop", nv_policy,
    KOKKOS_LAMBDA (size_t v) // 1d
    {
      // TODO: Currently defaults to host space...
      // TODO: Using the object and getting the state multiple times was giving
      // me the same numbers...
      //_RNG::RandomNumberProvider<_RNG::KokkosRNG> rp;
      _RNG::KokkosRNG rp;
      auto rg = rp.rand_pool.get_state();

    //Kokkos::Experimental::UniqueToken<
        //Kokkos::HostSpace, Kokkos::Experimental::UniqueTokenScope::Global>
        //unique_token{Kokkos::HostSpace()};
        //int thread_id = unique_token.acquire();
        //std::cout << "Token is " << thread_id << std::endl;
        int thread_id = omp_get_thread_num();
        auto& rng = _rng[thread_id]; // TODO: remove

  /*
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nz+1,ny+1,nx+1});
  Kokkos::parallel_for("collisions cell v loop", xyz_policy,
    KOKKOS_LAMBDA (size_t v) // 1d
    KOKKOS_LAMBDA(const int z, const int y, const int x) // 3d
    {
        int v = VOXEL(x, y, z, nx, ny, nz);
  */

        int nl, l0;
        /* Find the species i computational particles, k, and the species j
           computational particles, l, in this voxel, determine the number
           of computational particle pairs, np and the number of candidate
           pairs, nc, to test for collisions within this voxel.  Also,
           split the range of the fastest integer rng into intervals
           suitable for sampling pairs. */

        int k0 = spi_partition(v);
        int nk = spi_partition(v+1) - k0;

        //std::cout << " thread " << omp_get_thread_num() << " of " << omp_get_num_threads() << " has " << nk << " particles at v=" << v << std::endl;
        //std::cout << " x " << x << " y " << y << " z " << z << std::endl;

        // TODO: convert this to be a more explicit check on if we have work
        if( !nk ) return; /* Nothing to do */

        // Compute the species density for this cell while doing a Fisher-Yates
        // shuffle. NOTE: shuffling here instead of computing random indicies allows
        // for better optimization of the collision loop and an overall speedup.
        // [True on CPU with AoS, unlikely to be true for GPU with SoA]
        float density_k = 0;
        int k1 = k0+nk;

        int i;

        // TODO: do we really want to do the shuffle like this on the GPU?
        // Fisher-Yates shuffles
        for( i=k0; i < k1-1; ++i )
        {
            int rn = UINT32_MAX / (uint32_t)(k1-i);

            int j;
            do {
                // Generate a random unsigned int between [0, UINTTYPE_MAX]
                j = i + (int)(uirand(rng)/rn);
                //j = i + (rp.uint(UINT32_MAX)/rn);
            } while( j>=k1 );

            // Swap spi_p[j] with spi_p[i]
            /*
               particle_t* ptemp = spi_p[j];
               spi_p[j] = spi_p[i];
               spi_p[i] = ptemp;
               */

            // Gather temps for swap
            float dx = spi_p(i, particle_var::dx);
            float dy = spi_p(i, particle_var::dy);
            float dz = spi_p(i, particle_var::dz);
            float ux = spi_p(i, particle_var::ux);
            float uy = spi_p(i, particle_var::uy);
            float uz = spi_p(i, particle_var::uz);
            float w = spi_p(i, particle_var::w);
            int ii = spi_p_i(i);

            // Write to i, where we got the temps from
            spi_p(i, particle_var::dx) = spi_p(j, particle_var::dx);
            spi_p(i, particle_var::dy) = spi_p(j, particle_var::dy);
            spi_p(i, particle_var::dz) = spi_p(j, particle_var::dz);
            spi_p(i, particle_var::ux) = spi_p(j, particle_var::ux);
            spi_p(i, particle_var::uy) = spi_p(j, particle_var::uy);
            spi_p(i, particle_var::uz) = spi_p(j, particle_var::uz);
            spi_p(i, particle_var::w) = spi_p(j, particle_var::w);
            spi_p_i(i) = spi_p_i(j);

            // Write temps to j
            spi_p(j, particle_var::dx) = dx;
            spi_p(j, particle_var::dy) = dy;
            spi_p(j, particle_var::dz) = dz;
            spi_p(j, particle_var::ux) = ux;
            spi_p(j, particle_var::uy) = uy;
            spi_p(j, particle_var::uz) = uz;
            spi_p(j, particle_var::w) = w;
            spi_p_i(j) = ii;

            density_k += spi_p(i, particle_var::w);
        }

        //density_k += spi_p[i].w;
        density_k += spi_p(i, particle_var::w);

        float std;
        if( spi==spj )
        {
            if( nk%2 && nk >= 3 ) {
                std = sqrtf(0.5*density_k*cvar*dtinterval_dV);
                takizuka_abe_collision( spi_p,
                        spi_p,
                        k0,
                        k0 + 1,
                        mu_i, mu_j, std, rng, rg );
                takizuka_abe_collision( spi_p,
                        spi_p,
                        k0,
                        k0 + 1,
                        mu_i, mu_j, std, rng, rg );
                takizuka_abe_collision( spi_p,
                        spi_p,
                        k0 + 1,
                        k0 + 2,
                        mu_i, mu_j, std, rng, rg );
                nk -= 3;
                k0 += 3;
            }

            nl = nk = nk/2;
            l0 = k0 + nk;
            if( !nk ) return; /* We had exactly 1 or 3 paticles. */
            std = sqrtf(cvar*density_k*dtinterval_dV);

        }
        else {

            l0 = spj_partition(v);
            nl = spj_partition(v+1) - l0;

            if( !nl ) return; /* Nothing to do */

            // Since spi_p is already randomized, setting spi_j to any specific
            // permutataion will still give a perfectly valid and random set of
            // particle pairs. Which means we can just leave it as is.

            // Compute the species density for this cell.
            float density_l = 0;
            for(i=0 ; i < nl ; ++i)
            {
                //density_l += spj_p[l0+i].w;
                density_l += spj_p(l0+i, particle_var::w);
            }

            // Compute the standard deviation of the collision angle.
            std = sqrtf( cvar*(density_l > density_k ? density_k : density_l)*dtinterval_dV );
        }

        if (nk > nl) {

            int ii = nk/nl;
            int rn = nk - ii*nl;

            for( i=0; i < rn; ++i, ++l0 )
            {
                for( int j=0; j <= ii; ++j, ++k0 )
                {
                    takizuka_abe_collision(
                            spi_p,
                            spj_p,
                            k0,
                            l0,
                            mu_i, mu_j, std, rng, rg);
                }
            }

            for( ; i < nl ; ++i, ++l0 )
            {
                for( int j=0; j < ii; ++j, ++k0 )
                {
                    takizuka_abe_collision(
                            spi_p,
                            spj_p,
                            k0,
                            l0,
                            mu_i, mu_j, std, rng, rg);
                }
            }

        }
        else
        {
            int ii = nl/nk;
            int rn = nl - ii*nk;

            for( i=0; i < rn; ++i, ++k0 )
            {
                for( int j=0; j <= ii; ++j, ++l0 )
                {
                    takizuka_abe_collision(
                            spi_p,
                            spj_p,
                            k0,
                            l0,
                            mu_i, mu_j, std, rng, rg);
                }
            }

            for( ; i < nk ; ++i, ++k0 )
            {

                for( int j = 0; j < ii; ++j, ++l0 )
                {
                    takizuka_abe_collision(
                            spi_p,
                            spj_p,
                            k0,
                            l0,
                            mu_i, mu_j, std, rng, rg);
                }
            }

        }
    }); // end cell loop

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "TA took " << time_span.count() << std::endl;
}

#undef takizuka_abe_collision

void apply_takizuka_abe_pipeline( takizuka_abe_t* cm )
{
    takizuka_abe_pipeline_scalar_kokkos(cm);
    //EXEC_PIPELINES( takizuka_abe, cm, 0 );
    //WAIT_PIPELINES();
}
