// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/kokkos_tuning.hpp"
#include <Kokkos_SIMD.hpp>
#include "../../vpic/kokkos_simd_extensions.h"
#include "advance_p_helpers.hpp"

#undef ENABLE_CABANA

//#define ADVANCE_P_GPU
#define ADVANCE_P_UNIFIED
//#define ADVANCE_P_SIMD_COMPUTE
//#define ENABLE_SIMD_INTERPOLATORS
//#define ENABLE_SIMD_ACCUMULATORS
//#define ACCUMULATOR_LAYOUT Kokkos::LayoutRight
//#define ADVANCE_P_SIMD
//#define ADVANCE_P_SIMD_UNION
//#define ADVANCE_P_SIMD_CABANA

#ifdef  ADVANCE_P_CABANA
#define ENABLE_CABANA
#endif

#ifdef ADVANCE_P_UNIFIED
void
advance_p_kokkos_unified(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sv_t k_f_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
        const float qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  k_field_t k_field = fa->k_f_d;
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

  #define p_dx    k_particles( p_index, particle_var::dx)
  #define p_dy    k_particles( p_index, particle_var::dy)
  #define p_dz    k_particles( p_index, particle_var::dz)
  #define p_ux    k_particles( p_index, particle_var::ux)
  #define p_uy    k_particles( p_index, particle_var::uy)
  #define p_uz    k_particles( p_index, particle_var::uz)
  #define p_w     k_particles( p_index, particle_var::w )
  #define pii     k_particles_i(p_index)

  #define f_cbx k_interp(ii[LANE], interpolator_var::cbx)
  #define f_cby k_interp(ii[LANE], interpolator_var::cby)
  #define f_cbz k_interp(ii[LANE], interpolator_var::cbz)
  #define f_ex  k_interp(ii[LANE], interpolator_var::ex)
  #define f_ey  k_interp(ii[LANE], interpolator_var::ey)
  #define f_ez  k_interp(ii[LANE], interpolator_var::ez)

  #define f_dexdy    k_interp(ii[LANE], interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii[LANE], interpolator_var::dexdz)

  #define f_d2exdydz k_interp(ii[LANE], interpolator_var::d2exdydz)
  #define f_deydx    k_interp(ii[LANE], interpolator_var::deydx)
  #define f_deydz    k_interp(ii[LANE], interpolator_var::deydz)

  #define f_d2eydzdx k_interp(ii[LANE], interpolator_var::d2eydzdx)
  #define f_dezdx    k_interp(ii[LANE], interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii[LANE], interpolator_var::dezdy)

  #define f_d2ezdxdy k_interp(ii[LANE], interpolator_var::d2ezdxdy)
  #define f_dcbxdx   k_interp(ii[LANE], interpolator_var::dcbxdx)
  #define f_dcbydy   k_interp(ii[LANE], interpolator_var::dcbydy)
  #define f_dcbzdz   k_interp(ii[LANE], interpolator_var::dcbzdz)

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::View<float*[12]> accumulator("Accumulator", k_field.extent(0));
  Kokkos::deep_copy(accumulator, 0);
  auto current_sv = Kokkos::Experimental::create_scatter_view(accumulator);
#else
  k_field_sv_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);;
#endif

// Setting up work distribution settings
#if defined( VPIC_ENABLE_VECTORIZATION ) && !defined( USE_GPU )
  constexpr int num_lanes = 16;
  int chunk_size = num_lanes;
  int num_chunks = np/num_lanes;
  if(num_chunks*num_lanes < np)
    num_chunks += 1;
  auto policy = Kokkos::TeamPolicy<>(num_chunks, 1, num_lanes);
#elif defined( VPIC_ENABLE_HIERARCHICAL )
  auto policy = Kokkos::TeamPolicy<>(LEAGUE_SIZE, TEAM_SIZE);
  int chunk_size = np/LEAGUE_SIZE;
  if(chunk_size*LEAGUE_SIZE < np)
    chunk_size += 1;
  constexpr int num_lanes = 1;
  int num_chunks = LEAGUE_SIZE;
#else
  constexpr int num_lanes = 1;
#endif

  KOKKOS_TIC();

// Outermost parallel loop
#if defined(VPIC_ENABLE_HIERARCHICAL) || defined(VPIC_ENABLE_VECTORIZATION)
  Kokkos::parallel_for("advance_p", policy, 
  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
      auto current_sa = current_sv.access();
      int chunk = team_member.league_rank();
      int num_iters = chunk_size;
      if((chunk+1)*chunk_size > np)
        num_iters = np - chunk*chunk_size;
      size_t pi_offset = chunk*chunk_size;
#else
  auto policy = Kokkos::RangePolicy<>(0,np);
  Kokkos::parallel_for("advance_p", policy, KOKKOS_LAMBDA (const size_t pi_offset) {
      auto current_sa = current_sv.access();
#endif

// Inner parallelization loop
#if defined ( VPIC_ENABLE_HIERARCHICAL ) && !defined( VPIC_ENABLE_VECTORIZATION )
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_iters), [&] (const size_t index) {
      size_t pi_offset = chunk*chunk_size + index;
#endif
      int num_particles = num_lanes;
      if(pi_offset+num_particles > np)
        num_particles = np - pi_offset;
      float dx[num_lanes];
      float dy[num_lanes];
      float dz[num_lanes];
      float ux[num_lanes];
      float uy[num_lanes];
      float uz[num_lanes];
      float hax[num_lanes];
      float hay[num_lanes];
      float haz[num_lanes];
      float cbx[num_lanes];
      float cby[num_lanes];
      float cbz[num_lanes];
      float q[num_lanes];
      int   ii[num_lanes];
      int   inbnds[num_lanes];

      float fcbx[num_lanes];
      float fcby[num_lanes];
      float fcbz[num_lanes];
      float fex[num_lanes];
      float fey[num_lanes];
      float fez[num_lanes];
      float fdexdy[num_lanes];
      float fdexdz[num_lanes];
      float fd2exdydz[num_lanes];
      float fdeydx[num_lanes];
      float fdeydz[num_lanes];
      float fd2eydzdx[num_lanes];
      float fdezdx[num_lanes];
      float fdezdy[num_lanes];
      float fd2ezdxdy[num_lanes];
      float fdcbxdx[num_lanes];
      float fdcbydy[num_lanes];
      float fdcbzdz[num_lanes];
      float  v0[num_lanes];
      float  v1[num_lanes];
      float  v2[num_lanes];
      float  v3[num_lanes];
      float  v4[num_lanes];
      float  v5[num_lanes];
      float *v6 = fex;
      float *v7 = fdexdy;
      float *v8 = fdexdz;
      float *v9 = fd2exdydz;
      float *v10 = fey;
      float *v11 = fdeydz;
      float *v12 = fdeydx;
      float *v13 = fd2eydzdx;

      size_t p_index = pi_offset;

      BEGIN_VECTOR_BLOCK {
        p_index = pi_offset + LANE;
        // Load position
        dx[LANE] = p_dx;
        dy[LANE] = p_dy;
        dz[LANE] = p_dz;
        // Load momentum
        ux[LANE] = p_ux;
        uy[LANE] = p_uy;
        uz[LANE] = p_uz;
        // Load weight
        q[LANE]  = p_w;
        // Load index
        ii[LANE] = pii;
      } END_VECTOR_BLOCK;

      load_interpolators<num_lanes>( fex, fdexdy, fdexdz, fd2exdydz,
                                     fey, fdeydz, fdeydx, fd2eydzdx,
                                     fez, fdezdx, fdezdy, fd2ezdxdy,
                                     fcbx, fdcbxdx,
                                     fcby, fdcbydy,
                                     fcbz, fdcbzdz,
                                     ii, num_particles, k_interp);

      BEGIN_VECTOR_BLOCK {
        // Interpolate E
        hax[LANE] = qdt_2mc*( (fex[LANE] + dy[LANE]*fdexdy[LANE] ) + dz[LANE]*(fdexdz[LANE] + dy[LANE]*fd2exdydz[LANE]) );
        hay[LANE] = qdt_2mc*( (fey[LANE] + dz[LANE]*fdeydz[LANE] ) + dx[LANE]*(fdeydx[LANE] + dz[LANE]*fd2eydzdx[LANE]) );
        haz[LANE] = qdt_2mc*( (fez[LANE] + dx[LANE]*fdezdx[LANE] ) + dy[LANE]*(fdezdy[LANE] + dx[LANE]*fd2ezdxdy[LANE]) );
  
        // Interpolate B
        cbx[LANE] = fcbx[LANE] + dx[LANE]*fdcbxdx[LANE];
        cby[LANE] = fcby[LANE] + dy[LANE]*fdcbydy[LANE];
        cbz[LANE] = fcbz[LANE] + dz[LANE]*fdcbzdz[LANE];
  
        // Half advance e
        ux[LANE] += hax[LANE];
        uy[LANE] += hay[LANE];
        uz[LANE] += haz[LANE];
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
        v0[LANE] = qdt_2mc/sqrtf(one + (ux[LANE]*ux[LANE] + (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE])));
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
        p_index = pi_offset + LANE;

        // Boris - scalars
        v1[LANE] = cbx[LANE]*cbx[LANE] + (cby[LANE]*cby[LANE] + cbz[LANE]*cbz[LANE]);
        v2[LANE] = (v0[LANE]*v0[LANE])*v1[LANE];
        v3[LANE] = v0[LANE]*(one+v2[LANE]*(one_third+v2[LANE]*two_fifteenths));
        v4[LANE] = v3[LANE]/(one+v1[LANE]*(v3[LANE]*v3[LANE]));
        v4[LANE] += v4[LANE];
        // Boris - uprime
        v0[LANE] = ux[LANE] + v3[LANE]*(uy[LANE]*cbz[LANE] - uz[LANE]*cby[LANE]);
        v1[LANE] = uy[LANE] + v3[LANE]*(uz[LANE]*cbx[LANE] - ux[LANE]*cbz[LANE]);
        v2[LANE] = uz[LANE] + v3[LANE]*(ux[LANE]*cby[LANE] - uy[LANE]*cbx[LANE]);
        // Boris - rotation
        ux[LANE] += v4[LANE]*(v1[LANE]*cbz[LANE] - v2[LANE]*cby[LANE]);
        uy[LANE] += v4[LANE]*(v2[LANE]*cbx[LANE] - v0[LANE]*cbz[LANE]);
        uz[LANE] += v4[LANE]*(v0[LANE]*cby[LANE] - v1[LANE]*cbx[LANE]);
        // Half advance e
        ux[LANE] += hax[LANE];
        uy[LANE] += hay[LANE];
        uz[LANE] += haz[LANE];
        // Store momentum
        p_ux = ux[LANE];
        p_uy = uy[LANE];
        p_uz = uz[LANE];
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
        v0[LANE]   = one/sqrtf(one + (ux[LANE]*ux[LANE]+ (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE])));
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {

        /**/                                      // Get norm displacement
        ux[LANE]  *= cdt_dx;
        uy[LANE]  *= cdt_dy;
        uz[LANE]  *= cdt_dz;
        ux[LANE]  *= v0[LANE];
        uy[LANE]  *= v0[LANE];
        uz[LANE]  *= v0[LANE];
        v0[LANE]   = dx[LANE] + ux[LANE];                           // Streak midpoint (inbnds)
        v1[LANE]   = dy[LANE] + uy[LANE];
        v2[LANE]   = dz[LANE] + uz[LANE];
        v3[LANE]   = v0[LANE] + ux[LANE];                           // New position
        v4[LANE]   = v1[LANE] + uy[LANE];
        v5[LANE]   = v2[LANE] + uz[LANE];
  
        inbnds[LANE] = v3[LANE]<=one &&  v4[LANE]<=one &&  v5[LANE]<=one &&
                      -v3[LANE]<=one && -v4[LANE]<=one && -v5[LANE]<=one;
      } END_VECTOR_BLOCK;
    
#ifdef VPIC_ENABLE_TEAM_REDUCTION
      int in_cell = particles_in_same_cell(team_member, ii, inbnds, num_iters);
#endif

      BEGIN_VECTOR_BLOCK {
        p_index = pi_offset + LANE;

        v3[LANE] = static_cast<float>(inbnds[LANE])*v3[LANE] + (1.0-static_cast<float>(inbnds[LANE]))*p_dx;
        v4[LANE] = static_cast<float>(inbnds[LANE])*v4[LANE] + (1.0-static_cast<float>(inbnds[LANE]))*p_dy;
        v5[LANE] = static_cast<float>(inbnds[LANE])*v5[LANE] + (1.0-static_cast<float>(inbnds[LANE]))*p_dz;
        q[LANE]  = static_cast<float>(inbnds[LANE])*q[LANE]*qsp;

        p_dx = v3[LANE];
        p_dy = v4[LANE];
        p_dz = v5[LANE];
        dx[LANE] = v0[LANE];
        dy[LANE] = v1[LANE];
        dz[LANE] = v2[LANE];
        v5[LANE] = q[LANE]*ux[LANE]*uy[LANE]*uz[LANE]*one_third;

#       define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                                              \
        v4[LANE]  = q[LANE]*u##X[LANE];   /* v2 = q ux                            */        \
        v1[LANE]  = v4[LANE]*d##Y[LANE];  /* v1 = q ux dy                         */        \
        v0[LANE]  = v4[LANE]-v1[LANE];    /* v0 = q ux (1-dy)                     */        \
        v1[LANE] += v4[LANE];             /* v1 = q ux (1+dy)                     */        \
        v4[LANE]  = one+d##Z[LANE];       /* v4 = 1+dz                            */        \
        v2[LANE]  = v0[LANE]*v4[LANE];    /* v2 = q ux (1-dy)(1+dz)               */        \
        v3[LANE]  = v1[LANE]*v4[LANE];    /* v3 = q ux (1+dy)(1+dz)               */        \
        v4[LANE]  = one-d##Z[LANE];       /* v4 = 1-dz                            */        \
        v0[LANE] *= v4[LANE];             /* v0 = q ux (1-dy)(1-dz)               */        \
        v1[LANE] *= v4[LANE];             /* v1 = q ux (1+dy)(1-dz)               */        \
        v0[LANE] += v5[LANE];             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
        v1[LANE] -= v5[LANE];             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
        v2[LANE] -= v5[LANE];             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
        v3[LANE] += v5[LANE];             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

        ACCUMULATE_J( x,y,z, v6,v7,v8,v9 );

        ACCUMULATE_J( y,z,x, v10,v11,v12,v13 );

        ACCUMULATE_J( z,x,y, v0,v1,v2,v3 );
      } END_VECTOR_BLOCK;

#ifdef VPIC_ENABLE_TEAM_REDUCTION
      if(in_cell) {
        int first = ii[0];
        reduce_and_accumulate_current(team_member, current_sa, num_iters, first, 
                                      nx, ny, nz, cx, cy, cz,
                                      v6, v7, v8, v9,
                                      v10, v11, v12, v13,
                                      v0, v1, v2, v3);
      } else {
#endif
        BEGIN_VECTOR_BLOCK {
          accumulate_current(current_sa, ii[LANE],
                       nx, ny, nz, cx, cy, cz, 
                       v6[LANE], v7[LANE], v8[LANE], v9[LANE],
                       v10[LANE], v11[LANE], v12[LANE], v13[LANE],
                       v0[LANE], v1[LANE], v2[LANE], v3[LANE]);
        } END_VECTOR_BLOCK;
#ifdef VPIC_ENABLE_TEAM_REDUCTION
      }
#endif
#       undef ACCUMULATE_J
      BEGIN_THREAD_BLOCK {
        if(!inbnds[LANE]) {
          p_index = pi_offset + LANE;

          DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
          local_pm->dispx = ux[LANE];
          local_pm->dispy = uy[LANE];
          local_pm->dispz = uz[LANE];
          local_pm->i     = p_index;

          if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                             current_sa, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
          {
            if( k_nm(0)<max_nm ) {
              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
              k_particle_movers_i(nm)   = local_pm->i;

              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
              k_particle_copy(nm, particle_var::dx) = p_dx;
              k_particle_copy(nm, particle_var::dy) = p_dy;
              k_particle_copy(nm, particle_var::dz) = p_dz;
              k_particle_copy(nm, particle_var::ux) = p_ux;
              k_particle_copy(nm, particle_var::uy) = p_uy;
              k_particle_copy(nm, particle_var::uz) = p_uz;
              k_particle_copy(nm, particle_var::w) = p_w;
              k_particle_i_copy(nm) = pii;
            }
          }
        }
      } END_THREAD_BLOCK;
#if defined( VPIC_ENABLE_HIERARCHICAL ) && !defined( VPIC_ENABLE_VECTORIZATION )
      });
#endif
  });

  KOKKOS_TOC(advance_p, 1);
  KOKKOS_TIC();

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::Experimental::contribute(accumulator, current_sv);
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
      int f0  = VOXEL(x, y, z, nx, ny, nz);
      int a0  = VOXEL(x, y, z, nx, ny, nz);
      int ax  = VOXEL(x-1, y,   z,   nx, ny, nz);
      int ay  = VOXEL(x,   y-1, z,   nx, ny, nz);
      int az  = VOXEL(x,   y,   z-1, nx, ny, nz);
      int ayz = VOXEL(x,   y-1, z-1, nx, ny, nz);
      int azx = VOXEL(x-1, y,   z-1, nx, ny, nz);
      int axy = VOXEL(x-1, y-1, z,   nx, ny, nz);
      k_field(f0, field_var::jfx) += ( accumulator(a0,  0) +
                                       accumulator(ay,  1) +
                                       accumulator(az,  2) +
                                       accumulator(ayz, 3) );
      k_field(f0, field_var::jfy) += ( accumulator(a0,  4) +
                                       accumulator(az,  5) +
                                       accumulator(ax,  6) +
                                       accumulator(azx, 7) );
      k_field(f0, field_var::jfz) += ( accumulator(a0,  8) +
                                       accumulator(ax,  9) +
                                       accumulator(ay,  10) +
                                       accumulator(axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif

  KOKKOS_TOC(reduce_accumulators, 1);

#undef p_dx
#undef p_dy
#undef p_dz
#undef p_ux
#undef p_uy
#undef p_uz
#undef p_w 
#undef pii 

#undef f_cbx
#undef f_cby
#undef f_cbz
#undef f_ex 
#undef f_ey 
#undef f_ez 

#undef f_dexdy
#undef f_dexdz

#undef f_d2exdydz
#undef f_deydx   
#undef f_deydz   

#undef f_d2eydzdx
#undef f_dezdx   
#undef f_dezdy   

#undef f_d2ezdxdy
#undef f_dcbxdx  
#undef f_dcbydy  
#undef f_dcbzdz  
}
#endif // ADVANCE_P_UNIFIED

#ifdef ADVANCE_P_GPU
void
advance_p_kokkos_gpu(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sv_t k_f_sa,
        k_interpolator_t& k_interp,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
        const float qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;
  const float cx = 0.25 * g->rdy * g->rdz / g->dt;
  const float cy = 0.25 * g->rdz * g->rdx / g->dt;
  const float cz = 0.25 * g->rdx * g->rdy / g->dt;

  // Process particles for this pipeline

  #define p_dx       k_particles(p_index, particle_var::dx)
  #define p_dy       k_particles(p_index, particle_var::dy)
  #define p_dz       k_particles(p_index, particle_var::dz)
  #define p_ux       k_particles(p_index, particle_var::ux)
  #define p_uy       k_particles(p_index, particle_var::uy)
  #define p_uz       k_particles(p_index, particle_var::uz)
  #define p_w        k_particles(p_index, particle_var::w )
  #define pii        k_particles_i(p_index)

  #define f_cbx      k_interp(ii, interpolator_var::cbx)
  #define f_cby      k_interp(ii, interpolator_var::cby)
  #define f_cbz      k_interp(ii, interpolator_var::cbz)
  #define f_ex       k_interp(ii, interpolator_var::ex)
  #define f_ey       k_interp(ii, interpolator_var::ey)
  #define f_ez       k_interp(ii, interpolator_var::ez)

  #define f_dexdy    k_interp(ii, interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii, interpolator_var::dexdz)

  #define f_d2exdydz k_interp(ii, interpolator_var::d2exdydz)
  #define f_deydx    k_interp(ii, interpolator_var::deydx)
  #define f_deydz    k_interp(ii, interpolator_var::deydz)

  #define f_d2eydzdx k_interp(ii, interpolator_var::d2eydzdx)
  #define f_dezdx    k_interp(ii, interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii, interpolator_var::dezdy)

  #define f_d2ezdxdy k_interp(ii, interpolator_var::d2ezdxdy)
  #define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
  #define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
  #define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)

  k_field_t k_field = fa->k_f_d;
// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::View<float*[12]> accumulator("Accumulator", k_field.extent(0));
  Kokkos::deep_copy(accumulator, 0);
  auto current_sv = Kokkos::Experimental::create_scatter_view(accumulator);
#else
  k_field_sv_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
#endif

  // copy local memmbers from grid
  //auto nfaces_per_voxel = 6;
  //auto nvoxels = g->nv;
  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // zero out nm, we could probably do this earlier if we're worried about it
  // slowing things down
  Kokkos::deep_copy(k_nm, 0);

  KOKKOS_TIC();

#ifdef VPIC_ENABLE_HIERARCHICAL
  auto team_policy = Kokkos::TeamPolicy<>(LEAGUE_SIZE, TEAM_SIZE);
  int per_league = np/LEAGUE_SIZE;
  if(np%LEAGUE_SIZE > 0)
    per_league += 1;
  Kokkos::parallel_for("advance_p", team_policy, KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [=] (size_t pindex) {
      int p_index = team_member.league_rank()*per_league + pindex;
      if(p_index < np) {
#else
  auto range_policy = Kokkos::RangePolicy<>(0,np);
  Kokkos::parallel_for("advance_p", range_policy, KOKKOS_LAMBDA (size_t p_index) {
#endif
      
    float v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13;
    auto  current_sa = current_sv.access();

    float dx   = p_dx;                             // Load position
    float dy   = p_dy;
    float dz   = p_dz;
    int   ii   = pii;
    float hax  = qdt_2mc*(    ( f_ex    + dy*f_dexdy    ) +
                           dz*( f_dexdz + dy*f_d2exdydz ) );
    float hay  = qdt_2mc*(    ( f_ey    + dz*f_deydz    ) +
                           dx*( f_deydx + dz*f_d2eydzdx ) );
    float haz  = qdt_2mc*(    ( f_ez    + dx*f_dezdx    ) +
                           dy*( f_dezdy + dx*f_d2ezdxdy ) );

    float cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
    float cby  = f_cby + dy*f_dcbydy;
    float cbz  = f_cbz + dz*f_dcbzdz;
    float ux   = p_ux;                             // Load momentum
    float uy   = p_uy;
    float uz   = p_uz;
    float q    = p_w;
    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;
    v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
    /**/                                      // Boris - scalars
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = (v0*v0)*v1;
    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4   = v3/(one+v1*(v3*v3));
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
    p_ux = ux;                               // Store momentum
    p_uy = uy;
    p_uz = uz;

    v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;
    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;
    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

#ifdef VPIC_ENABLE_TEAM_REDUCTION
    int reduce = 0;
    int inbnds = v3<=one && v4<=one && v5<=one && -v3<=one && -v4<=one && -v5<=one;
    int min_inbnds = inbnds;
    int max_inbnds = inbnds;
    team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
    team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
    int min_index = ii;
    int max_index = ii;
    team_member.team_reduce(Kokkos::Max<int>(max_index));
    team_member.team_reduce(Kokkos::Min<int>(min_index));
    reduce = min_inbnds == max_inbnds && min_index == max_index;
#endif

    // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
    if(  v3<=one &&  v4<=one &&  v5<=one &&   // Check if inbnds
        -v3<=one && -v4<=one && -v5<=one ) {

      // Common case (inbnds).  Note: accumulator values are 4 times
      // the total physical charge that passed through the appropriate
      // current quadrant in a time-step

      q *= qsp;
      p_dx = v3;                             // Store new position
      p_dy = v4;
      p_dz = v5;
      dx = v0;                                // Streak midpoint
      dy = v1;
      dz = v2;
      v13 = q*ux*uy*uz*one_third;              // Compute correction
      v12 = v4;

#     define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                            \
      v12  = q*u##X;   /* v2 = q ux                            */       \
      v1  = v12*d##Y;  /* v1 = q ux dy                         */       \
      v0  = v12-v1;    /* v0 = q ux (1-dy)                     */       \
      v1 += v12;       /* v1 = q ux (1+dy)                     */       \
      v12  = one+d##Z; /* v12 = 1+dz                           */       \
      v2  = v0*v12;    /* v2 = q ux (1-dy)(1+dz)               */       \
      v3  = v1*v12;    /* v3 = q ux (1+dy)(1+dz)               */       \
      v12  = one-d##Z; /* v12 = 1-dz                           */       \
      v0 *= v12;       /* v0 = q ux (1-dy)(1-dz)               */       \
      v1 *= v12;       /* v1 = q ux (1+dy)(1-dz)               */       \
      v0 += v13;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */       \
      v1 -= v13;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */       \
      v2 -= v13;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */       \
      v3 += v13;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

      ACCUMULATE_J( x,y,z, v0,v1,v2,v3 );

      ACCUMULATE_J( y,z,x, v4,v5,v6,v7 );

      ACCUMULATE_J( z,x,y, v8,v9,v10,v11 );

#ifdef VPIC_ENABLE_TEAM_REDUCTION
      if(reduce) {
#ifdef __CUDA_ARCH__
  int mask = 0xffffffff;
  int team_rank = team_member.team_rank();
  for(int i=16; i>0; i=i/2) {
    v0  += __shfl_down_sync(mask, v0, i);
    v1  += __shfl_down_sync(mask, v1, i);
    v2  += __shfl_down_sync(mask, v2, i);
    v3  += __shfl_down_sync(mask, v3, i);
    v4  += __shfl_down_sync(mask, v4, i);
    v5  += __shfl_down_sync(mask, v5, i);
    v6  += __shfl_down_sync(mask, v6, i);
    v7  += __shfl_down_sync(mask, v7, i);
    v8  += __shfl_down_sync(mask, v8, i);
    v9  += __shfl_down_sync(mask, v9, i);
    v10 += __shfl_down_sync(mask, v10, i);
    v11 += __shfl_down_sync(mask, v11, i);
  }
#else
  team_member.team_reduce(Kokkos::Sum<float>(v0));
  team_member.team_reduce(Kokkos::Sum<float>(v1));
  team_member.team_reduce(Kokkos::Sum<float>(v2));
  team_member.team_reduce(Kokkos::Sum<float>(v3));
  team_member.team_reduce(Kokkos::Sum<float>(v4));
  team_member.team_reduce(Kokkos::Sum<float>(v5));
  team_member.team_reduce(Kokkos::Sum<float>(v6));
  team_member.team_reduce(Kokkos::Sum<float>(v7));
  team_member.team_reduce(Kokkos::Sum<float>(v8));
  team_member.team_reduce(Kokkos::Sum<float>(v9));
  team_member.team_reduce(Kokkos::Sum<float>(v10));
  team_member.team_reduce(Kokkos::Sum<float>(v11));
#endif
  if(team_member.team_rank() == 0) {
    accumulate_current(current_sa, ii, nx, ny, nz, cx, cy, cz, 
                       v0, v1, v2, v3,
                       v4, v5, v6, v7,
                       v8, v9, v10, v11);
  }
//        int iii = ii;
//        int zi = iii/((nx+2)*(ny+2));
//        iii -= zi*(nx+2)*(ny+2);
//        int yi = iii/(nx+2);
//        int xi = iii - yi*(nx+2);
//
//        int i0 = ii;
//        int i1 = VOXEL(xi,yi+1,zi,nx,ny,nz);
//        int i2 = VOXEL(xi,yi,zi+1,nx,ny,nz);
//        int i3 = VOXEL(xi,yi+1,zi+1,nx,ny,nz);
////        ACCUMULATE_J( x,y,z );
//        contribute_current(team_member, current_sa, i0, i1, i2, i3, 
//                            field_var::jfx, cx*v0, cx*v1, cx*v2, cx*v3);
//
//        i1 = VOXEL(xi,yi,zi+1,nx,ny,nz);
//        i2 = VOXEL(xi+1,yi,zi,nx,ny,nz);
//        i3 = VOXEL(xi+1,yi,zi+1,nx,ny,nz);
////        ACCUMULATE_J( y,z,x );
//        contribute_current(team_member, current_sa, i0, i1, i2, i3, 
//                            field_var::jfy, cy*v4, cy*v5, cy*v6, cy*v7);
//
//        i1 = VOXEL(xi+1,yi,zi,nx,ny,nz);
//        i2 = VOXEL(xi,yi+1,zi,nx,ny,nz);
//        i3 = VOXEL(xi+1,yi+1,zi,nx,ny,nz);
////        ACCUMULATE_J( z,x,y );
//        contribute_current(team_member, current_sa, i0, i1, i2, i3, 
//                            field_var::jfz, cz*v8, cz*v9, cz*v10, cz*v11);
      } else {
#endif
        // TODO: That 2 needs to be 2*NGHOST eventually
        accumulate_current(current_sa, ii, nx, ny, nz, cx, cy, cz, 
                           v0, v1, v2, v3,
                           v4, v5, v6, v7,
                           v8, v9, v10, v11);
//        int iii = ii;
//        int zi = iii/((nx+2)*(ny+2));
//        iii -= zi*(nx+2)*(ny+2);
//        int yi = iii/(nx+2);
//        int xi = iii - yi*(nx+2);
//        ACCUMULATE_J( x,y,z );
//        current_sa(ii, field_var::jfx) += cx*v0;
//        current_sa(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
//        current_sa(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
//        current_sa(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
//
//        ACCUMULATE_J( y,z,x );
//        current_sa(ii, field_var::jfy) += cy*v0;
//        current_sa(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
//        current_sa(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
//        current_sa(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;
//
//        ACCUMULATE_J( z,x,y );
//        current_sa(ii, field_var::jfz) += cz*v0;
//        current_sa(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
//        current_sa(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
//        current_sa(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
#ifdef VPIC_ENABLE_TEAM_REDUCTION
      }
#endif

#     undef ACCUMULATE_J
    } else {
      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
      local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;
      local_pm->i     = p_index;

      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
      if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                         current_sa, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
      {
        if( k_nm(0) < max_nm )
        {
            const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
            if (nm >= max_nm) Kokkos::abort("overran max_nm");

            k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
            k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
            k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
            k_particle_movers_i(nm)   = local_pm->i;

            // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
            k_particle_copy(nm, particle_var::dx) = p_dx;
            k_particle_copy(nm, particle_var::dy) = p_dy;
            k_particle_copy(nm, particle_var::dz) = p_dz;
            k_particle_copy(nm, particle_var::ux) = p_ux;
            k_particle_copy(nm, particle_var::uy) = p_uy;
            k_particle_copy(nm, particle_var::uz) = p_uz;
            k_particle_copy(nm, particle_var::w) = p_w;
            k_particle_i_copy(nm) = pii;

            // Tag this one as having left
            //k_particles(p_index, particle_var::pi) = 999999;

            // Copy local local_pm back
            //local_pm_dispx = local_pm->dispx;
            //local_pm_dispy = local_pm->dispy;
            //local_pm_dispz = local_pm->dispz;
            //local_pm_i = local_pm->i;
            //printf("rank copying %d to nm %d \n", local_pm_i, nm);
            //copy_local_to_pm(nm);
        }
      }
    }
#ifdef VPIC_ENABLE_HIERARCHICAL
  }
  });
#endif
  });

  KOKKOS_TOC(advance_p, 1);
  KOKKOS_TIC();

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::Experimental::contribute(accumulator, current_sv);
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
      int f0  = VOXEL(x, y, z, nx, ny, nz);
      int a0  = VOXEL(x, y, z, nx, ny, nz);
      int ax  = VOXEL(x-1, y,   z,   nx, ny, nz);
      int ay  = VOXEL(x,   y-1, z,   nx, ny, nz);
      int az  = VOXEL(x,   y,   z-1, nx, ny, nz);
      int ayz = VOXEL(x,   y-1, z-1, nx, ny, nz);
      int azx = VOXEL(x-1, y,   z-1, nx, ny, nz);
      int axy = VOXEL(x-1, y-1, z,   nx, ny, nz);
      k_field(f0, field_var::jfx) += ( accumulator(a0,  0) +
                                       accumulator(ay,  1) +
                                       accumulator(az,  2) +
                                       accumulator(ayz, 3) );
      k_field(f0, field_var::jfy) += ( accumulator(a0,  4) +
                                       accumulator(az,  5) +
                                       accumulator(ax,  6) +
                                       accumulator(azx, 7) );
      k_field(f0, field_var::jfz) += ( accumulator(a0,  8) +
                                       accumulator(ax,  9) +
                                       accumulator(ay,  10) +
                                       accumulator(axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif

  KOKKOS_TOC(reduce_accumulators, 1);

  // TODO: abstract this manual data copy
  //Kokkos::deep_copy(h_nm, k_nm);

  //args->seg[pipeline_rank].pm        = pm;
  //args->seg[pipeline_rank].max_nm    = max_nm;
  //args->seg[pipeline_rank].nm        = h_nm(0);
  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
  //delete(k_local_particle_movers_p);
  //return h_nm(0);

  #undef p_dx 
  #undef p_dy 
  #undef p_dz 
  #undef p_ux 
  #undef p_uy 
  #undef p_uz 
  #undef p_w  
  #undef pii  

  #undef f_cbx 
  #undef f_cby 
  #undef f_cbz 
  #undef f_ex  
  #undef f_ey  
  #undef f_ez  

  #undef f_dexdy    
  #undef f_dexdz    

  #undef f_d2exdydz 
  #undef f_deydx    
  #undef f_deydz    

  #undef f_d2eydzdx 
  #undef f_dezdx    
  #undef f_dezdy    

  #undef f_d2ezdxdy 
  #undef f_dcbxdx   
  #undef f_dcbydy   
  #undef f_dcbzdz   
}
#endif // ADVANCE_P_GPU

#ifdef ADVANCE_P_SIMD_COMPUTE
void
advance_p_kokkos_simd_compute(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sv_t k_f_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
        const float _qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{
  constexpr auto num_lanes = SIMD_LEN;

  const simd_float_t one = 1.0f;
  const simd_float_t one_third = 1.0f/3.0f;
  const simd_float_t two_fifteenths = 2.0f/15.0f;

  k_field_t k_field = fa->k_f_d;

  const simd_float_t cx = 0.25 * g->rdy * g->rdz / g->dt;
  const simd_float_t cy = 0.25 * g->rdz * g->rdx / g->dt;
  const simd_float_t cz = 0.25 * g->rdx * g->rdy / g->dt;
  const simd_float_t qdt_2mc(_qdt_2mc);

  const auto rangel = g->rangel;
  const auto rangeh = g->rangeh;

  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT+PAD_SIZE_INTERPOLATOR], k_interpolator_t::array_layout> const_interp = k_interp;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  size_t n_accum = k_field.extent(0) % 16 == 0 ? k_field.extent(0) : 16*((k_field.extent(0)/16)+1);
  Kokkos::View<float***, ACCUMULATOR_LAYOUT> accumulator("Accumulator", Kokkos::num_threads(), n_accum, 16);
  Kokkos::deep_copy(accumulator, 0);
#else
  k_field_sv_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);;
#endif

  Kokkos::Profiling::pushRegion("advance_p_internal");
  KOKKOS_TIC();

// Setting up work distribution settings
  const int num_threads = 1;//Kokkos::num_threads();
  const int chunk_size = SIMD_LEN*num_threads;
  int num_chunks = np/chunk_size;
  if(num_chunks*chunk_size < np)
    num_chunks += 1;
  auto policy = Kokkos::TeamPolicy<>(num_chunks, num_threads);
  Kokkos::parallel_for("advance_p", policy, 
  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    int leagueID = team_member.league_rank();
    int threadID = team_member.team_rank();
    size_t p_index = team_member.league_rank()*team_member.team_size() + team_member.team_rank();

//  Kokkos::Experimental::UniqueToken<> token;
//  auto num_chunks = np/SIMD_LEN;
//  if(num_chunks*SIMD_LEN < np)
//    num_chunks += 1;
//  auto policy = Kokkos::RangePolicy<size_t>(0,num_chunks);
//  Kokkos::parallel_for("advance_p", policy, KOKKOS_LAMBDA (const size_t p_index) {
//    int threadID = token.acquire();

#if defined( VPIC_ENABLE_ACCUMULATORS )
    auto current = Kokkos::subview(accumulator, threadID, Kokkos::ALL, Kokkos::ALL);
#else
    auto current = current_sv.access();
#endif

    simd_float_t v0, v1, v2, v3, v4, v5;
    simd_float_t v6, v7, v8, v9, v10, v11;
    simd_float_t v12, v13, v14, v15;

    simd_float_t dx,dy,dz,ux,uy,uz,q;
    simd_int32_t ii;
    simd_float_mask_t inbnds;

//    simd_float_t fex, fdexdy, fdexdz, fd2exdydz;
//    simd_float_t fey, fdeydz, fdeydx, fd2eydzdx;
//    simd_float_t fez, fdezdx, fdezdy, fd2ezdxdy;
//    simd_float_t fcbx , fcby, fcbz;     
//    simd_float_t fdcbxdx, fdcbydy, fdcbzdz;

    simd_float_mask_t mask([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    simd_int32_mask_t mask_int([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    size_t active_lanes = num_lanes;
    if(p_index*num_lanes+active_lanes >= np)
      active_lanes = np - p_index*num_lanes;

    // Load particles
    load_particles(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
                   dx, dy, dz, ii, ux, uy, uz, q);

    // Load interpolators
    simd_float_t hax, hay, haz;
    simd_float_t cbx, cby, cbz;
#ifdef ENABLE_SIMD_INTERPOLATORS
    load_interpolators(const_interp, active_lanes, mask, ii,
                             dx, dy, dz,
                             hax, hay, haz,
                             cbx, cby, cbz,
                             qdt_2mc);
#else
    load_interpolators_basic(const_interp, active_lanes, mask, ii,
                             dx, dy, dz,
                             hax, hay, haz,
                             cbx, cby, cbz,
                             qdt_2mc);
#endif

//    load_interpolators(const_interp, active_lanes, mask, ii,
//                       fex, fdexdy, fdexdz, fd2exdydz,
//                       fey, fdeydz, fdeydx, fd2eydzdx,
//                       fez, fdezdx, fdezdy, fd2ezdxdy,
//                       fcbx, fdcbxdx, fcby, fdcbydy, fcbz, fdcbzdz);
//
//    // Interpolate E
//    simd_float_t hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
//    simd_float_t hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
//    simd_float_t haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
//    // Interpolate B
//    simd_float_t cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
//    simd_float_t cby  = Kokkos::fma(dy, fdcbydy, fcby);
//    simd_float_t cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    //v0  = qdt_2mc/Kokkos::sqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    v0  = qdt_2mc * rsqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    // Boris - scalars
    v1  = Kokkos::fma(cbx, cbx, Kokkos::fma(cby, cby, cbz*cbz));
    v2  = (v0*v0)*v1;
    v3  = v0*Kokkos::fma(v2, Kokkos::fma(v2, two_fifteenths, one_third), one);
    v4  = v3/Kokkos::fma(v1, (v3*v3), one);
    v4 += v4;
    // Boris - uprime
    v0  = Kokkos::fma( Kokkos::fma( uy, cbz, -uz*cby ), v3, ux);
    v1  = Kokkos::fma( Kokkos::fma( uz, cbx, -ux*cbz ), v3, uy);
    v2  = Kokkos::fma( Kokkos::fma( ux, cby, -uy*cbx ), v3, uz);
    // Boris - rotation
    ux  = Kokkos::fma( Kokkos::fma( v1, cbz, -v2*cby ), v4, ux);
    uy  = Kokkos::fma( Kokkos::fma( v2, cbx, -v0*cbz ), v4, uy);
    uz  = Kokkos::fma( Kokkos::fma( v0, cby, -v1*cbx ), v4, uz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    // Store momentum in registers for later storage
    v6  = ux; 
    v7  = uy;
    v8  = uz;

    //v0   = one/Kokkos::sqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));
    v0   = rsqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;
    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;
    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    inbnds = v3<=one &&  v4<=one &&  v5<=one && 
            -v3<=one && -v4<=one && -v5<=one;

#if KOKKOS_VERSION_MAJOR == 5
    v3 = KokkosSIMD::condition(!inbnds, dx, v3);
    v4 = KokkosSIMD::condition(!inbnds, dy, v4);
    v5 = KokkosSIMD::condition(!inbnds, dz, v5);
#else
    KokkosSIMD::where(!inbnds, v3) = dx;
    KokkosSIMD::where(!inbnds, v4) = dy;
    KokkosSIMD::where(!inbnds, v5) = dz;
#endif

    // Store updated particles
    store_particles(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
                    v3, v4, v5, ii, v6, v7, v8, q);

    q *= qsp;
#if KOKKOS_VERSION_MAJOR == 5
    q = KokkosSIMD::condition(!inbnds, simd_float_t(0.0f), q);
#else
    KokkosSIMD::where(!inbnds, q) = 0.0;
#endif

    dx = v0;
    dy = v1;
    dz = v2;
    v15 = q*ux*uy*uz*one_third;

#   define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                            \
    v12  = q*u##X;   /* v2 = q ux                            */        \
    v1  = v12*d##Y;  /* v1 = q ux dy                         */        \
    v0  = v12-v1;    /* v0 = q ux (1-dy)                     */        \
    v1 += v12;       /* v1 = q ux (1+dy)                     */        \
    v12  = one+d##Z; /* v12 = 1+dz                            */       \
    v2  = v0*v12;    /* v2 = q ux (1-dy)(1+dz)               */        \
    v3  = v1*v12;    /* v3 = q ux (1+dy)(1+dz)               */        \
    v12  = one-d##Z; /* v12 = 1-dz                            */       \
    v0 *= v12;       /* v0 = q ux (1-dy)(1-dz)               */        \
    v1 *= v12;       /* v1 = q ux (1+dy)(1-dz)               */        \
    v0 += v15;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
    v1 -= v15;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
    v2 -= v15;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
    v3 += v15;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

    // Accumulate current density
    ACCUMULATE_J( x,y,z, v0,v1,v2,v3 );
    v0 *= cx; v1 *= cx; v2  *= cx; v3  *= cx;

    ACCUMULATE_J( y,z,x, v4,v5,v6,v7 );
    v4 *= cy; v5 *= cy; v6  *= cy; v7  *= cy;

    ACCUMULATE_J( z,x,y, v8,v9,v10,v11 );
    v8 *= cz; v9 *= cz; v10 *= cz; v11 *= cz;

#ifdef ENABLE_SIMD_ACCUMULATORS
    // Store current contributions
    accumulate_simd<ACCUMULATOR_LAYOUT>(current, active_lanes, mask,
                    ii,  nx,  ny,  nz,
                    v0,  v1,  v2,  v3,
                    v4,  v5,  v6,  v7,
                    v8,  v9,  v10, v11);
#else
#ifdef VPIC_ENABLE_ACCUMULATORS 
    for(size_t idx=0; idx<active_lanes; idx++) {
      current((int)ii[idx], 0)  += v0[idx];
      current((int)ii[idx], 1)  += v1[idx];
      current((int)ii[idx], 2)  += v2[idx];
      current((int)ii[idx], 3)  += v3[idx];
      current((int)ii[idx], 4)  += v4[idx];
      current((int)ii[idx], 5)  += v5[idx];
      current((int)ii[idx], 6)  += v6[idx];
      current((int)ii[idx], 7)  += v7[idx];
      current((int)ii[idx], 8)  += v8[idx];
      current((int)ii[idx], 9)  += v9[idx];
      current((int)ii[idx], 10) += v10[idx];
      current((int)ii[idx], 11) += v11[idx];
    }
#else
    for(size_t idx=0; idx<active_lanes; idx++) {
      int iii = ii[idx];
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);
      
      current(ii[idx],                      field_var::jfx) += v0[idx];
      current(VOXEL(xi,yi+1,zi,nx,ny,nz),   field_var::jfx) += v1[idx];
      current(VOXEL(xi,yi,zi+1,nx,ny,nz),   field_var::jfx) += v2[idx];
      current(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += v3[idx];
      
      current(ii[idx],                      field_var::jfy) += v4[idx];
      current(VOXEL(xi,yi,zi+1,nx,ny,nz),   field_var::jfy) += v5[idx];
      current(VOXEL(xi+1,yi,zi,nx,ny,nz),   field_var::jfy) += v6[idx];
      current(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += v7[idx];
      
      current(ii[idx],                      field_var::jfz) += v8[idx];
      current(VOXEL(xi+1,yi,zi,nx,ny,nz),   field_var::jfz) += v9[idx];
      current(VOXEL(xi,yi+1,zi,nx,ny,nz),   field_var::jfz) += v10[idx];
      current(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += v11[idx];
    }
#endif
#endif

#   undef ACCUMULATE_J

//    if(KokkosSIMD::any_of(!inbnds)) {
//      simd_float_t dispx = ux;
//      simd_float_t dispy = uy;
//      simd_float_t dispz = uz;
//      simd_int32_t pm_i([p_index,np, num_lanes] (std::size_t lane) { return p_index*num_lanes + int(lane); });
//      simd_float_mask_t outbnds = !inbnds && mask;
//      simd_float_mask_t moved = move_p_kokkos_simd_a(k_particles, k_particles_i, 
//                                                     dispx, dispy, dispz, pm_i, 
//                                                     outbnds, active_lanes,
//                                                     current, g, k_neighbors, 
//                                                     rangel, rangeh, qsp, 
//                                                     cx[0], cy[0], cz[0], 
//                                                     nx, ny, nz);
//
//      if(KokkosSIMD::any_of(moved)) {
//        for(size_t idx=0; idx<active_lanes; idx++) {
//          if(moved[idx]) {
//            if( k_nm(0) < max_nm )
//            {
//                const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//                if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//                k_particle_movers(nm, particle_mover_var::dispx) = dispx[idx]; //local_pm->dispx; //mover_list(mover_idx, particle_mover_var::dispx);
//                k_particle_movers(nm, particle_mover_var::dispy) = dispy[idx]; //local_pm->dispy; //mover_list(mover_idx, particle_mover_var::dispy);
//                k_particle_movers(nm, particle_mover_var::dispz) = dispz[idx]; //local_pm->dispz; //mover_list(mover_idx, particle_mover_var::dispz);
//                k_particle_movers_i(nm)                          = pm_i[idx];  //local_pm->i; // mover_list_i(mover_idx);
//
//                // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//                k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
//                k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
//                k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
//                k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
//                k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
//                k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
//                k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
//                k_particle_i_copy(nm) = k_particles_i(p_index*SIMD_LEN+idx);
//            }
//          }
//        }
//      }
//    }

    for(size_t idx=0; idx<active_lanes; idx++) {
      if(!inbnds[idx]) {
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux[idx];
        local_pm->dispy = uy[idx];
        local_pm->dispz = uz[idx];
        local_pm->i     = p_index*num_lanes+idx;

        if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) ) [[unlikely]]
//        if( move_p_kokkos_simd_b( k_particles, k_particles_i, local_pm, // Unlikely
//                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) )
        {
          if( k_nm(0) < max_nm )
          {
              const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx; 
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy; 
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz; 
              k_particle_movers_i(nm)   = local_pm->i; 

              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
              k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
              k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
              k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
              k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
              k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
              k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
              k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
              k_particle_i_copy(nm)                 = k_particles_i(p_index*SIMD_LEN+idx);
          }
        }
      }
    }
//    token.release(threadID);
  });

  KOKKOS_TOC(advance_p, 1);
  KOKKOS_TIC();

  Kokkos::Profiling::popRegion();

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::MDRangePolicy<size_t, Kokkos::Rank<2>> reduce_policy({0LU,0LU}, {accumulator.extent(1), 12LU});
  Kokkos::parallel_for("reduce accumulator", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, const uint32_t j) {
    for(int tid=1; tid<accumulator.extent(0); tid++) {
      accumulator(0, i, j) += accumulator(tid, i,  j);
    }
  });
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    int f0  = VOXEL(x,   y,   z, nx, ny, nz);
    int a0  = VOXEL(x,   y,   z, nx, ny, nz);
    int ax  = VOXEL(x-1, y,   z, nx, ny, nz);
    int ay  = VOXEL(x,   y-1, z, nx, ny, nz);
    int az  = VOXEL(x,   y,   z-1, nx, ny, nz);
    int ayz = VOXEL(x,   y-1, z-1, nx, ny, nz);
    int azx = VOXEL(x-1, y,   z-1, nx, ny, nz);
    int axy = VOXEL(x-1, y-1, z, nx, ny, nz);
    k_field(f0, field_var::jfx) += ( accumulator(0, a0,  0) +
                                     accumulator(0, ay,  1) +
                                     accumulator(0, az,  2) +
                                     accumulator(0, ayz, 3) );
    k_field(f0, field_var::jfy) += ( accumulator(0, a0,  4) +
                                     accumulator(0, az,  5) +
                                     accumulator(0, ax,  6) +
                                     accumulator(0, azx, 7) );
    k_field(f0, field_var::jfz) += ( accumulator(0, a0,  8) +
                                     accumulator(0, ax,  9) +
                                     accumulator(0, ay,  10) +
                                     accumulator(0, axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif
  KOKKOS_TOC(reduce_accumulators, 1);
}
#endif // ADVANCE_P_SIMD_COMPUTE

#ifdef ADVANCE_P_SIMD
void
advance_p_kokkos_simd(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sv_t k_f_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
        const float _qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{
  constexpr auto num_lanes = SIMD_LEN;

  const simd_float_t one = 1.0f;
  const simd_float_t one_third = 1.0f/3.0f;
  const simd_float_t two_fifteenths = 2.0f/15.0f;

  k_field_t k_field = fa->k_f_d;

  const simd_float_t cx = 0.25 * g->rdy * g->rdz / g->dt;
  const simd_float_t cy = 0.25 * g->rdz * g->rdx / g->dt;
  const simd_float_t cz = 0.25 * g->rdx * g->rdy / g->dt;
  const simd_float_t qdt_2mc(_qdt_2mc);

  const auto rangel = g->rangel;
  const auto rangeh = g->rangeh;

  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT+PAD_SIZE_INTERPOLATOR], k_interpolator_t::array_layout> const_interp = k_interp;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::View<float***, Kokkos::LayoutRight> accumulator("Accumulator", Kokkos::num_threads(), k_field.extent(0), 16);
  Kokkos::deep_copy(accumulator, 0);
#else
  k_field_sv_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);;
#endif

  Kokkos::Profiling::pushRegion("advance_p_internal");
  KOKKOS_TIC();

// Setting up work distribution settings
  const int num_threads = 1;//Kokkos::num_threads();
  const int chunk_size = SIMD_LEN*num_threads;
  int num_chunks = np/chunk_size;
  if(num_chunks*chunk_size < np)
    num_chunks += 1;
  auto policy = Kokkos::TeamPolicy<>(num_chunks, num_threads);
  Kokkos::parallel_for("advance_p", policy, 
  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    int leagueID = team_member.league_rank();
    int threadID = team_member.team_rank();
    size_t p_index = team_member.league_rank()*team_member.team_size() + team_member.team_rank();

//  Kokkos::Experimental::UniqueToken<> token;
//  auto num_chunks = np/SIMD_LEN;
//  if(num_chunks*SIMD_LEN < np)
//    num_chunks += 1;
//  auto policy = Kokkos::RangePolicy<size_t>(0,num_chunks);
//  Kokkos::parallel_for("advance_p", policy, KOKKOS_LAMBDA (const size_t p_index) {
//    int threadID = token.acquire();

#if defined( VPIC_ENABLE_ACCUMULATORS )
    auto current = Kokkos::subview(accumulator, threadID, Kokkos::ALL, Kokkos::ALL);
#else
    auto current = current_sv.access();
#endif

    simd_float_t v0, v1, v2, v3, v4, v5;
    simd_float_t v6, v7, v8, v9, v10, v11;
    simd_float_t v12, v13, v14, v15;

    simd_float_t dx,dy,dz,ux,uy,uz,q;
    simd_int32_t ii;
    simd_float_mask_t inbnds;

//    simd_float_t fex, fdexdy, fdexdz, fd2exdydz;
//    simd_float_t fey, fdeydz, fdeydx, fd2eydzdx;
//    simd_float_t fez, fdezdx, fdezdy, fd2ezdxdy;
//    simd_float_t fcbx , fcby, fcbz;     
//    simd_float_t fdcbxdx, fdcbydy, fdcbzdz;

    simd_float_mask_t mask([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    simd_int32_mask_t mask_int([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    size_t active_lanes = num_lanes;
    if(p_index*num_lanes+active_lanes >= np)
      active_lanes = np - p_index*num_lanes;

    // Load particles
    load_particles_simd(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
                        dx, dy, dz, ii, ux, uy, uz, q);

    // Load interpolators
    simd_float_t hax, hay, haz;
    simd_float_t cbx, cby, cbz;
    load_interpolators(const_interp, active_lanes, mask, ii,
                       dx, dy, dz,
                       hax, hay, haz,
                       cbx, cby, cbz,
                       qdt_2mc);
    //load_interpolators_basic(const_interp, active_lanes, mask, ii,
    //                         dx, dy, dz,
    //                         hax, hay, haz,
    //                         cbx, cby, cbz,
    //                         qdt_2mc);

//    load_interpolators(const_interp, active_lanes, mask, ii,
//                       fex, fdexdy, fdexdz, fd2exdydz,
//                       fey, fdeydz, fdeydx, fd2eydzdx,
//                       fez, fdezdx, fdezdy, fd2ezdxdy,
//                       fcbx, fdcbxdx, fcby, fdcbydy, fcbz, fdcbzdz);
//
//    // Interpolate E
//    simd_float_t hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
//    simd_float_t hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
//    simd_float_t haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
//    // Interpolate B
//    simd_float_t cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
//    simd_float_t cby  = Kokkos::fma(dy, fdcbydy, fcby);
//    simd_float_t cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    //v0  = qdt_2mc/Kokkos::sqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    v0  = qdt_2mc * rsqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    // Boris - scalars
    v1  = Kokkos::fma(cbx, cbx, Kokkos::fma(cby, cby, cbz*cbz));
    v2  = (v0*v0)*v1;
    v3  = v0*Kokkos::fma(v2, Kokkos::fma(v2, two_fifteenths, one_third), one);
    v4  = v3/Kokkos::fma(v1, (v3*v3), one);
    v4 += v4;
    // Boris - uprime
    v0  = Kokkos::fma( Kokkos::fma( uy, cbz, -uz*cby ), v3, ux);
    v1  = Kokkos::fma( Kokkos::fma( uz, cbx, -ux*cbz ), v3, uy);
    v2  = Kokkos::fma( Kokkos::fma( ux, cby, -uy*cbx ), v3, uz);
    // Boris - rotation
    ux  = Kokkos::fma( Kokkos::fma( v1, cbz, -v2*cby ), v4, ux);
    uy  = Kokkos::fma( Kokkos::fma( v2, cbx, -v0*cbz ), v4, uy);
    uz  = Kokkos::fma( Kokkos::fma( v0, cby, -v1*cbx ), v4, uz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    // Store momentum in registers for later storage
    v6  = ux; 
    v7  = uy;
    v8  = uz;

    //v0   = one/Kokkos::sqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));
    v0   = rsqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;
    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;
    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    inbnds = v3<=one &&  v4<=one &&  v5<=one && 
            -v3<=one && -v4<=one && -v5<=one;

#if KOKKOS_VERSION_MAJOR == 5
    v3 = KokkosSIMD::condition(!inbnds, dx, v3);
    v4 = KokkosSIMD::condition(!inbnds, dy, v4);
    v5 = KokkosSIMD::condition(!inbnds, dz, v5);
#else
    KokkosSIMD::where(!inbnds, v3) = dx;
    KokkosSIMD::where(!inbnds, v4) = dy;
    KokkosSIMD::where(!inbnds, v5) = dz;
#endif

    // Store updated particles
    //store_particles(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
    //                     v3, v4, v5, ii, v6, v7, v8, q);
    store_particles_simd(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
                         v3, v4, v5, ii, v6, v7, v8, q);

    q *= qsp;
#if KOKKOS_VERSION_MAJOR == 5
    q = KokkosSIMD::condition(!inbnds, simd_float_t(0.0f), q);
#else
    KokkosSIMD::where(!inbnds, q) = 0.0;
#endif

    dx = v0;
    dy = v1;
    dz = v2;
    v15 = q*ux*uy*uz*one_third;

#   define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                            \
    v12  = q*u##X;   /* v2 = q ux                            */        \
    v1  = v12*d##Y;  /* v1 = q ux dy                         */        \
    v0  = v12-v1;    /* v0 = q ux (1-dy)                     */        \
    v1 += v12;       /* v1 = q ux (1+dy)                     */        \
    v12  = one+d##Z; /* v12 = 1+dz                            */       \
    v2  = v0*v12;    /* v2 = q ux (1-dy)(1+dz)               */        \
    v3  = v1*v12;    /* v3 = q ux (1+dy)(1+dz)               */        \
    v12  = one-d##Z; /* v12 = 1-dz                            */       \
    v0 *= v12;       /* v0 = q ux (1-dy)(1-dz)               */        \
    v1 *= v12;       /* v1 = q ux (1+dy)(1-dz)               */        \
    v0 += v15;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
    v1 -= v15;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
    v2 -= v15;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
    v3 += v15;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

    // Accumulate current density
    ACCUMULATE_J( x,y,z, v0,v1,v2,v3 );
    v0 *= cx; v1 *= cx; v2  *= cx; v3  *= cx;

    ACCUMULATE_J( y,z,x, v4,v5,v6,v7 );
    v4 *= cy; v5 *= cy; v6  *= cy; v7  *= cy;

    ACCUMULATE_J( z,x,y, v8,v9,v10,v11 );
    v8 *= cz; v9 *= cz; v10 *= cz; v11 *= cz;

#   undef ACCUMULATE_J
    // Store current contributions
    accumulate_simd<Kokkos::LayoutRight>(current, active_lanes, mask,
                    ii,  nx,  ny,  nz,
                    v0,  v1,  v2,  v3,
                    v4,  v5,  v6,  v7,
                    v8,  v9,  v10, v11);

//    if(KokkosSIMD::any_of(!inbnds)) {
//      simd_float_t dispx = ux;
//      simd_float_t dispy = uy;
//      simd_float_t dispz = uz;
//      simd_int32_t pm_i([p_index,np, num_lanes] (std::size_t lane) { return p_index*num_lanes + int(lane); });
//      simd_float_mask_t outbnds = !inbnds && mask;
//      simd_float_mask_t moved = move_p_kokkos_simd_a(k_particles, k_particles_i, 
//                                                     dispx, dispy, dispz, pm_i, 
//                                                     outbnds, active_lanes,
//                                                     current, g, k_neighbors, 
//                                                     rangel, rangeh, qsp, 
//                                                     cx[0], cy[0], cz[0], 
//                                                     nx, ny, nz);
//
//      if(KokkosSIMD::any_of(moved)) {
//        for(size_t idx=0; idx<active_lanes; idx++) {
//          if(moved[idx]) {
//            if( k_nm(0) < max_nm )
//            {
//                const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//                if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//                k_particle_movers(nm, particle_mover_var::dispx) = dispx[idx]; //local_pm->dispx; //mover_list(mover_idx, particle_mover_var::dispx);
//                k_particle_movers(nm, particle_mover_var::dispy) = dispy[idx]; //local_pm->dispy; //mover_list(mover_idx, particle_mover_var::dispy);
//                k_particle_movers(nm, particle_mover_var::dispz) = dispz[idx]; //local_pm->dispz; //mover_list(mover_idx, particle_mover_var::dispz);
//                k_particle_movers_i(nm)                          = pm_i[idx];  //local_pm->i; // mover_list_i(mover_idx);
//
//                // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//                k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
//                k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
//                k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
//                k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
//                k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
//                k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
//                k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
//                k_particle_i_copy(nm) = k_particles_i(p_index*SIMD_LEN+idx);
//            }
//          }
//        }
//      }
//    }

    for(size_t idx=0; idx<active_lanes; idx++) {
      if(!inbnds[idx]) {
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux[idx];
        local_pm->dispy = uy[idx];
        local_pm->dispz = uz[idx];
        local_pm->i     = p_index*num_lanes+idx;

        if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) ) [[unlikely]]
//        if( move_p_kokkos_simd_b( k_particles, k_particles_i, local_pm, // Unlikely
//                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) )
        {
          if( k_nm(0) < max_nm )
          {
              const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx; 
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy; 
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz; 
              k_particle_movers_i(nm)   = local_pm->i; 

              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
              k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
              k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
              k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
              k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
              k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
              k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
              k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
              k_particle_i_copy(nm)                 = k_particles_i(p_index*SIMD_LEN+idx);
          }
        }
      }
    }
//    token.release(threadID);
  });

  KOKKOS_TOC(advance_p, 1);
  KOKKOS_TIC();

  Kokkos::Profiling::popRegion();

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::MDRangePolicy<size_t, Kokkos::Rank<2>> reduce_policy({0LU,0LU}, {accumulator.extent(1), 12LU});
  Kokkos::parallel_for("reduce accumulator", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, const uint32_t j) {
    for(int tid=1; tid<accumulator.extent(0); tid++) {
      accumulator(0, i, j) += accumulator(tid, i,  j);
    }
  });
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    int f0  = VOXEL(x,   y,   z, nx, ny, nz);
    int a0  = VOXEL(x,   y,   z, nx, ny, nz);
    int ax  = VOXEL(x-1, y,   z, nx, ny, nz);
    int ay  = VOXEL(x,   y-1, z, nx, ny, nz);
    int az  = VOXEL(x,   y,   z-1, nx, ny, nz);
    int ayz = VOXEL(x,   y-1, z-1, nx, ny, nz);
    int azx = VOXEL(x-1, y,   z-1, nx, ny, nz);
    int axy = VOXEL(x-1, y-1, z, nx, ny, nz);
    k_field(f0, field_var::jfx) += ( accumulator(0, a0,  0) +
                                     accumulator(0, ay,  1) +
                                     accumulator(0, az,  2) +
                                     accumulator(0, ayz, 3) );
    k_field(f0, field_var::jfy) += ( accumulator(0, a0,  4) +
                                     accumulator(0, az,  5) +
                                     accumulator(0, ax,  6) +
                                     accumulator(0, azx, 7) );
    k_field(f0, field_var::jfz) += ( accumulator(0, a0,  8) +
                                     accumulator(0, ax,  9) +
                                     accumulator(0, ay,  10) +
                                     accumulator(0, axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif
  KOKKOS_TOC(reduce_accumulators, 1);
}
#endif // ADVANCE_P_GPU

#ifdef ADVANCE_P_SIMD_UNION
void
advance_p_kokkos_simd_union(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sv_t k_f_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
        const float _qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{
  constexpr auto num_lanes = SIMD_LEN;

  const simd_float_t one = 1.0f;
  const simd_float_t one_third = 1.0f/3.0f;
  const simd_float_t two_fifteenths = 2.0f/15.0f;

  k_field_t k_field = fa->k_f_d;

  const simd_float_t cx = 0.25 * g->rdy * g->rdz / g->dt;
  const simd_float_t cy = 0.25 * g->rdz * g->rdx / g->dt;
  const simd_float_t cz = 0.25 * g->rdx * g->rdy / g->dt;
  const simd_float_t qdt_2mc(_qdt_2mc);

  const int64_t rangel = g->rangel;
  const int64_t rangeh = g->rangeh;

  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT+PAD_SIZE_INTERPOLATOR], k_interpolator_t::array_layout> const_interp = k_interp;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::View<float***, ACCUMULATOR_LAYOUT> accumulator("Accumulator", Kokkos::num_threads(), k_field.extent(0), 16);
  Kokkos::deep_copy(accumulator, 0);
#else
  k_field_sv_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);;
#endif

  particles_union_t particle_data("Particle union", k_particles.extent(0));
  Kokkos::parallel_for("Copy to Union View", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
    particle_data(i, 0).f32 = k_particles(i, particle_var::dx);
    particle_data(i, 1).f32 = k_particles(i, particle_var::dy);
    particle_data(i, 2).f32 = k_particles(i, particle_var::dz);
    particle_data(i, 3).i32 = k_particles_i(i);
    particle_data(i, 4).f32 = k_particles(i, particle_var::ux);
    particle_data(i, 5).f32 = k_particles(i, particle_var::uy);
    particle_data(i, 6).f32 = k_particles(i, particle_var::uz);
    particle_data(i, 7).f32 = k_particles(i, particle_var::w );
  });
  //particles_union_copy_t particle_copy("Particle union copy", k_particle_copy.extent(0));

  Kokkos::Profiling::pushRegion("advance_p_internal");
  KOKKOS_TIC();

// Setting up work distribution settings
  const int num_threads = 1;//Kokkos::num_threads();
  const int chunk_size = SIMD_LEN*num_threads;
  int num_chunks = np/chunk_size;
  if(num_chunks*chunk_size < np)
    num_chunks += 1;
  auto policy = Kokkos::TeamPolicy<>(num_chunks, num_threads);
  Kokkos::parallel_for("advance_p", policy, 
  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    int leagueID = team_member.league_rank();
    int threadID = team_member.team_rank();
    size_t p_index = team_member.league_rank()*team_member.team_size() + team_member.team_rank();

//  Kokkos::Experimental::UniqueToken<> token;
//  auto num_chunks = np/SIMD_LEN;
//  if(num_chunks*SIMD_LEN < np)
//    num_chunks += 1;
//  auto policy = Kokkos::RangePolicy<size_t>(0,num_chunks);
//  Kokkos::parallel_for("advance_p", policy, KOKKOS_LAMBDA (const size_t p_index) {
//    int threadID = token.acquire();

#if defined( VPIC_ENABLE_ACCUMULATORS )
    auto current = Kokkos::subview(accumulator, threadID, Kokkos::ALL, Kokkos::ALL);
#else
    auto current = current_sv.access();
#endif

    simd_float_t v0, v1, v2, v3, v4, v5;
    simd_float_t v6, v7, v8, v9, v10, v11;
    simd_float_t v12, v13, v14, v15;

    simd_float_t dx,dy,dz,ux,uy,uz,q;
    simd_int32_t ii;
    simd_float_mask_t inbnds;

//    simd_float_t fex, fdexdy, fdexdz, fd2exdydz;
//    simd_float_t fey, fdeydz, fdeydx, fd2eydzdx;
//    simd_float_t fez, fdezdx, fdezdy, fd2ezdxdy;
//    simd_float_t fcbx , fcby, fcbz;     
//    simd_float_t fdcbxdx, fdcbydy, fdcbzdz;

    simd_float_mask_t mask([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    simd_int32_mask_t mask_int([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    size_t active_lanes = num_lanes;
    if(p_index*num_lanes+active_lanes >= np)
      active_lanes = np - p_index*num_lanes;

    // Load particles
    load_particles_union(particle_data, active_lanes, p_index*num_lanes, mask,
                   dx, dy, dz, ii, ux, uy, uz, q);
    //load_particles(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
    //               dx, dy, dz, ii, ux, uy, uz, q);

    // Load interpolators
    simd_float_t hax, hay, haz;
    simd_float_t cbx, cby, cbz;
    load_interpolators(const_interp, active_lanes, mask, ii,
                       dx, dy, dz,
                       hax, hay, haz,
                       cbx, cby, cbz,
                       qdt_2mc);

//    load_interpolators(const_interp, active_lanes, mask, ii,
//                       fex, fdexdy, fdexdz, fd2exdydz,
//                       fey, fdeydz, fdeydx, fd2eydzdx,
//                       fez, fdezdx, fdezdy, fd2ezdxdy,
//                       fcbx, fdcbxdx, fcby, fdcbydy, fcbz, fdcbzdz);
//
//    // Interpolate E
//    simd_float_t hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
//    simd_float_t hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
//    simd_float_t haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
//    // Interpolate B
//    simd_float_t cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
//    simd_float_t cby  = Kokkos::fma(dy, fdcbydy, fcby);
//    simd_float_t cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    //v0  = qdt_2mc/Kokkos::sqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    v0  = qdt_2mc * rsqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    // Boris - scalars
    v1  = Kokkos::fma(cbx, cbx, Kokkos::fma(cby, cby, cbz*cbz));
    v2  = (v0*v0)*v1;
    v3  = v0*Kokkos::fma(v2, Kokkos::fma(v2, two_fifteenths, one_third), one);
    v4  = v3/Kokkos::fma(v1, (v3*v3), one);
    v4 += v4;
    // Boris - uprime
    v0  = Kokkos::fma( Kokkos::fma( uy, cbz, -uz*cby ), v3, ux);
    v1  = Kokkos::fma( Kokkos::fma( uz, cbx, -ux*cbz ), v3, uy);
    v2  = Kokkos::fma( Kokkos::fma( ux, cby, -uy*cbx ), v3, uz);
    // Boris - rotation
    ux  = Kokkos::fma( Kokkos::fma( v1, cbz, -v2*cby ), v4, ux);
    uy  = Kokkos::fma( Kokkos::fma( v2, cbx, -v0*cbz ), v4, uy);
    uz  = Kokkos::fma( Kokkos::fma( v0, cby, -v1*cbx ), v4, uz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    // Store momentum in registers for later storage
    v6  = ux; 
    v7  = uy;
    v8  = uz;

    //v0   = one/Kokkos::sqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));
    v0   = rsqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;
    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;
    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    inbnds = v3<=one &&  v4<=one &&  v5<=one && 
            -v3<=one && -v4<=one && -v5<=one;

#if KOKKOS_VERSION_MAJOR == 5
    v3 = KokkosSIMD::condition(!inbnds, dx, v3);
    v4 = KokkosSIMD::condition(!inbnds, dy, v4);
    v5 = KokkosSIMD::condition(!inbnds, dz, v5);
#else
    KokkosSIMD::where(!inbnds, v3) = dx;
    KokkosSIMD::where(!inbnds, v4) = dy;
    KokkosSIMD::where(!inbnds, v5) = dz;
#endif

    // Store updated particles
    store_particles_union(particle_data, active_lanes, p_index*num_lanes, mask,
                    v3, v4, v5, ii, v6, v7, v8, q);
    //store_particles(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
    //                v3, v4, v5, ii, v6, v7, v8, q);

    q *= qsp;
#if KOKKOS_VERSION_MAJOR == 5
    q = KokkosSIMD::condition(!inbnds, simd_float_t(0.0f), q);
#else
    KokkosSIMD::where(!inbnds, q) = 0.0;
#endif

    dx = v0;
    dy = v1;
    dz = v2;
    v15 = q*ux*uy*uz*one_third;

#   define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                            \
    v12  = q*u##X;   /* v2 = q ux                            */        \
    v1  = v12*d##Y;  /* v1 = q ux dy                         */        \
    v0  = v12-v1;    /* v0 = q ux (1-dy)                     */        \
    v1 += v12;       /* v1 = q ux (1+dy)                     */        \
    v12  = one+d##Z; /* v12 = 1+dz                            */       \
    v2  = v0*v12;    /* v2 = q ux (1-dy)(1+dz)               */        \
    v3  = v1*v12;    /* v3 = q ux (1+dy)(1+dz)               */        \
    v12  = one-d##Z; /* v12 = 1-dz                            */       \
    v0 *= v12;       /* v0 = q ux (1-dy)(1-dz)               */        \
    v1 *= v12;       /* v1 = q ux (1+dy)(1-dz)               */        \
    v0 += v15;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
    v1 -= v15;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
    v2 -= v15;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
    v3 += v15;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

    // Accumulate current density
    ACCUMULATE_J( x,y,z, v0,v1,v2,v3 );
    v0 *= cx; v1 *= cx; v2  *= cx; v3  *= cx;

    ACCUMULATE_J( y,z,x, v4,v5,v6,v7 );
    v4 *= cy; v5 *= cy; v6  *= cy; v7  *= cy;

    ACCUMULATE_J( z,x,y, v8,v9,v10,v11 );
    v8 *= cz; v9 *= cz; v10 *= cz; v11 *= cz;

#   undef ACCUMULATE_J
    // Store current contributions
    accumulate_simd<ACCUMULATOR_LAYOUT>(current, active_lanes, mask,
                    ii,  nx,  ny,  nz,
                    v0,  v1,  v2,  v3,
                    v4,  v5,  v6,  v7,
                    v8,  v9,  v10, v11);

//    if(KokkosSIMD::any_of(!inbnds)) {
//      simd_float_t dispx = ux;
//      simd_float_t dispy = uy;
//      simd_float_t dispz = uz;
//      simd_int32_t pm_i([p_index,np, num_lanes] (std::size_t lane) { return p_index*num_lanes + int(lane); });
//      simd_float_mask_t outbnds = !inbnds && mask;
//      simd_float_mask_t moved = move_p_kokkos_simd_a(k_particles, k_particles_i, 
//                                                     dispx, dispy, dispz, pm_i, 
//                                                     outbnds, active_lanes,
//                                                     current, g, k_neighbors, 
//                                                     rangel, rangeh, qsp, 
//                                                     cx[0], cy[0], cz[0], 
//                                                     nx, ny, nz);
//
//      if(KokkosSIMD::any_of(moved)) {
//        for(size_t idx=0; idx<active_lanes; idx++) {
//          if(moved[idx]) {
//            if( k_nm(0) < max_nm )
//            {
//                const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//                if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//                k_particle_movers(nm, particle_mover_var::dispx) = dispx[idx]; //local_pm->dispx; //mover_list(mover_idx, particle_mover_var::dispx);
//                k_particle_movers(nm, particle_mover_var::dispy) = dispy[idx]; //local_pm->dispy; //mover_list(mover_idx, particle_mover_var::dispy);
//                k_particle_movers(nm, particle_mover_var::dispz) = dispz[idx]; //local_pm->dispz; //mover_list(mover_idx, particle_mover_var::dispz);
//                k_particle_movers_i(nm)                          = pm_i[idx];  //local_pm->i; // mover_list_i(mover_idx);
//
//                // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//                k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
//                k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
//                k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
//                k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
//                k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
//                k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
//                k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
//                k_particle_i_copy(nm) = k_particles_i(p_index*SIMD_LEN+idx);
//            }
//          }
//        }
//      }
//    }

    for(size_t idx=0; idx<active_lanes; idx++) {
      if(!inbnds[idx]) {
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux[idx];
        local_pm->dispy = uy[idx];
        local_pm->dispz = uz[idx];
        local_pm->i     = p_index*num_lanes+idx;

        if( move_p_kokkos_union( particle_data, local_pm, // Unlikely
                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) ) [[unlikely]]
//        if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
//                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) ) [[unlikely]]
//        if( move_p_kokkos_simd_b( k_particles, k_particles_i, local_pm, // Unlikely
//                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) )
        {
          if( k_nm(0) < max_nm )
          {
              const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx; 
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy; 
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz; 
              k_particle_movers_i(nm)   = local_pm->i; 

              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
              //k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
              //k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
              //k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
              //k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
              //k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
              //k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
              //k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
              //k_particle_i_copy(nm)                 = k_particles_i(p_index*SIMD_LEN+idx);

              k_particle_copy(nm, particle_var::dx) = particle_data( p_index*SIMD_LEN+idx, 0).f32;
              k_particle_copy(nm, particle_var::dy) = particle_data( p_index*SIMD_LEN+idx, 1).f32;
              k_particle_copy(nm, particle_var::dz) = particle_data( p_index*SIMD_LEN+idx, 2).f32;
              k_particle_i_copy(nm)                 = particle_data( p_index*SIMD_LEN+idx, 3).i32;
              k_particle_copy(nm, particle_var::ux) = particle_data( p_index*SIMD_LEN+idx, 4).f32;
              k_particle_copy(nm, particle_var::uy) = particle_data( p_index*SIMD_LEN+idx, 5).f32;
              k_particle_copy(nm, particle_var::uz) = particle_data( p_index*SIMD_LEN+idx, 6).f32;
              k_particle_copy(nm, particle_var::w)  = particle_data( p_index*SIMD_LEN+idx, 7).f32;
          }
        }
      }
    }
//    token.release(threadID);
  });

  KOKKOS_TOC(advance_p, 1);
  KOKKOS_TIC();

  Kokkos::Profiling::popRegion();

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::MDRangePolicy<size_t, Kokkos::Rank<2>> reduce_policy({0LU,0LU}, {accumulator.extent(1), 12LU});
  Kokkos::parallel_for("reduce accumulator", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, const uint32_t j) {
    for(size_t tid=1; tid<accumulator.extent(0); tid++) {
      accumulator(0, i, j) += accumulator(tid, i,  j);
    }
  });
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    int f0  = VOXEL(x,   y,   z, nx, ny, nz);
    int a0  = VOXEL(x,   y,   z, nx, ny, nz);
    int ax  = VOXEL(x-1, y,   z, nx, ny, nz);
    int ay  = VOXEL(x,   y-1, z, nx, ny, nz);
    int az  = VOXEL(x,   y,   z-1, nx, ny, nz);
    int ayz = VOXEL(x,   y-1, z-1, nx, ny, nz);
    int azx = VOXEL(x-1, y,   z-1, nx, ny, nz);
    int axy = VOXEL(x-1, y-1, z, nx, ny, nz);
    k_field(f0, field_var::jfx) += ( accumulator(0, a0,  0) +
                                     accumulator(0, ay,  1) +
                                     accumulator(0, az,  2) +
                                     accumulator(0, ayz, 3) );
    k_field(f0, field_var::jfy) += ( accumulator(0, a0,  4) +
                                     accumulator(0, az,  5) +
                                     accumulator(0, ax,  6) +
                                     accumulator(0, azx, 7) );
    k_field(f0, field_var::jfz) += ( accumulator(0, a0,  8) +
                                     accumulator(0, ax,  9) +
                                     accumulator(0, ay,  10) +
                                     accumulator(0, axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif
  KOKKOS_TOC(reduce_accumulators, 1);

  Kokkos::parallel_for("Copy to particle View", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
    k_particles(i, particle_var::dx) = particle_data(i, 0).f32;
    k_particles(i, particle_var::dy) = particle_data(i, 1).f32;
    k_particles(i, particle_var::dz) = particle_data(i, 2).f32;
    k_particles_i(i)                 = particle_data(i, 3).i32;
    k_particles(i, particle_var::ux) = particle_data(i, 4).f32;
    k_particles(i, particle_var::uy) = particle_data(i, 5).f32;
    k_particles(i, particle_var::uz) = particle_data(i, 6).f32;
    k_particles(i, particle_var::w ) = particle_data(i, 7).f32;
  });

//  Kokkos::parallel_for("Copy dup particles to particle View", Kokkos::RangePolicy<>(0, k_particle_copy.extent(0)), KOKKOS_LAMBDA(const int i) {
//    k_particle_copy(i, particle_var::dx) = particle_copy(i, 0).f32;
//    k_particle_copy(i, particle_var::dy) = particle_copy(i, 1).f32;
//    k_particle_copy(i, particle_var::dz) = particle_copy(i, 2).f32;
//    k_particle_i_copy(i)                 = particle_copy(i, 3).i32;
//    k_particle_copy(i, particle_var::ux) = particle_copy(i, 4).f32;
//    k_particle_copy(i, particle_var::uy) = particle_copy(i, 5).f32;
//    k_particle_copy(i, particle_var::uz) = particle_copy(i, 6).f32;
//    k_particle_copy(i, particle_var::w ) = particle_copy(i, 7).f32;
//  });
}
#endif // ADVANCE_P_SIMD_UNION

#ifdef ADVANCE_P_SIMD_CABANA
void
advance_p_kokkos_simd_cabana(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sv_t k_f_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
        const float _qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{
  constexpr auto num_lanes = SIMD_LEN;

  const simd_float_t one = 1.;
  const simd_float_t one_third = 1./3.;
  const simd_float_t two_fifteenths = 2./15.;

  k_field_t k_field = fa->k_f_d;

  const simd_float_t cx = 0.25 * g->rdy * g->rdz / g->dt;
  const simd_float_t cy = 0.25 * g->rdz * g->rdx / g->dt;
  const simd_float_t cz = 0.25 * g->rdx * g->rdy / g->dt;
  const simd_float_t qdt_2mc(_qdt_2mc);

  const auto rangel = g->rangel;
  const auto rangeh = g->rangeh;

  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT+PAD_SIZE_INTERPOLATOR], k_interpolator_t::array_layout> const_interp = k_interp;

  size_t np_align = k_particles.extent(0) % 16 == 0 ? k_particles.extent(0) : 16*((k_particles.extent(0)/16)+1);
  particle_aosoa_t part_aosoa("AoSoA for particle data", np_align);
  auto vec_length = part_aosoa.vector_length;
  auto pos_slice = Cabana::slice<0>(part_aosoa, "Position slice");
  auto cel_slice = Cabana::slice<1>(part_aosoa, "Cell idx slice");
  auto mom_slice = Cabana::slice<2>(part_aosoa, "Momentum slice");
  auto wgt_slice = Cabana::slice<3>(part_aosoa, "Weight slice");
  Kokkos::parallel_for(k_particles.extent(0), KOKKOS_LAMBDA(const int i) {
    pos_slice(i, 0) = k_particles(i, particle_var::dx);
    pos_slice(i, 1) = k_particles(i, particle_var::dy);
    pos_slice(i, 2) = k_particles(i, particle_var::dz);
    cel_slice(i)    = k_particles_i(i);
    mom_slice(i, 0) = k_particles(i, particle_var::ux);
    mom_slice(i, 1) = k_particles(i, particle_var::uy);
    mom_slice(i, 2) = k_particles(i, particle_var::uz);
    wgt_slice(i)    = k_particles(i, particle_var::w );
  });

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::View<float***, ACCUMULATOR_LAYOUT> accumulator("Accumulator", Kokkos::num_threads(), k_field.extent(0), 16);
  Kokkos::deep_copy(accumulator, 0);
#else
  k_field_sv_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);;
#endif

  Kokkos::Profiling::pushRegion("advance_p_internal");
  KOKKOS_TIC();

// Setting up work distribution settings
  const int num_threads = Kokkos::num_threads();
  const int chunk_size = SIMD_LEN*num_threads;
  int num_chunks = np/chunk_size;
  if(num_chunks*chunk_size < np)
    num_chunks += 1;
  auto policy = Kokkos::TeamPolicy<>(num_chunks, num_threads);
  Kokkos::parallel_for("advance_p", policy, 
  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    int leagueID = team_member.league_rank();
    int threadID = team_member.team_rank();
    size_t p_index = team_member.league_rank()*team_member.team_size() + team_member.team_rank();

//  Kokkos::Experimental::UniqueToken<> token;
//  auto num_chunks = np/SIMD_LEN;
//  if(num_chunks*SIMD_LEN < np)
//    num_chunks += 1;
//  auto policy = Kokkos::RangePolicy<size_t>(0,num_chunks);
//  Kokkos::parallel_for("advance_p", policy, KOKKOS_LAMBDA (const size_t p_index) {
//    int threadID = token.acquire();

#if defined( VPIC_ENABLE_ACCUMULATORS )
    auto current = Kokkos::subview(accumulator, threadID, Kokkos::ALL, Kokkos::ALL);
#else
    auto current = current_sv.access();
#endif

    simd_float_t v0, v1, v2, v3, v4, v5;
    simd_float_t v6, v7, v8, v9, v10, v11;
    simd_float_t v12, v13, v14, v15;

    simd_float_t dx,dy,dz,ux,uy,uz,q;
    simd_int32_t ii;
    simd_float_mask_t inbnds;

//    simd_float_t fex, fdexdy, fdexdz, fd2exdydz;
//    simd_float_t fey, fdeydz, fdeydx, fd2eydzdx;
//    simd_float_t fez, fdezdx, fdezdy, fd2ezdxdy;
//    simd_float_t fcbx , fcby, fcbz;     
//    simd_float_t fdcbxdx, fdcbydy, fdcbzdz;

    simd_float_mask_t mask([p_index,np] (std::size_t lane) { return p_index*num_lanes + int(lane) < np; });
    size_t active_lanes = num_lanes;
    if(p_index*num_lanes+active_lanes >= np)
      active_lanes = np - p_index*num_lanes;

    // Load particles
    //load_particles(k_particles, k_particles_i, active_lanes, p_index*num_lanes, mask,
    //               dx, dy, dz, ii, ux, uy, uz, q);
    load_particles_aosoa(pos_slice, cel_slice, mom_slice, wgt_slice, 
                         active_lanes, p_index, mask,
                         dx, dy, dz, ii, ux, uy, uz, q);

    // Load interpolators
    simd_float_t hax, hay, haz;
    simd_float_t cbx, cby, cbz;
    load_interpolators(const_interp, active_lanes, mask, ii,
                       dx, dy, dz,
                       hax, hay, haz,
                       cbx, cby, cbz,
                       qdt_2mc);

//    load_interpolators(const_interp, active_lanes, mask, ii,
//                       fex, fdexdy, fdexdz, fd2exdydz,
//                       fey, fdeydz, fdeydx, fd2eydzdx,
//                       fez, fdezdx, fdezdy, fd2ezdxdy,
//                       fcbx, fdcbxdx, fcby, fdcbydy, fcbz, fdcbzdz);
//
//    // Interpolate E
//    simd_float_t hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
//    simd_float_t hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
//    simd_float_t haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
//    // Interpolate B
//    simd_float_t cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
//    simd_float_t cby  = Kokkos::fma(dy, fdcbydy, fcby);
//    simd_float_t cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    //v0  = qdt_2mc/Kokkos::sqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    v0  = qdt_2mc * rsqrt(one + Kokkos::fma(ux, ux, Kokkos::fma(uy, uy, uz*uz)));
    // Boris - scalars
    v1  = Kokkos::fma(cbx, cbx, Kokkos::fma(cby, cby, cbz*cbz));
    v2  = (v0*v0)*v1;
    v3  = v0*Kokkos::fma(v2, Kokkos::fma(v2, two_fifteenths, one_third), one);
    v4  = v3/Kokkos::fma(v1, (v3*v3), one);
    v4 += v4;
    // Boris - uprime
    v0  = Kokkos::fma( Kokkos::fma( uy, cbz, -uz*cby ), v3, ux);
    v1  = Kokkos::fma( Kokkos::fma( uz, cbx, -ux*cbz ), v3, uy);
    v2  = Kokkos::fma( Kokkos::fma( ux, cby, -uy*cbx ), v3, uz);
    // Boris - rotation
    ux  = Kokkos::fma( Kokkos::fma( v1, cbz, -v2*cby ), v4, ux);
    uy  = Kokkos::fma( Kokkos::fma( v2, cbx, -v0*cbz ), v4, uy);
    uz  = Kokkos::fma( Kokkos::fma( v0, cby, -v1*cbx ), v4, uz);
    // Half advance E
    ux += hax;
    uy += hay;
    uz += haz;
    // Store momentum in registers for later storage
    v6  = ux; 
    v7  = uy;
    v8  = uz;

    //v0   = one/Kokkos::sqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));
    v0   = rsqrt(one + Kokkos::fma(ux, ux , Kokkos::fma(uy, uy, uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;
    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;
    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    inbnds = v3<=one &&  v4<=one &&  v5<=one && 
            -v3<=one && -v4<=one && -v5<=one;

#if KOKKOS_VERSION_MAJOR == 5
    v3 = KokkosSIMD::condition(!inbnds, dx, v3);
    v4 = KokkosSIMD::condition(!inbnds, dy, v4);
    v5 = KokkosSIMD::condition(!inbnds, dz, v5);
#else
    KokkosSIMD::where(!inbnds, v3) = dx;
    KokkosSIMD::where(!inbnds, v4) = dy;
    KokkosSIMD::where(!inbnds, v5) = dz;
#endif

    // Store updated particles
    store_particles_aosoa(pos_slice, cel_slice, mom_slice, wgt_slice, 
                          active_lanes, p_index, mask,
                          v3, v4, v5, ii, v6, v7, v8, q);
    //store_particles(k_particles, k_particles_i, 
    //                active_lanes, p_index*num_lanes, mask,
    //                v3, v4, v5, ii, v6, v7, v8, q);

    q *= qsp;
#if KOKKOS_VERSION_MAJOR == 5
    q = KokkosSIMD::condition(!inbnds, 0.0f, q);
#else
    KokkosSIMD::where(!inbnds, q) = 0.0;
#endif

    dx = v0;
    dy = v1;
    dz = v2;
    v15 = q*ux*uy*uz*one_third;

#   define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                            \
    v12  = q*u##X;   /* v2 = q ux                            */        \
    v1  = v12*d##Y;  /* v1 = q ux dy                         */        \
    v0  = v12-v1;    /* v0 = q ux (1-dy)                     */        \
    v1 += v12;       /* v1 = q ux (1+dy)                     */        \
    v12  = one+d##Z; /* v12 = 1+dz                            */       \
    v2  = v0*v12;    /* v2 = q ux (1-dy)(1+dz)               */        \
    v3  = v1*v12;    /* v3 = q ux (1+dy)(1+dz)               */        \
    v12  = one-d##Z; /* v12 = 1-dz                            */       \
    v0 *= v12;       /* v0 = q ux (1-dy)(1-dz)               */        \
    v1 *= v12;       /* v1 = q ux (1+dy)(1-dz)               */        \
    v0 += v15;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
    v1 -= v15;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
    v2 -= v15;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
    v3 += v15;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

    // Accumulate current density
    ACCUMULATE_J( x,y,z, v0,v1,v2,v3 );
    v0 *= cx; v1 *= cx; v2  *= cx; v3  *= cx;

    ACCUMULATE_J( y,z,x, v4,v5,v6,v7 );
    v4 *= cy; v5 *= cy; v6  *= cy; v7  *= cy;

    ACCUMULATE_J( z,x,y, v8,v9,v10,v11 );
    v8 *= cz; v9 *= cz; v10 *= cz; v11 *= cz;

#   undef ACCUMULATE_J
    // Store current contributions
    accumulate_simd<ACCUMULATOR_LAYOUT>(current, active_lanes, mask,
                    ii,  nx,  ny,  nz,
                    v0,  v1,  v2,  v3,
                    v4,  v5,  v6,  v7,
                    v8,  v9,  v10, v11);

//    if(KokkosSIMD::any_of(!inbnds)) {
//      simd_float_t dispx = ux;
//      simd_float_t dispy = uy;
//      simd_float_t dispz = uz;
//      simd_int32_t pm_i([p_index,np, num_lanes] (std::size_t lane) { return p_index*num_lanes + int(lane); });
//      simd_float_mask_t outbnds = !inbnds && mask;
//      simd_float_mask_t moved = move_p_kokkos_simd_a(k_particles, k_particles_i, 
//                                                     dispx, dispy, dispz, pm_i, 
//                                                     outbnds, active_lanes,
//                                                     current, g, k_neighbors, 
//                                                     rangel, rangeh, qsp, 
//                                                     cx[0], cy[0], cz[0], 
//                                                     nx, ny, nz);
//
//      if(KokkosSIMD::any_of(moved)) {
//        for(size_t idx=0; idx<active_lanes; idx++) {
//          if(moved[idx]) {
//            if( k_nm(0) < max_nm )
//            {
//                const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//                if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//                k_particle_movers(nm, particle_mover_var::dispx) = dispx[idx]; //local_pm->dispx; //mover_list(mover_idx, particle_mover_var::dispx);
//                k_particle_movers(nm, particle_mover_var::dispy) = dispy[idx]; //local_pm->dispy; //mover_list(mover_idx, particle_mover_var::dispy);
//                k_particle_movers(nm, particle_mover_var::dispz) = dispz[idx]; //local_pm->dispz; //mover_list(mover_idx, particle_mover_var::dispz);
//                k_particle_movers_i(nm)                          = pm_i[idx];  //local_pm->i; // mover_list_i(mover_idx);
//
//                // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//                k_particle_copy(nm, particle_var::dx) = k_particles( p_index*SIMD_LEN+idx, particle_var::dx);
//                k_particle_copy(nm, particle_var::dy) = k_particles( p_index*SIMD_LEN+idx, particle_var::dy);
//                k_particle_copy(nm, particle_var::dz) = k_particles( p_index*SIMD_LEN+idx, particle_var::dz);
//                k_particle_copy(nm, particle_var::ux) = k_particles( p_index*SIMD_LEN+idx, particle_var::ux);
//                k_particle_copy(nm, particle_var::uy) = k_particles( p_index*SIMD_LEN+idx, particle_var::uy);
//                k_particle_copy(nm, particle_var::uz) = k_particles( p_index*SIMD_LEN+idx, particle_var::uz);
//                k_particle_copy(nm, particle_var::w)  = k_particles( p_index*SIMD_LEN+idx, particle_var::w );
//                k_particle_i_copy(nm) = k_particles_i(p_index*SIMD_LEN+idx);
//            }
//          }
//        }
//      }
//    }

    for(size_t idx=0; idx<active_lanes; idx++) {
      if(!inbnds[idx]) {
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux[idx];
        local_pm->dispy = uy[idx];
        local_pm->dispz = uz[idx];
        local_pm->i     = p_index*num_lanes+idx;

        if( move_p_kokkos_cabana( pos_slice, cel_slice, mom_slice, wgt_slice, local_pm, // Unlikely
                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) ) [[unlikely]]
//        if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
//                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) ) [[unlikely]]
//        if( move_p_kokkos_simd_b( k_particles, k_particles_i, local_pm, // Unlikely
//                           current, g, k_neighbors, rangel, rangeh, qsp, cx[0], cy[0], cz[0], nx, ny, nz ) )
        {
          if( k_nm(0) < max_nm )
          {
              const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx; 
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy; 
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz; 
              k_particle_movers_i(nm)   = local_pm->i; 

              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
              k_particle_copy(nm, particle_var::dx) = pos_slice(p_index*SIMD_LEN+idx, 0); //k_particles( p_index*SIMD_LEN+idx, particle_var::dx); //
              k_particle_copy(nm, particle_var::dy) = pos_slice(p_index*SIMD_LEN+idx, 1); //k_particles( p_index*SIMD_LEN+idx, particle_var::dy); //
              k_particle_copy(nm, particle_var::dz) = pos_slice(p_index*SIMD_LEN+idx, 2); //k_particles( p_index*SIMD_LEN+idx, particle_var::dz); //
              k_particle_i_copy(nm)                 = cel_slice(p_index*SIMD_LEN+idx);    //k_particles_i(p_index*SIMD_LEN+idx);                  //
              k_particle_copy(nm, particle_var::ux) = mom_slice(p_index*SIMD_LEN+idx, 0); //k_particles( p_index*SIMD_LEN+idx, particle_var::ux); //
              k_particle_copy(nm, particle_var::uy) = mom_slice(p_index*SIMD_LEN+idx, 1); //k_particles( p_index*SIMD_LEN+idx, particle_var::uy); //
              k_particle_copy(nm, particle_var::uz) = mom_slice(p_index*SIMD_LEN+idx, 2); //k_particles( p_index*SIMD_LEN+idx, particle_var::uz); //
              k_particle_copy(nm, particle_var::w)  = wgt_slice(p_index*SIMD_LEN+idx);    //k_particles( p_index*SIMD_LEN+idx, particle_var::w ); //
          }
        }
      }
    }
//    token.release(threadID);
  });

  KOKKOS_TOC(advance_p, 1);
  KOKKOS_TIC();

  Kokkos::Profiling::popRegion();

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::MDRangePolicy<size_t, Kokkos::Rank<2>> reduce_policy({0LU,0LU}, {accumulator.extent(1), 12LU});
  Kokkos::parallel_for("reduce accumulator", reduce_policy, KOKKOS_LAMBDA(const uint32_t i, const uint32_t j) {
    for(int tid=1; tid<accumulator.extent(0); tid++) {
      accumulator(0, i, j) += accumulator(tid, i,  j);
    }
  });
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nx+2, ny+2, nz+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int x, const int y, const int z) {
    int f0  = VOXEL(x,   y,   z, nx, ny, nz);
    int a0  = VOXEL(x,   y,   z, nx, ny, nz);
    int ax  = VOXEL(x-1, y,   z, nx, ny, nz);
    int ay  = VOXEL(x,   y-1, z, nx, ny, nz);
    int az  = VOXEL(x,   y,   z-1, nx, ny, nz);
    int ayz = VOXEL(x,   y-1, z-1, nx, ny, nz);
    int azx = VOXEL(x-1, y,   z-1, nx, ny, nz);
    int axy = VOXEL(x-1, y-1, z, nx, ny, nz);
    k_field(f0, field_var::jfx) += ( accumulator(0, a0,  0) +
                                     accumulator(0, ay,  1) +
                                     accumulator(0, az,  2) +
                                     accumulator(0, ayz, 3) );
    k_field(f0, field_var::jfy) += ( accumulator(0, a0,  4) +
                                     accumulator(0, az,  5) +
                                     accumulator(0, ax,  6) +
                                     accumulator(0, azx, 7) );
    k_field(f0, field_var::jfz) += ( accumulator(0, a0,  8) +
                                     accumulator(0, ax,  9) +
                                     accumulator(0, ay,  10) +
                                     accumulator(0, axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif
 
  KOKKOS_TOC(reduce_accumulators, 1);

  Kokkos::parallel_for(k_particles.extent(0), KOKKOS_LAMBDA(const int i) {
    k_particles(i, particle_var::dx) = pos_slice(i, 0);
    k_particles(i, particle_var::dy) = pos_slice(i, 1);
    k_particles(i, particle_var::dz) = pos_slice(i, 2);
    k_particles_i(i)                 = cel_slice(i);   
    k_particles(i, particle_var::ux) = mom_slice(i, 0);
    k_particles(i, particle_var::uy) = mom_slice(i, 1);
    k_particles(i, particle_var::uz) = mom_slice(i, 2);
    k_particles(i, particle_var::w ) = wgt_slice(i);
  });
}
#endif

void
advance_p( /**/  species_t            * RESTRICT sp,
//           accumulator_array_t * RESTRICT aa,
           interpolator_array_t * RESTRICT ia,
           field_array_t* RESTRICT fa ) {
  //DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );
  //DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE+1 );
  //int rank;

  if( !sp )
  {
    ERROR(( "Bad args" ));
  }
  if( !ia  )
  {
    ERROR(( "Bad args" ));
  }
  if( sp->g!=ia->g )
  {
    ERROR(( "Bad args" ));
  }


  float qdt_2mc  = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  float cdt_dx   = sp->g->cvac*sp->g->dt*sp->g->rdx;
  float cdt_dy   = sp->g->cvac*sp->g->dt*sp->g->rdy;
  float cdt_dz   = sp->g->cvac*sp->g->dt*sp->g->rdz;

  #ifdef USE_GPU
    // Use the gpu kernel for slightly better performance
    #define ADVANCE_P advance_p_kokkos_gpu
    //#define ADVANCE_P advance_p_kokkos_unified
    //#define ADVANCE_P advance_p_kokkos_simd
  #else
    // Portable kernel with additional vectorization options
    #ifdef ADVANCE_P_UNIFIED
      #define ADVANCE_P advance_p_kokkos_unified
    #elif defined( ADVANCE_P_GPU )
      #define ADVANCE_P advance_p_kokkos_gpu
    #elif defined( ADVANCE_P_SIMD_COMPUTE )
      #define ADVANCE_P advance_p_kokkos_simd_compute
    #elif defined( ADVANCE_P_SIMD )
      #define ADVANCE_P advance_p_kokkos_simd
    #elif defined( ADVANCE_P_SIMD_UNION )
      #define ADVANCE_P advance_p_kokkos_simd_union
    #elif defined( ADVANCE_P_SIMD_CABANA )
      #define ADVANCE_P advance_p_kokkos_simd_cabana
    #endif
  #endif
//  KOKKOS_TIC();
  ADVANCE_P(
          sp->k_p_d,
          sp->k_p_i_d,
          sp->k_pc_d,
          sp->k_pc_i_d,
          sp->k_pm_d,
          sp->k_pm_i_d,
          fa->k_field_sv_d,
          ia->k_i_d,
          sp->k_nm_d,
          sp->g->k_neighbor_d,
          fa,
          sp->g,
          qdt_2mc,
          cdt_dx,
          cdt_dy,
          cdt_dz,
          sp->q,
          sp->np,
          sp->max_nm,
          sp->g->nx,
          sp->g->ny,
          sp->g->nz
  );
//  KOKKOS_TOC( advance_p, 1);

  KOKKOS_TIC();
  // I need to know the number of movers that got populated so I can call the
  // compress. Let's copy it back
  Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
  // TODO: which way round should this copy be?

  //  int nm = sp->k_nm_h(0);

  //  printf("nm = %d \n", nm);

  // Copy particle mirror movers back so we have their data safe. Ready for
  // boundary_p_kokkos
  auto pc_d_subview = Kokkos::subview(sp->k_pc_d, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
  auto pci_d_subview = Kokkos::subview(sp->k_pc_i_d, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_h_subview = Kokkos::subview(sp->k_pc_h, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
  auto pci_h_subview = Kokkos::subview(sp->k_pc_i_h, std::make_pair(0, sp->k_nm_h(0)));

  Kokkos::deep_copy(pc_h_subview, pc_d_subview);
  Kokkos::deep_copy(pci_h_subview, pci_d_subview);
  //  Kokkos::deep_copy(sp->k_pc_h, sp->k_pc_d);
  //  Kokkos::deep_copy(sp->k_pc_i_h, sp->k_pc_i_d);

  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
}
