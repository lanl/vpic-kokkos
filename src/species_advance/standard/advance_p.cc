// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include <math.h>

//void
//advance_p_kokkos_vector(
//        k_particles_t& k_particles,
//        k_particles_i_t& k_particles_i,
//        k_particle_copy_t& k_particle_copy,
//        k_particle_i_copy_t& k_particle_i_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_interpolator_t& k_interp,
//        //k_particle_movers_t k_local_particle_movers,
//        k_iterator_t& k_nm,
//        k_neighbor_t& k_neighbors,
//        const grid_t *g,
//        const float qdt_2mc,
//        const float cdt_dx,
//        const float cdt_dy,
//        const float cdt_dz,
//        const float qsp,
//        const int na,
//        const int np,
//        const int max_nm,
//        const int nx,
//        const int ny,
//        const int nz)
//{
//
//  constexpr float one            = 1.;
//  constexpr float one_third      = 1./3.;
//  constexpr float two_fifteenths = 2./15.;
//
//  #define p_dx    k_particles(p_index, particle_var::dx)
//  #define p_dy    k_particles(p_index, particle_var::dy)
//  #define p_dz    k_particles(p_index, particle_var::dz)
//  #define p_ux    k_particles(p_index, particle_var::ux)
//  #define p_uy    k_particles(p_index, particle_var::uy)
//  #define p_uz    k_particles(p_index, particle_var::uz)
//  #define p_w     k_particles(p_index, particle_var::w)
//  #define pii     k_particles_i(p_index)
//
//  #define f_cbx k_interp(ii[lane], interpolator_var::cbx)
//  #define f_cby k_interp(ii[lane], interpolator_var::cby)
//  #define f_cbz k_interp(ii[lane], interpolator_var::cbz)
//  #define f_ex  k_interp(ii[lane], interpolator_var::ex)
//  #define f_ey  k_interp(ii[lane], interpolator_var::ey)
//  #define f_ez  k_interp(ii[lane], interpolator_var::ez)
//
//  #define f_dexdy    k_interp(ii[lane], interpolator_var::dexdy)
//  #define f_dexdz    k_interp(ii[lane], interpolator_var::dexdz)
//
//  #define f_d2exdydz k_interp(ii[lane], interpolator_var::d2exdydz)
//  #define f_deydx    k_interp(ii[lane], interpolator_var::deydx)
//  #define f_deydz    k_interp(ii[lane], interpolator_var::deydz)
//
//  #define f_d2eydzdx k_interp(ii[lane], interpolator_var::d2eydzdx)
//  #define f_dezdx    k_interp(ii[lane], interpolator_var::dezdx)
//  #define f_dezdy    k_interp(ii[lane], interpolator_var::dezdy)
//
//  #define f_d2ezdxdy k_interp(ii[lane], interpolator_var::d2ezdxdy)
//  #define f_dcbxdx   k_interp(ii[lane], interpolator_var::dcbxdx)
//  #define f_dcbydy   k_interp(ii[lane], interpolator_var::dcbydy)
//  #define f_dcbzdz   k_interp(ii[lane], interpolator_var::dcbzdz)
//
//  // copy local memmbers from grid
//  //auto nfaces_per_voxel = 6;
//  //auto nvoxels = g->nv;
//  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
//  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);
//
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::deep_copy(k_nm, 0);
//
//  constexpr int num_leagues = 1;
//  constexpr int num_lanes = 64;
//
////  Kokkos::View<float[num_lanes]> v0("v0"), v1("v1"), v2("v2"), v3("v3"), v4("v4"), v5("v5");
////  Kokkos::View<float[num_lanes]> dx("dx"), dy("dy"), dz("dz"), ux("ux"), uy("uy"), uz("uz");
////  Kokkos::View<float[num_lanes]> hax("hax"), hay("hay"), haz("haz"), cbx("cbx"), cby("cby"), cbz("cbz");
////  Kokkos::View<float[num_lanes]> q("q");
////  Kokkos::View<int[num_lanes]> ii("ii");
////  Kokkos::View<bool[num_lanes]> inbnds("inbnds");
//  int num_chunks = np/num_lanes;
////  if(np%num_lanes > 0)
////    num_chunks++;
//  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(num_leagues, Kokkos::AUTO(), num_lanes).set_scratch_size(0, Kokkos::PerThread(21*num_lanes*4));
////  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(num_leagues, Kokkos::AUTO(), num_lanes);
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
//  typedef Kokkos::View<float[num_lanes], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_float;
//  typedef Kokkos::View<int[num_lanes],   ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_int;
//  typedef Kokkos::View<bool[num_lanes],  ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_bool;
//  Kokkos::parallel_for("advance_p", policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    simd_float v0(team_member.team_scratch(0));
//    simd_float v1(team_member.team_scratch(0));
//    simd_float v2(team_member.team_scratch(0));
//    simd_float v3(team_member.team_scratch(0));
//    simd_float v4(team_member.team_scratch(0));
//    simd_float v5(team_member.team_scratch(0));
//    simd_float dx(team_member.team_scratch(0));
//    simd_float dy(team_member.team_scratch(0));
//    simd_float dz(team_member.team_scratch(0));
//    simd_float ux(team_member.team_scratch(0));
//    simd_float uy(team_member.team_scratch(0));
//    simd_float uz(team_member.team_scratch(0));
//    simd_float hax(team_member.team_scratch(0));
//    simd_float hay(team_member.team_scratch(0));
//    simd_float haz(team_member.team_scratch(0));
//    simd_float cbx(team_member.team_scratch(0));
//    simd_float cby(team_member.team_scratch(0));
//    simd_float cbz(team_member.team_scratch(0));
//    simd_int   ii(team_member.team_scratch(0));
//    simd_float q(team_member.team_scratch(0));
//    simd_bool  inbnds(team_member.team_scratch(0));
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_chunks), [=] (size_t chunk) {
////Kokkos::parallel_for("advance_p", Kokkos::RangePolicy<>(0,num_chunks), KOKKOS_LAMBDA(const size_t chunk) {
//      auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        size_t p_index = chunk*num_lanes + lane;
//  
//        // Load position
//        dx[lane] = p_dx;
//        dy[lane] = p_dy;
//        dz[lane] = p_dz;
//        ii[lane] = pii;
//  
//        hax[lane] = qdt_2mc*( (f_ex + dy[lane]*f_dexdy ) + dz[lane]*(f_dexdz + dy[lane]*f_d2exdydz) );
//        hay[lane] = qdt_2mc*( (f_ey + dz[lane]*f_deydz ) + dx[lane]*(f_deydx + dz[lane]*f_d2eydzdx) );
//        haz[lane] = qdt_2mc*( (f_ez + dx[lane]*f_dezdx ) + dy[lane]*(f_dezdy + dx[lane]*f_d2ezdxdy) );
//  
//        // Interpolate B
//        cbx[lane] = f_cbx + dx[lane]*f_dcbxdx;
//        cby[lane] = f_cby + dy[lane]*f_dcbydy;
//        cbz[lane] = f_cbz + dz[lane]*f_dcbzdz;
//  
//        // Load momentum
//        ux[lane] = p_ux;
//        uy[lane] = p_uy;
//        uz[lane] = p_uz;
//        q[lane] = p_w;
//  
//        // Half advance e
//        ux[lane] += hax[lane];
//        uy[lane] += hay[lane];
//        uz[lane] += haz[lane];
//      }
//  
//      for(int lane=0; lane<num_lanes; lane++) {
//        v0[lane] = qdt_2mc/sqrtf(one + (ux[lane]*ux[lane] + (uy[lane]*uy[lane] + uz[lane]*uz[lane])));
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        size_t p_index = chunk*num_lanes + lane;
//        // Boris - scalars
//        v1[lane] = cbx[lane]*cbx[lane] + (cby[lane]*cby[lane] + cbz[lane]*cbz[lane]);
//        v2[lane] = (v0[lane]*v0[lane])*v1[lane];
//        v3[lane] = v0[lane]*(one+v2[lane]*(one_third+v2[lane]*two_fifteenths));
//        v4[lane] = v3[lane]/(one+v1[lane]*(v3[lane]*v3[lane]));
//        v4[lane] += v4[lane];
//        // Boris - uprime
//        v0[lane] = ux[lane] + v3[lane]*(uy[lane]*cbz[lane] - uz[lane]*cby[lane]);
//        v1[lane] = uy[lane] + v3[lane]*(uz[lane]*cbx[lane] - ux[lane]*cbz[lane]);
//        v2[lane] = uz[lane] + v3[lane]*(ux[lane]*cby[lane] - uy[lane]*cbx[lane]);
//        // Boris - rotation
//        ux[lane] += v4[lane]*(v1[lane]*cbz[lane] - v2[lane]*cby[lane]);
//        uy[lane] += v4[lane]*(v2[lane]*cbx[lane] - v0[lane]*cbz[lane]);
//        uz[lane] += v4[lane]*(v0[lane]*cby[lane] - v1[lane]*cbx[lane]);
//        // Half advance e
//        ux[lane] += hax[lane];
//        uy[lane] += hay[lane];
//        uz[lane] += haz[lane];
//        // Store momentum
//        p_ux = ux[lane];
//        p_uy = uy[lane];
//        p_uz = uz[lane];
//      }
//  
//      for(int lane=0; lane<num_lanes; lane++) {
//        v0[lane]   = one/sqrtf(one + (ux[lane]*ux[lane]+ (uy[lane]*uy[lane] + uz[lane]*uz[lane])));
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        /**/                                      // Get norm displacement
//        ux[lane]  *= cdt_dx;
//        uy[lane]  *= cdt_dy;
//        uz[lane]  *= cdt_dz;
//        ux[lane]  *= v0[lane];
//        uy[lane]  *= v0[lane];
//        uz[lane]  *= v0[lane];
//        v0[lane]   = dx[lane] + ux[lane];                           // Streak midpoint (inbnds)
//        v1[lane]   = dy[lane] + uy[lane];
//        v2[lane]   = dz[lane] + uz[lane];
//        v3[lane]   = v0[lane] + ux[lane];                           // New position
//        v4[lane]   = v1[lane] + uy[lane];
//        v5[lane]   = v2[lane] + uz[lane];
//  
//        inbnds[lane] = v3[lane]<=one &&  v4[lane]<=one &&  v5[lane]<=one &&
//                      -v3[lane]<=one && -v4[lane]<=one && -v5[lane]<=one;
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        size_t p_index = chunk*num_lanes + lane;
//
//        v3[lane] = inbnds[lane] ? v3[lane] : p_dx;
//        v4[lane] = inbnds[lane] ? v4[lane] : p_dy;
//        v5[lane] = inbnds[lane] ? v5[lane] : p_dz;
//
//        q[lane] = inbnds[lane] ? q[lane]*qsp : 0.0;
//        p_dx = v3[lane];
//        p_dy = v4[lane];
//        p_dz = v5[lane];
//        dx[lane] = v0[lane];
//        dy[lane] = v1[lane];
//        dz[lane] = v2[lane];
//        v5[lane] = q[lane]*ux[lane]*uy[lane]*uz[lane]*one_third;
//
//#       define ACCUMULATE_J(X,Y,Z)                                 \
//        v4[lane]  = q[lane]*u##X[lane];   /* v2 = q ux                            */        \
//        v1[lane]  = v4[lane]*d##Y[lane];  /* v1 = q ux dy                         */        \
//        v0[lane]  = v4[lane]-v1[lane];    /* v0 = q ux (1-dy)                     */        \
//        v1[lane] += v4[lane];             /* v1 = q ux (1+dy)                     */        \
//        v4[lane]  = one+d##Z[lane];       /* v4 = 1+dz                            */        \
//        v2[lane]  = v0[lane]*v4[lane];    /* v2 = q ux (1-dy)(1+dz)               */        \
//        v3[lane]  = v1[lane]*v4[lane];    /* v3 = q ux (1+dy)(1+dz)               */        \
//        v4[lane]  = one-d##Z[lane];       /* v4 = 1-dz                            */        \
//        v0[lane] *= v4[lane];             /* v0 = q ux (1-dy)(1-dz)               */        \
//        v1[lane] *= v4[lane];             /* v1 = q ux (1+dy)(1-dz)               */        \
//        v0[lane] += v5[lane];             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//        v1[lane] -= v5[lane];             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//        v2[lane] -= v5[lane];             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//        v3[lane] += v5[lane];             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//        ACCUMULATE_J( x,y,z );
////        v6[lane] = v0[lane];
////        v7[lane] = v1[lane];
////        v8[lane] = v2[lane];
////        v9[lane] = v3[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 0) += v0[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 1) += v1[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 2) += v2[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 3) += v3[lane];
//
//        ACCUMULATE_J( y,z,x );
////        v10[lane] = v0[lane];
////        v11[lane] = v1[lane];
////        v12[lane] = v2[lane];
////        v13[lane] = v3[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 0) += v0[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 1) += v1[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 2) += v2[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 3) += v3[lane];
//
//        ACCUMULATE_J( z,x,y );
////        v14[lane] = v0[lane];
////        v15[lane] = v1[lane];
////        v16[lane] = v2[lane];
////        v17[lane] = v3[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 0) += v0[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 1) += v1[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 2) += v2[lane];
//        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 3) += v3[lane];
//#       undef ACCUMULATE_J
//      }
////      for(int lane=0; lane<num_lanes; lane++) {
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 0) += v6[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 1) += v7[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 2) += v8[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 3) += v9[lane];
////
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 0) += v10[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 1) += v11[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 2) += v12[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 3) += v13[lane];
////
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 0) += v14[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 1) += v15[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 2) += v16[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 3) += v17[lane];
////      }
//      for(int lane=0; lane<num_lanes; lane++) {
//        if(!inbnds[lane]) {
//          size_t p_index = chunk*num_lanes + lane;
//          DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//          local_pm->dispx = ux[lane];
//          local_pm->dispy = uy[lane];
//          local_pm->dispz = uz[lane];
//          local_pm->i     = p_index;
//
//          if( move_p_kokkos( k_particles, k_particles_i, local_pm,
//                             k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
//            if( k_nm(0)<max_nm ) {
//              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//              if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
//              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
//              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
//              k_particle_movers_i(nm)   = local_pm->i;
//
//              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//              k_particle_copy(nm, particle_var::dx) = p_dx;
//              k_particle_copy(nm, particle_var::dy) = p_dy;
//              k_particle_copy(nm, particle_var::dz) = p_dz;
//              k_particle_copy(nm, particle_var::ux) = p_ux;
//              k_particle_copy(nm, particle_var::uy) = p_uy;
//              k_particle_copy(nm, particle_var::uz) = p_uz;
//              k_particle_copy(nm, particle_var::w) = p_w;
//              k_particle_i_copy(nm) = pii;
//            }
//          }
//        }
//      }
//    });
//  });
////});
//#undef p_dx
//#undef p_dy
//#undef p_dz
//#undef p_ux
//#undef p_uy
//#undef p_uz
//#undef p_w 
//#undef pii 
//
//#undef f_cbx
//#undef f_cby
//#undef f_cbz
//#undef f_ex 
//#undef f_ey 
//#undef f_ez 
//
//#undef f_dexdy
//#undef f_dexdz
//
//#undef f_d2exdydz
//#undef f_deydx   
//#undef f_deydz   
//
//#undef f_d2eydzdx
//#undef f_dezdx   
//#undef f_dezdy   
//
//#undef f_d2ezdxdy
//#undef f_dcbxdx  
//#undef f_dcbydy  
//#undef f_dcbzdz  
//
//#define p_dx    k_particles(p_index, particle_var::dx)
//#define p_dy    k_particles(p_index, particle_var::dy)
//#define p_dz    k_particles(p_index, particle_var::dz)
//#define p_ux    k_particles(p_index, particle_var::ux)
//#define p_uy    k_particles(p_index, particle_var::uy)
//#define p_uz    k_particles(p_index, particle_var::uz)
//#define p_w     k_particles(p_index, particle_var::w)
//#define pii     k_particles_i(p_index)
//
//#define f_cbx k_interp(ii, interpolator_var::cbx)
//#define f_cby k_interp(ii, interpolator_var::cby)
//#define f_cbz k_interp(ii, interpolator_var::cbz)
//#define f_ex  k_interp(ii, interpolator_var::ex)
//#define f_ey  k_interp(ii, interpolator_var::ey)
//#define f_ez  k_interp(ii, interpolator_var::ez)
//
//#define f_dexdy    k_interp(ii, interpolator_var::dexdy)
//#define f_dexdz    k_interp(ii, interpolator_var::dexdz)
//
//#define f_d2exdydz k_interp(ii, interpolator_var::d2exdydz)
//#define f_deydx    k_interp(ii, interpolator_var::deydx)
//#define f_deydz    k_interp(ii, interpolator_var::deydz)
//
//#define f_d2eydzdx k_interp(ii, interpolator_var::d2eydzdx)
//#define f_dezdx    k_interp(ii, interpolator_var::dezdx)
//#define f_dezdy    k_interp(ii, interpolator_var::dezdy)
//
//#define f_d2ezdxdy k_interp(ii, interpolator_var::d2ezdxdy)
//#define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
//#define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
//#define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)
//  if(num_chunks*num_lanes < np) {
//    for(int p_index=num_chunks*num_lanes; p_index<np; p_index++) {
//      float v0, v1, v2, v3, v4, v5;
//      auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//
//      // Load position
//      float dx = p_dx;
//      float dy = p_dy;
//      float dz = p_dz;
//      int ii = pii;
//
//      float hax = qdt_2mc*( (f_ex + dy*f_dexdy ) + dz*(f_dexdz + dy*f_d2exdydz) );
//      float hay = qdt_2mc*( (f_ey + dz*f_deydz ) + dx*(f_deydx + dz*f_d2eydzdx) );
//      float haz = qdt_2mc*( (f_ez + dx*f_dezdx ) + dy*(f_dezdy + dx*f_d2ezdxdy) );
//
//      // Interpolate B
//      float cbx = f_cbx + dx*f_dcbxdx;
//      float cby = f_cby + dy*f_dcbydy;
//      float cbz = f_cbz + dz*f_dcbzdz;
//
//      // Load momentum
//      float ux = p_ux;
//      float uy = p_uy;
//      float uz = p_uz;
//      float q = p_w;
//
//      // Half advance e
//      ux += hax;
//      uy += hay;
//      uz += haz;
//
//      v0 = qdt_2mc/std::sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//
//      // Boris - scalars
//      v1 = cbx*cbx + (cby*cby + cbz*cbz);
//      v2 = (v0*v0)*v1;
//      v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
//      v4 = v3/(one+v1*(v3*v3));
//      v4 += v4;
//      // Boris - uprime
//      v0 = ux + v3*(uy*cbz - uz*cby);
//      v1 = uy + v3*(uz*cbx - ux*cbz);
//      v2 = uz + v3*(ux*cby - uy*cbx);
//      // Boris - rotation
//      ux += v4*(v1*cbz - v2*cby);
//      uy += v4*(v2*cbx - v0*cbz);
//      uz += v4*(v0*cby - v1*cbx);
//      // Half advance e
//      ux += hax;
//      uy += hay;
//      uz += haz;
//      // Store momentum
//      p_ux = ux;
//      p_uy = uy;
//      p_uz = uz;
//
//      v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));
//
//      /**/                                      // Get norm displacement
//      ux  *= cdt_dx;
//      uy  *= cdt_dy;
//      uz  *= cdt_dz;
//      ux  *= v0;
//      uy  *= v0;
//      uz  *= v0;
//      v0   = dx + ux;                           // Streak midpoint (inbnds)
//      v1   = dy + uy;
//      v2   = dz + uz;
//      v3   = v0 + ux;                           // New position
//      v4   = v1 + uy;
//      v5   = v2 + uz;
//
//      bool inbnds = v3<=one &&  v4<=one &&  v5<=one &&
//                    -v3<=one && -v4<=one && -v5<=one;
//      if(inbnds) {
//        // Common case (inbnds).  Note: accumulator values are 4 times
//        // the total physical charge that passed through the appropriate
//        // current quadrant in a time-step
//
//        q *= qsp;
//        p_dx = v3;                             // Store new position
//        p_dy = v4;
//        p_dz = v5;
//        dx = v0;                                // Streak midpoint
//        dy = v1;
//        dz = v2;
//        v5 = q*ux*uy*uz*one_third;              // Compute correction
//
//#       define ACCUMULATE_J(X,Y,Z)                                 \
//        v4  = q*u##X;   /* v2 = q ux                            */        \
//        v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//        v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//        v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//        v4  = one+d##Z; /* v4 = 1+dz                            */        \
//        v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//        v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//        v4  = one-d##Z; /* v4 = 1-dz                            */        \
//        v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//        v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//        v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//        v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//        v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//        v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//        ACCUMULATE_J( x,y,z );
//        k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
//        k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
//        k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
//        k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;
//
//        ACCUMULATE_J( y,z,x );
//        k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
//        k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
//        k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
//        k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;
//
//        ACCUMULATE_J( z,x,y );
//        k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
//        k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
//        k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
//        k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
//#       undef ACCUMULATE_J
//      } else {
//        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//        local_pm->dispx = ux;
//        local_pm->dispy = uy;
//        local_pm->dispz = uz;
//        local_pm->i     = p_index;
//
//        if( move_p_kokkos( k_particles, k_particles_i, local_pm,
//                           k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
//          if( k_nm(0)<max_nm ) {
//            const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//            if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//            k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
//            k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
//            k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
//            k_particle_movers_i(nm)   = local_pm->i;
//
//            // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//            k_particle_copy(nm, particle_var::dx) = p_dx;
//            k_particle_copy(nm, particle_var::dy) = p_dy;
//            k_particle_copy(nm, particle_var::dz) = p_dz;
//            k_particle_copy(nm, particle_var::ux) = p_ux;
//            k_particle_copy(nm, particle_var::uy) = p_uy;
//            k_particle_copy(nm, particle_var::uz) = p_uz;
//            k_particle_copy(nm, particle_var::w) = p_w;
//            k_particle_i_copy(nm) = pii;
//          }
//        }
//      }
//    }
//  }
//#undef p_dx
//#undef p_dy
//#undef p_dz
//#undef p_ux
//#undef p_uy
//#undef p_uz
//#undef p_w 
//#undef pii 
//
//#undef f_cbx
//#undef f_cby
//#undef f_cbz
//#undef f_ex 
//#undef f_ey 
//#undef f_ez 
//
//#undef f_dexdy
//#undef f_dexdz
//
//#undef f_d2exdydz
//#undef f_deydx   
//#undef f_deydz   
//
//#undef f_d2eydzdx
//#undef f_dezdx   
//#undef f_dezdy   
//
//#undef f_d2ezdxdy
//#undef f_dcbxdx  
//#undef f_dcbydy  
//#undef f_dcbzdz  
//}

void
advance_p_kokkos_omp_simd(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_accumulators_sa_t k_accumulators_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_iterator_t& k_nm,
        k_neighbor_t& k_neighbors,
        const grid_t *g,
        const float qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int na,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux)
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define p_w     k_particles(p_index, particle_var::w)
  #define pii     k_particles_i(p_index)

  #define f_cbx k_interp(ii[lane], interpolator_var::cbx)
  #define f_cby k_interp(ii[lane], interpolator_var::cby)
  #define f_cbz k_interp(ii[lane], interpolator_var::cbz)
  #define f_ex  k_interp(ii[lane], interpolator_var::ex)
  #define f_ey  k_interp(ii[lane], interpolator_var::ey)
  #define f_ez  k_interp(ii[lane], interpolator_var::ez)

  #define f_dexdy    k_interp(ii[lane], interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii[lane], interpolator_var::dexdz)

  #define f_d2exdydz k_interp(ii[lane], interpolator_var::d2exdydz)
  #define f_deydx    k_interp(ii[lane], interpolator_var::deydx)
  #define f_deydz    k_interp(ii[lane], interpolator_var::deydz)

  #define f_d2eydzdx k_interp(ii[lane], interpolator_var::d2eydzdx)
  #define f_dezdx    k_interp(ii[lane], interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii[lane], interpolator_var::dezdy)

  #define f_d2ezdxdy k_interp(ii[lane], interpolator_var::d2ezdxdy)
  #define f_dcbxdx   k_interp(ii[lane], interpolator_var::dcbxdx)
  #define f_dcbydy   k_interp(ii[lane], interpolator_var::dcbydy)
  #define f_dcbzdz   k_interp(ii[lane], interpolator_var::dcbzdz)

  // copy local memmbers from grid
  //auto nfaces_per_voxel = 6;
  //auto nvoxels = g->nv;
  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

  constexpr int num_lanes = 16;
  int num_chunks = np/num_lanes;
  auto  k_accumulators_scatter_access = k_accumulators_sa.access();

  for(size_t chunk=0; chunk<num_chunks; chunk++) {
    float v0[num_lanes], v1[num_lanes], v2[num_lanes], v3[num_lanes], v4[num_lanes], v5[num_lanes];
    float v6[num_lanes], v7[num_lanes], v8[num_lanes], v9[num_lanes];
    float v10[num_lanes], v11[num_lanes], v12[num_lanes], v13[num_lanes];
    float v14[num_lanes], v15[num_lanes], v16[num_lanes], v17[num_lanes];
    float dx[num_lanes], dy[num_lanes], dz[num_lanes];
    float ux[num_lanes], uy[num_lanes], uz[num_lanes];
    float hax[num_lanes], hay[num_lanes], haz[num_lanes];
    float cbx[num_lanes], cby[num_lanes], cbz[num_lanes];
    float q[num_lanes];
    int ii[num_lanes];
    bool inbnds[num_lanes];

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
      // Load position
      dx[lane] = p_dx;
      dy[lane] = p_dy;
      dz[lane] = p_dz;

      // Load momentum
      ux[lane] = p_ux;
      uy[lane] = p_uy;
      uz[lane] = p_uz;
      q[lane] = p_w;
      ii[lane] = pii;
    }

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
//      // Load position
//      dx[lane] = p_dx;
//      dy[lane] = p_dy;
//      dz[lane] = p_dz;
//      ii[lane] = pii;

      hax[lane] = qdt_2mc*( (f_ex + dy[lane]*f_dexdy ) + dz[lane]*(f_dexdz + dy[lane]*f_d2exdydz) );
      hay[lane] = qdt_2mc*( (f_ey + dz[lane]*f_deydz ) + dx[lane]*(f_deydx + dz[lane]*f_d2eydzdx) );
      haz[lane] = qdt_2mc*( (f_ez + dx[lane]*f_dezdx ) + dy[lane]*(f_dezdy + dx[lane]*f_d2ezdxdy) );

      // Interpolate B
      cbx[lane] = f_cbx + dx[lane]*f_dcbxdx;
      cby[lane] = f_cby + dy[lane]*f_dcbydy;
      cbz[lane] = f_cbz + dz[lane]*f_dcbzdz;

//      // Load momentum
//      ux[lane] = p_ux;
//      uy[lane] = p_uy;
//      uz[lane] = p_uz;
//      q[lane] = p_w;

      // Half advance e
      ux[lane] += hax[lane];
      uy[lane] += hay[lane];
      uz[lane] += haz[lane];
    }

#pragma omp simd 
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      v0[lane] = qdt_2mc/sqrt(one + (ux[lane]*ux[lane] + (uy[lane]*uy[lane] + uz[lane]*uz[lane])));
    }

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
      // Boris - scalars
      v1[lane] = cbx[lane]*cbx[lane] + (cby[lane]*cby[lane] + cbz[lane]*cbz[lane]);
      v2[lane] = (v0[lane]*v0[lane])*v1[lane];
      v3[lane] = v0[lane]*(one+v2[lane]*(one_third+v2[lane]*two_fifteenths));
      v4[lane] = v3[lane]/(one+v1[lane]*(v3[lane]*v3[lane]));
      v4[lane] += v4[lane];
      // Boris - uprime
      v0[lane] = ux[lane] + v3[lane]*(uy[lane]*cbz[lane] - uz[lane]*cby[lane]);
      v1[lane] = uy[lane] + v3[lane]*(uz[lane]*cbx[lane] - ux[lane]*cbz[lane]);
      v2[lane] = uz[lane] + v3[lane]*(ux[lane]*cby[lane] - uy[lane]*cbx[lane]);
      // Boris - rotation
      ux[lane] += v4[lane]*(v1[lane]*cbz[lane] - v2[lane]*cby[lane]);
      uy[lane] += v4[lane]*(v2[lane]*cbx[lane] - v0[lane]*cbz[lane]);
      uz[lane] += v4[lane]*(v0[lane]*cby[lane] - v1[lane]*cbx[lane]);
      // Half advance e
      ux[lane] += hax[lane];
      uy[lane] += hay[lane];
      uz[lane] += haz[lane];
      // Store momentum
      p_ux = ux[lane];
      p_uy = uy[lane];
      p_uz = uz[lane];
    }

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      v0[lane]   = one/std::sqrt(one + (ux[lane]*ux[lane]+ (uy[lane]*uy[lane] + uz[lane]*uz[lane])));
    }

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      /**/                                      // Get norm displacement
      ux[lane]  *= cdt_dx;
      uy[lane]  *= cdt_dy;
      uz[lane]  *= cdt_dz;
      ux[lane]  *= v0[lane];
      uy[lane]  *= v0[lane];
      uz[lane]  *= v0[lane];
      v0[lane]   = dx[lane] + ux[lane];                           // Streak midpoint (inbnds)
      v1[lane]   = dy[lane] + uy[lane];
      v2[lane]   = dz[lane] + uz[lane];
      v3[lane]   = v0[lane] + ux[lane];                           // New position
      v4[lane]   = v1[lane] + uy[lane];
      v5[lane]   = v2[lane] + uz[lane];

      inbnds[lane] = v3[lane]<=one &&  v4[lane]<=one &&  v5[lane]<=one &&
                    -v3[lane]<=one && -v4[lane]<=one && -v5[lane]<=one;
    }

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;

      v3[lane] = inbnds[lane] ? v3[lane] : p_dx;
      v4[lane] = inbnds[lane] ? v4[lane] : p_dy;
      v5[lane] = inbnds[lane] ? v5[lane] : p_dz;

//      q[lane] = inbnds[lane] ? q[lane]*qsp : 0.0;
      q[lane] = static_cast<float>(inbnds[lane])*q[lane]*qsp;
      p_dx = v3[lane];
      p_dy = v4[lane];
      p_dz = v5[lane];
      dx[lane] = v0[lane];
      dy[lane] = v1[lane];
      dz[lane] = v2[lane];
      v5[lane] = q[lane]*ux[lane]*uy[lane]*uz[lane]*one_third;
    }

//#pragma omp simd
#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
#     define ACCUMULATE_J(X,Y,Z)                                 \
      v4[lane]  = q[lane]*u##X[lane];   /* v2 = q ux                            */        \
      v1[lane]  = v4[lane]*d##Y[lane];  /* v1 = q ux dy                         */        \
      v0[lane]  = v4[lane]-v1[lane];    /* v0 = q ux (1-dy)                     */        \
      v1[lane] += v4[lane];             /* v1 = q ux (1+dy)                     */        \
      v4[lane]  = one+d##Z[lane];       /* v4 = 1+dz                            */        \
      v2[lane]  = v0[lane]*v4[lane];    /* v2 = q ux (1-dy)(1+dz)               */        \
      v3[lane]  = v1[lane]*v4[lane];    /* v3 = q ux (1+dy)(1+dz)               */        \
      v4[lane]  = one-d##Z[lane];       /* v4 = 1-dz                            */        \
      v0[lane] *= v4[lane];             /* v0 = q ux (1-dy)(1-dz)               */        \
      v1[lane] *= v4[lane];             /* v1 = q ux (1+dy)(1-dz)               */        \
      v0[lane] += v5[lane];             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
      v1[lane] -= v5[lane];             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
      v2[lane] -= v5[lane];             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
      v3[lane] += v5[lane];             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

      ACCUMULATE_J( x,y,z );
      v6[lane] = v0[lane];
      v7[lane] = v1[lane];
      v8[lane] = v2[lane];
      v9[lane] = v3[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 0) += v0[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 1) += v1[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 2) += v2[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 3) += v3[lane];

      ACCUMULATE_J( y,z,x );
      v10[lane] = v0[lane];
      v11[lane] = v1[lane];
      v12[lane] = v2[lane];
      v13[lane] = v3[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 0) += v0[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 1) += v1[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 2) += v2[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 3) += v3[lane];

      ACCUMULATE_J( z,x,y );
      v14[lane] = v0[lane];
      v15[lane] = v1[lane];
      v16[lane] = v2[lane];
      v17[lane] = v3[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 0) += v0[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 1) += v1[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 2) += v2[lane];
//      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 3) += v3[lane];
#     undef ACCUMULATE_J
    }

#pragma omp simd
//#pragma ivdep
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 0) += v6[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 1) += v7[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 2) += v8[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 3) += v9[lane];

      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 0) += v10[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 1) += v11[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 2) += v12[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 3) += v13[lane];

      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 0) += v14[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 1) += v15[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 2) += v16[lane];
      k_accumulators_scatter_access(ii[lane], accumulator_var::jz, 3) += v17[lane];
    }

    for(int lane=0; lane<num_lanes; lane++) {
      if(!inbnds[lane]) {
        size_t p_index = chunk*num_lanes + lane;
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux[lane];
        local_pm->dispy = uy[lane];
        local_pm->dispz = uz[lane];
        local_pm->i     = p_index;

        if( move_p_kokkos( k_particles, k_particles_i, local_pm,
                           k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
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
    }
  }
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

#define p_dx    k_particles(p_index, particle_var::dx)
#define p_dy    k_particles(p_index, particle_var::dy)
#define p_dz    k_particles(p_index, particle_var::dz)
#define p_ux    k_particles(p_index, particle_var::ux)
#define p_uy    k_particles(p_index, particle_var::uy)
#define p_uz    k_particles(p_index, particle_var::uz)
#define p_w     k_particles(p_index, particle_var::w)
#define pii     k_particles_i(p_index)

#define f_cbx k_interp(ii, interpolator_var::cbx)
#define f_cby k_interp(ii, interpolator_var::cby)
#define f_cbz k_interp(ii, interpolator_var::cbz)
#define f_ex  k_interp(ii, interpolator_var::ex)
#define f_ey  k_interp(ii, interpolator_var::ey)
#define f_ez  k_interp(ii, interpolator_var::ez)

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
  if(num_chunks*num_lanes < np) {
    for(int p_index=num_chunks*num_lanes; p_index<np; p_index++) {
      float v0, v1, v2, v3, v4, v5;
      auto  k_accumulators_scatter_access = k_accumulators_sa.access();

      // Load position
      float dx = p_dx;
      float dy = p_dy;
      float dz = p_dz;
      int ii = pii;

      float hax = qdt_2mc*( (f_ex + dy*f_dexdy ) + dz*(f_dexdz + dy*f_d2exdydz) );
      float hay = qdt_2mc*( (f_ey + dz*f_deydz ) + dx*(f_deydx + dz*f_d2eydzdx) );
      float haz = qdt_2mc*( (f_ez + dx*f_dezdx ) + dy*(f_dezdy + dx*f_d2ezdxdy) );

      // Interpolate B
      float cbx = f_cbx + dx*f_dcbxdx;
      float cby = f_cby + dy*f_dcbydy;
      float cbz = f_cbz + dz*f_dcbzdz;

      // Load momentum
      float ux = p_ux;
      float uy = p_uy;
      float uz = p_uz;
      float q = p_w;

      // Half advance e
      ux += hax;
      uy += hay;
      uz += haz;

      v0 = qdt_2mc/sqrt(one + (ux*ux + (uy*uy + uz*uz)));

      // Boris - scalars
      v1 = cbx*cbx + (cby*cby + cbz*cbz);
      v2 = (v0*v0)*v1;
      v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
      v4 = v3/(one+v1*(v3*v3));
      v4 += v4;
      // Boris - uprime
      v0 = ux + v3*(uy*cbz - uz*cby);
      v1 = uy + v3*(uz*cbx - ux*cbz);
      v2 = uz + v3*(ux*cby - uy*cbx);
      // Boris - rotation
      ux += v4*(v1*cbz - v2*cby);
      uy += v4*(v2*cbx - v0*cbz);
      uz += v4*(v0*cby - v1*cbx);
      // Half advance e
      ux += hax;
      uy += hay;
      uz += haz;
      // Store momentum
      p_ux = ux;
      p_uy = uy;
      p_uz = uz;

      v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));

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

      bool inbnds = v3<=one &&  v4<=one &&  v5<=one &&
                    -v3<=one && -v4<=one && -v5<=one;
      if(inbnds) {
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
        v5 = q*ux*uy*uz*one_third;              // Compute correction

#       define ACCUMULATE_J(X,Y,Z)                                 \
        v4  = q*u##X;   /* v2 = q ux                            */        \
        v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
        v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
        v1 += v4;       /* v1 = q ux (1+dy)                     */        \
        v4  = one+d##Z; /* v4 = 1+dz                            */        \
        v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
        v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
        v4  = one-d##Z; /* v4 = 1-dz                            */        \
        v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
        v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
        v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
        v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
        v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
        v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

        ACCUMULATE_J( x,y,z );
        k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
        k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
        k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
        k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;

        ACCUMULATE_J( y,z,x );
        k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
        k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
        k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
        k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

        ACCUMULATE_J( z,x,y );
        k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
        k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
        k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
        k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
#       undef ACCUMULATE_J
      } else {
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux;
        local_pm->dispy = uy;
        local_pm->dispz = uz;
        local_pm->i     = p_index;

        if( move_p_kokkos( k_particles, k_particles_i, local_pm,
                           k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
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
    }
  }
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

//void
//advance_p_kokkos_devel_serial(
//        k_particles_t& k_particles,
//        k_particles_i_t& k_particles_i,
//        k_particle_copy_t& k_particle_copy,
//        k_particle_i_copy_t& k_particle_i_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_interpolator_t& k_interp,
//        //k_particle_movers_t k_local_particle_movers,
//        k_iterator_t& k_nm,
//        k_neighbor_t& k_neighbors,
//        const grid_t *g,
//        const float qdt_2mc,
//        const float cdt_dx,
//        const float cdt_dy,
//        const float cdt_dz,
//        const float qsp,
//        const int na,
//        const int np,
//        const int max_nm,
//        const int nx,
//        const int ny,
//        const int nz)
//{
//
//  constexpr float one            = 1.;
//  constexpr float one_third      = 1./3.;
//  constexpr float two_fifteenths = 2./15.;
//
//  #define p_dx    k_particles(p_index, particle_var::dx)
//  #define p_dy    k_particles(p_index, particle_var::dy)
//  #define p_dz    k_particles(p_index, particle_var::dz)
//  #define p_ux    k_particles(p_index, particle_var::ux)
//  #define p_uy    k_particles(p_index, particle_var::uy)
//  #define p_uz    k_particles(p_index, particle_var::uz)
//  #define p_w     k_particles(p_index, particle_var::w)
//  #define pii     k_particles_i(p_index)
//
//  #define f_cbx k_interp(ii, interpolator_var::cbx)
//  #define f_cby k_interp(ii, interpolator_var::cby)
//  #define f_cbz k_interp(ii, interpolator_var::cbz)
//  #define f_ex  k_interp(ii, interpolator_var::ex)
//  #define f_ey  k_interp(ii, interpolator_var::ey)
//  #define f_ez  k_interp(ii, interpolator_var::ez)
//
//  #define f_dexdy    k_interp(ii, interpolator_var::dexdy)
//  #define f_dexdz    k_interp(ii, interpolator_var::dexdz)
//
//  #define f_d2exdydz k_interp(ii, interpolator_var::d2exdydz)
//  #define f_deydx    k_interp(ii, interpolator_var::deydx)
//  #define f_deydz    k_interp(ii, interpolator_var::deydz)
//
//  #define f_d2eydzdx k_interp(ii, interpolator_var::d2eydzdx)
//  #define f_dezdx    k_interp(ii, interpolator_var::dezdx)
//  #define f_dezdy    k_interp(ii, interpolator_var::dezdy)
//
//  #define f_d2ezdxdy k_interp(ii, interpolator_var::d2ezdxdy)
//  #define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
//  #define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
//  #define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)
//
//  // copy local memmbers from grid
//  //auto nfaces_per_voxel = 6;
//  //auto nvoxels = g->nv;
//  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
//  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);
//
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::deep_copy(k_nm, 0);
//
////  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(const size_t p_index) {
//  for(size_t p_index=0; p_index<np; p_index++) {
//    float v0, v1, v2, v3, v4, v5;
//    auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//
//    // Load position
//    float dx = p_dx;
//    float dy = p_dy;
//    float dz = p_dz;
//    int ii = pii;
//
//    float hax = qdt_2mc*( (f_ex + dy*f_dexdy ) + dz*(f_dexdz + dy*f_d2exdydz) );
//    float hay = qdt_2mc*( (f_ey + dz*f_deydz ) + dx*(f_deydx + dz*f_d2eydzdx) );
//    float haz = qdt_2mc*( (f_ez + dx*f_dezdx ) + dy*(f_dezdy + dx*f_d2ezdxdy) );
//
//    // Interpolate B
//    float cbx = f_cbx + dx*f_dcbxdx;
//    float cby = f_cby + dy*f_dcbydy;
//    float cbz = f_cbz + dz*f_dcbzdz;
//
//    // Load momentum
//    float ux = p_ux;
//    float uy = p_uy;
//    float uz = p_uz;
//    float q = p_w;
//
//    // Half advance e
//    ux += hax;
//    uy += hay;
//    uz += haz;
//
//    v0 = qdt_2mc/std::sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//
//    // Boris - scalars
//    v1 = cbx*cbx + (cby*cby + cbz*cbz);
//    v2 = (v0*v0)*v1;
//    v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
//    v4 = v3/(one+v1*(v3*v3));
//    v4 += v4;
//    // Boris - uprime
//    v0 = ux + v3*(uy*cbz - uz*cby);
//    v1 = uy + v3*(uz*cbx - ux*cbz);
//    v2 = uz + v3*(ux*cby - uy*cbx);
//    // Boris - rotation
//    ux += v4*(v1*cbz - v2*cby);
//    uy += v4*(v2*cbx - v0*cbz);
//    uz += v4*(v0*cby - v1*cbx);
//    // Half advance e
//    ux += hax;
//    uy += hay;
//    uz += haz;
//    // Store momentum
//    p_ux = ux;
//    p_uy = uy;
//    p_uz = uz;
//
//    v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));
//
//    /**/                                      // Get norm displacement
//    ux  *= cdt_dx;
//    uy  *= cdt_dy;
//    uz  *= cdt_dz;
//    ux  *= v0;
//    uy  *= v0;
//    uz  *= v0;
//    v0   = dx + ux;                           // Streak midpoint (inbnds)
//    v1   = dy + uy;
//    v2   = dz + uz;
//    v3   = v0 + ux;                           // New position
//    v4   = v1 + uy;
//    v5   = v2 + uz;
//
//    bool inbnds = v3<=one &&  v4<=one &&  v5<=one &&
//                  -v3<=one && -v4<=one && -v5<=one;
//    if(inbnds) {
//      // Common case (inbnds).  Note: accumulator values are 4 times
//      // the total physical charge that passed through the appropriate
//      // current quadrant in a time-step
//
//      q *= qsp;
//      p_dx = v3;                             // Store new position
//      p_dy = v4;
//      p_dz = v5;
//      dx = v0;                                // Streak midpoint
//      dy = v1;
//      dz = v2;
//      v5 = q*ux*uy*uz*one_third;              // Compute correction
//
//#     define ACCUMULATE_J(X,Y,Z)                                 \
//      v4  = q*u##X;   /* v2 = q ux                            */        \
//      v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//      v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//      v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//      v4  = one+d##Z; /* v4 = 1+dz                            */        \
//      v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//      v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//      v4  = one-d##Z; /* v4 = 1-dz                            */        \
//      v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//      v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//      v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//      v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//      v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//      v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//      ACCUMULATE_J( x,y,z );
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;
//
//      ACCUMULATE_J( y,z,x );
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;
//
//      ACCUMULATE_J( z,x,y );
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
//#     undef ACCUMULATE_J
//    } else {
//      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//      local_pm->dispx = ux;
//      local_pm->dispy = uy;
//      local_pm->dispz = uz;
//      local_pm->i     = p_index;
//
//      if( move_p_kokkos( k_particles, k_particles_i, local_pm,
//                         k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
//        if( k_nm(0)<max_nm ) {
//          const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//          if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//          k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
//          k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
//          k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
//          k_particle_movers_i(nm)   = local_pm->i;
//
//          // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//          k_particle_copy(nm, particle_var::dx) = p_dx;
//          k_particle_copy(nm, particle_var::dy) = p_dy;
//          k_particle_copy(nm, particle_var::dz) = p_dz;
//          k_particle_copy(nm, particle_var::ux) = p_ux;
//          k_particle_copy(nm, particle_var::uy) = p_uy;
//          k_particle_copy(nm, particle_var::uz) = p_uz;
//          k_particle_copy(nm, particle_var::w) = p_w;
//          k_particle_i_copy(nm) = pii;
//        }
//      }
//    }
//  }
////  });
//#undef p_dx
//#undef p_dy
//#undef p_dz
//#undef p_ux
//#undef p_uy
//#undef p_uz
//#undef p_w 
//#undef pii 
//
//#undef f_cbx
//#undef f_cby
//#undef f_cbz
//#undef f_ex 
//#undef f_ey 
//#undef f_ez 
//
//#undef f_dexdy
//#undef f_dexdz
//
//#undef f_d2exdydz
//#undef f_deydx   
//#undef f_deydz   
//
//#undef f_d2eydzdx
//#undef f_dezdx   
//#undef f_dezdy   
//
//#undef f_d2ezdxdy
//#undef f_dcbxdx  
//#undef f_dcbydy  
//#undef f_dcbzdz  
//}

void
advance_p_kokkos_devel(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_accumulators_sa_t k_accumulators_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_iterator_t& k_nm,
        k_neighbor_t& k_neighbors,
        const grid_t *g,
        const float qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int na,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux)
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define p_w     k_particles(p_index, particle_var::w)
  #define pii     k_particles_i(p_index)

  #define f_cbx k_interp(ii, interpolator_var::cbx)
  #define f_cby k_interp(ii, interpolator_var::cby)
  #define f_cbz k_interp(ii, interpolator_var::cbz)
  #define f_ex  k_interp(ii, interpolator_var::ex)
  #define f_ey  k_interp(ii, interpolator_var::ey)
  #define f_ez  k_interp(ii, interpolator_var::ez)

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

  // copy local memmbers from grid
  //auto nfaces_per_voxel = 6;
  //auto nvoxels = g->nv;
  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(const size_t p_index) {
    float v0, v1, v2, v3, v4, v5;
    auto  k_accumulators_scatter_access = k_accumulators_sa.access();

    // Load position
    float dx = p_dx;
    float dy = p_dy;
    float dz = p_dz;
    int ii = pii;

    float hax = qdt_2mc*( (f_ex + dy*f_dexdy ) + dz*(f_dexdz + dy*f_d2exdydz) );
    float hay = qdt_2mc*( (f_ey + dz*f_deydz ) + dx*(f_deydx + dz*f_d2eydzdx) );
    float haz = qdt_2mc*( (f_ez + dx*f_dezdx ) + dy*(f_dezdy + dx*f_d2ezdxdy) );

    // Interpolate B
    float cbx = f_cbx + dx*f_dcbxdx;
    float cby = f_cby + dy*f_dcbydy;
    float cbz = f_cbz + dz*f_dcbzdz;

    // Load momentum
    float ux = p_ux;
    float uy = p_uy;
    float uz = p_uz;
    float q = p_w;

    // Half advance e
    ux += hax;
    uy += hay;
    uz += haz;

    v0 = qdt_2mc/std::sqrt(one + (ux*ux + (uy*uy + uz*uz)));

    // Boris - scalars
    v1 = cbx*cbx + (cby*cby + cbz*cbz);
    v2 = (v0*v0)*v1;
    v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4 = v3/(one+v1*(v3*v3));
    v4 += v4;
    // Boris - uprime
    v0 = ux + v3*(uy*cbz - uz*cby);
    v1 = uy + v3*(uz*cbx - ux*cbz);
    v2 = uz + v3*(ux*cby - uy*cbx);
    // Boris - rotation
    ux += v4*(v1*cbz - v2*cby);
    uy += v4*(v2*cbx - v0*cbz);
    uz += v4*(v0*cby - v1*cbx);
    // Half advance e
    ux += hax;
    uy += hay;
    uz += haz;
    // Store momentum
    p_ux = ux;
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

    bool inbnds = v3<=one &&  v4<=one &&  v5<=one &&
                  -v3<=one && -v4<=one && -v5<=one;
    if(inbnds) {
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
      v5 = q*ux*uy*uz*one_third;              // Compute correction

#     define ACCUMULATE_J(X,Y,Z)                                 \
      v4  = q*u##X;   /* v2 = q ux                            */        \
      v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
      v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
      v1 += v4;       /* v1 = q ux (1+dy)                     */        \
      v4  = one+d##Z; /* v4 = 1+dz                            */        \
      v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
      v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
      v4  = one-d##Z; /* v4 = 1-dz                            */        \
      v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
      v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
      v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
      v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
      v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
      v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

      ACCUMULATE_J( x,y,z );
      k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
      k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
      k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
      k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;

      ACCUMULATE_J( y,z,x );
      k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
      k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
      k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
      k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;

      ACCUMULATE_J( z,x,y );
      k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
      k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
      k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
      k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
#     undef ACCUMULATE_J
    } else {
      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
      local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;
      local_pm->i     = p_index;

      if( move_p_kokkos( k_particles, k_particles_i, local_pm,
                         k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
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
  });
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

//void
//advance_p_kokkos(
//        k_particles_t& k_particles,
//        k_particles_i_t& k_particles_i,
//        k_particle_copy_t& k_particle_copy,
//        k_particle_i_copy_t& k_particle_i_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_interpolator_t& k_interp,
//        //k_particle_movers_t k_local_particle_movers,
//        k_iterator_t& k_nm,
//        k_neighbor_t& k_neighbors,
//        const grid_t *g,
//        const float qdt_2mc,
//        const float cdt_dx,
//        const float cdt_dy,
//        const float cdt_dz,
//        const float qsp,
//        const int na,
//        const int np,
//        const int max_nm,
//        const int nx,
//        const int ny,
//        const int nz)
//{
//
//  constexpr float one            = 1.;
//  constexpr float one_third      = 1./3.;
//  constexpr float two_fifteenths = 2./15.;
//
//  #define p_dx    k_particles(p_index, particle_var::dx)
//  #define p_dy    k_particles(p_index, particle_var::dy)
//  #define p_dz    k_particles(p_index, particle_var::dz)
//  #define p_ux    k_particles(p_index, particle_var::ux)
//  #define p_uy    k_particles(p_index, particle_var::uy)
//  #define p_uz    k_particles(p_index, particle_var::uz)
//  #define p_w     k_particles(p_index, particle_var::w)
//  #define pii     k_particles_i(p_index)
//
//  #define f_cbx k_interp(ii(lane), interpolator_var::cbx)
//  #define f_cby k_interp(ii(lane), interpolator_var::cby)
//  #define f_cbz k_interp(ii(lane), interpolator_var::cbz)
//  #define f_ex  k_interp(ii(lane), interpolator_var::ex)
//  #define f_ey  k_interp(ii(lane), interpolator_var::ey)
//  #define f_ez  k_interp(ii(lane), interpolator_var::ez)
//
//  #define f_dexdy    k_interp(ii(lane), interpolator_var::dexdy)
//  #define f_dexdz    k_interp(ii(lane), interpolator_var::dexdz)
//
//  #define f_d2exdydz k_interp(ii(lane), interpolator_var::d2exdydz)
//  #define f_deydx    k_interp(ii(lane), interpolator_var::deydx)
//  #define f_deydz    k_interp(ii(lane), interpolator_var::deydz)
//
//  #define f_d2eydzdx k_interp(ii(lane), interpolator_var::d2eydzdx)
//  #define f_dezdx    k_interp(ii(lane), interpolator_var::dezdx)
//  #define f_dezdy    k_interp(ii(lane), interpolator_var::dezdy)
//
//  #define f_d2ezdxdy k_interp(ii(lane), interpolator_var::d2ezdxdy)
//  #define f_dcbxdx   k_interp(ii(lane), interpolator_var::dcbxdx)
//  #define f_dcbydy   k_interp(ii(lane), interpolator_var::dcbydy)
//  #define f_dcbzdz   k_interp(ii(lane), interpolator_var::dcbzdz)
//
//  // copy local memmbers from grid
//  //auto nfaces_per_voxel = 6;
//  //auto nvoxels = g->nv;
//  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
//  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);
//
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::deep_copy(k_nm, 0);
//
//  Kokkos::View<float[16]> v0("v0"), v1("v1"), v2("v2"), v3("v3"), v4("v4"), v5("v5");
//  Kokkos::View<float[16]> dx("dx"), dy("dy"), dz("dz"), ux("ux"), uy("uy"), uz("uz");
//  Kokkos::View<float[16]> hax("hax"), hay("hay"), haz("haz"), cbx("cbx"), cby("cby"), cbz("cbz");
//  Kokkos::View<float[16]> q("q");
//  Kokkos::View<int[16]> ii("ii");
//  Kokkos::View<bool[16]> inbnds("inbnds");
//
//  constexpr int num_leagues = 1;
//  constexpr int num_threads = 1;
//  constexpr int num_lanes = 16;
//  int num_chunks = np/num_lanes;
////  if(np%num_lanes > 0)
////    num_chunks++;
//  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, num_lanes).set_scratch_size(0, Kokkos::PerThread(21*num_lanes*4));
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
//  typedef Kokkos::View<float[num_lanes], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_float;
//  typedef Kokkos::View<int[num_lanes], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_int;
//  typedef Kokkos::View<bool[num_lanes], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_bool;
////Kokkos::parallel_for("advance_p", Kokkos::RangePolicy<>(0,num_chunks), KOKKOS_LAMBDA(const size_t chunk) {
//  Kokkos::parallel_for("advance_p", policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
////    simd_float v0(team_member.team_scratch(0));
////    simd_float v1(team_member.team_scratch(0));
////    simd_float v2(team_member.team_scratch(0));
////    simd_float v3(team_member.team_scratch(0));
////    simd_float v4(team_member.team_scratch(0));
////    simd_float v5(team_member.team_scratch(0));
////    simd_float dx(team_member.team_scratch(0));
////    simd_float dy(team_member.team_scratch(0));
////    simd_float dz(team_member.team_scratch(0));
////    simd_float ux(team_member.team_scratch(0));
////    simd_float uy(team_member.team_scratch(0));
////    simd_float uz(team_member.team_scratch(0));
////    simd_float hax(team_member.team_scratch(0));
////    simd_float hay(team_member.team_scratch(0));
////    simd_float haz(team_member.team_scratch(0));
////    simd_float cbx(team_member.team_scratch(0));
////    simd_float cby(team_member.team_scratch(0));
////    simd_float cbz(team_member.team_scratch(0));
////    simd_int   ii(team_member.team_scratch(0));
////    simd_float q(team_member.team_scratch(0));
////    simd_bool  inbnds(team_member.team_scratch(0));
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_chunks), [=] (size_t chunk) {
//      auto  k_accumulators_scatter_access = k_accumulators_sa.access();
////      for(int lane=0; lane<num_lanes; lane++) {
////        int p_index = chunk*num_lanes + lane;
////        dx(lane) = p_dx;
////        dy(lane) = p_dy;
////        dz(lane) = p_dz;
////        ux(lane) = p_ux;
////        uy(lane) = p_uy;
////        uz(lane) = p_uz;
////        q(lane) = p_w;
////        ii(lane) = pii;
////      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        int p_index = chunk*num_lanes + lane;
//        // Load position
//        dx(lane) = p_dx;
//        dy(lane) = p_dy;
//        dz(lane) = p_dz;
//        // Load momentum
//        ux(lane) = p_ux;
//        uy(lane) = p_uy;
//        uz(lane) = p_uz;
//        q(lane) = p_w;
//        ii(lane) = pii;
//
//        hax(lane) = qdt_2mc*( (f_ex + dy(lane)*f_dexdy ) + dz(lane)*(f_dexdz + dy(lane)*f_d2exdydz) );
//        hay(lane) = qdt_2mc*( (f_ey + dz(lane)*f_deydz ) + dx(lane)*(f_deydx + dz(lane)*f_d2eydzdx) );
//        haz(lane) = qdt_2mc*( (f_ez + dx(lane)*f_dezdx ) + dy(lane)*(f_dezdy + dx(lane)*f_d2ezdxdy) );
//
//        // Interpolate B
//        cbx(lane) = f_cbx + dx(lane)*f_dcbxdx;
//        cby(lane) = f_cby + dy(lane)*f_dcbydy;
//        cbz(lane) = f_cbz + dz(lane)*f_dcbzdz;
//
//        // Half advance e
//        ux(lane) += hax(lane);
//        uy(lane) += hay(lane);
//        uz(lane) += haz(lane);
//      }
//
//      for(int lane=0; lane<num_lanes; lane++) {
//        v0(lane) = qdt_2mc/sqrtf(one + (ux(lane)*ux(lane) + (uy(lane)*uy(lane) + uz(lane)*uz(lane))));
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        int p_index = chunk*num_lanes + lane;
//        // Boris - scalars
//        v1(lane) = cbx(lane)*cbx(lane) + (cby(lane)*cby(lane) + cbz(lane)*cbz(lane));
//        v2(lane) = (v0(lane)*v0(lane))*v1(lane);
//        v3(lane) = v0(lane)*(one+v2(lane)*(one_third+v2(lane)*two_fifteenths));
//        v4(lane) = v3(lane)/(one+v1(lane)*(v3(lane)*v3(lane)));
//        v4(lane) += v4(lane);
//        // Boris - uprime
//        v0(lane) = ux(lane) + v3(lane)*(uy(lane)*cbz(lane) - uz(lane)*cby(lane));
//        v1(lane) = uy(lane) + v3(lane)*(uz(lane)*cbx(lane) - ux(lane)*cbz(lane));
//        v2(lane) = uz(lane) + v3(lane)*(ux(lane)*cby(lane) - uy(lane)*cbx(lane));
//        // Boris - rotation
//        ux(lane) += v4(lane)*(v1(lane)*cbz(lane) - v2(lane)*cby(lane));
//        uy(lane) += v4(lane)*(v2(lane)*cbx(lane) - v0(lane)*cbz(lane));
//        uz(lane) += v4(lane)*(v0(lane)*cby(lane) - v1(lane)*cbx(lane));
//        // Half advance e
//        ux(lane) += hax(lane);
//        uy(lane) += hay(lane);
//        uz(lane) += haz(lane);
//        // Store momentum
//        p_ux = ux(lane);
//        p_uy = uy(lane);
//        p_uz = uz(lane);
//      }
//
//      for(int lane=0; lane<num_lanes; lane++) {
//        v0(lane)   = one/sqrtf(one + (ux(lane)*ux(lane)+ (uy(lane)*uy(lane) + uz(lane)*uz(lane))));
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        /**/                                      // Get norm displacement
//        ux(lane)  *= cdt_dx;
//        uy(lane)  *= cdt_dy;
//        uz(lane)  *= cdt_dz;
//        ux(lane)  *= v0(lane);
//        uy(lane)  *= v0(lane);
//        uz(lane)  *= v0(lane);
//        v0(lane)   = dx(lane) + ux(lane);                           // Streak midpoint (inbnds)
//        v1(lane)   = dy(lane) + uy(lane);
//        v2(lane)   = dz(lane) + uz(lane);
//        v3(lane)   = v0(lane) + ux(lane);                           // New position
//        v4(lane)   = v1(lane) + uy(lane);
//        v5(lane)   = v2(lane) + uz(lane);
//
//        inbnds(lane) = v3(lane)<=one &&  v4(lane)<=one &&  v5(lane)<=one &&
//                      -v3(lane)<=one && -v4(lane)<=one && -v5(lane)<=one;
//      }
//      for(int lane=0; lane<num_lanes; lane++) {
//        int p_index = chunk*num_lanes + lane;
//        if(inbnds(lane)) {
//          // Common case (inbnds).  Note: accumulator values are 4 times
//          // the total physical charge that passed through the appropriate
//          // current quadrant in a time-step
//
//          q(lane) *= qsp;
//          p_dx = v3(lane);                             // Store new position
//          p_dy = v4(lane);
//          p_dz = v5(lane);
//          dx(lane) = v0(lane);                                // Streak midpoint
//          dy(lane) = v1(lane);
//          dz(lane) = v2(lane);
//          v5(lane) = q(lane)*ux(lane)*uy(lane)*uz(lane)*one_third;              // Compute correction
//
//#         define ACCUMULATE_J(X,Y,Z)                                 \
//          v4(lane)  = q(lane)*u##X(lane);   /* v2 = q ux                            */        \
//          v1(lane)  = v4(lane)*d##Y(lane);  /* v1 = q ux dy                         */        \
//          v0(lane)  = v4(lane)-v1(lane);    /* v0 = q ux (1-dy)                     */        \
//          v1(lane) += v4(lane);       /* v1 = q ux (1+dy)                     */        \
//          v4(lane)  = one+d##Z(lane); /* v4 = 1+dz                            */        \
//          v2(lane)  = v0(lane)*v4(lane);    /* v2 = q ux (1-dy)(1+dz)               */        \
//          v3(lane)  = v1(lane)*v4(lane);    /* v3 = q ux (1+dy)(1+dz)               */        \
//          v4(lane)  = one-d##Z(lane); /* v4 = 1-dz                            */        \
//          v0(lane) *= v4(lane);       /* v0 = q ux (1-dy)(1-dz)               */        \
//          v1(lane) *= v4(lane);       /* v1 = q ux (1+dy)(1-dz)               */        \
//          v0(lane) += v5(lane);       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//          v1(lane) -= v5(lane);       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//          v2(lane) -= v5(lane);       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//          v3(lane) += v5(lane);       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//          ACCUMULATE_J( x,y,z );
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jx, 0) += v0(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jx, 1) += v1(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jx, 2) += v2(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jx, 3) += v3(lane);
//
//          ACCUMULATE_J( y,z,x );
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jy, 0) += v0(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jy, 1) += v1(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jy, 2) += v2(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jy, 3) += v3(lane);
//
//          ACCUMULATE_J( z,x,y );
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jz, 0) += v0(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jz, 1) += v1(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jz, 2) += v2(lane);
//          k_accumulators_scatter_access(ii(lane), accumulator_var::jz, 3) += v3(lane);
//#         undef ACCUMULATE_J
//        } else {
//          DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//          local_pm->dispx = ux(lane);
//          local_pm->dispy = uy(lane);
//          local_pm->dispz = uz(lane);
//          local_pm->i     = p_index;
//
//          if( move_p_kokkos( k_particles, k_particles_i, local_pm,
//                             k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
//            if( k_nm(0)<max_nm ) {
//              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//              if (nm >= max_nm) Kokkos::abort("overran max_nm");
//
//              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
//              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
//              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
//              k_particle_movers_i(nm)   = local_pm->i;
//
//              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
//              k_particle_copy(nm, particle_var::dx) = p_dx;
//              k_particle_copy(nm, particle_var::dy) = p_dy;
//              k_particle_copy(nm, particle_var::dz) = p_dz;
//              k_particle_copy(nm, particle_var::ux) = p_ux;
//              k_particle_copy(nm, particle_var::uy) = p_uy;
//              k_particle_copy(nm, particle_var::uz) = p_uz;
//              k_particle_copy(nm, particle_var::w) = p_w;
//              k_particle_i_copy(nm) = pii;
//            }
//          }
//        }
//      }
//    });
//  });
////});
//#undef p_dx
//#undef p_dy
//#undef p_dz
//#undef p_ux
//#undef p_uy
//#undef p_uz
//#undef p_w 
//#undef pii 
//
//#undef f_cbx
//#undef f_cby
//#undef f_cbz
//#undef f_ex 
//#undef f_ey 
//#undef f_ez 
//
//#undef f_dexdy
//#undef f_dexdz
//
//#undef f_d2exdydz
//#undef f_deydx   
//#undef f_deydz   
//
//#undef f_d2eydzdx
//#undef f_dezdx   
//#undef f_dezdy   
//
//#undef f_d2ezdxdy
//#undef f_dcbxdx  
//#undef f_dcbydy  
//#undef f_dcbzdz  
//}

void
advance_p( /**/  species_t            * RESTRICT sp,
           /**/  accumulator_array_t  * RESTRICT aa,
           interpolator_array_t * RESTRICT ia ) {
  //DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );
  //DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE+1 );
  //int rank;

  if( !sp )
  {
    ERROR(( "Bad args" ));
  }
  if( !aa )
  {
    ERROR(( "Bad args" ));
  }
  if( !ia  )
  {
    ERROR(( "Bad args" ));
  }
  if( sp->g!=aa->g )
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

  KOKKOS_TIC();
  advance_p_kokkos_omp_simd(
//  advance_p_kokkos_vector(
//  advance_p_kokkos_devel(
          sp->k_p_d,
          sp->k_p_i_d,
          sp->k_pc_d,
          sp->k_pc_i_d,
          sp->k_pm_d,
          sp->k_pm_i_d,
          aa->k_a_sa,
          ia->k_i_d,
          sp->k_nm_d,
          sp->g->k_neighbor_d,
          sp->g,
          qdt_2mc,
          cdt_dx,
          cdt_dy,
          cdt_dz,
          sp->q,
          aa->na,
          sp->np,
          sp->max_nm,
          sp->g->nx,
          sp->g->ny,
          sp->g->nz
  );
  KOKKOS_TOC( advance_p, 1);

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

  //print_nm(sp->k_pm_d, nm);

/*
  args->p0       = sp->p;
  args->pm       = sp->pm;
  args->a0       = aa->a;
  args->f0       = ia->i;
  args->seg      = seg;
  args->g        = sp->g;

  args->qdt_2mc  = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  args->cdt_dx   = sp->g->cvac*sp->g->dt*sp->g->rdx;
  args->cdt_dy   = sp->g->cvac*sp->g->dt*sp->g->rdy;
  args->cdt_dz   = sp->g->cvac*sp->g->dt*sp->g->rdz;
  args->qsp      = sp->q;

  args->np       = sp->np;
  args->max_nm   = sp->max_nm;
  args->nx       = sp->g->nx;
  args->ny       = sp->g->ny;
  args->nz       = sp->g->nz;

  // Have the host processor do the last incomplete bundle if necessary.
  // Note: This is overlapped with the pipelined processing.  As such,
  // it uses an entire accumulator.  Reserving an entire accumulator
  // for the host processor to handle at most 15 particles is wasteful
  // of memory.  It is anticipated that it may be useful at some point
  // in the future have pipelines accumulating currents while the host
  // processor is doing other more substantive work (e.g. accumulating
  // currents from particles received from neighboring nodes).
  // However, it is worth reconsidering this at some point in the
  // future.

  EXEC_PIPELINES( advance_p, args, 0 );
  // of memory.  It is anticipated that it may be useful at some point
  // in the future have pipelines accumulating currents while the host
  // processor is doing other more substantive work (e.g. accumulating
  // currents from particles received from neighboring nodes).
  // However, it is worth reconsidering this at some point in the
  // future.

  EXEC_PIPELINES( advance_p, args, 0 );
  WAIT_PIPELINES();

  // FIXME: HIDEOUS HACK UNTIL BETTER PARTICLE MOVER SEMANTICS
  // INSTALLED FOR DEALING WITH PIPELINES.  COMPACT THE PARTICLE
  // MOVERS TO ELIMINATE HOLES FROM THE PIPELINING.


  sp->nm = 0;
  for( rank=0; rank<=N_PIPELINE; rank++ ) {
    if( args->seg[rank].n_ignored )
      WARNING(( "Pipeline %i ran out of storage for %i movers",
                rank, args->seg[rank].n_ignored ));
    if( sp->pm+sp->nm != args->seg[rank].pm )
      MOVE( sp->pm+sp->nm, args->seg[rank].pm, args->seg[rank].nm );
    sp->nm += args->seg[rank].nm;
  }
  */
  //sp->nm = nm;
}
