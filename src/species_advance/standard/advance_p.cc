// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/kokkos_tuning.hpp"

//template<class TeamMember, class accumulator_sa_t, class accumulator_var>
//void
//KOKKOS_INLINE_FUNCTION
//contribute_current(TeamMember& team_member, accumulator_sa_t& access, int ii, accumulator_var j, float v0, float v1,  float v2, float v3) {
//#ifdef __CUDA_ARCH__
//  int mask = 0xffffffff;
//  int team_rank = team_member.team_rank();
//  for(int i=16; i>0; i=i/2) {
//    v0 += __shfl_down_sync(mask, v0, i);
//    v1 += __shfl_down_sync(mask, v1, i);
//    v2 += __shfl_down_sync(mask, v2, i);
//    v3 += __shfl_down_sync(mask, v3, i);
//  }
//  if(team_rank%32 == 0) {
//    access(ii, j, 0) += v0;
//    access(ii, j, 1) += v1;
//    access(ii, j, 2) += v2;
//    access(ii, j, 3) += v3;
//  }
//#else
//  team_member.team_reduce(Kokkos::Sum<float>(v0));
//  team_member.team_reduce(Kokkos::Sum<float>(v1));
//  team_member.team_reduce(Kokkos::Sum<float>(v2));
//  team_member.team_reduce(Kokkos::Sum<float>(v3));
//  if(team_member.team_rank() == 0) {
//    access(ii, j, 0) += v0;
//    access(ii, j, 1) += v1;
//    access(ii, j, 2) += v2;
//    access(ii, j, 3) += v3;
//  }
//#endif
//}
//
//void
//advance_p_kokkos_vector(
//        k_particles_t& k_particles,
//        k_particles_i_t& k_particles_i,
//        k_particle_copy_t& k_particle_copy,
//        k_particle_i_copy_t& k_particle_i_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
////        k_accumulators_sa_t k_accumulators_sa,
//        k_field_sa_t k_f_sa,
//        k_interpolator_t& k_interp,
//        //k_particle_movers_t k_local_particle_movers,
//        k_counter_t& k_nm,
//        k_neighbor_t& k_neighbors,
//        field_array_t* RESTRICT fa,
//        const grid_t *g,
//        const float qdt_2mc,
//        const float cdt_dx,
//        const float cdt_dy,
//        const float cdt_dz,
//        const float qsp,
////        const int na,
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
//  k_field_t k_field = fa->k_f_d;
//  k_field_sa_t k_f_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
//  float cx = 0.25 * g->rdy * g->rdz / g->dt;
//  float cy = 0.25 * g->rdz * g->rdx / g->dt;
//  float cz = 0.25 * g->rdx * g->rdy / g->dt;
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
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::deep_copy(k_nm, 0);
//
//  int num_leagues = g->nv/4;
//  constexpr int num_lanes = 16;
//
//  int num_chunks = np/num_lanes;
//  int chunks_per_league = num_chunks/num_leagues;
//
//  Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(num_leagues, Kokkos::AUTO(), num_lanes).set_scratch_size(0, Kokkos::PerThread(39*num_lanes*4));
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
//  typedef Kokkos::View<float[num_lanes], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_float;
//  typedef Kokkos::View<int[num_lanes],   ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> simd_int;
//  Kokkos::parallel_for("advance_p", policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    simd_float v0(team_member.thread_scratch(0));
//    simd_float v1(team_member.thread_scratch(0));
//    simd_float v2(team_member.thread_scratch(0));
//    simd_float v3(team_member.thread_scratch(0));
//    simd_float v4(team_member.thread_scratch(0));
//    simd_float v5(team_member.thread_scratch(0));
//    simd_float dx(team_member.thread_scratch(0));
//    simd_float dy(team_member.thread_scratch(0));
//    simd_float dz(team_member.thread_scratch(0));
//    simd_float ux(team_member.thread_scratch(0));
//    simd_float uy(team_member.thread_scratch(0));
//    simd_float uz(team_member.thread_scratch(0));
//    simd_float hax(team_member.thread_scratch(0));
//    simd_float hay(team_member.thread_scratch(0));
//    simd_float haz(team_member.thread_scratch(0));
//    simd_float cbx(team_member.thread_scratch(0));
//    simd_float cby(team_member.thread_scratch(0));
//    simd_float cbz(team_member.thread_scratch(0));
//    simd_int   ii(team_member.thread_scratch(0));
//    simd_float q(team_member.thread_scratch(0));
//    simd_int   inbnds(team_member.thread_scratch(0));
//
//    simd_float fcbx(team_member.thread_scratch(0));
//    simd_float fcby(team_member.thread_scratch(0));
//    simd_float fcbz(team_member.thread_scratch(0));
//    simd_float fex(team_member.thread_scratch(0));
//    simd_float fey(team_member.thread_scratch(0));
//    simd_float fez(team_member.thread_scratch(0));
//    simd_float fdexdy(team_member.thread_scratch(0));
//    simd_float fdexdz(team_member.thread_scratch(0));
//    simd_float fd2exdydz(team_member.thread_scratch(0));
//    simd_float fdeydx(team_member.thread_scratch(0));
//    simd_float fdeydz(team_member.thread_scratch(0));
//    simd_float fd2eydzdx(team_member.thread_scratch(0));
//    simd_float fdezdx(team_member.thread_scratch(0));
//    simd_float fdezdy(team_member.thread_scratch(0));
//    simd_float fd2ezdxdy(team_member.thread_scratch(0));
//    simd_float fdcbxdx(team_member.thread_scratch(0));
//    simd_float fdcbydy(team_member.thread_scratch(0));
//    simd_float fdcbzdz(team_member.thread_scratch(0));
//
////    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_chunks), [=] (size_t chunk) {
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, chunks_per_league), [=] (size_t chunk_offset) {
//      auto k_field_scatter_access = k_f_sv.access();
//      size_t chunk = team_member.league_rank()*chunks_per_league + chunk_offset;
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        size_t p_index = chunk*num_lanes + lane;
//        // Load position
//        dx[lane] = p_dx;
//        dy[lane] = p_dy;
//        dz[lane] = p_dz;
//  
//        // Load momentum
//        ux[lane] = p_ux;
//        uy[lane] = p_uy;
//        uz[lane] = p_uz;
//        q[lane] = p_w;
//
//        // Load index
//        ii[lane] = pii;
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        // Load interpolators
//        fex[lane]       = f_ex;     
//        fdexdy[lane]    = f_dexdy;  
//        fdexdz[lane]    = f_dexdz;  
//        fd2exdydz[lane] = f_d2exdydz;
//        fey[lane]       = f_ey;     
//        fdeydz[lane]    = f_deydz;  
//        fdeydx[lane]    = f_deydx;  
//        fd2eydzdx[lane] = f_d2eydzdx;
//        fez[lane]       = f_ez;     
//        fdezdx[lane]    = f_dezdx;  
//        fdezdy[lane]    = f_dezdy;  
//        fd2ezdxdy[lane] = f_d2ezdxdy;
//        fcbx[lane]      = f_cbx;    
//        fdcbxdx[lane]   = f_dcbxdx; 
//        fcby[lane]      = f_cby;    
//        fdcbydy[lane]   = f_dcbydy; 
//        fcbz[lane]      = f_cbz;    
//        fdcbzdz[lane]   = f_dcbzdz; 
//      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        size_t p_index = chunk*num_lanes + lane;
//  
////        hax[lane] = qdt_2mc*( (f_ex + dy[lane]*f_dexdy ) + dz[lane]*(f_dexdz + dy[lane]*f_d2exdydz) );
////        hay[lane] = qdt_2mc*( (f_ey + dz[lane]*f_deydz ) + dx[lane]*(f_deydx + dz[lane]*f_d2eydzdx) );
////        haz[lane] = qdt_2mc*( (f_ez + dx[lane]*f_dezdx ) + dy[lane]*(f_dezdy + dx[lane]*f_d2ezdxdy) );
//        hax[lane] = qdt_2mc*( (fex[lane] + dy[lane]*fdexdy[lane] ) + dz[lane]*(fdexdz[lane] + dy[lane]*fd2exdydz[lane]) );
//        hay[lane] = qdt_2mc*( (fey[lane] + dz[lane]*fdeydz[lane] ) + dx[lane]*(fdeydx[lane] + dz[lane]*fd2eydzdx[lane]) );
//        haz[lane] = qdt_2mc*( (fez[lane] + dx[lane]*fdezdx[lane] ) + dy[lane]*(fdezdy[lane] + dx[lane]*fd2ezdxdy[lane]) );
//  
//        // Interpolate B
////        cbx[lane] = f_cbx + dx[lane]*f_dcbxdx;
////        cby[lane] = f_cby + dy[lane]*f_dcbydy;
////        cbz[lane] = f_cbz + dz[lane]*f_dcbzdz;
//        cbx[lane] = fcbx[lane] + dx[lane]*fdcbxdx[lane];
//        cby[lane] = fcby[lane] + dy[lane]*fdcbydy[lane];
//        cbz[lane] = fcbz[lane] + dz[lane]*fdcbzdz[lane];
//  
//        // Half advance e
//        ux[lane] += hax[lane];
//        uy[lane] += hay[lane];
//        uz[lane] += haz[lane];
//      }
//  
//#pragma omp simd
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
//#pragma omp simd
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
////        v3[lane] = inbnds[lane] ? v3[lane] : p_dx;
////        v4[lane] = inbnds[lane] ? v4[lane] : p_dy;
////        v5[lane] = inbnds[lane] ? v5[lane] : p_dz;
////        q[lane] = inbnds[lane] ? q[lane]*qsp : 0.0;
//
//        v3[lane] = static_cast<float>(inbnds[lane])*v3[lane] + (1.0-static_cast<float>(inbnds[lane]))*p_dx;
//        v4[lane] = static_cast<float>(inbnds[lane])*v4[lane] + (1.0-static_cast<float>(inbnds[lane]))*p_dy;
//        v5[lane] = static_cast<float>(inbnds[lane])*v5[lane] + (1.0-static_cast<float>(inbnds[lane]))*p_dz;
//        q[lane] = static_cast<float>(inbnds[lane])*q[lane]*qsp;
//
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
//      }
//const simd_float& v6 = fex;
//const simd_float& v7 = fdexdy;
//const simd_float& v8 = fdexdz;
//const simd_float& v9 = fd2exdydz;
//#pragma omp simd
//for(int lane=0; lane<num_lanes; lane++) {
//  v6(lane) = v0(lane);
//  v7(lane) = v1(lane);
//  v8(lane) = v2(lane);
//  v9(lane) = v3(lane);
//}
//
////#pragma omp simd
////      for(int lane=0; lane<num_lanes; lane++) {
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 0) += v0[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 1) += v1[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 2) += v2[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jx, 3) += v3[lane];
////      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//
//        ACCUMULATE_J( y,z,x );
////        v10[lane] = v0[lane];
////        v11[lane] = v1[lane];
////        v12[lane] = v2[lane];
////        v13[lane] = v3[lane];
//      }
//
//const simd_float& v10 = fey;
//const simd_float& v11 = fdeydz;
//const simd_float& v12 = fdeydx;
//const simd_float& v13 = fd2eydzdx;
//#pragma omp simd
//for(int lane=0; lane<num_lanes; lane++) {
//  v10(lane) = v0(lane);
//  v11(lane) = v1(lane);
//  v12(lane) = v2(lane);
//  v13(lane) = v3(lane);
//}
//
////#pragma omp simd
////      for(int lane=0; lane<num_lanes; lane++) {
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 0) += v0[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 1) += v1[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 2) += v2[lane];
////        k_accumulators_scatter_access(ii[lane], accumulator_var::jy, 3) += v3[lane];
////
////      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        ACCUMULATE_J( z,x,y );
////        v14[lane] = v0[lane];
////        v15[lane] = v1[lane];
////        v16[lane] = v2[lane];
////        v17[lane] = v3[lane];
//      }
//
//const simd_float& v14 = fez;
//const simd_float& v15 = fdezdx;
//const simd_float& v16 = fdezdy;
//const simd_float& v17 = fd2ezdxdy;
//#pragma omp simd
//for(int lane=0; lane<num_lanes; lane++) {
//  v14(lane) = v0(lane);
//  v15(lane) = v1(lane);
//  v16(lane) = v2(lane);
//  v17(lane) = v3(lane);
//}
//
////#pragma omp simd
////      for(int lane=0; lane<num_lanes; lane++) {
////        k_field_scatter_access(ii[lane], accumulator_var::jz, 0) += v0[lane];
////        k_field_scatter_access(ii[lane], accumulator_var::jz, 1) += v1[lane];
////        k_field_scatter_access(ii[lane], accumulator_var::jz, 2) += v2[lane];
////        k_field_scatter_access(ii[lane], accumulator_var::jz, 3) += v3[lane];
////      }
//
//#pragma omp simd
//      for(int lane=0; lane<num_lanes; lane++) {
//        // TODO: That 2 needs to be 2*NGHOST eventually
//        int iii = ii(lane);
//        int zi = iii/((nx+2)*(ny+2));
//        iii -= zi*(nx+2)*(ny+2);
//        int yi = iii/(nx+2);
//        int xi = iii - yi*(nx+2);
//        ACCUMULATE_J( x,y,z );
//        k_field_scatter_access(ii(lane), field_var::jfx) += cx*v6(lane);
//        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v7(lane);
//        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v8(lane);
//        k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v9(lane);
//
//        ACCUMULATE_J( y,z,x );
//        k_field_scatter_access(ii(lane), field_var::jfy) += cy*v10(lane);
//        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v11(lane);
//        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v12(lane);
//        k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v13(lane);
//
//        ACCUMULATE_J( z,x,y );
//        k_field_scatter_access(ii(lane), field_var::jfz) += cz*v14(lane);
//        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v15(lane);
//        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v16(lane);
//        k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v17(lane);
//      }
//#       undef ACCUMULATE_J
//
//      for(int lane=0; lane<num_lanes; lane++) {
//        if(!inbnds[lane]) {
//          size_t p_index = chunk*num_lanes + lane;
//          DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//          local_pm->dispx = ux[lane];
//          local_pm->dispy = uy[lane];
//          local_pm->dispz = uz[lane];
//          local_pm->i     = p_index;
//
//          if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
//                             k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
//          {
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
//      auto k_field_scatter_access = k_f_sv.access();
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
//        // TODO: That 2 needs to be 2*NGHOST eventually
//        int iii = ii;
//        int zi = iii/((nx+2)*(ny+2));
//        iii -= zi*(nx+2)*(ny+2);
//        int yi = iii/(nx+2);
//        int xi = iii - yi*(nx+2);
//        ACCUMULATE_J( x,y,z );
//        k_field_scatter_access(ii, field_var::jfx) += cx*v0;
//        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
//        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
//        k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
//
//        ACCUMULATE_J( y,z,x );
//        k_field_scatter_access(ii, field_var::jfy) += cy*v0;
//        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
//        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
//        k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;
//
//        ACCUMULATE_J( z,x,y );
//        k_field_scatter_access(ii, field_var::jfz) += cz*v0;
//        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
//        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
//        k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
//#       undef ACCUMULATE_J
//      } else {
//        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//        local_pm->dispx = ux;
//        local_pm->dispy = uy;
//        local_pm->dispz = uz;
//        local_pm->i     = p_index;
//
//        if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
//                           k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
//        {
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
//  Kokkos::Experimental::contribute(k_field, k_f_sv);
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
//
//void KOKKOS_INLINE_FUNCTION sw(float& x, float& y) {
//  float z = x;
//  x = y;
//  y = z;
//}
//void KOKKOS_INLINE_FUNCTION transpose(float* v0, float* v1, float* v2, float* v3, 
//                                      float* v4, float* v5, float* v6, float* v7) {
//  sw(v0[1], v1[0]); sw(v0[2], v2[0]); sw(v0[3], v3[0]); sw(v0[4], v4[0]); sw(v0[5], v5[0]); sw(v0[6], v6[0]); sw(v0[7], v7[0]);
//                    sw(v1[2], v2[1]); sw(v1[3], v3[1]); sw(v1[4], v4[1]); sw(v1[5], v5[1]); sw(v1[6], v6[1]); sw(v1[7], v7[1]);
//                                      sw(v2[3], v3[2]); sw(v2[4], v4[2]); sw(v2[5], v5[2]); sw(v2[6], v6[2]); sw(v2[7], v7[2]);
//                                                        sw(v3[4], v4[3]); sw(v3[5], v5[3]); sw(v3[6], v6[3]); sw(v3[7], v7[3]);
//                                                                          sw(v4[5], v5[4]); sw(v4[6], v6[4]); sw(v4[7], v7[4]);
//                                                                                            sw(v5[6], v6[5]); sw(v5[7], v7[5]);
//                                                                                                              sw(v6[7], v7[6]);
//}
//
void
advance_p_kokkos_omp_simd(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
        k_field_sa_t k_f_sa,
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
//        const int na,
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
  k_field_sa_t k_f_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

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
  auto k_field_scatter_access = k_f_sv.access();
  for(size_t chunk=0; chunk<num_chunks; chunk++) {
    float v0[num_lanes], v1[num_lanes], v2[num_lanes], v3[num_lanes], v4[num_lanes], v5[num_lanes];
    float v6[num_lanes], v7[num_lanes], v8[num_lanes], v9[num_lanes];
    float v10[num_lanes], v11[num_lanes], v12[num_lanes], v13[num_lanes];
    float dx[num_lanes], dy[num_lanes], dz[num_lanes];
    float ux[num_lanes], uy[num_lanes], uz[num_lanes];
    float hax[num_lanes], hay[num_lanes], haz[num_lanes];
    float cbx[num_lanes], cby[num_lanes], cbz[num_lanes];
    float q[num_lanes];
    int ii[num_lanes];
    int inbnds[num_lanes];

    float fex[num_lanes];      
    float fdexdy[num_lanes];   
    float fdexdz[num_lanes];   
    float fd2exdydz[num_lanes];
    float fey[num_lanes];      
    float fdeydz[num_lanes];   
    float fdeydx[num_lanes];   
    float fd2eydzdx[num_lanes];
    float fez[num_lanes];      
    float fdezdx[num_lanes];   
    float fdezdy[num_lanes];   
    float fd2ezdxdy[num_lanes];
    float fcbx[num_lanes];     
    float fdcbxdx[num_lanes];  
    float fcby[num_lanes];     
    float fdcbydy[num_lanes];  
    float fcbz[num_lanes];     
    float fdcbzdz[num_lanes];  

#pragma omp simd
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
    for(int lane=0; lane<num_lanes; lane++) {
      // Load interpolators
      fex[lane]       = f_ex;     
      fdexdy[lane]    = f_dexdy;  
      fdexdz[lane]    = f_dexdz;  
      fd2exdydz[lane] = f_d2exdydz;
      fey[lane]       = f_ey;     
      fdeydz[lane]    = f_deydz;  
      fdeydx[lane]    = f_deydx;  
      fd2eydzdx[lane] = f_d2eydzdx;
      fez[lane]       = f_ez;     
      fdezdx[lane]    = f_dezdx;  
      fdezdy[lane]    = f_dezdy;  
      fd2ezdxdy[lane] = f_d2ezdxdy;
      fcbx[lane]      = f_cbx;    
      fdcbxdx[lane]   = f_dcbxdx; 
      fcby[lane]      = f_cby;    
      fdcbydy[lane]   = f_dcbydy; 
      fcbz[lane]      = f_cbz;    
      fdcbzdz[lane]   = f_dcbzdz; 
    }

#pragma omp simd
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
//      hax[lane] = qdt_2mc*( (f_ex + dy[lane]*f_dexdy ) + dz[lane]*(f_dexdz + dy[lane]*f_d2exdydz) );
//      hay[lane] = qdt_2mc*( (f_ey + dz[lane]*f_deydz ) + dx[lane]*(f_deydx + dz[lane]*f_d2eydzdx) );
//      haz[lane] = qdt_2mc*( (f_ez + dx[lane]*f_dezdx ) + dy[lane]*(f_dezdy + dx[lane]*f_d2ezdxdy) );
      hax[lane] = qdt_2mc*( (fex[lane] + dy[lane]*fdexdy[lane] ) + dz[lane]*(fdexdz[lane] + dy[lane]*fd2exdydz[lane]) );
      hay[lane] = qdt_2mc*( (fey[lane] + dz[lane]*fdeydz[lane] ) + dx[lane]*(fdeydx[lane] + dz[lane]*fd2eydzdx[lane]) );
      haz[lane] = qdt_2mc*( (fez[lane] + dx[lane]*fdezdx[lane] ) + dy[lane]*(fdezdy[lane] + dx[lane]*fd2ezdxdy[lane]) );

      // Interpolate B
//      cbx[lane] = f_cbx + dx[lane]*f_dcbxdx;
//      cby[lane] = f_cby + dy[lane]*f_dcbydy;
//      cbz[lane] = f_cbz + dz[lane]*f_dcbzdz;
      cbx[lane] = fcbx[lane] + dx[lane]*fdcbxdx[lane];
      cby[lane] = fcby[lane] + dy[lane]*fdcbydy[lane];
      cbz[lane] = fcbz[lane] + dz[lane]*fdcbzdz[lane];

      // Half advance e
      ux[lane] += hax[lane];
      uy[lane] += hay[lane];
      uz[lane] += haz[lane];
    }

#pragma omp simd 
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
      v0[lane] = qdt_2mc/sqrtf(one + (ux[lane]*ux[lane] + (uy[lane]*uy[lane] + uz[lane]*uz[lane])));
    }

#pragma omp simd
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
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
      v0[lane]   = one/sqrtf(one + (ux[lane]*ux[lane]+ (uy[lane]*uy[lane] + uz[lane]*uz[lane])));
    }

#pragma omp simd
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;

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
    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
//      v3[lane] = ( v3[lane] & inbnds[lane] ) | ( dx[lane] & ~inbnds[lane] );
//      v4[lane] = ( v4[lane] & inbnds[lane] ) | ( dy[lane] & ~inbnds[lane] );
//      v5[lane] = ( v5[lane] & inbnds[lane] ) | ( dz[lane] & ~inbnds[lane] );

//      v3[lane] = inbnds[lane] ? v3[lane] : p_dx;
//      v4[lane] = inbnds[lane] ? v4[lane] : p_dy;
//      v5[lane] = inbnds[lane] ? v5[lane] : p_dz;
//      q[lane] = inbnds[lane] ? q[lane]*qsp : 0.0;

      v3[lane] = static_cast<float>(inbnds[lane])*v3[lane] + (1.0-static_cast<float>(inbnds[lane]))*dx[lane];
      v4[lane] = static_cast<float>(inbnds[lane])*v4[lane] + (1.0-static_cast<float>(inbnds[lane]))*dy[lane];
      v5[lane] = static_cast<float>(inbnds[lane])*v5[lane] + (1.0-static_cast<float>(inbnds[lane]))*dz[lane];
      q[lane]  = static_cast<float>(inbnds[lane])*q[lane]*qsp;

      p_dx = v3[lane];
      p_dy = v4[lane];
      p_dz = v5[lane];
      dx[lane] = v0[lane];
      dy[lane] = v1[lane];
      dz[lane] = v2[lane];
      v5[lane] = q[lane]*ux[lane]*uy[lane]*uz[lane]*one_third;
    }

#pragma omp simd
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

//        int iii = ii[lane];
//        int zi = iii/((nx+2)*(ny+2));
//        iii -= zi*(nx+2)*(ny+2);
//        int yi = iii/(nx+2);
//        int xi = iii - yi*(nx+2);
//        ACCUMULATE_J( x,y,z );
//        //Kokkos::atomic_add(&k_field(ii, field_var::jfx), cx*v0);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx), cx*v1);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx), cx*v2);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx), cx*v3);
//        k_field_scatter_access(ii[lane], field_var::jfx) += cx*v0[lane];
//        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1[lane];
//        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2[lane];
//        k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3[lane];
//
//        ACCUMULATE_J( y,z,x );
//        //Kokkos::atomic_add(&k_field(ii, field_var::jfy), cy*v0);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v1);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy), cy*v2);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v3);
//        k_field_scatter_access(ii[lane], field_var::jfy) += cy*v0[lane];
//        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1[lane];
//        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2[lane];
//        k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3[lane];
//
//        ACCUMULATE_J( z,x,y );
//        //Kokkos::atomic_add(&k_field(ii, field_var::jfz), cz*v0);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz), cz*v1);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v2);
//        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v3);
//        k_field_scatter_access(ii[lane], field_var::jfz) += cz*v0[lane];
//        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1[lane];
//        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2[lane];
//        k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3[lane];

      ACCUMULATE_J( x,y,z );
      v6[lane] = v0[lane];
      v7[lane] = v1[lane];
      v8[lane] = v2[lane];
      v9[lane] = v3[lane];

      ACCUMULATE_J( y,z,x );
      v10[lane] = v0[lane];
      v11[lane] = v1[lane];
      v12[lane] = v2[lane];
      v13[lane] = v3[lane];

      ACCUMULATE_J( z,x,y );
//      v14[lane] = v0[lane];
//      v15[lane] = v1[lane];
//      v16[lane] = v2[lane];
//      v17[lane] = v3[lane];
#     undef ACCUMULATE_J
    }

//    transpose(v6, v7, v8, v9, v10, v11, v12, v13);
//    float* a0_ptr = &(k_accumulators(ii[0], accumulator_var::jx, 0));
//    float* a1_ptr = &(k_accumulators(ii[1], accumulator_var::jx, 0));
//    float* a2_ptr = &(k_accumulators(ii[2], accumulator_var::jx, 0));
//    float* a3_ptr = &(k_accumulators(ii[3], accumulator_var::jx, 0));
//    float* a4_ptr = &(k_accumulators(ii[4], accumulator_var::jx, 0));
//    float* a5_ptr = &(k_accumulators(ii[5], accumulator_var::jx, 0));
//    float* a6_ptr = &(k_accumulators(ii[6], accumulator_var::jx, 0));
//    float* a7_ptr = &(k_accumulators(ii[7], accumulator_var::jx, 0));

//#pragma omp simd
//    for(int lane=0; lane<num_lanes; lane++) {
//      a0_ptr[lane] += v6[lane];
//      a1_ptr[lane] += v7[lane];
//      a2_ptr[lane] += v8[lane];
//      a3_ptr[lane] += v9[lane];
//      a4_ptr[lane] += v10[lane];
//      a5_ptr[lane] += v11[lane];
//      a6_ptr[lane] += v12[lane];
//      a7_ptr[lane] += v13[lane];
//    }

    for(int lane=0; lane<num_lanes; lane++) {
      size_t p_index = chunk*num_lanes + lane;
      int iii = ii[lane];
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);

      k_field_scatter_access(ii[lane], field_var::jfx) += cx*v6[lane];
      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v7[lane];
      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v8[lane];
      k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v9[lane];

      k_field_scatter_access(ii[lane], field_var::jfy) += cy*v10[lane];
      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v11[lane];
      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v12[lane];
      k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v13[lane];

      k_field_scatter_access(ii[lane], field_var::jfz) += cz*v0[lane];
      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1[lane];
      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2[lane];
      k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3[lane];
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
                           k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) ) { // Unlikely
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
      auto  k_field_scatter_access = k_f_sv.access();

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

// Bypass accumulators TODO: enable warp reduction

        // TODO: That 2 needs to be 2*NGHOST eventually
        int iii = ii;
        int zi = iii/((nx+2)*(ny+2));
        iii -= zi*(nx+2)*(ny+2);
        int yi = iii/(nx+2);
        int xi = iii - yi*(nx+2);
        ACCUMULATE_J( x,y,z );
        //Kokkos::atomic_add(&k_field(ii, field_var::jfx), cx*v0);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx), cx*v1);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx), cx*v2);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx), cx*v3);
        k_field_scatter_access(ii, field_var::jfx) += cx*v0;
        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
        k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;

        ACCUMULATE_J( y,z,x );
        //Kokkos::atomic_add(&k_field(ii, field_var::jfy), cy*v0);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v1);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy), cy*v2);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v3);
        k_field_scatter_access(ii, field_var::jfy) += cy*v0;
        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
        k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;

        ACCUMULATE_J( z,x,y );
        //Kokkos::atomic_add(&k_field(ii, field_var::jfz), cz*v0);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz), cz*v1);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v2);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v3);
        k_field_scatter_access(ii, field_var::jfz) += cz*v0;
        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
        k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
#       undef ACCUMULATE_J
      } else {
        DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
        local_pm->dispx = ux;
        local_pm->dispy = uy;
        local_pm->dispz = uz;
        local_pm->i     = p_index;

        //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
        if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                       k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
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
    }
  }
  Kokkos::Experimental::contribute(k_field, k_f_sv);
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

void
advance_p_kokkos_devel(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
        k_field_sa_t k_f_sa,
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
//        const int na,
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
  k_field_sa_t k_f_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

  // Process particles for this pipeline

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

  // zero out nm, we could probably do this earlier if we're worried about it
  // slowing things down
  Kokkos::deep_copy(k_nm, 0);

#ifdef VPIC_ENABLE_HIERARCHICAL
  auto team_policy = Kokkos::TeamPolicy<>(LEAGUE_SIZE, TEAM_SIZE);
//printf("Team size: %d\n", team_policy.team_size());
  int per_league = np/LEAGUE_SIZE;
  if(np%LEAGUE_SIZE > 0)
    per_league++;
  Kokkos::parallel_for("advance_p", team_policy, KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [=] (size_t pindex) {
      int p_index = team_member.league_rank()*per_league + pindex;
      if(p_index < np) {
#else
  auto range_policy = Kokkos::RangePolicy<>(0,np);
  Kokkos::parallel_for("advance_p", range_policy, KOKKOS_LAMBDA (size_t p_index) {
#endif
      
    float v0, v1, v2, v3, v4, v5;
    auto  k_field_scatter_access = k_f_sv.access();

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
    int inbnds = v3<=one && v4<=one && v5<=one && -v3<=one && -v4<=one && -v5<=one;
    int min_inbnds = inbnds;
    int max_inbnds = inbnds;
    team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
    team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
    int min_index = ii;
    int max_index = ii;
    team_member.team_reduce(Kokkos::Max<int>(max_index));
    team_member.team_reduce(Kokkos::Min<int>(min_index));
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

// Bypass accumulators TODO: enable warp reduction

      // TODO: That 2 needs to be 2*NGHOST eventually
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);
      ACCUMULATE_J( x,y,z );
      //Kokkos::atomic_add(&k_field(ii, field_var::jfx), cx*v0);
      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx), cx*v1);
      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx), cx*v2);
      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx), cx*v3);
      k_field_scatter_access(ii, field_var::jfx) += cx*v0;
      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
      k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;

      ACCUMULATE_J( y,z,x );
      //Kokkos::atomic_add(&k_field(ii, field_var::jfy), cy*v0);
      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v1);
      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy), cy*v2);
      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v3);
      k_field_scatter_access(ii, field_var::jfy) += cy*v0;
      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
      k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;

      ACCUMULATE_J( z,x,y );
      //Kokkos::atomic_add(&k_field(ii, field_var::jfz), cz*v0);
      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz), cz*v1);
      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v2);
      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v3);
      k_field_scatter_access(ii, field_var::jfz) += cz*v0;
      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
      k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;

// TODO: Make this optimization work with the accumulator bypass
//#ifdef VPIC_ENABLE_TEAM_REDUCTION
//      if(min_inbnds == max_inbnds && min_index == max_index) {
//        ACCUMULATE_J( x,y,z );
//        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jx, v0, v1, v2, v3);
//        ACCUMULATE_J( y,z,x );
//        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jy, v0, v1, v2, v3);
//        ACCUMULATE_J( z,x,y );
//        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jz, v0, v1, v2, v3);
//      } else {
//#endif
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
//#ifdef VPIC_ENABLE_TEAM_REDUCTION
//      }
//#endif

#     undef ACCUMULATE_J
    } else {
      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
      local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;
      local_pm->i     = p_index;

      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
      if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                     k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
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
  Kokkos::Experimental::contribute(k_field, k_f_sv);


  // TODO: abstract this manual data copy
  //Kokkos::deep_copy(h_nm, k_nm);

  //args->seg[pipeline_rank].pm        = pm;
  //args->seg[pipeline_rank].max_nm    = max_nm;
  //args->seg[pipeline_rank].nm        = h_nm(0);
  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
  //delete(k_local_particle_movers_p);
  //return h_nm(0);

}

void
advance_p( /**/  species_t            * RESTRICT sp,
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
          //aa->k_a_sa,
          fa->k_field_sa_d,
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
}
