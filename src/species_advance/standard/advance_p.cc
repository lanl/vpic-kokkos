// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include <Kokkos_OffsetView.hpp>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/vpic.h"
#include "advance_p_cuda.cuh"

//// Half 2
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t k_accumulator,
//        k_interpolator_t& k_interp,
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
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    k_nm(0) = 0;
//  });
//
////  k_particle_movers_soa_t mover_particles(np);
//
////  int num_leagues = 2048;
////  int num_threads = 1024;
////  int per_league = np/(2*num_leagues);
////  if((np/2)%num_leagues > 0)
////    per_league++;
////  int per_thread = per_league/num_threads;
////  Kokkos::parallel_for("advance_p", Kokkos::TeamPolicy<>(num_leagues, num_threads, 1), 
////  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
////    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [=] (size_t pindex) {
////      int p_index = team_member.league_rank()*per_league + pindex;
//
//  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np/2),
//    KOKKOS_LAMBDA (size_t p_index)
//    {
//
//    packed_t v0, v1, v2, v3, v4, v5;
//
////    packed_t one = packed_t(one_f);
////    packed_t one_third = packed_t(one_third_f);
////    packed_t two_fifteenths = packed_t(two_fifteenths_f);
//
//    packed_t dx = packed_t(k_part.dx(p_index*2), k_part.dx(p_index*2+1)); // Load position
//    packed_t dy = packed_t(k_part.dy(p_index*2), k_part.dy(p_index*2+1));
//    packed_t dz = packed_t(k_part.dz(p_index*2), k_part.dz(p_index*2+1));
//    const int pi0 = k_part.i(p_index*2);
//    const int pi1 = k_part.i(p_index*2+1);
//
//    packed_t f_ex        = packed_t(k_interp(pi0, interpolator_var::ex), 
//                                    k_interp(pi1, interpolator_var::ex));
//    packed_t f_ey        = packed_t(k_interp(pi0, interpolator_var::ey), 
//                                    k_interp(pi1, interpolator_var::ey));
//    packed_t f_ez        = packed_t(k_interp(pi0, interpolator_var::ez), 
//                                    k_interp(pi1, interpolator_var::ez));
//
//    packed_t f_dexdy     = packed_t(k_interp(pi0, interpolator_var::dexdy), 
//                                    k_interp(pi1, interpolator_var::dexdy));
//    packed_t f_dexdz     = packed_t(k_interp(pi0, interpolator_var::dexdz), 
//                                    k_interp(pi1, interpolator_var::dexdz));
//    packed_t f_deydz     = packed_t(k_interp(pi0, interpolator_var::deydz), 
//                                    k_interp(pi1, interpolator_var::deydz));
//    packed_t f_deydx     = packed_t(k_interp(pi0, interpolator_var::deydx), 
//                                    k_interp(pi1, interpolator_var::deydx));
//    packed_t f_dezdx     = packed_t(k_interp(pi0, interpolator_var::dezdx), 
//                                    k_interp(pi1, interpolator_var::dezdx));
//    packed_t f_dezdy     = packed_t(k_interp(pi0, interpolator_var::dezdy), 
//                                    k_interp(pi1, interpolator_var::dezdy));
//    packed_t f_d2exdydz  = packed_t(k_interp(pi0, interpolator_var::d2exdydz), 
//                                    k_interp(pi1, interpolator_var::d2exdydz));
//    packed_t f_d2eydzdx  = packed_t(k_interp(pi0, interpolator_var::d2eydzdx), 
//                                    k_interp(pi1, interpolator_var::d2eydzdx));
//    packed_t f_d2ezdxdy  = packed_t(k_interp(pi0, interpolator_var::d2ezdxdy), 
//                                    k_interp(pi1, interpolator_var::d2ezdxdy));
//    packed_t f_cbx       = packed_t(k_interp(pi0, interpolator_var::cbx), 
//                                    k_interp(pi1, interpolator_var::cbx));
//    packed_t f_cby       = packed_t(k_interp(pi0, interpolator_var::cby), 
//                                    k_interp(pi1, interpolator_var::cby));
//    packed_t f_cbz       = packed_t(k_interp(pi0, interpolator_var::cbz), 
//                                    k_interp(pi1, interpolator_var::cbz));
//    packed_t f_dcbxdx    = packed_t(k_interp(pi0, interpolator_var::dcbxdx), 
//                                    k_interp(pi1, interpolator_var::dcbxdx));
//    packed_t f_dcbydy    = packed_t(k_interp(pi0, interpolator_var::dcbydy), 
//                                    k_interp(pi1, interpolator_var::dcbydy));
//    packed_t f_dcbzdz    = packed_t(k_interp(pi0, interpolator_var::dcbzdz), 
//                                    k_interp(pi1, interpolator_var::dcbzdz));
//
//    packed_t hax  = packed_t(qdt_2mc)*( (f_ex    + dy*f_dexdy) +
//                                     dz*(f_dexdz + dy*f_d2exdydz) );
//    packed_t hay  = packed_t(qdt_2mc)*( (f_ey    + dz*f_deydz) +
//                                     dx*(f_deydx + dz*f_d2eydzdx) );
//    packed_t haz  = packed_t(qdt_2mc)*( (f_ez    + dx*f_dezdx) +
//                                     dy*(f_dezdy + dx*f_d2ezdxdy) );
//
////    packed_t cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
////    packed_t cby  = f_cby + dy*f_dcbydy;
////    packed_t cbz  = f_cbz + dz*f_dcbzdz;
//    packed_t cbx  = fma(dx, f_dcbxdx, f_cbx); // Interpolate B
//    packed_t cby  = fma(dy, f_dcbydy, f_cby); 
//    packed_t cbz  = fma(dz, f_dcbzdz, f_cbz); 
//
//    packed_t ux   = packed_t(k_part.ux(p_index*2), k_part.ux(p_index*2+1));       // Load momentum
//    packed_t uy   = packed_t(k_part.uy(p_index*2), k_part.uy(p_index*2+1));
//    packed_t uz   = packed_t(k_part.uz(p_index*2), k_part.uz(p_index*2+1));
//    packed_t q    = packed_t(k_part.w(p_index*2), k_part.w(p_index*2+1));
//
//    ux  += hax;                               // Half advance E
//    uy  += hay;
//    uz  += haz;
//    v0   = packed_t(qdt_2mc)/sqrt(packed_t(one) + (ux*ux + (uy*uy + uz*uz)));
//    v1   = cbx*cbx + (cby*cby + cbz*cbz);
//    v2   = ( v0*v0 ) * v1;
////    v3   = v0*(packed_t(one)+v2*(packed_t(one_third)+v2*two_fifteenths));
////    v4   = v3/(packed_t(one)+v1*(v3*v3));
//    v3   = v0*fma(v2, fma(v2, packed_t(two_fifteenths), packed_t(one_third)), packed_t(one));
//    v4   = v3/(fma(v1, (v3*v3), packed_t(one)));
//    v4  += v4;
//
//    v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
//    v1   = uy + v3*( uz*cbx - ux*cbz );
//    v2   = uz + v3*( ux*cby - uy*cbx );
//
//    ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
//    uy  += v4*( v2*cbx - v0*cbz );
//    uz  += v4*( v0*cby - v1*cbx );
//
//    ux  += hax;                               // Half advance E
//    uy  += hay;
//    uz  += haz;
//
//    k_part.ux(p_index*2)   = ux.low2half();                               // Store momentum
//    k_part.ux(p_index*2+1) = ux.high2half();
//    k_part.uy(p_index*2)   = uy.low2half();
//    k_part.uy(p_index*2+1) = uy.high2half();
//    k_part.uz(p_index*2)   = uz.low2half();
//    k_part.uz(p_index*2+1) = uz.high2half();
//
//    v0   = packed_t(one)/sqrt(packed_t(one) + (ux*ux+ (uy*uy + uz*uz)));
//
//    /**/                                      // Get norm displacement
//    ux  *= packed_t(cdt_dx);
//    uy  *= packed_t(cdt_dy);
//    uz  *= packed_t(cdt_dz);
//
//    ux  *= v0;
//    uy  *= v0;
//    uz  *= v0;
//
//    v0   = dx + ux;                           // Streak midpoint (inbnds)
//    v1   = dy + uy;
//    v2   = dz + uz;
//
//    v3   = v0 + ux;                           // New position
//    v4   = v1 + uy;
//    v5   = v2 + uz;
//
//    packed_t v3_inbnds = leq( v3, packed_t(one));
//            v3_inbnds += leq(-v3, packed_t(one));
//    packed_t v4_inbnds = leq( v4, packed_t(one));
//            v4_inbnds += leq(-v4, packed_t(one));
//    packed_t v5_inbnds = leq( v5, packed_t(one));
//            v5_inbnds += leq(-v5, packed_t(one));
//    packed_t inbnds = eq(v3_inbnds+v4_inbnds+v5_inbnds, packed_t(6.0, 6.0));
//    
//    // Common case (inbnds).  Note: accumulator values are 4 times
//    // the total physical charge that passed through the appropriate
//    // current quadrant in a time-step
//
//    q *= packed_t(qsp);
//    // Store new position
//    if(inbnds.low2float()) {
//      k_part.dx(p_index*2)    = v3.low2half();
//      k_part.dy(p_index*2)    = v4.low2half();
//      k_part.dz(p_index*2)    = v5.low2half();
//    }
//    if(inbnds.high2float()) {
//      k_part.dx(p_index*2+1)  = v3.high2half();
//      k_part.dy(p_index*2+1)  = v4.high2half();
//      k_part.dz(p_index*2+1)  = v5.high2half();
//    }
//
//    dx = v0;                                // Streak midpoint
//    dy = v1;
//    dz = v2;
//    v5 = q*ux*uy*uz*packed_t(one_third);              // Compute correction
//
//#   define ACCUMULATE_J(X,Y,Z)                                        \
//    v4  = q*u##X;   /* v2 = q ux                            */        \
//    v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//    v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//    v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//    v4  = packed_t(one)+d##Z; /* v4 = 1+dz                  */        \
//    v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//    v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//    v4  = packed_t(one)-d##Z; /* v4 = 1-dz                  */        \
//    v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//    v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//    v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//    v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//    v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//    v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
////    if(inbnds.low2float() && inbnds.high2float()) {
////      ACCUMULATE_J( x,y,z );
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 0), v0.low2float()+v0.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 1), v1.low2float()+v1.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 2), v2.low2float()+v2.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 3), v3.low2float()+v3.high2float());
////
////      ACCUMULATE_J( y,z,x );
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 0), v0.low2float()+v0.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 1), v1.low2float()+v1.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 2), v2.low2float()+v2.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 3), v3.low2float()+v3.high2float());
////
////      ACCUMULATE_J( z,x,y );
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 0), v0.low2float()+v0.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 1), v1.low2float()+v1.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 2), v2.low2float()+v2.high2float());
////      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 3), v3.low2float()+v3.high2float());
////    } else {
//    if(inbnds.low2float()) {
//      ACCUMULATE_J( x,y,z );
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 3), v3.low2float());
////      k_accumulator(pi0, accumulator_var::jx, 0) += v0.low2float();
////      k_accumulator(pi0, accumulator_var::jx, 1) += v1.low2float();
////      k_accumulator(pi0, accumulator_var::jx, 2) += v2.low2float();
////      k_accumulator(pi0, accumulator_var::jx, 3) += v3.low2float();
//      ACCUMULATE_J( y,z,x );
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 3), v3.low2float());
////      k_accumulator(pi0, accumulator_var::jy, 0) += v0.low2float();
////      k_accumulator(pi0, accumulator_var::jy, 1) += v1.low2float();
////      k_accumulator(pi0, accumulator_var::jy, 2) += v2.low2float();
////      k_accumulator(pi0, accumulator_var::jy, 3) += v3.low2float();
//      ACCUMULATE_J( z,x,y );
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 3), v3.low2float());
////      k_accumulator(pi0, accumulator_var::jz, 0) += v0.low2float();
////      k_accumulator(pi0, accumulator_var::jz, 1) += v1.low2float();
////      k_accumulator(pi0, accumulator_var::jz, 2) += v2.low2float();
////      k_accumulator(pi0, accumulator_var::jz, 3) += v3.low2float();
//    }
//    if(inbnds.high2float()) {
//      ACCUMULATE_J( x,y,z );
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 3), v3.high2float());
////      k_accumulator(pi1, accumulator_var::jx, 0) += v0.high2float();
////      k_accumulator(pi1, accumulator_var::jx, 1) += v1.high2float();
////      k_accumulator(pi1, accumulator_var::jx, 2) += v2.high2float();
////      k_accumulator(pi1, accumulator_var::jx, 3) += v3.high2float();
//      ACCUMULATE_J( y,z,x );
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 3), v3.high2float());
////      k_accumulator(pi1, accumulator_var::jy, 0) += v0.high2float();
////      k_accumulator(pi1, accumulator_var::jy, 1) += v1.high2float();
////      k_accumulator(pi1, accumulator_var::jy, 2) += v2.high2float();
////      k_accumulator(pi1, accumulator_var::jy, 3) += v3.high2float();
//      ACCUMULATE_J( z,x,y );
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 3), v3.high2float());
////      k_accumulator(pi1, accumulator_var::jz, 0) += v0.high2float();
////      k_accumulator(pi1, accumulator_var::jz, 1) += v1.high2float();
////      k_accumulator(pi1, accumulator_var::jz, 2) += v2.high2float();
////      k_accumulator(pi1, accumulator_var::jz, 3) += v3.high2float();
//    }
////    }
//#   undef ACCUMULATE_J
//
//    k_particle_mover_t local_pm[1];
//    if(!inbnds.low2float()) {
//      local_pm->dispx = ux.low2half();
//      local_pm->dispy = uy.low2half();
//      local_pm->dispz = uz.low2half();
//      local_pm->i     = p_index*2;
//      if( move_p_kokkos(k_part, local_pm,
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
//          k_particle_copy.dx(nm) = k_part.dx(p_index*2);
//          k_particle_copy.dy(nm) = k_part.dy(p_index*2);
//          k_particle_copy.dz(nm) = k_part.dz(p_index*2);
//          k_particle_copy.ux(nm) = k_part.ux(p_index*2);
//          k_particle_copy.uy(nm) = k_part.uy(p_index*2);
//          k_particle_copy.uz(nm) = k_part.uz(p_index*2);
//          k_particle_copy.w(nm)  = k_part.w(p_index*2);
//          k_particle_copy.i(nm)  = k_part.i(p_index*2);
//        }
//      }
//    }
//    if(!inbnds.high2float()) {
//      local_pm->dispx = ux.high2half();
//      local_pm->dispy = uy.high2half();
//      local_pm->dispz = uz.high2half();
//      local_pm->i     = p_index*2+1;
//      if( move_p_kokkos(k_part, local_pm,
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
//          k_particle_copy.dx(nm) = k_part.dx(p_index*2+1);
//          k_particle_copy.dy(nm) = k_part.dy(p_index*2+1);
//          k_particle_copy.dz(nm) = k_part.dz(p_index*2+1);
//          k_particle_copy.ux(nm) = k_part.ux(p_index*2+1);
//          k_particle_copy.uy(nm) = k_part.uy(p_index*2+1);
//          k_particle_copy.uz(nm) = k_part.uz(p_index*2+1);
//          k_particle_copy.w(nm)  = k_part.w(p_index*2+1);
//          k_particle_copy.i(nm)  = k_part.i(p_index*2+1);
//        }
//      }
//    }
////  });
//  });
//}

// Half
void
advance_p_kokkos(
        k_particles_soa_t& k_part,
        k_particles_soa_t& k_particle_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_accumulators_sa_t k_accumulators_sa,
        k_accumulators_t k_accumulator,
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

//  const mixed_t one            = mixed_t(1.0);
//  const mixed_t one_third      = mixed_t(1./3.);
//  const mixed_t two_fifteenths = mixed_t(2./15.);
  const mixed_t qdt_2mc_h = static_cast<mixed_t>(qdt_2mc);
  const mixed_t cdt_dx_h = static_cast<mixed_t>(cdt_dx);
  const mixed_t cdt_dy_h = static_cast<mixed_t>(cdt_dy);
  const mixed_t cdt_dz_h = static_cast<mixed_t>(cdt_dz);
  const mixed_t qsp_h = static_cast<mixed_t>(qsp);

  /*
  k_particle_movers_t *k_local_particle_movers_p = new k_particle_movers_t("k_local_pm", 1);
  k_particle_movers_t  k_local_particle_movers("k_local_pm", 1);

  k_iterator_t k_nm("k_nm");
  k_iterator_t::HostMirror h_nm = Kokkos::create_mirror_view(k_nm);
  h_nm(0) = 0;
  Kokkos::deep_copy(k_nm, h_nm);
  */
  // Determine which quads of particles quads this pipeline processes

  //DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, itmp, n );
  //p = args->p0 + itmp;

  /*
  printf("original value %f\n\n", k_accumulators(0, 0, 0));
sp_[id]->
  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int i) {

      auto scatter_access = k_accumulators_sa.access();
      //auto scatter_access_atomic = scatter_view.template access<Kokkos::Experimental::ScatterAtomic>();
          printf("Writing to %d\n", i);
          scatter_access(i, 0, 0) += 4;
          //scatter_access_atomic(i, 1) += 2.0;
          //scatter_access(k, 2) += 1.0;
          //
  });

  // copy back
  Kokkos::Experimental::contribute(k_accumulators, k_accumulators_sa);
  printf("changed value %f\n", k_accumulators(0, 0, 0));
  */

  // Determine which movers are reserved for this pipeline
  // Movers (16 bytes) should be reserved for pipelines in at least
  // multiples of 8 such that the set of particle movers reserved for
  // a pipeline is 128-byte aligned and a multiple of 128-byte in
  // size.  The host is guaranteed to get enough movers to process its
  // particles with this allocation.
/*
  max_nm = args->max_nm - (args->np&15);
  if( max_nm<0 ) max_nm = 0;
  DISTRIBUTE( max_nm, 8, pipeline_rank, n_pipeline, itmp, max_nm );
  if( pipeline_rank==n_pipeline ) max_nm = args->max_nm - itmp;
  pm   = args->pm + itmp;
  nm   = 0;
  itmp = 0;

  // Determine which accumulator array to use
  // The host gets the first accumulator array

  if( pipeline_rank!=n_pipeline )
    a0 += (1+pipeline_rank)*
          POW2_CEIL((args->nx+2)*(args->ny+2)*(args->nz+2),2);
*/
  // Process particles for this pipeline

  #define p_dx    k_part.dx(p_index)
  #define p_dy    k_part.dy(p_index)
  #define p_dz    k_part.dz(p_index)
  #define p_ux    k_part.ux(p_index)
  #define p_uy    k_part.uy(p_index)
  #define p_uz    k_part.uz(p_index)
  #define p_w     k_part.w(p_index)
  #define pii     k_part.i(p_index)

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
  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
    //printf("how many times does this run %d", i);
    k_nm(0) = 0;
    //local_pm_dispx = 0;
    //local_pm_dispy = 0;
    //local_pm_dispz = 0;
    //local_pm_i = 0;
  });

//printf("np: %d\n", np);

//  Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>
//  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy<>(0, np),
    KOKKOS_LAMBDA (size_t p_index)
    {
//if(p_index == 0) {
//#ifdef __CUDA_ARCH__
//printf("gridDims (%d,%d,%d), blockDims (%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
//#endif
//}

    auto  k_accumulators_scatter_access = k_accumulators_sa.access();
    mixed_t v0, v1, v2, v3, v4, v5;

    mixed_t dx = p_dx; // Load position
    mixed_t dy = p_dy;
    mixed_t dz = p_dz;
    int   ii   = pii;

//#ifdef __CUDA_ARCH__
//if(p_index < 4096) {
//  printf("blockIdx (x,y,z): (%d,%d,%d), threadIdx (x,y,z): (%d,%d,%d), p_index: %llu, cell_index: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, p_index, ii);
//}
//#endif

    mixed_t hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
                            dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
    mixed_t hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
                            dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
    mixed_t haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
                            dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
//    mixed_t hax  = qdt_2mc_h*fma(dz, fma(dy, f_d2exdydz_h, f_dexdz_h), fma(dy, f_dexdy_h, f_ex_h));
//    mixed_t hay  = qdt_2mc_h*fma(dx, fma(dz, f_d2eydzdx_h, f_deydx_h), fma(dz, f_deydz_h, f_ey_h));
//    mixed_t haz  = qdt_2mc_h*fma(dy, fma(dx, f_d2ezdxdy_h, f_dezdy_h), fma(dx, f_dezdx_h, f_ez_h));

    mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
    mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
    mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
//    mixed_t cbx  = fma(dx, f_dcbxdx_h, f_cbx_h);             // Interpolate B
//    mixed_t cby  = fma(dy, f_dcbydy_h, f_cby_h);
//    mixed_t cbz  = fma(dz, f_dcbzdz_h, f_cbz_h);

    mixed_t ux   = p_ux;                             // Load momentum
    mixed_t uy   = p_uy;
    mixed_t uz   = p_uz;
    mixed_t q    = static_cast<mixed_t>(p_w);

    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;
    v0   = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = ( v0*v0 ) * v1;
    v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
//    v3   = v0*(fma(v2, fma(v2, two_fifteenths, one_third), 1));
    v4   = v3/(mixed_t(one)+v1*(v3*v3));
//    v4   = v3/(fma((v3*v3), v1, one));
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
    v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
    ux  *= cdt_dx_h;
    uy  *= cdt_dy_h;
    uz  *= cdt_dz_h;

    /**/                                      // Get norm displacement
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;

    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;

    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
        -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {

      // Common case (inbnds).  Note: accumulator values are 4 times
      // the total physical charge that passed through the appropriate
      // current quadrant in a time-step

      q *= qsp_h;
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
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
//      k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//      k_accumulator(ii, accumulator_var::jx, 0) += static_cast<float>(v0);
//      k_accumulator(ii, accumulator_var::jx, 1) += static_cast<float>(v1);
//      k_accumulator(ii, accumulator_var::jx, 2) += static_cast<float>(v2);
//      k_accumulator(ii, accumulator_var::jx, 3) += static_cast<float>(v3);

      ACCUMULATE_J( y,z,x );
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
//      k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//      k_accumulator(ii, accumulator_var::jy, 0) += static_cast<float>(v0);
//      k_accumulator(ii, accumulator_var::jy, 1) += static_cast<float>(v1);
//      k_accumulator(ii, accumulator_var::jy, 2) += static_cast<float>(v2);
//      k_accumulator(ii, accumulator_var::jy, 3) += static_cast<float>(v3);

      ACCUMULATE_J( z,x,y );
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
//      k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
      Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//      k_accumulator(ii, accumulator_var::jz, 0) += static_cast<float>(v0);
//      k_accumulator(ii, accumulator_var::jz, 1) += static_cast<float>(v1);
//      k_accumulator(ii, accumulator_var::jz, 2) += static_cast<float>(v2);
//      k_accumulator(ii, accumulator_var::jz, 3) += static_cast<float>(v3);

#     undef ACCUMULATE_J

    } 
    else
    {                                    // Unlikely

      k_particle_mover_t local_pm[1];
      local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;
      local_pm->i     = p_index;

      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
      if( move_p_kokkos(k_part, local_pm,
                         k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
        if( k_nm(0)<max_nm ) {
          const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
          if (nm >= max_nm) Kokkos::abort("overran max_nm");

          k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
          k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
          k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
          k_particle_movers_i(nm)   = local_pm->i;

          // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
          k_particle_copy.dx(nm) = k_part.dx(p_index);
          k_particle_copy.dy(nm) = k_part.dy(p_index);
          k_particle_copy.dz(nm) = k_part.dz(p_index);
          k_particle_copy.ux(nm) = k_part.ux(p_index);
          k_particle_copy.uy(nm) = k_part.uy(p_index);
          k_particle_copy.uz(nm) = k_part.uz(p_index);
          k_particle_copy.w(nm) = k_part.w(p_index);
          k_particle_copy.i(nm) = k_part.i(p_index);

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
  });
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

//// Hierarchical parallelism
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t k_accumulator,
//        k_interpolator_t& k_interp,
////        Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>>& k_interp,
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
//  constexpr mixed_t one            = 1.;
//  constexpr mixed_t one_third      = 1./3.;
//  constexpr mixed_t two_fifteenths = 2./15.;
//
////  const mixed_t one            = mixed_t(1.0);
////  const mixed_t one_third      = mixed_t(1./3.);
////  const mixed_t two_fifteenths = mixed_t(2./15.);
//  const mixed_t qdt_2mc_h = static_cast<mixed_t>(qdt_2mc);
//  const mixed_t cdt_dx_h = static_cast<mixed_t>(cdt_dx);
//  const mixed_t cdt_dy_h = static_cast<mixed_t>(cdt_dy);
//  const mixed_t cdt_dz_h = static_cast<mixed_t>(cdt_dz);
//  const mixed_t qsp_h = static_cast<mixed_t>(qsp);
//
//  /*
//  k_particle_movers_t *k_local_particle_movers_p = new k_particle_movers_t("k_local_pm", 1);
//  k_particle_movers_t  k_local_particle_movers("k_local_pm", 1);
//
//  k_iterator_t k_nm("k_nm");
//  k_iterator_t::HostMirror h_nm = Kokkos::create_mirror_view(k_nm);
//  h_nm(0) = 0;
//  Kokkos::deep_copy(k_nm, h_nm);
//  */
//  // Determine which quads of particles quads this pipeline processes
//
//  //DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, itmp, n );
//  //p = args->p0 + itmp;
//
//  /*
//  printf("original value %f\n\n", k_accumulators(0, 0, 0));
//sp_[id]->
//  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int i) {
//
//      auto scatter_access = k_accumulators_sa.access();
//      //auto scatter_access_atomic = scatter_view.template access<Kokkos::Experimental::ScatterAtomic>();
//          printf("Writing to %d\n", i);
//          scatter_access(i, 0, 0) += 4;
//          //scatter_access_atomic(i, 1) += 2.0;
//          //scatter_access(k, 2) += 1.0;
//          //
//  });
//
//  // copy back
//  Kokkos::Experimental::contribute(k_accumulators, k_accumulators_sa);
//  printf("changed value %f\n", k_accumulators(0, 0, 0));
//  */
//
//  // Determine which movers are reserved for this pipeline
//  // Movers (16 bytes) should be reserved for pipelines in at least
//  // multiples of 8 such that the set of particle movers reserved for
//  // a pipeline is 128-byte aligned and a multiple of 128-byte in
//  // size.  The host is guaranteed to get enough movers to process its
//  // particles with this allocation.
///*
//  max_nm = args->max_nm - (args->np&15);
//  if( max_nm<0 ) max_nm = 0;
//  DISTRIBUTE( max_nm, 8, pipeline_rank, n_pipeline, itmp, max_nm );
//  if( pipeline_rank==n_pipeline ) max_nm = args->max_nm - itmp;
//  pm   = args->pm + itmp;
//  nm   = 0;
//  itmp = 0;
//
//  // Determine which accumulator array to use
//  // The host gets the first accumulator array
//
//  if( pipeline_rank!=n_pipeline )
//    a0 += (1+pipeline_rank)*
//          POW2_CEIL((args->nx+2)*(args->ny+2)*(args->nz+2),2);
//*/
//  // Process particles for this pipeline
//
//  #define p_dx    k_part.dx(p_index)
//  #define p_dy    k_part.dy(p_index)
//  #define p_dz    k_part.dz(p_index)
//  #define p_ux    k_part.ux(p_index)
//  #define p_uy    k_part.uy(p_index)
//  #define p_uz    k_part.uz(p_index)
//  #define p_w     k_part.w(p_index)
//  #define pii     k_part.i(p_index)
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
////Kokkos::View<int[1]> movers_counter("Counter for # of moving particles");
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    k_nm(0) = 0;
////    movers_counter(0) = 0;
//  });
//
//
////Kokkos::View<int*> accum_index = Kokkos::View<int*>("Accumulator indices", na);
////Kokkos::parallel_for("get accumulator counts", Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(size_t i) {
////  Kokkos::atomic_add(&(accum_index(i)), 1);
////});
//
////  int num_threads = 32;
////  int num_leagues = 2698187;
//////  if(num_threads*num_leagues < np)
//////    num_leagues++;
////  int per_league = num_threads;
//
////  int num_threads = 32;
////  int num_leagues = np/num_threads;
////  if(num_threads*num_leagues < np)
////    num_leagues++;
////  int per_league = num_threads;
//
//  int num_threads = 1024;
//  int num_leagues = 2048;
//  int per_league = np/num_leagues;
//  if(num_leagues*per_league < np)
//    per_league++;
////    num_leagues++;
//
////  int num_threads = 288;
////  int num_leagues = 8565;
////  int per_league = 10080;
//
//  int num_entries = num_threads;
////  int num_entries = 500;
////  const int nv = g->nv;
//
////cudaFuncCache cache_config;
////cudaDeviceGetCacheConfig(&cache_config);
////printf("Cache config: %d\n", cache_config);
////cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
////printf("Cache config post update: %d\n", cache_config);
//
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
//  typedef Kokkos::View<float*[18], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_interp;
//  typedef Kokkos::View<float*[3][4], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_accum;
//  typedef Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_cell_idx;
////  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1).set_scratch_size(0, Kokkos::PerTeam(num_entries*(sizeof(float)*12)));
////  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy <Kokkos::LaunchBounds<16,2> > (0, np),
//  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1);
//
////printf("np: %d, na: %d, max_nm: %d\n", np, na, max_nm);
//
//  Kokkos::parallel_for("advance_p", team_policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    const int league_rank = team_member.league_rank();
////    int start = k_part.i(league_rank*per_league);
////    int end = start+num_entries;
////    if(end >= nv) {
////      end = k_part.i(np-1);
////    } else {
////      end = k_part.i(league_rank*per_league + per_league);
////    }
//
////    shared_interp cached_interp(team_member.team_scratch(0), num_entries);
////    shared_accum atomic_cache(team_member.team_scratch(0), num_entries);
//
////    if(end-start < num_entries) {
////      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 18*num_entries), [=] (const int index) {
////        int idx = index / 18;
////        int var = index % 18;
////        cached_interp(idx, var) = k_interp(start+idx, var);
////      });
////
////      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end-start), [=] (const int idx) {
////        atomic_cache(idx,0,0)  = 0.0f; 
////        atomic_cache(idx,1,0)  = 0.0f; 
////        atomic_cache(idx,2,0)  = 0.0f; 
////
////        atomic_cache(idx,0,1)  = 0.0f; 
////        atomic_cache(idx,1,1)  = 0.0f; 
////        atomic_cache(idx,2,1)  = 0.0f; 
////
////        atomic_cache(idx,0,2)  = 0.0f; 
////        atomic_cache(idx,1,2)  = 0.0f; 
////        atomic_cache(idx,2,2)  = 0.0f; 
////
////        atomic_cache(idx,0,3)  = 0.0f; 
////        atomic_cache(idx,1,3)  = 0.0f; 
////        atomic_cache(idx,2,3)  = 0.0f; 
////      });
////    }
//
////    float jx[4] = {0.0, 0.0, 0.0, 0.0};
////    float jy[4] = {0.0, 0.0, 0.0, 0.0};
////    float jz[4] = {0.0, 0.0, 0.0, 0.0};
////    int cell_idx = 0;
//
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [&] (const int pindex) {
//      int p_index = league_rank*per_league + pindex;
//      const int team_rank = team_member.team_rank();
//
////if(p_index < 43510)
////    printf("league_rank: %d, team_rank: %d, pindex: %d, p_index: %d\n", team_member.league_rank(), team_member.team_rank(), pindex, p_index);
//
////      auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//      if(p_index < np) {
//        mixed_t v0, v1, v2, v3, v4, v5;
//    
//        const int   ii   = pii;
//
////#ifdef __CUDA_ARCH__
////if(p_index < 4096) {
////  printf("blockIdx (x,y,z): (%d,%d,%d), threadIdx (x,y,z): (%d,%d,%d), p_index: %d, cell_index: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, p_index, ii);
////}
////#endif
//
//        mixed_t dx = p_dx; // Load position
//        mixed_t dy = p_dy;
//        mixed_t dz = p_dz;
//
//        mixed_t ux   = p_ux;                             // Load momentum
//        mixed_t uy   = p_uy;
//        mixed_t uz   = p_uz;
//        mixed_t q    = static_cast<mixed_t>(p_w);
//    
//        mixed_t hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
//                                dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
//        mixed_t hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
//                                dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
//        mixed_t haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
//                                dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
//
//        mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
//        mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
//        mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
//
////if(ii < start) {
////printf("league rank: %d, team rank: %d, start: %d, end: %d, end-start: %d, ii: %d, true_start: %d, true_end: %d\n", team_member.league_rank(), team_member.team_rank(), start, end, end-start, ii, k_part.i(league_rank*per_league), k_part.i(league_rank*per_league + per_league));
////}
//
//////      Shared Interpolators
////        mixed_t hax, hay, haz;
////        mixed_t cbx, cby, cbz;
////        if(start <= ii && ii < end && end-start < num_entries) {
////          int index = ii-start;
////          hax  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ex)) + 
////                            dy*mixed_t(cached_interp(index, interpolator_var::dexdy))) +
////                            dz*(mixed_t(cached_interp(index, interpolator_var::dexdz)) + 
////                            dy*mixed_t(cached_interp(index, interpolator_var::d2exdydz))) );
////          hay  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ey)) + 
////                            dz*mixed_t(cached_interp(index, interpolator_var::deydz))) +
////                            dx*(mixed_t(cached_interp(index, interpolator_var::deydx)) + 
////                            dz*mixed_t(cached_interp(index, interpolator_var::d2eydzdx))) );
////          haz  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ez)) + 
////                            dx*mixed_t(cached_interp(index, interpolator_var::dezdx))) +
////                            dy*(mixed_t(cached_interp(index, interpolator_var::dezdy)) + 
////                            dx*mixed_t(cached_interp(index, interpolator_var::d2ezdxdy))) );
////
////          cbx = mixed_t( cached_interp(ii-start, interpolator_var::cbx)) + 
////              dx*mixed_t(cached_interp(ii-start, interpolator_var::dcbxdx));
////          cby = mixed_t( cached_interp(ii-start, interpolator_var::cby)) + 
////              dy*mixed_t(cached_interp(ii-start, interpolator_var::dcbydy));
////          cbz = mixed_t( cached_interp(ii-start, interpolator_var::cbz)) + 
////              dz*mixed_t(cached_interp(ii-start, interpolator_var::dcbzdz));
////        } else {
////          hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
////                          dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
////          hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
////                          dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
////          haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
////                          dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
////
////          cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
////          cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
////          cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
////        }
//  
////        mixed_t ux   = p_ux;                             // Load momentum
////        mixed_t uy   = p_uy;
////        mixed_t uz   = p_uz;
////        mixed_t q    = static_cast<mixed_t>(p_w);
//    
//        ux  += hax;                               // Half advance E
//        uy  += hay;
//        uz  += haz;
//        v0   = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//        v1   = cbx*cbx + (cby*cby + cbz*cbz);
//        v2   = ( v0*v0 ) * v1;
//        v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
//        v4   = v3/(mixed_t(one)+v1*(v3*v3));
//        v4  += v4;
//    
//        v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
//        v1   = uy + v3*( uz*cbx - ux*cbz );
//        v2   = uz + v3*( ux*cby - uy*cbx );
//    
//        ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
//        uy  += v4*( v2*cbx - v0*cbz );
//        uz  += v4*( v0*cby - v1*cbx );
//    
//        ux  += hax;                               // Half advance E
//        uy  += hay;
//        uz  += haz;
//    
//        p_ux = ux;                               // Store momentum
//        p_uy = uy;
//        p_uz = uz;
//
//        v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
//        ux  *= cdt_dx_h;
//        uy  *= cdt_dy_h;
//        uz  *= cdt_dz_h;
//    
//        /**/                                      // Get norm displacement
//        ux  *= v0;
//        uy  *= v0;
//        uz  *= v0;
//    
//        v0   = dx + ux;                           // Streak midpoint (inbnds)
//        v1   = dy + uy;
//        v2   = dz + uz;
//    
//        v3   = v0 + ux;                           // New position
//        v4   = v1 + uy;
//        v5   = v2 + uz;
//
//// Warp reduction
//bool inbnds = v3<=mixed_t(one) && v4<=mixed_t(one) && v5<=mixed_t(one) &&
//              -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one);
//int mask = 0xffffffff;
//int synced = 0;
//int same_idx = 0;
//#ifdef __CUDA_ARCH__
//__match_all_sync(0xffffffff, inbnds, &synced);
//__match_all_sync(0xffffffff, ii, &same_idx);
//#endif
//          if(inbnds) {
//    
////        if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
////            -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {
//    
//          // Common case (inbnds).  Note: accumulator values are 4 times
//          // the total physical charge that passed through the appropriate
//          // current quadrant in a time-step
//    
//          q *= qsp_h;
//          p_dx = v3;                             // Store new position
//          p_dy = v4;
//          p_dz = v5;
//          dx = v0;                                // Streak midpoint
//          dy = v1;
//          dz = v2;
//          v5 = q*ux*uy*uz*one_third;              // Compute correction
//    
//         #define ACCUMULATE_J(X,Y,Z)                                        \
//          v4  = q*u##X;   /* v2 = q ux                            */        \
//          v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//          v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//          v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//          v4  = one+d##Z; /* v4 = 1+dz                            */        \
//          v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//          v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//          v4  = one-d##Z; /* v4 = 1-dz                            */        \
//          v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//          v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//          v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//          v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//          v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//          v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//// Warp reduction
//          if(synced && same_idx) {
//#ifdef __CUDA_ARCH__
//            ACCUMULATE_J( x,y,z );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//            }
//            ACCUMULATE_J( y,z,x );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//            }
//            ACCUMULATE_J( z,x,y );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//            }
//#endif
//          } else {
//            ACCUMULATE_J( x,y,z );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//            
//            ACCUMULATE_J( y,z,x );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//            
//            ACCUMULATE_J( z,x,y );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//          }
//
//////        Non-atomic local with atomic global accumulation
////          if(cell_idx != ii) {
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 0)), jx[0]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 1)), jx[1]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 2)), jx[2]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 3)), jx[3]);
////            ACCUMULATE_J( x,y,z );
////            jx[0] = static_cast<float>(v0);
////            jx[1] = static_cast<float>(v1);
////            jx[2] = static_cast<float>(v2);
////            jx[3] = static_cast<float>(v3);
////
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 0)), jy[0]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 1)), jy[1]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 2)), jy[2]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 3)), jy[3]);
////            ACCUMULATE_J( y,z,x );
////            jy[0] = static_cast<float>(v0);
////            jy[1] = static_cast<float>(v1);
////            jy[2] = static_cast<float>(v2);
////            jy[3] = static_cast<float>(v3);
////
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 0)), jz[0]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 1)), jz[1]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 2)), jz[2]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 3)), jz[3]);
////            ACCUMULATE_J( z,x,y );
////            jz[0] = static_cast<float>(v0);
////            jz[1] = static_cast<float>(v1);
////            jz[2] = static_cast<float>(v2);
////            jz[3] = static_cast<float>(v3);
////            cell_idx = ii;
////          } else {
////            ACCUMULATE_J( x,y,z );
////            jx[0] += static_cast<float>(v0);
////            jx[1] += static_cast<float>(v1);
////            jx[2] += static_cast<float>(v2);
////            jx[3] += static_cast<float>(v3);
////            ACCUMULATE_J( y,z,x );
////            jy[0] += static_cast<float>(v0);
////            jy[1] += static_cast<float>(v1);
////            jy[2] += static_cast<float>(v2);
////            jy[3] += static_cast<float>(v3);
////            ACCUMULATE_J( z,x,y );
////            jz[0] += static_cast<float>(v0);
////            jz[1] += static_cast<float>(v1);
////            jz[2] += static_cast<float>(v2);
////            jz[3] += static_cast<float>(v3);
////          }
//
//////        Atomic Accumulation
////          ACCUMULATE_J( x,y,z );
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
////          ACCUMULATE_J( y,z,x );
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
////          ACCUMULATE_J( z,x,y );
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//
//////        Non-atomic accumulation (Not accurate, only for perf comparisions)
////          ACCUMULATE_J( x,y,z );
////          k_accumulator(ii, accumulator_var::jx, 0) += static_cast<float>(v0);
////          k_accumulator(ii, accumulator_var::jx, 1) += static_cast<float>(v1);
////          k_accumulator(ii, accumulator_var::jx, 2) += static_cast<float>(v2);
////          k_accumulator(ii, accumulator_var::jx, 3) += static_cast<float>(v3);
////          ACCUMULATE_J( y,z,x );
////          k_accumulator(ii, accumulator_var::jy, 0) += static_cast<float>(v0);
////          k_accumulator(ii, accumulator_var::jy, 1) += static_cast<float>(v1);
////          k_accumulator(ii, accumulator_var::jy, 2) += static_cast<float>(v2);
////          k_accumulator(ii, accumulator_var::jy, 3) += static_cast<float>(v3);
////          ACCUMULATE_J( z,x,y );
////          k_accumulator(ii, accumulator_var::jz, 0) += static_cast<float>(v0);
////          k_accumulator(ii, accumulator_var::jz, 1) += static_cast<float>(v1);
////          k_accumulator(ii, accumulator_var::jz, 2) += static_cast<float>(v2);
////          k_accumulator(ii, accumulator_var::jz, 3) += static_cast<float>(v3);
//
//////        Shared atomic accumulation (accumulate in shared memory instead of global)
////          if(start <= ii && ii < end && end-start < num_entries) {
////            ACCUMULATE_J( x,y,z );
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 0)), static_cast<float>(v0));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 1)), static_cast<float>(v1));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 2)), static_cast<float>(v2));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( y,z,x );
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 0)), static_cast<float>(v0));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 1)), static_cast<float>(v1));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 2)), static_cast<float>(v2));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( z,x,y );
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 0)), static_cast<float>(v0));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 1)), static_cast<float>(v1));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 2)), static_cast<float>(v2));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 3)), static_cast<float>(v3));
////          } else {
////            ACCUMULATE_J( x,y,z );
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 0)), static_cast<float>(v0));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 1)), static_cast<float>(v1));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 2)), static_cast<float>(v2));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( y,z,x );
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 0)), static_cast<float>(v0));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 1)), static_cast<float>(v1));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 2)), static_cast<float>(v2));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( z,x,y );
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 0)), static_cast<float>(v0));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 1)), static_cast<float>(v1));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 2)), static_cast<float>(v2));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 3)), static_cast<float>(v3));
////          }
//    
//////        ScatterView
////          ACCUMULATE_J( x,y,z );
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;
////          ACCUMULATE_J( y,z,x );
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;
////          ACCUMULATE_J( z,x,y );
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
//
//    #     undef ACCUMULATE_J
//        } 
//        else
//        {                                    // Unlikely
////Kokkos::atomic_add(&movers_counter(0), 1);
//          k_particle_mover_t local_pm[1];
//          local_pm->dispx = ux;
//          local_pm->dispy = uy;
//          local_pm->dispz = uz;
//          local_pm->i     = p_index;
//    
//          //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
////          if( move_p_kokkos(k_part, local_pm,
////                             k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
//          if( move_p_kokkos(k_part, local_pm,
//                             k_accumulator, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
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
//              k_particle_copy.dx(nm) = k_part.dx(p_index);
//              k_particle_copy.dy(nm) = k_part.dy(p_index);
//              k_particle_copy.dz(nm) = k_part.dz(p_index);
//              k_particle_copy.ux(nm) = k_part.ux(p_index);
//              k_particle_copy.uy(nm) = k_part.uy(p_index);
//              k_particle_copy.uz(nm) = k_part.uz(p_index);
//              k_particle_copy.w(nm) = k_part.w(p_index);
//              k_particle_copy.i(nm) = k_part.i(p_index);
//    
//              // Tag this one as having left
//              //k_particles(p_index, particle_var::pi) = 999999;
//    
//              // Copy local local_pm back
//              //local_pm_dispx = local_pm->dispx;
//              //local_pm_dispy = local_pm->dispy;
//              //local_pm_dispz = local_pm->dispz;
//              //local_pm_i = local_pm->i;
//              //printf("rank copying %d to nm %d \n", local_pm_i, nm);
//              //copy_local_to_pm(nm);
//            }
//          }
//        }
//      }
//    });
//
//////  Non-atomic local with atomic global accumulation
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 0)), jx[0]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 1)), jx[1]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 2)), jx[2]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 3)), jx[3]);
////
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 0)), jy[0]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 1)), jy[1]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 2)), jy[2]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 3)), jy[3]);
////
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 0)), jz[0]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 1)), jz[1]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 2)), jz[2]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 3)), jz[3]);
//
//////  Shared atomic accumulation (accumulate in shared memory instead of global)
////    if(end-start < num_entries) {
////      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end-start), [=] (const int idx) {
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 0)), atomic_cache(idx,0,0)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 0)), atomic_cache(idx,1,0)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 0)), atomic_cache(idx,2,0)); 
////
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 1)), atomic_cache(idx,0,1)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 1)), atomic_cache(idx,1,1)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 1)), atomic_cache(idx,2,1)); 
////
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 2)), atomic_cache(idx,0,2)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 2)), atomic_cache(idx,1,2)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 2)), atomic_cache(idx,2,2)); 
////
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 3)), atomic_cache(idx,0,3)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 3)), atomic_cache(idx,1,3)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 3)), atomic_cache(idx,2,3)); 
////      });
////    }
//  });
//
//  // TODO: abstract this manual data copy
//  //Kokkos::deep_copy(h_nm, k_nm);
//
//  //args->seg[pipeline_rank].pm        = pm;
//  //args->seg[pipeline_rank].max_nm    = max_nm;
//  //args->seg[pipeline_rank].nm        = h_nm(0);
//  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
//  //delete(k_local_particle_movers_p);
//  //return h_nm(0);
//
////Kokkos::parallel_for("print stuff", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int i) {
////  printf("max_nm: %d\tmovers: %d\n", max_nm, movers_counter(0));
////});
//
//  #undef p_dx  
//  #undef p_dy  
//  #undef p_dz  
//  #undef p_ux  
//  #undef p_uy  
//  #undef p_uz  
//  #undef p_w   
//  #undef pii   
//
//  #undef f_cbx 
//  #undef f_cby 
//  #undef f_cbz 
//  #undef f_ex  
//  #undef f_ey  
//  #undef f_ez  
//
//  #undef f_dexdy    
//  #undef f_dexdz    
//
//  #undef f_d2exdydz 
//  #undef f_deydx    
//  #undef f_deydz    
//
//  #undef f_d2eydzdx 
//  #undef f_dezdx    
//  #undef f_dezdy    
//
//  #undef f_d2ezdxdy 
//  #undef f_dcbxdx   
//  #undef f_dcbydy   
//  #undef f_dcbzdz   
//}

//// Iterate by Cell index
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t k_accumulator,
//        k_interpolator_t& k_interp,
////        Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>>& k_interp,
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
//  constexpr mixed_t one            = 1.;
//  constexpr mixed_t one_third      = 1./3.;
//  constexpr mixed_t two_fifteenths = 2./15.;
//
////  const mixed_t one            = mixed_t(1.0);
////  const mixed_t one_third      = mixed_t(1./3.);
////  const mixed_t two_fifteenths = mixed_t(2./15.);
//  const mixed_t qdt_2mc_h = static_cast<mixed_t>(qdt_2mc);
//  const mixed_t cdt_dx_h = static_cast<mixed_t>(cdt_dx);
//  const mixed_t cdt_dy_h = static_cast<mixed_t>(cdt_dy);
//  const mixed_t cdt_dz_h = static_cast<mixed_t>(cdt_dz);
//  const mixed_t qsp_h = static_cast<mixed_t>(qsp);
//
//  #define p_dx    k_part.dx(p_index)
//  #define p_dy    k_part.dy(p_index)
//  #define p_dz    k_part.dz(p_index)
//  #define p_ux    k_part.ux(p_index)
//  #define p_uy    k_part.uy(p_index)
//  #define p_uz    k_part.uz(p_index)
//  #define p_w     k_part.w(p_index)
//  #define pii     k_part.i(p_index)
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
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    k_nm(0) = 0;
//  });
//
//
////  int num_threads = 32;
////  int num_leagues = 2698187;
//////  if(num_threads*num_leagues < np)
//////    num_leagues++;
////  int per_league = num_threads;
//
////  int num_threads = 32;
////  int num_leagues = np/num_threads;
////  if(num_threads*num_leagues < np)
////    num_leagues++;
////  int per_league = num_threads;
//
////  int num_threads = 1024;
////  int num_leagues = 2048;
////  int per_league = np/num_leagues;
////  if(num_leagues*per_league < np)
////    per_league++;
//////    num_leagues++;
//
//  int num_threads = 32;
//  int num_leagues = g->nv;
//
////  int num_threads = 288;
////  int num_leagues = 8565;
////  int per_league = 10080;
//
//  int num_entries = num_threads;
////  int num_entries = 500;
//  const int nv = g->nv;
//
////cudaFuncCache cache_config;
////cudaDeviceGetCacheConfig(&cache_config);
////printf("Cache config: %d\n", cache_config);
////cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
////printf("Cache config post update: %d\n", cache_config);
//
//  Kokkos::View<int*> particle_counts("# of particles in each cell", nv);
//  Kokkos::parallel_for("get particle counts", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
//    Kokkos::atomic_add(&particle_counts(k_part.i(i)), 1);
//  });
//  Kokkos::parallel_scan("Get start indices", nv, KOKKOS_LAMBDA(const int i, int& update, const bool final) {
//    const float val_i = particle_counts(i);
//    if(final)
//      particle_counts(i) = update;
//    update += val_i;
//  });
//
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
//  typedef Kokkos::View<float*[18], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_interp;
//  typedef Kokkos::View<float*[3][4], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_accum;
//  typedef Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_cell_idx;
////  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1).set_scratch_size(0, Kokkos::PerTeam(num_entries*(sizeof(float)*12)));
////  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy <Kokkos::LaunchBounds<16,2> > (0, np),
//  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1);
//
//  Kokkos::parallel_for("advance_p", team_policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    const int league_rank = team_member.league_rank();
//    const int start = particle_counts(league_rank);
//    const int end = particle_counts(league_rank+1);
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end-start), [&] (const int idx) {
//      const int team_rank = team_member.team_rank();
//      int p_index = start + idx;
//      const int ii = pii;
//
//      mixed_t v0, v1, v2, v3, v4, v5;
//      
//      // Load position
//      mixed_t dx = p_dx;
//      mixed_t dy = p_dy;
//      mixed_t dz = p_dz;
//      
//      // Load momentum
//      mixed_t ux = p_ux;
//      mixed_t uy = p_uy;
//      mixed_t uz = p_uz;
//
//      // Load weight
//      mixed_t q = p_w;
//
//      mixed_t hax = qdt_2mc_h*((f_ex + dy*f_dexdy) + dz*(f_dexdz + dy*f_d2exdydz));
//      mixed_t hay = qdt_2mc_h*((f_ey + dz*f_deydz) + dx*(f_deydx + dz*f_d2eydzdx));
//      mixed_t haz = qdt_2mc_h*((f_ez + dx*f_dezdx) + dy*(f_dezdy + dx*f_d2ezdxdy));
//
//      // Interpolate B
//      mixed_t cbx = f_cbx + dx*f_dcbxdx;
//      mixed_t cby = f_cby + dy*f_dcbydy;
//      mixed_t cbz = f_cbz + dz*f_dcbzdz;
//
//      // Half advance E
//      ux += hax;
//      uy += hay;
//      uz += haz;
//      v0  = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//      v1  = cbx*cbx + (cby*cby + cbz*cbz);
//      v2  = (v0*v0)*v1;
//      v3  = v0*(one + v2*(one_third+v2*two_fifteenths));
//      v4  = v3/(one + v1*(v3*v3));
//      v4 += v4;
//
//      // Boris - uprime
//      v0 = ux + v3*(uy*cbz - uz*cby);
//      v1 = uy + v3*(uz*cbx - ux*cbz);
//      v2 = uz + v3*(ux*cby - uy*cbx);
//
//      // Boris - rotation
//      ux += v4*(v1*cbz -v2*cby);
//      uy += v4*(v2*cbx -v0*cbz);
//      uz += v4*(v0*cby -v1*cbx);
//
//      // Half advance E
//      ux += hax;
//      uy += hay;
//      uz += haz;
//      
//      // Store momentum
//      p_ux = ux;
//      p_uy = uy;
//      p_uz = uz;
//      
//      v0  = one/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//      ux *= cdt_dx_h;
//      uy *= cdt_dy_h;
//      uz *= cdt_dz_h;
//
//      // Get norm displacement
//      ux *= v0;
//      uy *= v0;
//      uz *= v0;
//
//      // Streak midpoint (inbnds)
//      v0 = dx + ux;
//      v1 = dy + uy;
//      v2 = dz + uz;
//
//      // New position
//      v3 = v0 + ux;
//      v4 = v1 + uy;
//      v5 = v2 + uz;
//
//      // In cell bounds
////      if( v3 <= one &&  v4 <= one &&  v5 <= one && 
////         -v3 <= one && -v4 <= one && -v5 <= one) {
//// Warp reduction
//bool inbnds = v3<=mixed_t(one) && v4<=mixed_t(one) && v5<=mixed_t(one) &&
//              -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one);
//int mask = 0xffffffff;
//int synced = 0;
//int same_idx = 0;
//#ifdef __CUDA_ARCH__
//__match_all_sync(0xffffffff, inbnds, &synced);
//__match_all_sync(0xffffffff, ii, &same_idx);
//#endif
//          if(inbnds) {
//        q *= qsp;
//        // Store new position
//        p_dx = v3;
//        p_dy = v4;
//        p_dz = v5;
//        // Streak midpoint
//        dx = v0;
//        dy = v1;
//        dz = v2;
//        // Compute correction
//        v5 = q*ux*uy*uz*one_third;
//
//       #define ACCUMULATE_J(X,Y,Z)                                        \
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
////        ACCUMULATE_J( x,y,z );
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), v0);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), v1);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), v2);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), v3);
////        ACCUMULATE_J( y,z,x );
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), v0);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), v1);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), v2);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), v3);
////        ACCUMULATE_J( z,x,y );
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), v0);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), v1);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), v2);
////        Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), v3);
//
//// Warp reduction
//          if(synced && same_idx) {
//#ifdef __CUDA_ARCH__
//            ACCUMULATE_J( x,y,z );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              k_accumulator(ii, accumulator_var::jx, 0) += static_cast<float>(v0);
//              k_accumulator(ii, accumulator_var::jx, 1) += static_cast<float>(v1);
//              k_accumulator(ii, accumulator_var::jx, 2) += static_cast<float>(v2);
//              k_accumulator(ii, accumulator_var::jx, 3) += static_cast<float>(v3);
//            }
//            ACCUMULATE_J( y,z,x );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              k_accumulator(ii, accumulator_var::jy, 0) += static_cast<float>(v0);
//              k_accumulator(ii, accumulator_var::jy, 1) += static_cast<float>(v1);
//              k_accumulator(ii, accumulator_var::jy, 2) += static_cast<float>(v2);
//              k_accumulator(ii, accumulator_var::jy, 3) += static_cast<float>(v3);
//            }
//            ACCUMULATE_J( z,x,y );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              k_accumulator(ii, accumulator_var::jz, 0) += static_cast<float>(v0);
//              k_accumulator(ii, accumulator_var::jz, 1) += static_cast<float>(v1);
//              k_accumulator(ii, accumulator_var::jz, 2) += static_cast<float>(v2);
//              k_accumulator(ii, accumulator_var::jz, 3) += static_cast<float>(v3);
//            }
//#endif
//          } else {
//            ACCUMULATE_J( x,y,z );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//            
//            ACCUMULATE_J( y,z,x );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//            
//            ACCUMULATE_J( z,x,y );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//          }
//      } 
//      else // Move particle out of cell
//      {
//        k_particle_mover_t local_pm[1];
//        local_pm->dispx = ux;
//        local_pm->dispy = uy;
//        local_pm->dispz = uz;
//        local_pm->i     = p_index;
//    
//        if( move_p_kokkos(k_part, local_pm,
//                           k_accumulator, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
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
//            k_particle_copy.dx(nm) = k_part.dx(p_index);
//            k_particle_copy.dy(nm) = k_part.dy(p_index);
//            k_particle_copy.dz(nm) = k_part.dz(p_index);
//            k_particle_copy.ux(nm) = k_part.ux(p_index);
//            k_particle_copy.uy(nm) = k_part.uy(p_index);
//            k_particle_copy.uz(nm) = k_part.uz(p_index);
//            k_particle_copy.w(nm) = k_part.w(p_index);
//            k_particle_copy.i(nm) = k_part.i(p_index);
//          }
//        }
//      }
//    });
//  });
//
//  #undef p_dx  
//  #undef p_dy  
//  #undef p_dz  
//  #undef p_ux  
//  #undef p_uy  
//  #undef p_uz  
//  #undef p_w   
//  #undef pii   
//
//  #undef f_cbx 
//  #undef f_cby 
//  #undef f_cbz 
//  #undef f_ex  
//  #undef f_ey  
//  #undef f_ez  
//
//  #undef f_dexdy    
//  #undef f_dexdz    
//
//  #undef f_d2exdydz 
//  #undef f_deydx    
//  #undef f_deydz    
//
//  #undef f_d2eydzdx 
//  #undef f_dezdx    
//  #undef f_dezdy    
//
//  #undef f_d2ezdxdy 
//  #undef f_dcbxdx   
//  #undef f_dcbydy   
//  #undef f_dcbzdz   
//}

//// Separate moving particles into different levels
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t k_accumulator,
//        k_interpolator_t& k_interp,
////        Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>>& k_interp,
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
//  constexpr mixed_t one            = 1.;
//  constexpr mixed_t one_third      = 1./3.;
//  constexpr mixed_t two_fifteenths = 2./15.;
//
////  const mixed_t one            = mixed_t(1.0);
////  const mixed_t one_third      = mixed_t(1./3.);
////  const mixed_t two_fifteenths = mixed_t(2./15.);
//  const mixed_t qdt_2mc_h = static_cast<mixed_t>(qdt_2mc);
//  const mixed_t cdt_dx_h = static_cast<mixed_t>(cdt_dx);
//  const mixed_t cdt_dy_h = static_cast<mixed_t>(cdt_dy);
//  const mixed_t cdt_dz_h = static_cast<mixed_t>(cdt_dz);
//  const mixed_t qsp_h = static_cast<mixed_t>(qsp);
//
//  // Process particles for this pipeline
//
//  #define p_dx    k_part.dx(p_index)
//  #define p_dy    k_part.dy(p_index)
//  #define p_dz    k_part.dz(p_index)
//  #define p_ux    k_part.ux(p_index)
//  #define p_uy    k_part.uy(p_index)
//  #define p_uz    k_part.uz(p_index)
//  #define p_w     k_part.w(p_index)
//  #define pii     k_part.i(p_index)
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
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    k_nm(0) = 0;
//  });
//
////Kokkos::View<int*> accum_index = Kokkos::View<int*>("Accumulator indices", na);
////Kokkos::parallel_for("get accumulator counts", Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(size_t i) {
////  Kokkos::atomic_add(&(accum_index(i)), 1);
////});
//
////  int num_threads = 32;
////  int num_leagues = 2698187;
//////  if(num_threads*num_leagues < np)
//////    num_leagues++;
////  int per_league = num_threads;
//
////  int num_threads = 32;
////  int num_leagues = np/num_threads;
////  if(num_threads*num_leagues < np)
////    num_leagues++;
////  int per_league = num_threads;
//
//  int num_threads = 1024;
//  int num_leagues = 2048;
//  int per_league = np/num_leagues;
//  if(num_leagues*per_league < np)
//    per_league++;
////    num_leagues++;
//
////  int num_threads = 288;
////  int num_leagues = 8565;
////  int per_league = 10080;
//
////  int num_entries = num_threads;
//  int num_entries = 2100;
////  const int nv = g->nv;
//
////cudaFuncCache cache_config;
////cudaDeviceGetCacheConfig(&cache_config);
////printf("Cache config: %d\n", cache_config);
////cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
////printf("Cache config post update: %d\n", cache_config);
//
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
////  typedef Kokkos::View<float*[18], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_interp;
////  typedef Kokkos::View<float*[3][4], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_accum;
////  typedef Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_cell_idx;
//  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1).set_scratch_size(0, Kokkos::PerTeam((num_entries+1)*(sizeof(float)*4)));
////  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy <Kokkos::LaunchBounds<16,2> > (0, np),
////  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1);
//
////printf("np: %d, na: %d, max_nm: %d\n", np, na, max_nm);
//
//  Kokkos::parallel_for("advance_p", team_policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    const int league_rank = team_member.league_rank();
////    int start = k_part.i(league_rank*per_league);
////    int end = start+num_entries;
////    if(end >= nv) {
////      end = k_part.i(np-1);
////    } else {
////      end = k_part.i(league_rank*per_league + per_league);
////    }
//
////    shared_interp cached_interp(team_member.team_scratch(0), num_entries);
////    shared_accum atomic_cache(team_member.team_scratch(0), num_entries);
//  
////    if(end-start < num_entries) {
////      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 18*num_entries), [=] (const int index) {
////        int idx = index / 18;
////        int var = index % 18;
////        cached_interp(idx, var) = k_interp(start+idx, var);
////      });
////
////      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end-start), [=] (const int idx) {
////        atomic_cache(idx,0,0)  = 0.0f; 
////        atomic_cache(idx,1,0)  = 0.0f; 
////        atomic_cache(idx,2,0)  = 0.0f; 
////
////        atomic_cache(idx,0,1)  = 0.0f; 
////        atomic_cache(idx,1,1)  = 0.0f; 
////        atomic_cache(idx,2,1)  = 0.0f; 
////
////        atomic_cache(idx,0,2)  = 0.0f; 
////        atomic_cache(idx,1,2)  = 0.0f; 
////        atomic_cache(idx,2,2)  = 0.0f; 
////
////        atomic_cache(idx,0,3)  = 0.0f; 
////        atomic_cache(idx,1,3)  = 0.0f; 
////        atomic_cache(idx,2,3)  = 0.0f; 
////      });
////    }
//
////    float jx[4] = {0.0, 0.0, 0.0, 0.0};
////    float jy[4] = {0.0, 0.0, 0.0, 0.0};
////    float jz[4] = {0.0, 0.0, 0.0, 0.0};
////    int cell_idx = 0;
//
//    Kokkos::View<int[1], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mover_flag(team_member.team_scratch(0));
//    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mover_dispx(team_member.team_scratch(0), num_entries);
//    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mover_dispy(team_member.team_scratch(0), num_entries);
//    Kokkos::View<float*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mover_dispz(team_member.team_scratch(0), num_entries);
//    Kokkos::View<int*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mover_i(team_member.team_scratch(0), num_entries);
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_entries), [&] (const int i) {
//      mover_dispx(i) = 0.0f;
//      mover_dispy(i) = 0.0f;
//      mover_dispz(i) = 0.0f;
//      mover_i(i) = 0;
//    });
//  
//    Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
//      mover_flag(0) = 0;
//    });
//
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [&] (const int pindex) {
//      int p_index = league_rank*per_league + pindex;
//      const int team_rank = team_member.team_rank();
//
////if(p_index < 43510)
////    printf("league_rank: %d, team_rank: %d, pindex: %d, p_index: %d\n", team_member.league_rank(), team_member.team_rank(), pindex, p_index);
//
////      auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//      if(p_index < np) {
//        mixed_t v0, v1, v2, v3, v4, v5;
//    
//        const int   ii   = pii;
//
////#ifdef __CUDA_ARCH__
////if(p_index < 4096) {
////  printf("blockIdx (x,y,z): (%d,%d,%d), threadIdx (x,y,z): (%d,%d,%d), p_index: %d, cell_index: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, p_index, ii);
////}
////#endif
//
//        mixed_t dx = p_dx; // Load position
//        mixed_t dy = p_dy;
//        mixed_t dz = p_dz;
//
//        mixed_t ux   = p_ux;                             // Load momentum
//        mixed_t uy   = p_uy;
//        mixed_t uz   = p_uz;
//        mixed_t q    = static_cast<mixed_t>(p_w);
//    
//        mixed_t hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
//                                dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
//        mixed_t hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
//                                dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
//        mixed_t haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
//                                dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
//
//        mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
//        mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
//        mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
//
////if(ii < start) {
////printf("league rank: %d, team rank: %d, start: %d, end: %d, end-start: %d, ii: %d, true_start: %d, true_end: %d\n", team_member.league_rank(), team_member.team_rank(), start, end, end-start, ii, k_part.i(league_rank*per_league), k_part.i(league_rank*per_league + per_league));
////}
//
//////      Shared Interpolators
////        mixed_t hax, hay, haz;
////        mixed_t cbx, cby, cbz;
////        if(start <= ii && ii < end && end-start < num_entries) {
////          int index = ii-start;
////          hax  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ex)) + 
////                            dy*mixed_t(cached_interp(index, interpolator_var::dexdy))) +
////                            dz*(mixed_t(cached_interp(index, interpolator_var::dexdz)) + 
////                            dy*mixed_t(cached_interp(index, interpolator_var::d2exdydz))) );
////          hay  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ey)) + 
////                            dz*mixed_t(cached_interp(index, interpolator_var::deydz))) +
////                            dx*(mixed_t(cached_interp(index, interpolator_var::deydx)) + 
////                            dz*mixed_t(cached_interp(index, interpolator_var::d2eydzdx))) );
////          haz  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ez)) + 
////                            dx*mixed_t(cached_interp(index, interpolator_var::dezdx))) +
////                            dy*(mixed_t(cached_interp(index, interpolator_var::dezdy)) + 
////                            dx*mixed_t(cached_interp(index, interpolator_var::d2ezdxdy))) );
////
////          cbx = mixed_t( cached_interp(ii-start, interpolator_var::cbx)) + 
////              dx*mixed_t(cached_interp(ii-start, interpolator_var::dcbxdx));
////          cby = mixed_t( cached_interp(ii-start, interpolator_var::cby)) + 
////              dy*mixed_t(cached_interp(ii-start, interpolator_var::dcbydy));
////          cbz = mixed_t( cached_interp(ii-start, interpolator_var::cbz)) + 
////              dz*mixed_t(cached_interp(ii-start, interpolator_var::dcbzdz));
////        } else {
////          hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
////                          dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
////          hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
////                          dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
////          haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
////                          dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
////
////          cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
////          cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
////          cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
////        }
//  
////        mixed_t ux   = p_ux;                             // Load momentum
////        mixed_t uy   = p_uy;
////        mixed_t uz   = p_uz;
////        mixed_t q    = static_cast<mixed_t>(p_w);
//    
//        ux  += hax;                               // Half advance E
//        uy  += hay;
//        uz  += haz;
//        v0   = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//        v1   = cbx*cbx + (cby*cby + cbz*cbz);
//        v2   = ( v0*v0 ) * v1;
//        v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
//        v4   = v3/(mixed_t(one)+v1*(v3*v3));
//        v4  += v4;
//    
//        v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
//        v1   = uy + v3*( uz*cbx - ux*cbz );
//        v2   = uz + v3*( ux*cby - uy*cbx );
//    
//        ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
//        uy  += v4*( v2*cbx - v0*cbz );
//        uz  += v4*( v0*cby - v1*cbx );
//    
//        ux  += hax;                               // Half advance E
//        uy  += hay;
//        uz  += haz;
//    
//        p_ux = ux;                               // Store momentum
//        p_uy = uy;
//        p_uz = uz;
//
//        v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
//        ux  *= cdt_dx_h;
//        uy  *= cdt_dy_h;
//        uz  *= cdt_dz_h;
//    
//        /**/                                      // Get norm displacement
//        ux  *= v0;
//        uy  *= v0;
//        uz  *= v0;
//    
//        v0   = dx + ux;                           // Streak midpoint (inbnds)
//        v1   = dy + uy;
//        v2   = dz + uz;
//    
//        v3   = v0 + ux;                           // New position
//        v4   = v1 + uy;
//        v5   = v2 + uz;
//
//// Warp reduction
//bool inbnds = v3<=mixed_t(one) && v4<=mixed_t(one) && v5<=mixed_t(one) &&
//              -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one);
//int mask = 0xffffffff;
//int synced = 0;
//int same_idx = 0;
//#ifdef __CUDA_ARCH__
//__match_all_sync(0xffffffff, inbnds, &synced);
//__match_all_sync(0xffffffff, ii, &same_idx);
//#endif
//          if(inbnds) {
//    
////        if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
////            -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {
//    
//          // Common case (inbnds).  Note: accumulator values are 4 times
//          // the total physical charge that passed through the appropriate
//          // current quadrant in a time-step
//    
//          q *= qsp_h;
//          p_dx = v3;                             // Store new position
//          p_dy = v4;
//          p_dz = v5;
//          dx = v0;                                // Streak midpoint
//          dy = v1;
//          dz = v2;
//          v5 = q*ux*uy*uz*one_third;              // Compute correction
//    
//         #define ACCUMULATE_J(X,Y,Z)                                        \
//          v4  = q*u##X;   /* v2 = q ux                            */        \
//          v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//          v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//          v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//          v4  = one+d##Z; /* v4 = 1+dz                            */        \
//          v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//          v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//          v4  = one-d##Z; /* v4 = 1-dz                            */        \
//          v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//          v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//          v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//          v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//          v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//          v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//// Warp reduction
//          if(synced && same_idx) {
//#ifdef __CUDA_ARCH__
//            ACCUMULATE_J( x,y,z );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//            }
//            ACCUMULATE_J( y,z,x );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//            }
//            ACCUMULATE_J( z,x,y );
//            for(int i=16; i>0; i=i/2) {
//              v0 += __shfl_down_sync(0xffffffff, v0, i);
//              v1 += __shfl_down_sync(0xffffffff, v1, i);
//              v2 += __shfl_down_sync(0xffffffff, v2, i);
//              v3 += __shfl_down_sync(0xffffffff, v3, i);
//            }
//            if(team_rank%32 == 0) {
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
//              Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//            }
//#endif
//          } else {
//            ACCUMULATE_J( x,y,z );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//            
//            ACCUMULATE_J( y,z,x );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//            
//            ACCUMULATE_J( z,x,y );
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
//            Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//          }
//
//////        Non-atomic local with atomic global accumulation
////          if(cell_idx != ii) {
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 0)), jx[0]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 1)), jx[1]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 2)), jx[2]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 3)), jx[3]);
////            ACCUMULATE_J( x,y,z );
////            jx[0] = static_cast<float>(v0);
////            jx[1] = static_cast<float>(v1);
////            jx[2] = static_cast<float>(v2);
////            jx[3] = static_cast<float>(v3);
////
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 0)), jy[0]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 1)), jy[1]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 2)), jy[2]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 3)), jy[3]);
////            ACCUMULATE_J( y,z,x );
////            jy[0] = static_cast<float>(v0);
////            jy[1] = static_cast<float>(v1);
////            jy[2] = static_cast<float>(v2);
////            jy[3] = static_cast<float>(v3);
////
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 0)), jz[0]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 1)), jz[1]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 2)), jz[2]); 
////            Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 3)), jz[3]);
////            ACCUMULATE_J( z,x,y );
////            jz[0] = static_cast<float>(v0);
////            jz[1] = static_cast<float>(v1);
////            jz[2] = static_cast<float>(v2);
////            jz[3] = static_cast<float>(v3);
////            cell_idx = ii;
////          } else {
////            ACCUMULATE_J( x,y,z );
////            jx[0] += static_cast<float>(v0);
////            jx[1] += static_cast<float>(v1);
////            jx[2] += static_cast<float>(v2);
////            jx[3] += static_cast<float>(v3);
////            ACCUMULATE_J( y,z,x );
////            jy[0] += static_cast<float>(v0);
////            jy[1] += static_cast<float>(v1);
////            jy[2] += static_cast<float>(v2);
////            jy[3] += static_cast<float>(v3);
////            ACCUMULATE_J( z,x,y );
////            jz[0] += static_cast<float>(v0);
////            jz[1] += static_cast<float>(v1);
////            jz[2] += static_cast<float>(v2);
////            jz[3] += static_cast<float>(v3);
////          }
//
//////        Atomic Accumulation
////          ACCUMULATE_J( x,y,z );
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
////          ACCUMULATE_J( y,z,x );
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
////          ACCUMULATE_J( z,x,y );
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
////          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//
//////        Non-atomic accumulation (Not accurate, only for perf comparisions)
////          ACCUMULATE_J( x,y,z );
////          k_accumulator(ii, accumulator_var::jx, 0) += static_cast<float>(v0);
////          k_accumulator(ii, accumulator_var::jx, 1) += static_cast<float>(v1);
////          k_accumulator(ii, accumulator_var::jx, 2) += static_cast<float>(v2);
////          k_accumulator(ii, accumulator_var::jx, 3) += static_cast<float>(v3);
////          ACCUMULATE_J( y,z,x );
////          k_accumulator(ii, accumulator_var::jy, 0) += static_cast<float>(v0);
////          k_accumulator(ii, accumulator_var::jy, 1) += static_cast<float>(v1);
////          k_accumulator(ii, accumulator_var::jy, 2) += static_cast<float>(v2);
////          k_accumulator(ii, accumulator_var::jy, 3) += static_cast<float>(v3);
////          ACCUMULATE_J( z,x,y );
////          k_accumulator(ii, accumulator_var::jz, 0) += static_cast<float>(v0);
////          k_accumulator(ii, accumulator_var::jz, 1) += static_cast<float>(v1);
////          k_accumulator(ii, accumulator_var::jz, 2) += static_cast<float>(v2);
////          k_accumulator(ii, accumulator_var::jz, 3) += static_cast<float>(v3);
//
//////        Shared atomic accumulation (accumulate in shared memory instead of global)
////          if(start <= ii && ii < end && end-start < num_entries) {
////            ACCUMULATE_J( x,y,z );
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 0)), static_cast<float>(v0));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 1)), static_cast<float>(v1));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 2)), static_cast<float>(v2));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jx, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( y,z,x );
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 0)), static_cast<float>(v0));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 1)), static_cast<float>(v1));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 2)), static_cast<float>(v2));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jy, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( z,x,y );
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 0)), static_cast<float>(v0));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 1)), static_cast<float>(v1));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 2)), static_cast<float>(v2));
////            atomic_add(&(atomic_cache(ii-start, accumulator_var::jz, 3)), static_cast<float>(v3));
////          } else {
////            ACCUMULATE_J( x,y,z );
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 0)), static_cast<float>(v0));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 1)), static_cast<float>(v1));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 2)), static_cast<float>(v2));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jx, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( y,z,x );
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 0)), static_cast<float>(v0));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 1)), static_cast<float>(v1));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 2)), static_cast<float>(v2));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jy, 3)), static_cast<float>(v3));
////            ACCUMULATE_J( z,x,y );
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 0)), static_cast<float>(v0));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 1)), static_cast<float>(v1));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 2)), static_cast<float>(v2));
////            atomic_add(&(k_accumulator(ii, accumulator_var::jz, 3)), static_cast<float>(v3));
////          }
//    
//////        ScatterView
////          ACCUMULATE_J( x,y,z );
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
////          k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;
////          ACCUMULATE_J( y,z,x );
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
////          k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;
////          ACCUMULATE_J( z,x,y );
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
////          k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
//
//    #     undef ACCUMULATE_J
//        } 
//        else
//        {
//          int mover_count = Kokkos::atomic_fetch_add(&mover_flag(0), 1);
//          mover_dispx(mover_count) = ux;
//          mover_dispy(mover_count) = uy;
//          mover_dispz(mover_count) = uz;
//          mover_i(mover_count) = p_index;
//        }
//        if(mover_flag(0) > 512) {
//if(team_rank < mover_flag(0)) {
//          k_particle_mover_t local_pm[1];
//          local_pm->dispx = mover_dispx(team_rank);
//          local_pm->dispy = mover_dispy(team_rank);
//          local_pm->dispz = mover_dispz(team_rank);
//          local_pm->i = mover_i(team_rank);
//          if( move_p_kokkos(k_part, local_pm,
//                             k_accumulator, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
//            if( k_nm(0)<max_nm ) {
//              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
//              if (nm >= max_nm) Kokkos::abort("overran max_nm");
//    
//              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
//              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
//              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
//              k_particle_movers_i(nm)   = local_pm->i;
//    
//              k_particle_copy.dx(nm) = k_part.dx(p_index);
//              k_particle_copy.dy(nm) = k_part.dy(p_index);
//              k_particle_copy.dz(nm) = k_part.dz(p_index);
//              k_particle_copy.ux(nm) = k_part.ux(p_index);
//              k_particle_copy.uy(nm) = k_part.uy(p_index);
//              k_particle_copy.uz(nm) = k_part.uz(p_index);
//              k_particle_copy.w(nm) = k_part.w(p_index);
//              k_particle_copy.i(nm) = k_part.i(p_index);
//            }
//          }
//          mover_flag(0) = 0;
//}
//        }
//        
////        {                                    // Unlikely
////          k_particle_mover_t local_pm[1];
////          local_pm->dispx = ux;
////          local_pm->dispy = uy;
////          local_pm->dispz = uz;
////          local_pm->i     = p_index;
////    
////          if( move_p_kokkos(k_part, local_pm,
////                             k_accumulator, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
////            if( k_nm(0)<max_nm ) {
////              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
////              if (nm >= max_nm) Kokkos::abort("overran max_nm");
////    
////              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
////              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
////              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
////              k_particle_movers_i(nm)   = local_pm->i;
////    
////              k_particle_copy.dx(nm) = k_part.dx(p_index);
////              k_particle_copy.dy(nm) = k_part.dy(p_index);
////              k_particle_copy.dz(nm) = k_part.dz(p_index);
////              k_particle_copy.ux(nm) = k_part.ux(p_index);
////              k_particle_copy.uy(nm) = k_part.uy(p_index);
////              k_particle_copy.uz(nm) = k_part.uz(p_index);
////              k_particle_copy.w(nm) = k_part.w(p_index);
////              k_particle_copy.i(nm) = k_part.i(p_index);
////            }
////          }
////        }
//      }
//    });
//
//////  Non-atomic local with atomic global accumulation
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 0)), jx[0]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 1)), jx[1]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 2)), jx[2]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jx, 3)), jx[3]);
////
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 0)), jy[0]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 1)), jy[1]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 2)), jy[2]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jy, 3)), jy[3]);
////
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 0)), jz[0]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 1)), jz[1]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 2)), jz[2]); 
////    Kokkos::atomic_add(&(k_accumulator(cell_idx, accumulator_var::jz, 3)), jz[3]);
//
//////  Shared atomic accumulation (accumulate in shared memory instead of global)
////    if(end-start < num_entries) {
////      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end-start), [=] (const int idx) {
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 0)), atomic_cache(idx,0,0)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 0)), atomic_cache(idx,1,0)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 0)), atomic_cache(idx,2,0)); 
////
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 1)), atomic_cache(idx,0,1)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 1)), atomic_cache(idx,1,1)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 1)), atomic_cache(idx,2,1)); 
////
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 2)), atomic_cache(idx,0,2)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 2)), atomic_cache(idx,1,2)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 2)), atomic_cache(idx,2,2)); 
////
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 0, 3)), atomic_cache(idx,0,3)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 1, 3)), atomic_cache(idx,1,3)); 
////        Kokkos::atomic_add(&(k_accumulator(start+idx, 2, 3)), atomic_cache(idx,2,3)); 
////      });
////    }
//  });
//
//  // TODO: abstract this manual data copy
//  //Kokkos::deep_copy(h_nm, k_nm);
//
//  //args->seg[pipeline_rank].pm        = pm;
//  //args->seg[pipeline_rank].max_nm    = max_nm;
//  //args->seg[pipeline_rank].nm        = h_nm(0);
//  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
//  //delete(k_local_particle_movers_p);
//  //return h_nm(0);
//
//  #undef p_dx  
//  #undef p_dy  
//  #undef p_dz  
//  #undef p_ux  
//  #undef p_uy  
//  #undef p_uz  
//  #undef p_w   
//  #undef pii   
//
//  #undef f_cbx 
//  #undef f_cby 
//  #undef f_cbz 
//  #undef f_ex  
//  #undef f_ey  
//  #undef f_ez  
//
//  #undef f_dexdy    
//  #undef f_dexdz    
//
//  #undef f_d2exdydz 
//  #undef f_deydx    
//  #undef f_deydz    
//
//  #undef f_d2eydzdx 
//  #undef f_dezdx    
//  #undef f_dezdy    
//
//  #undef f_d2ezdxdy 
//  #undef f_dcbxdx   
//  #undef f_dcbydy   
//  #undef f_dcbzdz   
//}

//// Team Policy, for loop
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t& k_accumulator,
//        k_interpolator_t& k_interp,
////        Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>>& k_interp,
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
//  constexpr mixed_t one            = 1.;
//  constexpr mixed_t one_third      = 1./3.;
//  constexpr mixed_t two_fifteenths = 2./15.;
//
////  const mixed_t one            = mixed_t(1.0);
////  const mixed_t one_third      = mixed_t(1./3.);
////  const mixed_t two_fifteenths = mixed_t(2./15.);
//  const mixed_t qdt_2mc_h = static_cast<mixed_t>(qdt_2mc);
//  const mixed_t cdt_dx_h = static_cast<mixed_t>(cdt_dx);
//  const mixed_t cdt_dy_h = static_cast<mixed_t>(cdt_dy);
//  const mixed_t cdt_dz_h = static_cast<mixed_t>(cdt_dz);
//  const mixed_t qsp_h = static_cast<mixed_t>(qsp);
//
//  #define p_dx    k_part.dx(p_index)
//  #define p_dy    k_part.dy(p_index)
//  #define p_dz    k_part.dz(p_index)
//  #define p_ux    k_part.ux(p_index)
//  #define p_uy    k_part.uy(p_index)
//  #define p_uz    k_part.uz(p_index)
//  #define p_w     k_part.w(p_index)
//  #define pii     k_part.i(p_index)
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
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    k_nm(0) = 0;
//  });
//
////  int num_threads = 32;
////  int num_leagues = np/num_threads;
////  if(num_threads*num_leagues < np)
////    num_leagues++;
////  int per_league = num_threads;
//
//  int num_threads = 1024;
//  int num_leagues = 2048;
//  int per_league = np/num_leagues;
//  if(np%num_leagues > 0)
//    per_league++;
//  int per_thread = per_league/num_threads;
//  if(per_league%num_threads > 0)
//    per_thread++;
//
//  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
//  typedef Kokkos::View<float*[18], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_interp;
//  typedef Kokkos::View<float*[3][4], ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_accum;
//
//  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1);
////  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1).set_scratch_size(0, Kokkos::PerTeam(100*sizeof(float)*18));
//
//  Kokkos::parallel_for("advance_p", team_policy, 
//  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    const int league_rank = team_member.league_rank();
//    const int team_rank = team_member.team_rank();
////    int start = k_part.i(league_rank*per_league);
////    int end = start+100;
//
////    shared_interp cached_interp(team_member.team_scratch(0), 100);
//
////    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 18*100), [=] (const int index) {
////      int idx = index / 18;
////      int var = index % 18;
////      cached_interp(idx, var) = k_interp(start+idx, var);
////    });
//
////    int cell_idx = k_part.i(league_rank*num_threads*per_thread + team_rank*per_thread);
////    int cell_idx = k_part.i(league_rank*num_threads*per_thread + team_rank);
////    float jx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
////    float jy[4] = {0.0f, 0.0f, 0.0f, 0.0f};
////    float jz[4] = {0.0f, 0.0f, 0.0f, 0.0f};
//
////    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_threads), [=] (const int i) {
//
//    for(int idx=0; idx<per_thread; idx++) {
////      int p_index = league_rank*num_threads*per_thread + team_rank*per_thread + idx;
//      int p_index = league_rank*num_threads*per_thread + idx*num_threads + team_rank;;
//      if(p_index < np) {
//        mixed_t v0, v1, v2, v3, v4, v5;
//    
//        const int   ii   = pii;
//        mixed_t dx = p_dx; // Load position
//        mixed_t dy = p_dy;
//        mixed_t dz = p_dz;
//
//        mixed_t ux   = p_ux;                             // Load momentum
//        mixed_t uy   = p_uy;
//        mixed_t uz   = p_uz;
//        mixed_t q    = static_cast<mixed_t>(p_w);
//    
//        mixed_t hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
//                                dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
//        mixed_t hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
//                                dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
//        mixed_t haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
//                                dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
//
//        mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
//        mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
//        mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
//
////        mixed_t hax, hay, haz;
////        mixed_t cbx, cby, cbz;
////
////        if(start <= ii && ii < end) {
////          int index = ii-start;
////          hax  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ex)) + 
////                            dy*mixed_t(cached_interp(index, interpolator_var::dexdy))) +
////                            dz*(mixed_t(cached_interp(index, interpolator_var::dexdz)) + 
////                            dy*mixed_t(cached_interp(index, interpolator_var::d2exdydz))) );
////          hay  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ey)) + 
////                            dz*mixed_t(cached_interp(index, interpolator_var::deydz))) +
////                            dx*(mixed_t(cached_interp(index, interpolator_var::deydx)) + 
////                            dz*mixed_t(cached_interp(index, interpolator_var::d2eydzdx))) );
////          haz  = qdt_2mc_h*( (mixed_t(cached_interp(index, interpolator_var::ez)) + 
////                            dx*mixed_t(cached_interp(index, interpolator_var::dezdx))) +
////                            dy*(mixed_t(cached_interp(index, interpolator_var::dezdy)) + 
////                            dx*mixed_t(cached_interp(index, interpolator_var::d2ezdxdy))) );
////
////          cbx = mixed_t( cached_interp(ii-start, interpolator_var::cbx)) + 
////              dx*mixed_t(cached_interp(ii-start, interpolator_var::dcbxdx));
////          cby = mixed_t( cached_interp(ii-start, interpolator_var::cby)) + 
////              dy*mixed_t(cached_interp(ii-start, interpolator_var::dcbydy));
////          cbz = mixed_t( cached_interp(ii-start, interpolator_var::cbz)) + 
////              dz*mixed_t(cached_interp(ii-start, interpolator_var::dcbzdz));
////        } else {
////          hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
////                          dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
////          hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
////                          dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
////          haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
////                          dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
////
////          cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
////          cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
////          cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
////        }
//    
//        ux  += hax;                               // Half advance E
//        uy  += hay;
//        uz  += haz;
//        v0   = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//        v1   = cbx*cbx + (cby*cby + cbz*cbz);
//        v2   = ( v0*v0 ) * v1;
//        v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
//        v4   = v3/(mixed_t(one)+v1*(v3*v3));
//        v4  += v4;
//    
//        v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
//        v1   = uy + v3*( uz*cbx - ux*cbz );
//        v2   = uz + v3*( ux*cby - uy*cbx );
//    
//        ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
//        uy  += v4*( v2*cbx - v0*cbz );
//        uz  += v4*( v0*cby - v1*cbx );
//    
//        ux  += hax;                               // Half advance E
//        uy  += hay;
//        uz  += haz;
//    
//        p_ux = ux;                               // Store momentum
//        p_uy = uy;
//        p_uz = uz;
//
//        v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
//        ux  *= cdt_dx_h;
//        uy  *= cdt_dy_h;
//        uz  *= cdt_dz_h;
//    
//        /**/                                      // Get norm displacement
//        ux  *= v0;
//        uy  *= v0;
//        uz  *= v0;
//    
//        v0   = dx + ux;                           // Streak midpoint (inbnds)
//        v1   = dy + uy;
//        v2   = dz + uz;
//    
//        v3   = v0 + ux;                           // New position
//        v4   = v1 + uy;
//        v5   = v2 + uz;
//    
//// Warp reduction
////bool inbnds = v3<=mixed_t(one) && v4<=mixed_t(one) && v5<=mixed_t(one) &&
////              -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one);
////int mask = 0xffffffff;
////int synced = 0;
////int same_idx = 0;
////#ifdef __CUDA_ARCH__
////__match_all_sync(mask, inbnds, &synced);
////__match_all_sync(mask, ii, &same_idx);
////#endif
////          if(inbnds) {
//
//        if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
//            -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {
//    
//          // Common case (inbnds).  Note: accumulator values are 4 times
//          // the total physical charge that passed through the appropriate
//          // current quadrant in a time-step
//    
//          q *= qsp_h;
//          p_dx = v3;                             // Store new position
//          p_dy = v4;
//          p_dz = v5;
//          dx = v0;                                // Streak midpoint
//          dy = v1;
//          dz = v2;
//          v5 = q*ux*uy*uz*one_third;              // Compute correction
//    
//    #     define ACCUMULATE_J(X,Y,Z)                                 \
//          v4  = q*u##X;   /* v2 = q ux                            */        \
//          v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//          v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//          v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//          v4  = one+d##Z; /* v4 = 1+dz                            */        \
//          v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//          v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//          v4  = one-d##Z; /* v4 = 1-dz                            */        \
//          v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//          v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//          v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//          v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//          v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//          v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//// Warp reduction
////if(synced && same_idx) {
////#ifdef __CUDA_ARCH__
////  ACCUMULATE_J( x,y,z );
////  for(int i=16; i>0; i=i/2) {
////    v0 += __shfl_down_sync(mask, v0, i);
////    v1 += __shfl_down_sync(mask, v1, i);
////    v2 += __shfl_down_sync(mask, v2, i);
////    v3 += __shfl_down_sync(mask, v3, i);
////  }
////  if(team_rank == 0) {
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
////  }
////  ACCUMULATE_J( y,z,x );
////  for(int i=16; i>0; i=i/2) {
////    v0 += __shfl_down_sync(mask, v0, i);
////    v1 += __shfl_down_sync(mask, v1, i);
////    v2 += __shfl_down_sync(mask, v2, i);
////    v3 += __shfl_down_sync(mask, v3, i);
////  }
////  if(team_rank == 0) {
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
////  }
////  ACCUMULATE_J( z,x,y );
////  for(int i=16; i>0; i=i/2) {
////    v0 += __shfl_down_sync(mask, v0, i);
////    v1 += __shfl_down_sync(mask, v1, i);
////    v2 += __shfl_down_sync(mask, v2, i);
////    v3 += __shfl_down_sync(mask, v3, i);
////  }
////  if(team_rank == 0) {
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
////    Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
////  }
////#endif
////} else {
////  ACCUMULATE_J( x,y,z );
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
////  
////  ACCUMULATE_J( y,z,x );
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
////  
////  ACCUMULATE_J( z,x,y );
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
////  Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
////}
//    
////// Non-atomic accumulation
////          if(cell_idx == ii) {
////            ACCUMULATE_J( x,y,z );
////            jx[0] += static_cast<float>(v0);
////            jx[1] += static_cast<float>(v1);
////            jx[2] += static_cast<float>(v2);
////            jx[3] += static_cast<float>(v3);
////            ACCUMULATE_J( y,z,x );
////            jy[0] += static_cast<float>(v0);
////            jy[1] += static_cast<float>(v1);
////            jy[2] += static_cast<float>(v2);
////            jy[3] += static_cast<float>(v3);
////            ACCUMULATE_J( z,x,y );
////            jz[0] += static_cast<float>(v0);
////            jz[1] += static_cast<float>(v1);
////            jz[2] += static_cast<float>(v2);
////            jz[3] += static_cast<float>(v3);
////          } else {
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 0), jx[0]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 1), jx[1]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 2), jx[2]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 3), jx[3]);
////
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 0), jy[0]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 1), jy[1]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 2), jy[2]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 3), jy[3]);
////
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 0), jz[0]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 1), jz[1]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 2), jz[2]);
////            Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 3), jz[3]);
////
////            ACCUMULATE_J( x,y,z );
////            jx[0] = static_cast<float>(v0);
////            jx[1] = static_cast<float>(v1);
////            jx[2] = static_cast<float>(v2);
////            jx[3] = static_cast<float>(v3);
////
////            ACCUMULATE_J( y,z,x );
////            jy[0] = static_cast<float>(v0);
////            jy[1] = static_cast<float>(v1);
////            jy[2] = static_cast<float>(v2);
////            jy[3] = static_cast<float>(v3);
////
////            ACCUMULATE_J( z,x,y );
////            jz[0] = static_cast<float>(v0);
////            jz[1] = static_cast<float>(v1);
////            jz[2] = static_cast<float>(v2);
////            jz[3] = static_cast<float>(v3);
////
////            cell_idx = ii;
////          }
//
//// Regular
//          ACCUMULATE_J( x,y,z );
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 0), static_cast<float>(v0));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 1), static_cast<float>(v1));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 2), static_cast<float>(v2));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jx, 3), static_cast<float>(v3));
//  
//          ACCUMULATE_J( y,z,x );
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 0), static_cast<float>(v0));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 1), static_cast<float>(v1));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 2), static_cast<float>(v2));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jy, 3), static_cast<float>(v3));
//  
//          ACCUMULATE_J( z,x,y );
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 0), static_cast<float>(v0));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 1), static_cast<float>(v1));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 2), static_cast<float>(v2));
//          Kokkos::atomic_add(&k_accumulator(ii, accumulator_var::jz, 3), static_cast<float>(v3));
//    #     undef ACCUMULATE_J
//        } 
//        else
//        {                                    // Unlikely
//          k_particle_mover_t local_pm[1];
//          local_pm->dispx = ux;
//          local_pm->dispy = uy;
//          local_pm->dispz = uz;
//          local_pm->i     = p_index;
//    
//          //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
//          if( move_p_kokkos(k_part, local_pm,
//                             k_accumulator, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
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
//              k_particle_copy.dx(nm) = k_part.dx(p_index);
//              k_particle_copy.dy(nm) = k_part.dy(p_index);
//              k_particle_copy.dz(nm) = k_part.dz(p_index);
//              k_particle_copy.ux(nm) = k_part.ux(p_index);
//              k_particle_copy.uy(nm) = k_part.uy(p_index);
//              k_particle_copy.uz(nm) = k_part.uz(p_index);
//              k_particle_copy.w(nm) = k_part.w(p_index);
//              k_particle_copy.i(nm) = k_part.i(p_index);
//            }
//          }
//        }
//      }
//    }
////// Non-atomic accumulation
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 0), jx[0]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 1), jx[1]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 2), jx[2]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jx, 3), jx[3]);
////
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 0), jy[0]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 1), jy[1]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 2), jy[2]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jy, 3), jy[3]);
////
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 0), jz[0]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 1), jz[1]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 2), jz[2]);
////    Kokkos::atomic_add(&k_accumulator(cell_idx, accumulator_var::jz, 3), jz[3]);
////    });
//  });
//
//  #undef p_dx  
//  #undef p_dy  
//  #undef p_dz  
//  #undef p_ux  
//  #undef p_uy  
//  #undef p_uz  
//  #undef p_w   
//  #undef pii   
//
//  #undef f_cbx 
//  #undef f_cby 
//  #undef f_cbz 
//  #undef f_ex  
//  #undef f_ey  
//  #undef f_ez  
//
//  #undef f_dexdy    
//  #undef f_dexdz    
//
//  #undef f_d2exdydz 
//  #undef f_deydx    
//  #undef f_deydz    
//
//  #undef f_d2eydzdx 
//  #undef f_dezdx    
//  #undef f_dezdy    
//
//  #undef f_d2ezdxdy 
//  #undef f_dcbxdx   
//  #undef f_dcbydy   
//  #undef f_dcbzdz   
//}

//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t& k_accumulator,
//        k_interpolator_t& k_interp,
////        Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>>& k_interp,
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
//  #define p_dx    k_part.dx(p_index)
//  #define p_dy    k_part.dy(p_index)
//  #define p_dz    k_part.dz(p_index)
//  #define p_ux    k_part.ux(p_index)
//  #define p_uy    k_part.uy(p_index)
//  #define p_uz    k_part.uz(p_index)
//  #define p_w     k_part.w(p_index)
//  #define pii     k_part.i(p_index)
//
//  constexpr float one            = 1.;
//  constexpr float one_third      = 1./3.;
//  constexpr float two_fifteenths = 2./15.;
//
//  const mixed_t qdt_2mc_h = static_cast<mixed_t>(qdt_2mc);
//  const mixed_t cdt_dx_h = static_cast<mixed_t>(cdt_dx);
//  const mixed_t cdt_dy_h = static_cast<mixed_t>(cdt_dy);
//  const mixed_t cdt_dz_h = static_cast<mixed_t>(cdt_dz);
//  const mixed_t qsp_h = static_cast<mixed_t>(qsp);
//
//  auto rangel = g->rangel;
//  auto rangeh = g->rangeh;
//  const int nv = g->nv;
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(size_t i) {
//    k_nm(0) = 0;
//  });
//
//  Kokkos::View<int*> start_indices("Start indices", nv);
//  Kokkos::parallel_for("find starts", Kokkos::RangePolicy<>(0,nv), KOKKOS_LAMBDA(int i) {
//    
//  });
//
////  const int num_threads = 32;
////  Kokkos::TeamPolicy<> team_policy = Kokkos::TeamPolicy<>(num_leagues, num_threads, 1);
////  Kokkos::parallel_for("advance_p", team_policy, 
////  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
////    const int league_rank = team_member.league_rank();
////
////    const float f_cbx = k_interp(league_rank, interpolator_var::cbx);
////    const float f_cby = k_interp(league_rank, interpolator_var::cby);
////    const float f_cbz = k_interp(league_rank, interpolator_var::cbz);
////
////    const float f_ex = k_interp(league_rank, interpolator_var::ex);
////    const float f_ey = k_interp(league_rank, interpolator_var::ey);
////    const float f_ez = k_interp(league_rank, interpolator_var::ez);
////
////    const float f_dexdy = k_interp(league_rank, interpolator_var::dexdy);
////    const float f_dexdz = k_interp(league_rank, interpolator_var::dexdz);
////    const float f_d2exdydz = k_interp(league_rank, interpolator_var::d2exdydz);
////
////    const float f_deydx = k_interp(league_rank, interpolator_var::deydx);
////    const float f_deydz = k_interp(league_rank, interpolator_var::deydz);
////    const float f_d2eydzdx = k_interp(league_rank, interpolator_var::d2eydzdx);
////
////    const float f_dezdx = k_interp(league_rank, interpolator_var::dezdx);
////    const float f_dezdy = k_interp(league_rank, interpolator_var::dezdy);
////    const float f_d2ezdxdy = k_interp(league_rank, interpolator_var::d2ezdxdy);
////
////    const float f_dcbxdx = k_interp(league_rank, interpolator_var::dcbxdx);
////    const float f_dcbydy = k_interp(league_rank, interpolator_var::dcbydy);
////    const float f_dcbzdz = k_interp(league_rank, interpolator_var::dcbzdz);
////
////    Kokkos::View<float[4]> jx;
////    Kokkos::View<float[4]> jy;
////    Kokkos::View<float[4]> jz;
////
////    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, np), [=] (const int p_index) {
////      const int ii = k_part.i(p_index);
////      if(ii == league_rank) {
////        mixed_t v0, v1, v2, v3, v4, v5;
////    
////        mixed_t dx = p_dx; // Load position
////        mixed_t dy = p_dy;
////        mixed_t dz = p_dz;
////
////        mixed_t ux   = p_ux;                             // Load momentum
////        mixed_t uy   = p_uy;
////        mixed_t uz   = p_uz;
////        mixed_t q    = static_cast<mixed_t>(p_w);
////    
////        mixed_t hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
////                                dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
////        mixed_t hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
////                                dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
////        mixed_t haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
////                                dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
////
////        mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
////        mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
////        mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
////    
////        ux  += hax;                               // Half advance E
////        uy  += hay;
////        uz  += haz;
////        v0   = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
////        v1   = cbx*cbx + (cby*cby + cbz*cbz);
////        v2   = ( v0*v0 ) * v1;
////        v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
////        v4   = v3/(mixed_t(one)+v1*(v3*v3));
////        v4  += v4;
////    
////        v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
////        v1   = uy + v3*( uz*cbx - ux*cbz );
////        v2   = uz + v3*( ux*cby - uy*cbx );
////    
////        ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
////        uy  += v4*( v2*cbx - v0*cbz );
////        uz  += v4*( v0*cby - v1*cbx );
////    
////        ux  += hax;                               // Half advance E
////        uy  += hay;
////        uz  += haz;
////    
////        p_ux = ux;                               // Store momentum
////        p_uy = uy;
////        p_uz = uz;
////
////        v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
////        ux  *= cdt_dx_h;
////        uy  *= cdt_dy_h;
////        uz  *= cdt_dz_h;
////    
////        /**/                                      // Get norm displacement
////        ux  *= v0;
////        uy  *= v0;
////        uz  *= v0;
////    
////        v0   = dx + ux;                           // Streak midpoint (inbnds)
////        v1   = dy + uy;
////        v2   = dz + uz;
////    
////        v3   = v0 + ux;                           // New position
////        v4   = v1 + uy;
////        v5   = v2 + uz;
////    
////        if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
////            -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {
////    
////          // Common case (inbnds).  Note: accumulator values are 4 times
////          // the total physical charge that passed through the appropriate
////          // current quadrant in a time-step
////    
////          q *= qsp_h;
////          p_dx = v3;                             // Store new position
////          p_dy = v4;
////          p_dz = v5;
////          dx = v0;                                // Streak midpoint
////          dy = v1;
////          dz = v2;
////          v5 = q*ux*uy*uz*one_third;              // Compute correction
////    
////    #     define ACCUMULATE_J(X,Y,Z)                                 \
////          v4  = q*u##X;   /* v2 = q ux                            */        \
////          v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
////          v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
////          v1 += v4;       /* v1 = q ux (1+dy)                     */        \
////          v4  = one+d##Z; /* v4 = 1+dz                            */        \
////          v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
////          v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
////          v4  = one-d##Z; /* v4 = 1-dz                            */        \
////          v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
////          v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
////          v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
////          v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
////          v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
////          v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
////
////// Regular
////          ACCUMULATE_J( x,y,z );
////          k_accumulator(league_rank, accumulator_var::jx, 0) += static_cast<float>(v0);
////          k_accumulator(league_rank, accumulator_var::jx, 1) += static_cast<float>(v1);
////          k_accumulator(league_rank, accumulator_var::jx, 2) += static_cast<float>(v2);
////          k_accumulator(league_rank, accumulator_var::jx, 3) += static_cast<float>(v3);
////  
////          ACCUMULATE_J( y,z,x );
////          k_accumulator(league_rank, accumulator_var::jy, 0) += static_cast<float>(v0);
////          k_accumulator(league_rank, accumulator_var::jy, 1) += static_cast<float>(v1);
////          k_accumulator(league_rank, accumulator_var::jy, 2) += static_cast<float>(v2);
////          k_accumulator(league_rank, accumulator_var::jy, 3) += static_cast<float>(v3);
////  
////          ACCUMULATE_J( z,x,y );
////          k_accumulator(league_rank, accumulator_var::jz, 0) += static_cast<float>(v0);
////          k_accumulator(league_rank, accumulator_var::jz, 1) += static_cast<float>(v1);
////          k_accumulator(league_rank, accumulator_var::jz, 2) += static_cast<float>(v2);
////          k_accumulator(league_rank, accumulator_var::jz, 3) += static_cast<float>(v3);
////    #     undef ACCUMULATE_J
////        } 
////        else
////        {                                    // Unlikely
////          k_particle_mover_t local_pm[1];
////          local_pm->dispx = ux;
////          local_pm->dispy = uy;
////          local_pm->dispz = uz;
////          local_pm->i     = p_index;
////    
////          //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
////          if( move_p_kokkos(k_part, local_pm,
////                             k_accumulator, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
////            if( k_nm(0)<max_nm ) {
////              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
////              if (nm >= max_nm) Kokkos::abort("overran max_nm");
////    
////              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
////              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
////              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
////              k_particle_movers_i(nm)   = local_pm->i;
////    
////              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
////              k_particle_copy.dx(nm) = k_part.dx(p_index);
////              k_particle_copy.dy(nm) = k_part.dy(p_index);
////              k_particle_copy.dz(nm) = k_part.dz(p_index);
////              k_particle_copy.ux(nm) = k_part.ux(p_index);
////              k_particle_copy.uy(nm) = k_part.uy(p_index);
////              k_particle_copy.uz(nm) = k_part.uz(p_index);
////              k_particle_copy.w(nm) = k_part.w(p_index);
////              k_particle_copy.i(nm) = k_part.i(p_index);
////            }
////          }
////        }
////      }
////    });
//////    k_accumulator(league_rank, accumulator_var::jx, 0) += jx(0);
//////    k_accumulator(league_rank, accumulator_var::jx, 1) += jx(1);
//////    k_accumulator(league_rank, accumulator_var::jx, 2) += jx(2);
//////    k_accumulator(league_rank, accumulator_var::jx, 3) += jx(3);
//////
//////    k_accumulator(league_rank, accumulator_var::jy, 0) += jy(0);
//////    k_accumulator(league_rank, accumulator_var::jy, 1) += jy(1);
//////    k_accumulator(league_rank, accumulator_var::jy, 2) += jy(2);
//////    k_accumulator(league_rank, accumulator_var::jy, 3) += jy(3);
//////
//////    k_accumulator(league_rank, accumulator_var::jz, 0) += jz(0);
//////    k_accumulator(league_rank, accumulator_var::jz, 1) += jz(1);
//////    k_accumulator(league_rank, accumulator_var::jz, 2) += jz(2);
//////    k_accumulator(league_rank, accumulator_var::jz, 3) += jz(3);
////  });
//  #undef p_dx
//  #undef p_dy
//  #undef p_dz
//  #undef p_ux
//  #undef p_uy
//  #undef p_uz
//  #undef p_w 
//  #undef pii 
//}

//// Float
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
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
//  /*
//  k_particle_movers_t *k_local_particle_movers_p = new k_particle_movers_t("k_local_pm", 1);
//  k_particle_movers_t  k_local_particle_movers("k_local_pm", 1);
//
//  k_iterator_t k_nm("k_nm");
//  k_iterator_t::HostMirror h_nm = Kokkos::create_mirror_view(k_nm);
//  h_nm(0) = 0;
//  Kokkos::deep_copy(k_nm, h_nm);
//  */
//  // Determine which quads of particles quads this pipeline processes
//
//  //DISTRIBUTE( args->np, 16, pipeline_rank, n_pipeline, itmp, n );
//  //p = args->p0 + itmp;
//
//  /*
//  printf("original value %f\n\n", k_accumulators(0, 0, 0));
//sp_[id]->
//  Kokkos::parallel_for(Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int i) {
//
//      auto scatter_access = k_accumulators_sa.access();
//      //auto scatter_access_atomic = scatter_view.template access<Kokkos::Experimental::ScatterAtomic>();
//          printf("Writing to %d\n", i);
//          scatter_access(i, 0, 0) += 4;
//          //scatter_access_atomic(i, 1) += 2.0;
//          //scatter_access(k, 2) += 1.0;
//          //
//  });
//
//  // copy back
//  Kokkos::Experimental::contribute(k_accumulators, k_accumulators_sa);
//  printf("changed value %f\n", k_accumulators(0, 0, 0));
//  */
//
//  // Determine which movers are reserved for this pipeline
//  // Movers (16 bytes) should be reserved for pipelines in at least
//  // multiples of 8 such that the set of particle movers reserved for
//  // a pipeline is 128-byte aligned and a multiple of 128-byte in
//  // size.  The host is guaranteed to get enough movers to process its
//  // particles with this allocation.
///*
//  max_nm = args->max_nm - (args->np&15);
//  if( max_nm<0 ) max_nm = 0;
//  DISTRIBUTE( max_nm, 8, pipeline_rank, n_pipeline, itmp, max_nm );
//  if( pipeline_rank==n_pipeline ) max_nm = args->max_nm - itmp;
//  pm   = args->pm + itmp;
//  nm   = 0;
//  itmp = 0;
//
//  // Determine which accumulator array to use
//  // The host gets the first accumulator array
//
//  if( pipeline_rank!=n_pipeline )
//    a0 += (1+pipeline_rank)*
//          POW2_CEIL((args->nx+2)*(args->ny+2)*(args->nz+2),2);
//*/
//  // Process particles for this pipeline
//
//  #define p_dx    k_part.dx(p_index)
//  #define p_dy    k_part.dy(p_index)
//  #define p_dz    k_part.dz(p_index)
//  #define p_ux    k_part.ux(p_index)
//  #define p_uy    k_part.uy(p_index)
//  #define p_uz    k_part.uz(p_index)
//  #define p_w     k_part.w(p_index)
//  #define pii     k_part.i(p_index)
//
////  #define p_dx    k_particles(p_index, particle_var::dx)
////  #define p_dy    k_particles(p_index, particle_var::dy)
////  #define p_dz    k_particles(p_index, particle_var::dz)
////  #define p_ux    k_particles(p_index, particle_var::ux)
////  #define p_uy    k_particles(p_index, particle_var::uy)
////  #define p_uz    k_particles(p_index, particle_var::uz)
////  #define p_w     k_particles(p_index, particle_var::w)
////  #define pii     k_particles_i(p_index)
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
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    //printf("how many times does this run %d", i);
//    k_nm(0) = 0;
//    //local_pm_dispx = 0;
//    //local_pm_dispy = 0;
//    //local_pm_dispz = 0;
//    //local_pm_i = 0;
//  });
//
//
////Kokkos::parallel_for(Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(int i) {
////  k_particles(i, particle_var::dx) = k_part.dx(i);
////  k_particles(i, particle_var::dy) = k_part.dy(i);
////  k_particles(i, particle_var::dz) = k_part.dz(i);
////  k_particles(i, particle_var::ux) = k_part.ux(i);
////  k_particles(i, particle_var::uy) = k_part.uy(i);
////  k_particles(i, particle_var::uz) = k_part.uz(i);
////  k_particles(i, particle_var::w) = k_part.w(i);
////  k_particles_i(i) = k_part.i(i);
////});
//
//  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
//    KOKKOS_LAMBDA (size_t p_index)
//    {
////for(int p_index=0; p_index<np; p_index++) {
//    float v0, v1, v2, v3, v4, v5;
//    auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//
//    float dx   = p_dx;                             // Load position
//    float dy   = p_dy;
//    float dz   = p_dz;
//    int   ii   = pii;
//    float hax  = qdt_2mc*(    ( f_ex    + dy*f_dexdy    ) +
//                           dz*( f_dexdz + dy*f_d2exdydz ) );
//    float hay  = qdt_2mc*(    ( f_ey    + dz*f_deydz    ) +
//                           dx*( f_deydx + dz*f_d2eydzdx ) );
//    float haz  = qdt_2mc*(    ( f_ez    + dx*f_dezdx    ) +
//                           dy*( f_dezdy + dx*f_d2ezdxdy ) );
//    //printf(" inter %d vs %ld \n", ii, k_interp.size());
//    float cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
//    float cby  = f_cby + dy*f_dcbydy;
//    float cbz  = f_cbz + dz*f_dcbzdz;
//    float ux   = p_ux;                             // Load momentum
//    float uy   = p_uy;
//    float uz   = p_uz;
//    float q    = p_w;
//    ux  += hax;                               // Half advance E
//    uy  += hay;
//    uz  += haz;
//    v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
//    /**/                                      // Boris - scalars
//    v1   = cbx*cbx + (cby*cby + cbz*cbz);
//    v2   = (v0*v0)*v1;
//    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
//    v4   = v3/(one+v1*(v3*v3));
//    v4  += v4;
//    v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
//    v1   = uy + v3*( uz*cbx - ux*cbz );
//    v2   = uz + v3*( ux*cby - uy*cbx );
//    ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
//    uy  += v4*( v2*cbx - v0*cbz );
//    uz  += v4*( v0*cby - v1*cbx );
//    ux  += hax;                               // Half advance E
//    uy  += hay;
//    uz  += haz;
//    p_ux = ux;                               // Store momentum
//    p_uy = uy;
//    p_uz = uz;
//
//    v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));
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
//    // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
//    if(  v3<=one &&  v4<=one &&  v5<=one &&   // Check if inbnds
//        -v3<=one && -v4<=one && -v5<=one ) {
//
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
//
//#     undef ACCUMULATE_J
//
//    } else
//    {                                    // Unlikely
//        /*
//           local_pm_dispx = ux;
//           local_pm_dispy = uy;
//           local_pm_dispz = uz;
//
//           local_pm_i     = p_index;
//        */
//      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
//      local_pm->dispx = ux;
//      local_pm->dispy = uy;
//      local_pm->dispz = uz;
//      local_pm->i     = p_index;
//
//      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
//      if( move_p_kokkos(k_part, k_particles, k_particles_i, local_pm,
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
//          k_particle_copy(nm, particle_var::dx) = float(p_dx);
//          k_particle_copy(nm, particle_var::dy) = p_dy;
//          k_particle_copy(nm, particle_var::dz) = p_dz;
//          k_particle_copy(nm, particle_var::ux) = p_ux;
//          k_particle_copy(nm, particle_var::uy) = p_uy;
//          k_particle_copy(nm, particle_var::uz) = p_uz;
//          k_particle_copy(nm, particle_var::w) = p_w;
//          k_particle_i_copy(nm) = pii;
//
//          // Tag this one as having left
//          //k_particles(p_index, particle_var::pi) = 999999;
//
//          // Copy local local_pm back
//          //local_pm_dispx = local_pm->dispx;
//          //local_pm_dispy = local_pm->dispy;
//          //local_pm_dispz = local_pm->dispz;
//          //local_pm_i = local_pm->i;
//          //printf("rank copying %d to nm %d \n", local_pm_i, nm);
//          //copy_local_to_pm(nm);
//        }
//      }
//    }
////}
//  });
//
////Kokkos::parallel_for(Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(int i) {
////  k_part.dx(i) = k_particles(i, particle_var::dx);
////  k_part.dy(i) = k_particles(i, particle_var::dy);
////  k_part.dz(i) = k_particles(i, particle_var::dz);
////  k_part.ux(i) = k_particles(i, particle_var::ux);
////  k_part.uy(i) = k_particles(i, particle_var::uy);
////  k_part.uz(i) = k_particles(i, particle_var::uz);
////  k_part.w(i) = k_particles(i, particle_var::w);
////  k_part.i(i) = k_particles_i(i);
////});
////Kokkos::parallel_for(Kokkos::RangePolicy<>(0,np), KOKKOS_LAMBDA(int i) {
////  k_particles(i, particle_var::dx) = k_part.dx(i);
////  k_particles(i, particle_var::dy) = k_part.dy(i);
////  k_particles(i, particle_var::dz) = k_part.dz(i);
////  k_particles(i, particle_var::ux) = k_part.ux(i);
////  k_particles(i, particle_var::uy) = k_part.uy(i);
////  k_particles(i, particle_var::uz) = k_part.uz(i);
////  k_particles(i, particle_var::w) = k_part.w(i);
////  k_particles_i(i) = k_part.i(i);
////});
//
//
//  // TODO: abstract this manual data copy
//  //Kokkos::deep_copy(h_nm, k_nm);
//
//  //args->seg[pipeline_rank].pm        = pm;
//  //args->seg[pipeline_rank].max_nm    = max_nm;
//  //args->seg[pipeline_rank].nm        = h_nm(0);
//  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
//  //delete(k_local_particle_movers_p);
//  //return h_nm(0);
//
////  #undef p_dx  
////  #undef p_dy  
////  #undef p_dz  
////  #undef p_ux  
////  #undef p_uy  
////  #undef p_uz  
////  #undef p_w   
////  #undef pii   
//
////  #undef p_dx
////  #undef p_dy
////  #undef p_dz
////  #undef p_ux
////  #undef p_uy
////  #undef p_uz
////  #undef p_w 
////  #undef pii 
//
////  #undef f_cbx 
////  #undef f_cby 
////  #undef f_cbz 
////  #undef f_ex  
////  #undef f_ey  
////  #undef f_ez  
////
////  #undef f_dexdy    
////  #undef f_dexdz    
////
////  #undef f_d2exdydz 
////  #undef f_deydx    
////  #undef f_deydz    
////
////  #undef f_d2eydzdx 
////  #undef f_dezdx    
////  #undef f_dezdy    
////
////  #undef f_d2ezdxdy 
////  #undef f_dcbxdx   
////  #undef f_dcbydy   
////  #undef f_dcbzdz   
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

//  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>> interp = ia->k_i_d;

//printf("Species: %s\n", sp->name);
  KOKKOS_TIC();
  advance_p_kokkos(
          sp->k_p_soa_d,
          sp->k_pc_soa_d,
          sp->k_pm_d,
          sp->k_pm_i_d,
          aa->k_a_sa,
          aa->k_a_d,
//          interp,
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
  KOKKOS_TOC( advance_p_kokkos, 1);

  // I need to know the number of movers that got populated so I can call the
  // compress. Let's copy it back
  Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
  // TODO: which way round should this copy be?

//  int nm = sp->k_nm_h(0);

//  printf("nm = %d \n", nm);

  // Copy particle mirror movers back so we have their data safe. Ready for
  // boundary_p_kokkos
  auto pc_dx_d_subview = Kokkos::subview(sp->k_pc_soa_d.dx, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dy_d_subview = Kokkos::subview(sp->k_pc_soa_d.dy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dz_d_subview = Kokkos::subview(sp->k_pc_soa_d.dz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_ux_d_subview = Kokkos::subview(sp->k_pc_soa_d.ux, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uy_d_subview = Kokkos::subview(sp->k_pc_soa_d.uy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uz_d_subview = Kokkos::subview(sp->k_pc_soa_d.uz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_w_d_subview = Kokkos::subview(sp->k_pc_soa_d.w, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_i_d_subview = Kokkos::subview(sp->k_pc_soa_d.i, std::make_pair(0, sp->k_nm_h(0)));

  auto pc_dx_h_subview = Kokkos::subview(sp->k_pc_soa_h.dx, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dy_h_subview = Kokkos::subview(sp->k_pc_soa_h.dy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dz_h_subview = Kokkos::subview(sp->k_pc_soa_h.dz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_ux_h_subview = Kokkos::subview(sp->k_pc_soa_h.ux, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uy_h_subview = Kokkos::subview(sp->k_pc_soa_h.uy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uz_h_subview = Kokkos::subview(sp->k_pc_soa_h.uz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_w_h_subview = Kokkos::subview(sp->k_pc_soa_h.w, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_i_h_subview = Kokkos::subview(sp->k_pc_soa_h.i, std::make_pair(0, sp->k_nm_h(0)));

  KOKKOS_TIC();
    Kokkos::deep_copy(pc_dx_h_subview, pc_dx_d_subview);
    Kokkos::deep_copy(pc_dy_h_subview, pc_dy_d_subview);
    Kokkos::deep_copy(pc_dz_h_subview, pc_dz_d_subview);
    Kokkos::deep_copy(pc_ux_h_subview, pc_ux_d_subview);
    Kokkos::deep_copy(pc_uy_h_subview, pc_uy_d_subview);
    Kokkos::deep_copy(pc_uz_h_subview, pc_uz_d_subview);
    Kokkos::deep_copy(pc_w_h_subview, pc_w_d_subview);
    Kokkos::deep_copy(pc_i_h_subview, pc_i_d_subview);
  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
//  auto pc_d_subview = Kokkos::subview(sp->k_pc_d, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
//  auto pci_d_subview = Kokkos::subview(sp->k_pc_i_d, std::make_pair(0, sp->k_nm_h(0)));
//  auto pc_h_subview = Kokkos::subview(sp->k_pc_h, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
//  auto pci_h_subview = Kokkos::subview(sp->k_pc_i_h, std::make_pair(0, sp->k_nm_h(0)));
//  KOKKOS_TIC();
//    Kokkos::deep_copy(pc_h_subview, pc_d_subview);
//    Kokkos::deep_copy(pci_h_subview, pci_d_subview);
////  Kokkos::deep_copy(sp->k_pc_h, sp->k_pc_d);
////  Kokkos::deep_copy(sp->k_pc_i_h, sp->k_pc_i_d);
//  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);

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

void
advance_p_profiling( /**/  species_t            * RESTRICT sp,
           /**/  accumulator_array_t  * RESTRICT aa,
           interpolator_array_t * RESTRICT ia, int64_t step ) {
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

//  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>> interp = ia->k_i_d;

Kokkos::Profiling::pushRegion(" " + std::to_string(step) + " " + std::string(sp->name) + " advance_p_kokkos");
  advance_p_kokkos(
          sp->k_p_soa_d,
          sp->k_pc_soa_d,
//          sp->k_p_d,
//          sp->k_p_i_d,
//          sp->k_pc_d,
//          sp->k_pc_i_d,
          sp->k_pm_d,
          sp->k_pm_i_d,
          aa->k_a_sa,
          aa->k_a_d,
//          interp,
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
Kokkos::Profiling::popRegion();

//  KOKKOS_TIC();
  // I need to know the number of movers that got populated so I can call the
  // compress. Let's copy it back
  Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
  // TODO: which way round should this copy be?

  //  int nm = sp->k_nm_h(0);

  //  printf("nm = %d \n", nm);

  // Copy particle mirror movers back so we have their data safe. Ready for
  // boundary_p_kokkos
  auto pc_dx_d_subview = Kokkos::subview(sp->k_pc_soa_d.dx, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dy_d_subview = Kokkos::subview(sp->k_pc_soa_d.dy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dz_d_subview = Kokkos::subview(sp->k_pc_soa_d.dz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_ux_d_subview = Kokkos::subview(sp->k_pc_soa_d.ux, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uy_d_subview = Kokkos::subview(sp->k_pc_soa_d.uy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uz_d_subview = Kokkos::subview(sp->k_pc_soa_d.uz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_w_d_subview = Kokkos::subview(sp->k_pc_soa_d.w, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_i_d_subview = Kokkos::subview(sp->k_pc_soa_d.i, std::make_pair(0, sp->k_nm_h(0)));

  auto pc_dx_h_subview = Kokkos::subview(sp->k_pc_soa_h.dx, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dy_h_subview = Kokkos::subview(sp->k_pc_soa_h.dy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dz_h_subview = Kokkos::subview(sp->k_pc_soa_h.dz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_ux_h_subview = Kokkos::subview(sp->k_pc_soa_h.ux, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uy_h_subview = Kokkos::subview(sp->k_pc_soa_h.uy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uz_h_subview = Kokkos::subview(sp->k_pc_soa_h.uz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_w_h_subview = Kokkos::subview(sp->k_pc_soa_h.w, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_i_h_subview = Kokkos::subview(sp->k_pc_soa_h.i, std::make_pair(0, sp->k_nm_h(0)));

  KOKKOS_TIC();
    Kokkos::deep_copy(pc_dx_h_subview, pc_dx_d_subview);
    Kokkos::deep_copy(pc_dy_h_subview, pc_dy_d_subview);
    Kokkos::deep_copy(pc_dz_h_subview, pc_dz_d_subview);
    Kokkos::deep_copy(pc_ux_h_subview, pc_ux_d_subview);
    Kokkos::deep_copy(pc_uy_h_subview, pc_uy_d_subview);
    Kokkos::deep_copy(pc_uz_h_subview, pc_uz_d_subview);
    Kokkos::deep_copy(pc_w_h_subview, pc_w_d_subview);
    Kokkos::deep_copy(pc_i_h_subview, pc_i_d_subview);
  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);

//  auto pc_d_subview = Kokkos::subview(sp->k_pc_d, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
//  auto pci_d_subview = Kokkos::subview(sp->k_pc_i_d, std::make_pair(0, sp->k_nm_h(0)));
//  auto pc_h_subview = Kokkos::subview(sp->k_pc_h, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
//  auto pci_h_subview = Kokkos::subview(sp->k_pc_i_h, std::make_pair(0, sp->k_nm_h(0)));
//Kokkos::Profiling::pushRegion(" " + std::to_string(step) + " " + std::string(sp->name) + " advance_p Copy particle copies to host");
//  KOKKOS_TIC();
//    Kokkos::deep_copy(pc_h_subview, pc_d_subview);
//    Kokkos::deep_copy(pci_h_subview, pci_d_subview);
//  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
//Kokkos::Profiling::popRegion();

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

void
advance_p_cuda( /**/  species_t            * RESTRICT sp,
           /**/  accumulator_array_t  * RESTRICT aa,
           interpolator_array_t * RESTRICT ia ) {

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

//  Kokkos::View<const float *[INTERPOLATOR_VAR_COUNT], Kokkos::MemoryTraits<Kokkos::RandomAccess>> interp = ia->k_i_d;

//printf("Species: %s\n", sp->name);
  KOKKOS_TIC();

  advance_p_cuda<<<4096, 1024>>>(
                  sp->k_p_soa_d.dx.data(),
                  sp->k_p_soa_d.dy.data(),
                  sp->k_p_soa_d.dz.data(),
                  sp->k_p_soa_d.ux.data(),
                  sp->k_p_soa_d.uy.data(),
                  sp->k_p_soa_d.uz.data(),
                  sp->k_p_soa_d.w.data(),
                  sp->k_p_soa_d.i.data(),
                  sp->k_pc_soa_d.dx.data(),
                  sp->k_pc_soa_d.dy.data(),
                  sp->k_pc_soa_d.dz.data(),
                  sp->k_pc_soa_d.ux.data(),
                  sp->k_pc_soa_d.uy.data(),
                  sp->k_pc_soa_d.uz.data(),
                  sp->k_pc_soa_d.w.data(),
                  sp->k_pc_soa_d.i.data(),
                  sp->k_pm_d.data(),
                  sp->k_pm_i_d.data(),
                  aa->k_a_d.data(),
                  ia->k_i_d.data(),
                  sp->k_nm_d.data(),
                  sp->g->k_neighbor_d.data(),
                  sp->g->rangel,
                  sp->g->rangeh,
                  qdt_2mc,
                  cdt_dx,
                  cdt_dy,
                  cdt_dz,
                  sp->q,
                  aa->na,
                  sp->g->nv,
                  sp->np,
                  sp->max_nm,
                  sp->g->nx,
                  sp->g->ny,
                  sp->g->nz
                );

  KOKKOS_TOC( advance_p_kokkos, 1);

  // I need to know the number of movers that got populated so I can call the
  // compress. Let's copy it back
  Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
  // TODO: which way round should this copy be?

//  int nm = sp->k_nm_h(0);

//  printf("nm = %d \n", nm);

  // Copy particle mirror movers back so we have their data safe. Ready for
  // boundary_p_kokkos
  auto pc_dx_d_subview = Kokkos::subview(sp->k_pc_soa_d.dx, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dy_d_subview = Kokkos::subview(sp->k_pc_soa_d.dy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dz_d_subview = Kokkos::subview(sp->k_pc_soa_d.dz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_ux_d_subview = Kokkos::subview(sp->k_pc_soa_d.ux, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uy_d_subview = Kokkos::subview(sp->k_pc_soa_d.uy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uz_d_subview = Kokkos::subview(sp->k_pc_soa_d.uz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_w_d_subview = Kokkos::subview(sp->k_pc_soa_d.w, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_i_d_subview = Kokkos::subview(sp->k_pc_soa_d.i, std::make_pair(0, sp->k_nm_h(0)));

  auto pc_dx_h_subview = Kokkos::subview(sp->k_pc_soa_h.dx, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dy_h_subview = Kokkos::subview(sp->k_pc_soa_h.dy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_dz_h_subview = Kokkos::subview(sp->k_pc_soa_h.dz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_ux_h_subview = Kokkos::subview(sp->k_pc_soa_h.ux, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uy_h_subview = Kokkos::subview(sp->k_pc_soa_h.uy, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_uz_h_subview = Kokkos::subview(sp->k_pc_soa_h.uz, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_w_h_subview = Kokkos::subview(sp->k_pc_soa_h.w, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_i_h_subview = Kokkos::subview(sp->k_pc_soa_h.i, std::make_pair(0, sp->k_nm_h(0)));

  KOKKOS_TIC();
    Kokkos::deep_copy(pc_dx_h_subview, pc_dx_d_subview);
    Kokkos::deep_copy(pc_dy_h_subview, pc_dy_d_subview);
    Kokkos::deep_copy(pc_dz_h_subview, pc_dz_d_subview);
    Kokkos::deep_copy(pc_ux_h_subview, pc_ux_d_subview);
    Kokkos::deep_copy(pc_uy_h_subview, pc_uy_d_subview);
    Kokkos::deep_copy(pc_uz_h_subview, pc_uz_d_subview);
    Kokkos::deep_copy(pc_w_h_subview, pc_w_d_subview);
    Kokkos::deep_copy(pc_i_h_subview, pc_i_d_subview);
  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
}
