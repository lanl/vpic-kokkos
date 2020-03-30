// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/vpic.h"

//struct AccumulatorReduction {
//
////  typedef float value_type[];
//  typedef Kokkos::View<float*[3][4]> value_type;
//
//  k_particles_soa_t& k_part;
//  const k_interpolator_t& k_interp;
//
//  int num_accum;
//  int num_var;
//  int length;
//  int value_count;
//  float qdt_2mc;
//  float cdt_dx;
//  float cdt_dy;
//  float cdt_dz;
//  float qsp;
//
//  AccumulatorReduction( const k_accumulators_t& k_accumulators, 
//                        k_particles_soa_t& k_particles, 
//                        const k_interpolator_t& k_interpolators, 
//                        const float qdt_2mc_, 
//                        const float cdt_dx_, 
//                        const float cdt_dy_, 
//                        const float cdt_dz_, 
//                        const float qsp_): 
//    num_accum(k_accumulators.extent(0)),
//    num_var(k_accumulators.extent(1)),
//    length(k_accumulators.extent(2)),
//    k_part(k_particles),
//    k_interp(k_interpolators),
//    qdt_2mc(qdt_2mc_),
//    cdt_dx(cdt_dx_),
//    cdt_dy(cdt_dy_),
//    cdt_dz(cdt_dz_),
//    qsp(qsp_)
//    {
//    }
//
//  KOKKOS_INLINE_FUNCTION void
//  operator() (const int p_index, value_type sum) const {
//    packed_t v0, v1, v2, v3, v4, v5;
//
//    packed_t dx = packed_t(k_part.dx(p_index*2), k_part.dx(p_index*2+1)); // Load position
//    packed_t dy = packed_t(k_part.dy(p_index*2), k_part.dy(p_index*2+1));
//    packed_t dz = packed_t(k_part.dz(p_index*2), k_part.dz(p_index*2+1));
//    int pi0 = k_part.i(p_index*2);
//    int pi1 = k_part.i(p_index*2+1);
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
//    packed_t cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
//    packed_t cby  = f_cby + dy*f_dcbydy;
//    packed_t cbz  = f_cbz + dz*f_dcbzdz;
//
//    packed_t ux   = packed_t(k_part.ux(p_index*2), k_part.ux(p_index*2+1));       // Load momentum
//    packed_t uy   = packed_t(k_part.uy(p_index*2), k_part.uy(p_index*2+1));
//    packed_t uz   = packed_t(k_part.uz(p_index*2), k_part.uz(p_index*2+1));
//    packed_t q    = packed_t(k_part.w(p_index*2), k_part.w(p_index*2+1));
//
//    const float one = 1.0;
//    const float one_third = 1.0/3.0;
//    const float two_fifteenths = 2.0/15.0;
//
//    ux  += hax;                               // Half advance E
//    uy  += hay;
//    uz  += haz;
//    v0   = packed_t(qdt_2mc)/sqrt(packed_t(one) + (ux*ux + (uy*uy + uz*uz)));
//    v1   = cbx*cbx + (cby*cby + cbz*cbz);
//    v2   = ( v0*v0 ) * v1;
//    v3   = v0*(packed_t(one)+v2*(packed_t(one_third)+v2*packed_t(two_fifteenths)));
//    v4   = v3/(packed_t(one)+v1*(v3*v3));
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
//    ux  *= packed_t(cdt_dx);
//    uy  *= packed_t(cdt_dy);
//    uz  *= packed_t(cdt_dz);
//
//    /**/                                      // Get norm displacement
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
//    packed_t v3_inbnds = leq(v3, packed_t(one));
//            v3_inbnds += leq(-v3, packed_t(one));
//    packed_t v4_inbnds = leq(v4, packed_t(one));
//            v4_inbnds += leq(-v4, packed_t(one));
//    packed_t v5_inbnds = leq(v5, packed_t(one));
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
//    v5 = q*ux*uy*uz*one_third;              // Compute correction
//
//#   define ACCUMULATE_J(X,Y,Z)                                 \
//    v4  = q*u##X;   /* v2 = q ux                            */        \
//    v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//    v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//    v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//    v4  = packed_t(one)+d##Z; /* v4 = 1+dz                            */        \
//    v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//    v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//    v4  = packed_t(one)-d##Z; /* v4 = 1-dz                            */        \
//    v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//    v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//    v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//    v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//    v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//    v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//    if(inbnds.low2float()) {
//      ACCUMULATE_J( x,y,z );
//      sum(pi0, accumulator_var::jx, 0) += v0.low2float();
//      sum(pi0, accumulator_var::jx, 1) += v1.low2float();
//      sum(pi0, accumulator_var::jx, 2) += v2.low2float();
//      sum(pi0, accumulator_var::jx, 3) += v3.low2float();
//      ACCUMULATE_J( y,z,x );
//      sum(pi0, accumulator_var::jy, 0) += v0.low2float();
//      sum(pi0, accumulator_var::jy, 1) += v1.low2float();
//      sum(pi0, accumulator_var::jy, 2) += v2.low2float();
//      sum(pi0, accumulator_var::jy, 3) += v3.low2float();
//      ACCUMULATE_J( z,x,y );
//      sum(pi0, accumulator_var::jz, 0) += v0.low2float();
//      sum(pi0, accumulator_var::jz, 1) += v1.low2float();
//      sum(pi0, accumulator_var::jz, 2) += v2.low2float();
//      sum(pi0, accumulator_var::jz, 3) += v3.low2float();
//    }
//    if(inbnds.high2float()) {
//      ACCUMULATE_J( x,y,z );
//      sum(pi1, accumulator_var::jx, 0) += v0.high2float();
//      sum(pi1, accumulator_var::jx, 1) += v1.high2float();
//      sum(pi1, accumulator_var::jx, 2) += v2.high2float();
//      sum(pi1, accumulator_var::jx, 3) += v3.high2float();
//      ACCUMULATE_J( y,z,x );
//      sum(pi1, accumulator_var::jy, 0) += v0.high2float();
//      sum(pi1, accumulator_var::jy, 1) += v1.high2float();
//      sum(pi1, accumulator_var::jy, 2) += v2.high2float();
//      sum(pi1, accumulator_var::jy, 3) += v3.high2float();
//      ACCUMULATE_J( z,x,y );
//      sum(pi1, accumulator_var::jz, 0) += v0.high2float();
//      sum(pi1, accumulator_var::jz, 1) += v1.high2float();
//      sum(pi1, accumulator_var::jz, 2) += v2.high2float();
//      sum(pi1, accumulator_var::jz, 3) += v3.high2float();
//    }
////    if(inbnds.low2float()) {
////      ACCUMULATE_J( x,y,z );
////      sum[pi0*num_var*length + accumulator_var::jx*length + 0] += v0.low2float();
////      sum[pi0*num_var*length + accumulator_var::jx*length + 1] += v1.low2float();
////      sum[pi0*num_var*length + accumulator_var::jx*length + 2] += v2.low2float();
////      sum[pi0*num_var*length + accumulator_var::jx*length + 3] += v3.low2float();
////      ACCUMULATE_J( y,z,x );
////      sum[pi0*num_var*length + accumulator_var::jy*length + 0] += v0.low2float();
////      sum[pi0*num_var*length + accumulator_var::jy*length + 1] += v1.low2float();
////      sum[pi0*num_var*length + accumulator_var::jy*length + 2] += v2.low2float();
////      sum[pi0*num_var*length + accumulator_var::jy*length + 3] += v3.low2float();
////      ACCUMULATE_J( z,x,y );
////      sum[pi0*num_var*length + accumulator_var::jz*length + 0] += v0.low2float();
////      sum[pi0*num_var*length + accumulator_var::jz*length + 1] += v1.low2float();
////      sum[pi0*num_var*length + accumulator_var::jz*length + 2] += v2.low2float();
////      sum[pi0*num_var*length + accumulator_var::jz*length + 3] += v3.low2float();
////    }
////    if(inbnds.high2float()) {
////      ACCUMULATE_J( x,y,z );
////      sum[pi1*num_var*length + accumulator_var::jx*length + 0] += v0.high2float();
////      sum[pi1*num_var*length + accumulator_var::jx*length + 1] += v1.high2float();
////      sum[pi1*num_var*length + accumulator_var::jx*length + 2] += v2.high2float();
////      sum[pi1*num_var*length + accumulator_var::jx*length + 3] += v3.high2float();
////      ACCUMULATE_J( y,z,x );
////      sum[pi1*num_var*length + accumulator_var::jy*length + 0] += v0.high2float();
////      sum[pi1*num_var*length + accumulator_var::jy*length + 1] += v1.high2float();
////      sum[pi1*num_var*length + accumulator_var::jy*length + 2] += v2.high2float();
////      sum[pi1*num_var*length + accumulator_var::jy*length + 3] += v3.high2float();
////      ACCUMULATE_J( z,x,y );
////      sum[pi1*num_var*length + accumulator_var::jz*length + 0] += v0.high2float();
////      sum[pi1*num_var*length + accumulator_var::jz*length + 1] += v1.high2float();
////      sum[pi1*num_var*length + accumulator_var::jz*length + 2] += v2.high2float();
////      sum[pi1*num_var*length + accumulator_var::jz*length + 3] += v3.high2float();
////    }
//#   undef ACCUMULATE_J
//  }
//
//  KOKKOS_INLINE_FUNCTION void
//  join (volatile value_type dst, const volatile value_type src) const {
//    for(int i=0; i<num_accum; i++) {
//      for(int j=0; j<num_var; j++) {
//        for(int k=0; k<length; k++) {
////          dst[i*num_var*length + j*length + k] += src[i*num_var*length + j*length + k];
//          dst(i,j,k) += src(i,j,k);
//        }
//      }
//    }
//  }
//
//  KOKKOS_INLINE_FUNCTION void
//  init(value_type sum) const {
//    for(int i=0; i<num_accum; i++) {
//      for(int j=0; j<num_var; j++) {
//        for(int k=0; k<length; k++) {
////          sum[i*num_var*length + j*length + k] = 0.0;
//          sum(i,j,k) = 0.0;
//        }
//      }
//    }
//  }
//};
//
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
////  constexpr float one            = 1.;
////  constexpr float one_third      = 1./3.;
////  constexpr float two_fifteenths = 2./15.;
////
////  auto rangel = g->rangel;
////  auto rangeh = g->rangeh;
//
//  // TODO: is this the right place to do this?
//  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
//    k_nm(0) = 0;
//  });
//
//  k_accumulators_t accumulator("Reduction result", k_accumulator.extent(0));
////  float* accumulator = (float*) malloc(sizeof(float)*k_accumulator.extent(0)*3*4);
//  AccumulatorReduction ar(k_accumulator, k_part, k_interp, qdt_2mc, cdt_dx, cdt_dy, cdt_dz, qsp); 
//  Kokkos::parallel_reduce("advance_p reduction", Kokkos::RangePolicy<>(0,np/2), ar, accumulator);
////  Kokkos::parallel_for( "advance_p accumulator contribution", 
////                        Kokkos::RangePolicy<>(0,k_accumulator.extent(0)), 
////                        KOKKOS_LAMBDA(size_t i) {
////    k_accumulator(i, accumulator_var::jx, 0) += accumulator(i, accumulator_var::jx, 0);
////    k_accumulator(i, accumulator_var::jx, 1) += accumulator(i, accumulator_var::jx, 1);
////    k_accumulator(i, accumulator_var::jx, 2) += accumulator(i, accumulator_var::jx, 2);
////    k_accumulator(i, accumulator_var::jx, 3) += accumulator(i, accumulator_var::jx, 3);
////
////    k_accumulator(i, accumulator_var::jy, 0) += accumulator(i, accumulator_var::jy, 0);
////    k_accumulator(i, accumulator_var::jy, 1) += accumulator(i, accumulator_var::jy, 1);
////    k_accumulator(i, accumulator_var::jy, 2) += accumulator(i, accumulator_var::jy, 2);
////    k_accumulator(i, accumulator_var::jy, 3) += accumulator(i, accumulator_var::jy, 3);
////
////    k_accumulator(i, accumulator_var::jz, 0) += accumulator(i, accumulator_var::jz, 0);
////    k_accumulator(i, accumulator_var::jz, 1) += accumulator(i, accumulator_var::jz, 1);
////    k_accumulator(i, accumulator_var::jz, 2) += accumulator(i, accumulator_var::jz, 2);
////    k_accumulator(i, accumulator_var::jz, 3) += accumulator(i, accumulator_var::jz, 3);
////  });
////  free(accumulator);
//}

// Half 2

void
advance_p_kokkos(
        k_particles_soa_t& k_part,
        k_particles_soa_t& k_particle_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_accumulators_sa_t k_accumulators_sa,
        k_accumulators_t k_accumulator,
        k_interpolator_t& k_interp,
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

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // TODO: is this the right place to do this?
  Kokkos::parallel_for("clear nm", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (size_t i) {
    k_nm(0) = 0;
  });

//  k_particle_movers_soa_t mover_particles(np);

  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np/2),
    KOKKOS_LAMBDA (size_t p_index)
    {

    packed_t v0, v1, v2, v3, v4, v5;

//    packed_t one = packed_t(one_f);
//    packed_t one_third = packed_t(one_third_f);
//    packed_t two_fifteenths = packed_t(two_fifteenths_f);

    packed_t dx = packed_t(k_part.dx(p_index*2), k_part.dx(p_index*2+1)); // Load position
    packed_t dy = packed_t(k_part.dy(p_index*2), k_part.dy(p_index*2+1));
    packed_t dz = packed_t(k_part.dz(p_index*2), k_part.dz(p_index*2+1));
    const int pi0 = k_part.i(p_index*2);
    const int pi1 = k_part.i(p_index*2+1);

    packed_t f_ex        = packed_t(k_interp(pi0, interpolator_var::ex), 
                                    k_interp(pi1, interpolator_var::ex));
    packed_t f_ey        = packed_t(k_interp(pi0, interpolator_var::ey), 
                                    k_interp(pi1, interpolator_var::ey));
    packed_t f_ez        = packed_t(k_interp(pi0, interpolator_var::ez), 
                                    k_interp(pi1, interpolator_var::ez));

    packed_t f_dexdy     = packed_t(k_interp(pi0, interpolator_var::dexdy), 
                                    k_interp(pi1, interpolator_var::dexdy));
    packed_t f_dexdz     = packed_t(k_interp(pi0, interpolator_var::dexdz), 
                                    k_interp(pi1, interpolator_var::dexdz));
    packed_t f_deydz     = packed_t(k_interp(pi0, interpolator_var::deydz), 
                                    k_interp(pi1, interpolator_var::deydz));
    packed_t f_deydx     = packed_t(k_interp(pi0, interpolator_var::deydx), 
                                    k_interp(pi1, interpolator_var::deydx));
    packed_t f_dezdx     = packed_t(k_interp(pi0, interpolator_var::dezdx), 
                                    k_interp(pi1, interpolator_var::dezdx));
    packed_t f_dezdy     = packed_t(k_interp(pi0, interpolator_var::dezdy), 
                                    k_interp(pi1, interpolator_var::dezdy));
    packed_t f_d2exdydz  = packed_t(k_interp(pi0, interpolator_var::d2exdydz), 
                                    k_interp(pi1, interpolator_var::d2exdydz));
    packed_t f_d2eydzdx  = packed_t(k_interp(pi0, interpolator_var::d2eydzdx), 
                                    k_interp(pi1, interpolator_var::d2eydzdx));
    packed_t f_d2ezdxdy  = packed_t(k_interp(pi0, interpolator_var::d2ezdxdy), 
                                    k_interp(pi1, interpolator_var::d2ezdxdy));
    packed_t f_cbx       = packed_t(k_interp(pi0, interpolator_var::cbx), 
                                    k_interp(pi1, interpolator_var::cbx));
    packed_t f_cby       = packed_t(k_interp(pi0, interpolator_var::cby), 
                                    k_interp(pi1, interpolator_var::cby));
    packed_t f_cbz       = packed_t(k_interp(pi0, interpolator_var::cbz), 
                                    k_interp(pi1, interpolator_var::cbz));
    packed_t f_dcbxdx    = packed_t(k_interp(pi0, interpolator_var::dcbxdx), 
                                    k_interp(pi1, interpolator_var::dcbxdx));
    packed_t f_dcbydy    = packed_t(k_interp(pi0, interpolator_var::dcbydy), 
                                    k_interp(pi1, interpolator_var::dcbydy));
    packed_t f_dcbzdz    = packed_t(k_interp(pi0, interpolator_var::dcbzdz), 
                                    k_interp(pi1, interpolator_var::dcbzdz));

    packed_t hax  = packed_t(qdt_2mc)*( (f_ex    + dy*f_dexdy) +
                                     dz*(f_dexdz + dy*f_d2exdydz) );
    packed_t hay  = packed_t(qdt_2mc)*( (f_ey    + dz*f_deydz) +
                                     dx*(f_deydx + dz*f_d2eydzdx) );
    packed_t haz  = packed_t(qdt_2mc)*( (f_ez    + dx*f_dezdx) +
                                     dy*(f_dezdy + dx*f_d2ezdxdy) );

//    packed_t cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
//    packed_t cby  = f_cby + dy*f_dcbydy;
//    packed_t cbz  = f_cbz + dz*f_dcbzdz;
    packed_t cbx  = fma(dx, f_dcbxdx, f_cbx); // Interpolate B
    packed_t cby  = fma(dy, f_dcbydy, f_cby); 
    packed_t cbz  = fma(dz, f_dcbzdz, f_cbz); 

    packed_t ux   = packed_t(k_part.ux(p_index*2), k_part.ux(p_index*2+1));       // Load momentum
    packed_t uy   = packed_t(k_part.uy(p_index*2), k_part.uy(p_index*2+1));
    packed_t uz   = packed_t(k_part.uz(p_index*2), k_part.uz(p_index*2+1));
    packed_t q    = packed_t(k_part.w(p_index*2), k_part.w(p_index*2+1));

    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;
    v0   = packed_t(qdt_2mc)/sqrt(packed_t(one) + (ux*ux + (uy*uy + uz*uz)));
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = ( v0*v0 ) * v1;
//    v3   = v0*(packed_t(one)+v2*(packed_t(one_third)+v2*two_fifteenths));
//    v4   = v3/(packed_t(one)+v1*(v3*v3));
    v3   = v0*fma(v2, fma(v2, packed_t(two_fifteenths), packed_t(one_third)), packed_t(one));
    v4   = v3/(fma(v1, (v3*v3), packed_t(one)));
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

    k_part.ux(p_index*2)   = ux.low2half();                               // Store momentum
    k_part.ux(p_index*2+1) = ux.high2half();
    k_part.uy(p_index*2)   = uy.low2half();
    k_part.uy(p_index*2+1) = uy.high2half();
    k_part.uz(p_index*2)   = uz.low2half();
    k_part.uz(p_index*2+1) = uz.high2half();

    v0   = packed_t(one)/sqrt(packed_t(one) + (ux*ux+ (uy*uy + uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= packed_t(cdt_dx);
    uy  *= packed_t(cdt_dy);
    uz  *= packed_t(cdt_dz);

    ux  *= v0;
    uy  *= v0;
    uz  *= v0;

    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;

    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

    packed_t v3_inbnds = leq( v3, packed_t(one));
            v3_inbnds += leq(-v3, packed_t(one));
    packed_t v4_inbnds = leq( v4, packed_t(one));
            v4_inbnds += leq(-v4, packed_t(one));
    packed_t v5_inbnds = leq( v5, packed_t(one));
            v5_inbnds += leq(-v5, packed_t(one));
    packed_t inbnds = eq(v3_inbnds+v4_inbnds+v5_inbnds, packed_t(6.0, 6.0));
    
    // Common case (inbnds).  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step

    q *= packed_t(qsp);
    // Store new position
    if(inbnds.low2float()) {
      k_part.dx(p_index*2)    = v3.low2half();
      k_part.dy(p_index*2)    = v4.low2half();
      k_part.dz(p_index*2)    = v5.low2half();
    }
    if(inbnds.high2float()) {
      k_part.dx(p_index*2+1)  = v3.high2half();
      k_part.dy(p_index*2+1)  = v4.high2half();
      k_part.dz(p_index*2+1)  = v5.high2half();
    }

    dx = v0;                                // Streak midpoint
    dy = v1;
    dz = v2;
    v5 = q*ux*uy*uz*packed_t(one_third);              // Compute correction

#   define ACCUMULATE_J(X,Y,Z)                                        \
    v4  = q*u##X;   /* v2 = q ux                            */        \
    v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
    v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
    v1 += v4;       /* v1 = q ux (1+dy)                     */        \
    v4  = packed_t(one)+d##Z; /* v4 = 1+dz                  */        \
    v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
    v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
    v4  = packed_t(one)-d##Z; /* v4 = 1-dz                  */        \
    v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
    v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
    v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
    v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
    v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
    v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

//    ACCUMULATE_J( x,y,z );
//    if(inbnds.low2float()) {
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 3), v3.low2float());
//    }
//    if(inbnds.high2float()) {
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 3), v3.high2float());
//    }
//    ACCUMULATE_J( y,z,x );
//    if(inbnds.low2float()) {
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 3), v3.low2float());
//    }
//    if(inbnds.high2float()) {
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 3), v3.high2float());
//    }
//    ACCUMULATE_J( z,x,y );
//    if(inbnds.low2float()) {
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 3), v3.low2float());
//    }
//    if(inbnds.high2float()) {
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 3), v3.high2float());
//    }

    if(inbnds.low2float()) {
      ACCUMULATE_J( x,y,z );
//      k_accumulators_scatter_access(pi0, accumulator_var::jx, 0) += v0.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jx, 1) += v1.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jx, 2) += v2.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jx, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jx, 3), v3.low2float());
      k_accumulator(pi0, accumulator_var::jx, 0) += v0.low2float();
      k_accumulator(pi0, accumulator_var::jx, 1) += v1.low2float();
      k_accumulator(pi0, accumulator_var::jx, 2) += v2.low2float();
      k_accumulator(pi0, accumulator_var::jx, 3) += v3.low2float();
      ACCUMULATE_J( y,z,x );
//      k_accumulators_scatter_access(pi0, accumulator_var::jy, 0) += v0.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jy, 1) += v1.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jy, 2) += v2.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jy, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 3), v3.low2float());
      k_accumulator(pi0, accumulator_var::jy, 0) += v0.low2float();
      k_accumulator(pi0, accumulator_var::jy, 1) += v1.low2float();
      k_accumulator(pi0, accumulator_var::jy, 2) += v2.low2float();
      k_accumulator(pi0, accumulator_var::jy, 3) += v3.low2float();
      ACCUMULATE_J( z,x,y );
//      k_accumulators_scatter_access(pi0, accumulator_var::jz, 0) += v0.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jz, 1) += v1.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jz, 2) += v2.low2float();
//      k_accumulators_scatter_access(pi0, accumulator_var::jz, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 3), v3.low2float());
      k_accumulator(pi0, accumulator_var::jz, 0) += v0.low2float();
      k_accumulator(pi0, accumulator_var::jz, 1) += v1.low2float();
      k_accumulator(pi0, accumulator_var::jz, 2) += v2.low2float();
      k_accumulator(pi0, accumulator_var::jz, 3) += v3.low2float();
    }
    if(inbnds.high2float()) {
      ACCUMULATE_J( x,y,z );
//      k_accumulators_scatter_access(pi1, accumulator_var::jx, 0) += v0.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jx, 1) += v1.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jx, 2) += v2.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jx, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jx, 3), v3.high2float());
      k_accumulator(pi1, accumulator_var::jx, 0) += v0.high2float();
      k_accumulator(pi1, accumulator_var::jx, 1) += v1.high2float();
      k_accumulator(pi1, accumulator_var::jx, 2) += v2.high2float();
      k_accumulator(pi1, accumulator_var::jx, 3) += v3.high2float();
      ACCUMULATE_J( y,z,x );
//      k_accumulators_scatter_access(pi1, accumulator_var::jy, 0) += v0.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jy, 1) += v1.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jy, 2) += v2.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jy, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 3), v3.high2float());
      k_accumulator(pi1, accumulator_var::jy, 0) += v0.high2float();
      k_accumulator(pi1, accumulator_var::jy, 1) += v1.high2float();
      k_accumulator(pi1, accumulator_var::jy, 2) += v2.high2float();
      k_accumulator(pi1, accumulator_var::jy, 3) += v3.high2float();
      ACCUMULATE_J( z,x,y );
//      k_accumulators_scatter_access(pi1, accumulator_var::jz, 0) += v0.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jz, 1) += v1.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jz, 2) += v2.high2float();
//      k_accumulators_scatter_access(pi1, accumulator_var::jz, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 3), v3.high2float());
      k_accumulator(pi1, accumulator_var::jz, 0) += v0.high2float();
      k_accumulator(pi1, accumulator_var::jz, 1) += v1.high2float();
      k_accumulator(pi1, accumulator_var::jz, 2) += v2.high2float();
      k_accumulator(pi1, accumulator_var::jz, 3) += v3.high2float();
    }
#   undef ACCUMULATE_J

    k_particle_mover_t local_pm[1];
    if(!inbnds.low2float()) {
      local_pm->dispx = ux.low2half();
      local_pm->dispy = uy.low2half();
      local_pm->dispz = uz.low2half();
      local_pm->i     = p_index*2;
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
          k_particle_copy.dx(nm) = k_part.dx(p_index*2);
          k_particle_copy.dy(nm) = k_part.dy(p_index*2);
          k_particle_copy.dz(nm) = k_part.dz(p_index*2);
          k_particle_copy.ux(nm) = k_part.ux(p_index*2);
          k_particle_copy.uy(nm) = k_part.uy(p_index*2);
          k_particle_copy.uz(nm) = k_part.uz(p_index*2);
          k_particle_copy.w(nm)  = k_part.w(p_index*2);
          k_particle_copy.i(nm)  = k_part.i(p_index*2);
        }
      }
    }
    if(!inbnds.high2float()) {
      local_pm->dispx = ux.high2half();
      local_pm->dispy = uy.high2half();
      local_pm->dispz = uz.high2half();
      local_pm->i     = p_index*2+1;
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
          k_particle_copy.dx(nm) = k_part.dx(p_index*2+1);
          k_particle_copy.dy(nm) = k_part.dy(p_index*2+1);
          k_particle_copy.dz(nm) = k_part.dz(p_index*2+1);
          k_particle_copy.ux(nm) = k_part.ux(p_index*2+1);
          k_particle_copy.uy(nm) = k_part.uy(p_index*2+1);
          k_particle_copy.uz(nm) = k_part.uz(p_index*2+1);
          k_particle_copy.w(nm)  = k_part.w(p_index*2+1);
          k_particle_copy.i(nm)  = k_part.i(p_index*2+1);
        }
      }
    }
  });
}

// Half
//void
//advance_p_kokkos(
//        k_particles_soa_t& k_part,
//        k_particles_soa_t& k_particle_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        k_accumulators_sa_t k_accumulators_sa,
//        k_accumulators_t k_accumulator,
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
//  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
//    KOKKOS_LAMBDA (size_t p_index)
//    {
//
//    auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//    mixed_t v0, v1, v2, v3, v4, v5;
//
//    mixed_t dx = p_dx; // Load position
//    mixed_t dy = p_dy;
//    mixed_t dz = p_dz;
//    int   ii   = pii;
//
//    mixed_t hax  = qdt_2mc_h*( (mixed_t(f_ex)    + dy*mixed_t(f_dexdy)) +
//                            dz*(mixed_t(f_dexdz) + dy*mixed_t(f_d2exdydz)) );
//    mixed_t hay  = qdt_2mc_h*( (mixed_t(f_ey)    + dz*mixed_t(f_deydz)) +
//                            dx*(mixed_t(f_deydx) + dz*mixed_t(f_d2eydzdx)) );
//    mixed_t haz  = qdt_2mc_h*( (mixed_t(f_ez)    + dx*mixed_t(f_dezdx)) +
//                            dy*(mixed_t(f_dezdy) + dx*mixed_t(f_d2ezdxdy)) );
////    mixed_t hax  = qdt_2mc_h*fma(dz, fma(dy, f_d2exdydz_h, f_dexdz_h), fma(dy, f_dexdy_h, f_ex_h));
////    mixed_t hay  = qdt_2mc_h*fma(dx, fma(dz, f_d2eydzdx_h, f_deydx_h), fma(dz, f_deydz_h, f_ey_h));
////    mixed_t haz  = qdt_2mc_h*fma(dy, fma(dx, f_d2ezdxdy_h, f_dezdy_h), fma(dx, f_dezdx_h, f_ez_h));
//
//    mixed_t cbx  = mixed_t(f_cbx) + dx*mixed_t(f_dcbxdx);             // Interpolate B
//    mixed_t cby  = mixed_t(f_cby) + dy*mixed_t(f_dcbydy);
//    mixed_t cbz  = mixed_t(f_cbz) + dz*mixed_t(f_dcbzdz);
////    mixed_t cbx  = fma(dx, f_dcbxdx_h, f_cbx_h);             // Interpolate B
////    mixed_t cby  = fma(dy, f_dcbydy_h, f_cby_h);
////    mixed_t cbz  = fma(dz, f_dcbzdz_h, f_cbz_h);
//
//    mixed_t ux   = p_ux;                             // Load momentum
//    mixed_t uy   = p_uy;
//    mixed_t uz   = p_uz;
//    mixed_t q    = static_cast<mixed_t>(p_w);
//
//    ux  += hax;                               // Half advance E
//    uy  += hay;
//    uz  += haz;
//    v0   = qdt_2mc_h/sqrt(one + (ux*ux + (uy*uy + uz*uz)));
//    v1   = cbx*cbx + (cby*cby + cbz*cbz);
//    v2   = ( v0*v0 ) * v1;
//    v3   = v0*(mixed_t(one)+v2*(mixed_t(one_third)+v2*mixed_t(two_fifteenths)));
////    v3   = v0*(fma(v2, fma(v2, two_fifteenths, one_third), 1));
//    v4   = v3/(mixed_t(one)+v1*(v3*v3));
////    v4   = v3/(fma((v3*v3), v1, one));
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
//    p_ux = ux;                               // Store momentum
//    p_uy = uy;
//    p_uz = uz;
//    v0   = one/sqrt(one + (ux*ux+ (uy*uy + uz*uz)));
//    ux  *= cdt_dx_h;
//    uy  *= cdt_dy_h;
//    uz  *= cdt_dz_h;
//
//    /**/                                      // Get norm displacement
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
//    if(  v3<=mixed_t(one) &&  v4<=mixed_t(one) &&  v5<=mixed_t(one) &&   // Check if inbnds
//        -v3<=mixed_t(one) && -v4<=mixed_t(one) && -v5<=mixed_t(one) ) {
//
//      // Common case (inbnds).  Note: accumulator values are 4 times
//      // the total physical charge that passed through the appropriate
//      // current quadrant in a time-step
//
//      q *= qsp_h;
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
//
//#     undef ACCUMULATE_J
//
//    } 
//    else
//    {                                    // Unlikely
//
//      k_particle_mover_t local_pm[1];
//      local_pm->dispx = ux;
//      local_pm->dispy = uy;
//      local_pm->dispz = uz;
//      local_pm->i     = p_index;
//
//      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
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
//          k_particle_copy.dx(nm) = k_part.dx(p_index);
//          k_particle_copy.dy(nm) = k_part.dy(p_index);
//          k_particle_copy.dz(nm) = k_part.dz(p_index);
//          k_particle_copy.ux(nm) = k_part.ux(p_index);
//          k_particle_copy.uy(nm) = k_part.uy(p_index);
//          k_particle_copy.uz(nm) = k_part.uz(p_index);
//          k_particle_copy.w(nm) = k_part.w(p_index);
//          k_particle_copy.i(nm) = k_part.i(p_index);
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
//  });
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

////  });
//
//  Kokkos::parallel_for("advance_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np/2),
//    KOKKOS_LAMBDA (size_t p_index)
//    {
//
////    auto  k_accumulators_scatter_access = k_accumulators_sa.access();
//    packed_t v0, v1, v2, v3, v4, v5;
//
//    packed_t dx = packed_t(k_part.dx(p_index*2), k_part.dx(p_index*2+1)); // Load position
//    packed_t dy = packed_t(k_part.dy(p_index*2), k_part.dy(p_index*2+1));
//    packed_t dz = packed_t(k_part.dz(p_index*2), k_part.dz(p_index*2+1));
//    int pi0 = k_part.i(p_index*2);
//    int pi1 = k_part.i(p_index*2+1);
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
//    packed_t cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
//    packed_t cby  = f_cby + dy*f_dcbydy;
//    packed_t cbz  = f_cbz + dz*f_dcbzdz;
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
//    v3   = v0*(packed_t(one)+v2*(packed_t(one_third)+v2*packed_t(two_fifteenths)));
//    v4   = v3/(packed_t(one)+v1*(v3*v3));
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
//    ux  *= packed_t(cdt_dx);
//    uy  *= packed_t(cdt_dy);
//    uz  *= packed_t(cdt_dz);
//
//    /**/                                      // Get norm displacement
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
//    packed_t v3_inbnds = leq(v3, packed_t(one));
//            v3_inbnds += leq(-v3, packed_t(one));
//    packed_t v4_inbnds = leq(v4, packed_t(one));
//            v4_inbnds += leq(-v4, packed_t(one));
//    packed_t v5_inbnds = leq(v5, packed_t(one));
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
////    k_part.dx(p_index*2)    = (pos_t(1.0) - inbnds.low2half())*k_part.dx(p_index*2) + 
////                              inbnds.low2half()*v3.low2half();
////    k_part.dx(p_index*2+1)  = (pos_t(1.0) - inbnds.high2half())*k_part.dx(p_index*2+1) + 
////                              inbnds.high2half()*v3.high2half();
////    k_part.dy(p_index*2)    = (pos_t(1.0) - inbnds.low2half())*k_part.dy(p_index*2) + 
////                              inbnds.low2half()*v4.low2half();
////    k_part.dy(p_index*2+1)  = (pos_t(1.0) - inbnds.high2half())*k_part.dy(p_index*2+1) + 
////                              inbnds.high2half()*v4.high2half();
////    k_part.dz(p_index*2)    = (pos_t(1.0) - inbnds.low2half())*k_part.dz(p_index*2) + 
////                              inbnds.low2half()*v5.low2half();
////    k_part.dz(p_index*2+1)  = (pos_t(1.0) - inbnds.high2half())*k_part.dz(p_index*2+1) + 
////                              inbnds.high2half()*v5.high2half();
//    dx = v0;                                // Streak midpoint
//    dy = v1;
//    dz = v2;
//    v5 = q*ux*uy*uz*one_third;              // Compute correction
//
//#   define ACCUMULATE_J(X,Y,Z)                                 \
//    v4  = q*u##X;   /* v2 = q ux                            */        \
//    v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
//    v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
//    v1 += v4;       /* v1 = q ux (1+dy)                     */        \
//    v4  = packed_t(one)+d##Z; /* v4 = 1+dz                            */        \
//    v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
//    v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
//    v4  = packed_t(one)-d##Z; /* v4 = 1-dz                            */        \
//    v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
//    v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
//    v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
//    v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
//    v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
//    v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
//
//    k_particle_mover_t local_pm[1];
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
//      ACCUMULATE_J( y,z,x );
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jy, 3), v3.low2float());
//      ACCUMULATE_J( z,x,y );
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 3) += v3.low2float();
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 0), v0.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 1), v1.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 2), v2.low2float());
//      Kokkos::atomic_add(&k_accumulator(pi0, accumulator_var::jz, 3), v3.low2float());
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
//      ACCUMULATE_J( y,z,x );
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jy, 3), v3.high2float());
//      ACCUMULATE_J( z,x,y );
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 3) += v3.high2float();
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 0), v0.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 1), v1.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 2), v2.high2float());
//      Kokkos::atomic_add(&k_accumulator(pi1, accumulator_var::jz, 3), v3.high2float());
//    }
//    if(inbnds.low2float()) {
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
//          k_particle_copy.w(nm) = k_part.w(p_index*2);
//          k_particle_copy.i(nm) = k_part.i(p_index*2);
//        }
//      }
//    }
//    if(inbnds.high2float()) {
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
//          k_particle_copy.w(nm) = k_part.w(p_index*2+1);
//          k_particle_copy.i(nm) = k_part.i(p_index*2+1);
//        }
//      }
//    }
//
//#   undef ACCUMULATE_J
//
//  
//
////    if(  v3<=packed_t(one) &&  v4<=packed_t(one) &&  v5<=packed_t(one) &&   // Check if inbnds
////        -v3<=packed_t(one) && -v4<=packed_t(one) && -v5<=packed_t(one) ) {
////
////      // Common case (inbnds).  Note: accumulator values are 4 times
////      // the total physical charge that passed through the appropriate
////      // current quadrant in a time-step
////
////      q *= packed_t(qsp);
////      k_part.dx(p_index*2)    = v3.low2half();                             // Store new position
////      k_part.dx(p_index*2+1)  = v3.high2half();
////      k_part.dy(p_index*2)    = v4.low2half();
////      k_part.dy(p_index*2+1)  = v4.high2half();
////      k_part.dz(p_index*2)    = v5.low2half();
////      k_part.dz(p_index*2+1)  = v5.high2half();
////      dx = v0;                                // Streak midpoint
////      dy = v1;
////      dz = v2;
////      v5 = q*ux*uy*uz*one_third;              // Compute correction
////
////#     define ACCUMULATE_J(X,Y,Z)                                 \
////      v4  = q*u##X;   /* v2 = q ux                            */        \
////      v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
////      v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
////      v1 += v4;       /* v1 = q ux (1+dy)                     */        \
////      v4  = packed_t(one)+d##Z; /* v4 = 1+dz                            */        \
////      v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
////      v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
////      v4  = packed_t(one)-d##Z; /* v4 = 1-dz                            */        \
////      v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
////      v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
////      v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
////      v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
////      v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
////      v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
////
////      ACCUMULATE_J( x,y,z );
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jx, 3) += v3.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jx, 3) += v3.high2float();
////
////      ACCUMULATE_J( y,z,x );
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jy, 3) += v3.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jy, 3) += v3.high2float();
////
////      ACCUMULATE_J( z,x,y );
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 0) += v0.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 0) += v0.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 1) += v1.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 1) += v1.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 2) += v2.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 2) += v2.high2float();
////      k_accumulators_scatter_access(pi0, accumulator_var::jz, 3) += v3.low2float();
////      k_accumulators_scatter_access(pi1, accumulator_var::jz, 3) += v3.high2float();
////
////#     undef ACCUMULATE_J
////
////    } 
////    else
////    {                                    // Unlikely
////      pos_t low_q   = q.low2half();
////      pos_t high_q  = q.high2half();
////      pos_t low_v0  = v0.low2half();
////      pos_t high_v0 = v0.high2half();
////      pos_t low_v1  = v1.low2half();
////      pos_t high_v1 = v1.high2half();
////      pos_t low_v2  = v2.low2half();
////      pos_t high_v2 = v2.high2half();
////      pos_t low_v3  = v3.low2half();
////      pos_t high_v3 = v3.high2half();
////      pos_t low_v4  = v4.low2half();
////      pos_t high_v4 = v4.high2half();
////      pos_t low_v5  = v5.low2half();
////      pos_t high_v5 = v5.high2half();
////
////      pos_t low_dx  = dx.low2half();
////      pos_t high_dx = dx.high2half();
////      pos_t low_dy  = dy.low2half();
////      pos_t high_dy = dy.high2half();
////      pos_t low_dz  = dz.low2half();
////      pos_t high_dz = dz.high2half();
////      pos_t low_ux  = ux.low2half();
////      pos_t high_ux = ux.high2half();
////      pos_t low_uy  = uy.low2half();
////      pos_t high_uy = uy.high2half();
////      pos_t low_uz  = uz.low2half();
////      pos_t high_uz = uz.high2half();
////
////      if(  low_v3<=pos_t(one) &&  low_v4<=pos_t(one) &&  low_v5<=pos_t(one) &&   // Check if inbnds
////          -low_v3<=pos_t(one) && -low_v4<=pos_t(one) && -low_v5<=pos_t(one) ) {
////        low_q *= qsp;
////        k_part.dx(p_index*2)    = low_v3;                             // Store new position
////        k_part.dy(p_index*2)    = low_v4;
////        k_part.dz(p_index*2)    = low_v5;
////        low_dx = low_v0;                                // Streak midpoint
////        low_dy = low_v1;
////        low_dz = low_v2;
////        low_v5 = low_q*low_ux*low_uy*low_uz*one_third;              // Compute correction
////
////#       define ACCUMULATE_J(X,Y,Z)                                 \
////        low_v4  = low_q*low_u##X;   /* v2 = q ux                            */        \
////        low_v1  = low_v4*low_d##Y;  /* v1 = q ux dy                         */        \
////        low_v0  = low_v4-low_v1;    /* v0 = q ux (1-dy)                     */        \
////        low_v1 += low_v4;       /* v1 = q ux (1+dy)                     */        \
////        low_v4  = one+low_d##Z; /* v4 = 1+dz                            */        \
////        low_v2  = low_v0*low_v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
////        low_v3  = low_v1*low_v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
////        low_v4  = one-low_d##Z; /* v4 = 1-dz                            */        \
////        low_v0 *= low_v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
////        low_v1 *= low_v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
////        low_v0 += low_v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
////        low_v1 -= low_v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
////        low_v2 -= low_v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
////        low_v3 += low_v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
////
////        ACCUMULATE_J( x,y,z );
////        k_accumulators_scatter_access(pi0, accumulator_var::jx, 0) += low_v0;
////        k_accumulators_scatter_access(pi0, accumulator_var::jx, 1) += low_v1;
////        k_accumulators_scatter_access(pi0, accumulator_var::jx, 2) += low_v2;
////        k_accumulators_scatter_access(pi0, accumulator_var::jx, 3) += low_v3;
////
////        ACCUMULATE_J( y,z,x );
////        k_accumulators_scatter_access(pi0, accumulator_var::jy, 0) += low_v0;
////        k_accumulators_scatter_access(pi0, accumulator_var::jy, 1) += low_v1;
////        k_accumulators_scatter_access(pi0, accumulator_var::jy, 2) += low_v2;
////        k_accumulators_scatter_access(pi0, accumulator_var::jy, 3) += low_v3;
////
////        ACCUMULATE_J( z,x,y );
////        k_accumulators_scatter_access(pi0, accumulator_var::jz, 0) += low_v0;
////        k_accumulators_scatter_access(pi0, accumulator_var::jz, 1) += low_v1;
////        k_accumulators_scatter_access(pi0, accumulator_var::jz, 2) += low_v2;
////        k_accumulators_scatter_access(pi0, accumulator_var::jz, 3) += low_v3;
////#       undef ACCUMULATE_J
////      } else {
////        k_particle_mover_t local_pm[1];
////        local_pm->dispx = low_ux;
////        local_pm->dispy = low_uy;
////        local_pm->dispz = low_uz;
////        local_pm->i     = p_index*2;
////
////        //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
////        if( move_p_kokkos(k_part, local_pm,
////                           k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
////          if( k_nm(0)<max_nm ) {
////            const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
////            if (nm >= max_nm) Kokkos::abort("overran max_nm");
////
////            k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
////            k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
////            k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
////            k_particle_movers_i(nm)   = local_pm->i;
////
////            // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
////            k_particle_copy(nm, particle_var::dx) = k_part.dx(p_index*2);
////            k_particle_copy(nm, particle_var::dy) = k_part.dy(p_index*2);
////            k_particle_copy(nm, particle_var::dz) = k_part.dz(p_index*2);
////            k_particle_copy(nm, particle_var::ux) = k_part.ux(p_index*2);
////            k_particle_copy(nm, particle_var::uy) = k_part.uy(p_index*2);
////            k_particle_copy(nm, particle_var::uz) = k_part.uz(p_index*2);
////            k_particle_copy(nm, particle_var::w) = k_part.w(p_index*2);
////            k_particle_i_copy(nm) = k_part.i(p_index*2);
////          }
////        }
////      }
////      if(  high_v3<=pos_t(one) &&  high_v4<=pos_t(one) &&  high_v5<=pos_t(one) &&   // Check if inbnds
////          -high_v3<=pos_t(one) && -high_v4<=pos_t(one) && -high_v5<=pos_t(one) ) {
////        high_q *= qsp;
////        k_part.dx(p_index*2+1)    = high_v3;                             // Store new position
////        k_part.dy(p_index*2+1)    = high_v4;
////        k_part.dz(p_index*2+1)    = high_v5;
////        high_dx = high_v0;                                // Streak midpoint
////        high_dy = high_v1;
////        high_dz = high_v2;
////        high_v5 = high_q*high_ux*high_uy*high_uz*one_third;              // Compute correction
////
////#       define ACCUMULATE_J(X,Y,Z)                                 \
////        high_v4  = high_q*high_u##X;   /* v2 = q ux                            */        \
////        high_v1  = high_v4*high_d##Y;  /* v1 = q ux dy                         */        \
////        high_v0  = high_v4-high_v1;    /* v0 = q ux (1-dy)                     */        \
////        high_v1 += high_v4;       /* v1 = q ux (1+dy)                     */        \
////        high_v4  = one+high_d##Z; /* v4 = 1+dz                            */        \
////        high_v2  = high_v0*high_v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
////        high_v3  = high_v1*high_v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
////        high_v4  = one-high_d##Z; /* v4 = 1-dz                            */        \
////        high_v0 *= high_v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
////        high_v1 *= high_v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
////        high_v0 += high_v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
////        high_v1 -= high_v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
////        high_v2 -= high_v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
////        high_v3 += high_v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */
////
////        ACCUMULATE_J( x,y,z );
////        k_accumulators_scatter_access(pi1, accumulator_var::jx, 0) += high_v0;
////        k_accumulators_scatter_access(pi1, accumulator_var::jx, 1) += high_v1;
////        k_accumulators_scatter_access(pi1, accumulator_var::jx, 2) += high_v2;
////        k_accumulators_scatter_access(pi1, accumulator_var::jx, 3) += high_v3;
////
////        ACCUMULATE_J( y,z,x );
////        k_accumulators_scatter_access(pi1, accumulator_var::jy, 0) += high_v0;
////        k_accumulators_scatter_access(pi1, accumulator_var::jy, 1) += high_v1;
////        k_accumulators_scatter_access(pi1, accumulator_var::jy, 2) += high_v2;
////        k_accumulators_scatter_access(pi1, accumulator_var::jy, 3) += high_v3;
////
////        ACCUMULATE_J( z,x,y );
////        k_accumulators_scatter_access(pi1, accumulator_var::jz, 0) += high_v0;
////        k_accumulators_scatter_access(pi1, accumulator_var::jz, 1) += high_v1;
////        k_accumulators_scatter_access(pi1, accumulator_var::jz, 2) += high_v2;
////        k_accumulators_scatter_access(pi1, accumulator_var::jz, 3) += high_v3;
////#       undef ACCUMULATE_J
////      } else {
////        k_particle_mover_t local_pm[1];
////        local_pm->dispx = high_ux;
////        local_pm->dispy = high_uy;
////        local_pm->dispz = high_uz;
////        local_pm->i     = p_index*2+1;
////
////        //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
////        if( move_p_kokkos(k_part, local_pm,
////                           k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
////          if( k_nm(0)<max_nm ) {
////            const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
////            if (nm >= max_nm) Kokkos::abort("overran max_nm");
////
////            k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
////            k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
////            k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
////            k_particle_movers_i(nm)   = local_pm->i;
////
////            // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
////            k_particle_copy(nm, particle_var::dx) = k_part.dx(p_index*2+1);
////            k_particle_copy(nm, particle_var::dy) = k_part.dy(p_index*2+1);
////            k_particle_copy(nm, particle_var::dz) = k_part.dz(p_index*2+1);
////            k_particle_copy(nm, particle_var::ux) = k_part.ux(p_index*2+1);
////            k_particle_copy(nm, particle_var::uy) = k_part.uy(p_index*2+1);
////            k_particle_copy(nm, particle_var::uz) = k_part.uz(p_index*2+1);
////            k_particle_copy(nm, particle_var::w) = k_part.w(p_index*2*1);
////            k_particle_i_copy(nm) = k_part.i(p_index*2+1);
////          }
////        }
////      }
////    }
//
////      k_particle_mover_t local_pm[1];
////      local_pm->dispx = ux;
////      local_pm->dispy = uy;
////      local_pm->dispz = uz;
////      local_pm->i     = p_index;
////
////      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
////      if( move_p_kokkos(k_part, local_pm,
////                         k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) ) { // Unlikely
////        if( k_nm(0)<max_nm ) {
////          const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
////          if (nm >= max_nm) Kokkos::abort("overran max_nm");
////
////          k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
////          k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
////          k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
////          k_particle_movers_i(nm)   = local_pm->i;
////
////          // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
////          k_particle_copy(nm, particle_var::dx) = p_dx;
////          k_particle_copy(nm, particle_var::dy) = p_dy;
////          k_particle_copy(nm, particle_var::dz) = p_dz;
////          k_particle_copy(nm, particle_var::ux) = p_ux;
////          k_particle_copy(nm, particle_var::uy) = p_uy;
////          k_particle_copy(nm, particle_var::uz) = p_uz;
////          k_particle_copy(nm, particle_var::w) = p_w;
////          k_particle_i_copy(nm) = pii;
////        }
////      }
////    }
//
//  });



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
