// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/kokkos_tuning.hpp"

template<class TeamMember, class field_sa_t, class field_var>
void
KOKKOS_INLINE_FUNCTION
contribute_current(TeamMember& team_member, field_sa_t& access, int i0, int i1, int i2, int i3, field_var j, float v0, float v1,  float v2, float v3) {
#ifdef __CUDA_ARCH__
  int mask = 0xffffffff;
  int team_rank = team_member.team_rank();
  for(int i=16; i>0; i=i/2) {
    v0 += __shfl_down_sync(mask, v0, i);
    v1 += __shfl_down_sync(mask, v1, i);
    v2 += __shfl_down_sync(mask, v2, i);
    v3 += __shfl_down_sync(mask, v3, i);
  }
  if(team_rank%32 == 0) {
    access(i0, j) += v0;
    access(i1, j) += v1;
    access(i2, j) += v2;
    access(i3, j) += v3;
  }
#else
  team_member.team_reduce(Kokkos::Sum<float>(v0));
  team_member.team_reduce(Kokkos::Sum<float>(v1));
  team_member.team_reduce(Kokkos::Sum<float>(v2));
  team_member.team_reduce(Kokkos::Sum<float>(v3));
  if(team_member.team_rank() == 0) {
    access(i0, j) += v0;
    access(i1, j) += v1;
    access(i2, j) += v2;
    access(i3, j) += v3;
  }
#endif
}

template <class TeamMember=KOKKOS_TEAM_POLICY_DEVICE::member_type>
void
KOKKOS_INLINE_FUNCTION
advance_p_kokkos(
        size_t p_index, 
        k_particles_t k_particles,
        k_particles_i_t k_particles_i,
        k_particle_copy_t k_particle_copy,
        k_particle_i_copy_t k_particle_i_copy,
        k_particle_movers_t k_particle_movers,
        k_particle_i_movers_t k_particle_movers_i,
        k_field_sa_t k_f_sv,
        k_interpolator_t k_interp,
        k_counter_t k_nm,
        k_neighbor_t k_neighbors,
        const float qdt_2mc,
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const float cx,
        const float cy,
        const float cz,
//        const int na,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz,
        const int64_t rangel,
        const int64_t rangeh,
        const bool enable_team_reduction,
        TeamMember* team_member_ptr=NULL
        )
{

  auto team_member = *team_member_ptr;

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

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

  int inbnds=0, reduce=0;
  inbnds = v3<=one && v4<=one && v5<=one && -v3<=one && -v4<=one && -v5<=one;
  if(enable_team_reduction) {
    int min_inbnds = inbnds;
    int max_inbnds = inbnds;
    team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
    team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
    int min_index = ii;
    int max_index = ii;
    team_member.team_reduce(Kokkos::Max<int>(max_index));
    team_member.team_reduce(Kokkos::Min<int>(min_index));
    reduce = min_inbnds == max_inbnds && min_index == max_index;
  }

  // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
  if( inbnds ) {

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


#   define ACCUMULATE_J(X,Y,Z)                                 \
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

    if(enable_team_reduction && reduce) {
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);

      int i0 = ii;
      int i1 = VOXEL(xi,yi+1,zi,nx,ny,nz);
      int i2 = VOXEL(xi,yi,zi+1,nx,ny,nz);
      int i3 = VOXEL(xi,yi+1,zi+1,nx,ny,nz);
      ACCUMULATE_J( x,y,z );
      contribute_current(team_member, k_field_scatter_access, i0, i1, i2, i3, field_var::jfx, v0, v1, v2, v3);

      i1 = VOXEL(xi,yi,zi+1,nx,ny,nz);
      i2 = VOXEL(xi+1,yi,zi,nx,ny,nz);
      i3 = VOXEL(xi+1,yi,zi+1,nx,ny,nz);
      ACCUMULATE_J( y,z,x );
      contribute_current(team_member, k_field_scatter_access, i0, i1, i2, i3, field_var::jfy, v0, v1, v2, v3);

      i1 = VOXEL(xi+1,yi,zi,nx,ny,nz);
      i2 = VOXEL(xi,yi+1,zi,nx,ny,nz);
      i3 = VOXEL(xi+1,yi+1,zi,nx,ny,nz);
      ACCUMULATE_J( z,x,y );
      contribute_current(team_member, k_field_scatter_access, i0, i1, i2, i3, field_var::jfz, v0, v1, v2, v3);
    } else {
      // TODO: That 2 needs to be 2*NGHOST eventually
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);
      ACCUMULATE_J( x,y,z );
      k_field_scatter_access(ii, field_var::jfx) += cx*v0;
      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
      k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
    
      ACCUMULATE_J( y,z,x );
      k_field_scatter_access(ii, field_var::jfy) += cy*v0;
      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
      k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;
    
      ACCUMULATE_J( z,x,y );
      k_field_scatter_access(ii, field_var::jfz) += cz*v0;
      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
      k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
    }
#   undef ACCUMULATE_J
  } else
  {                                    // Unlikely
    /*
       local_pm_dispx = ux;
       local_pm_dispy = uy;
       local_pm_dispz = uz;

       local_pm_i     = p_index;
    */
    DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
    local_pm->dispx = ux;
    local_pm->dispy = uy;
    local_pm->dispz = uz;
    local_pm->i     = p_index;

    //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
    if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                     k_f_sv, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
//                     k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
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

//void
//advance_p_kokkos(
//        k_particles_t& k_particles,
//        k_particles_i_t& k_particles_i,
//        k_particle_copy_t& k_particle_copy,
//        k_particle_i_copy_t& k_particle_i_copy,
//        k_particle_movers_t& k_particle_movers,
//        k_particle_i_movers_t& k_particle_movers_i,
//        //k_accumulators_sa_t k_accumulators_sa,
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
//        const int np,
//        const int max_nm,
//        const int nx,
//        const int ny,
//        const int nz,
//        OptimizationSettings* opt_settings
//        )
//{
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
//  // Process particles for this pipeline
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
//  // zero out nm, we could probably do this earlier if we're worried about it
//  // slowing things down
//  Kokkos::deep_copy(k_nm, 0);
//
//#ifdef VPIC_ENABLE_HIERARCHICAL
//  auto team_policy = Kokkos::TeamPolicy<>(LEAGUE_SIZE, TEAM_SIZE);
//  int per_league = np/LEAGUE_SIZE;
//  if(np%LEAGUE_SIZE > 0)
//    per_league++;
//  Kokkos::parallel_for("advance_p", team_policy, KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
//    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [=] (size_t pindex) {
//      int p_index = team_member.league_rank()*per_league + pindex;
//      if(p_index < np) {
//#else
//  auto range_policy = Kokkos::RangePolicy<>(0,np);
//  Kokkos::parallel_for("advance_p", range_policy, KOKKOS_LAMBDA (size_t p_index) {
//#endif
//      
//    float v0, v1, v2, v3, v4, v5;
//    auto  k_field_scatter_access = k_f_sv.access();
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
//
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
//    bool inbnds=0, reduce=0;
//    inbnds = v3<=one && v4<=one && v5<=one && -v3<=one && -v4<=one && -v5<=one;
//    if(!opt_settings && opt_settings->enable_team_reduction) {
//      int min_inbnds = inbnds;
//      int max_inbnds = inbnds;
//      team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
//      team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
//      int min_index = ii;
//      int max_index = ii;
//      team_member.team_reduce(Kokkos::Max<int>(max_index));
//      team_member.team_reduce(Kokkos::Min<int>(min_index));
//      reduce = min_inbnds == max_inbnds && min_index == max_index;
//    }
//
////#ifdef VPIC_ENABLE_TEAM_REDUCTION
////    int inbnds = v3<=one && v4<=one && v5<=one && -v3<=one && -v4<=one && -v5<=one;
////    int min_inbnds = inbnds;
////    int max_inbnds = inbnds;
////    team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
////    team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
////    int min_index = ii;
////    int max_index = ii;
////    team_member.team_reduce(Kokkos::Max<int>(max_index));
////    team_member.team_reduce(Kokkos::Min<int>(min_index));
////#endif
//
//    // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
////    if(  v3<=one &&  v4<=one &&  v5<=one &&   // Check if inbnds
////        -v3<=one && -v4<=one && -v5<=one ) {
//    if( inbnds ) {
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
//      if(!opt_settings && opt_settings->enable_team_reduction && reduce) {
//        ACCUMULATE_J( x,y,z );
//        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jx, v0, v1, v2, v3);
//        ACCUMULATE_J( y,z,x );
//        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jy, v0, v1, v2, v3);
//        ACCUMULATE_J( z,x,y );
//        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jz, v0, v1, v2, v3);
//      } else {
//// Bypass accumulators TODO: enable warp reduction
//
//      // TODO: That 2 needs to be 2*NGHOST eventually
//      int iii = ii;
//      int zi = iii/((nx+2)*(ny+2));
//      iii -= zi*(nx+2)*(ny+2);
//      int yi = iii/(nx+2);
//      int xi = iii - yi*(nx+2);
//      ACCUMULATE_J( x,y,z );
//      //Kokkos::atomic_add(&k_field(ii, field_var::jfx), cx*v0);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx), cx*v1);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx), cx*v2);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx), cx*v3);
//      k_field_scatter_access(ii, field_var::jfx) += cx*v0;
//      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
//      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
//      k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
//
//      ACCUMULATE_J( y,z,x );
//      //Kokkos::atomic_add(&k_field(ii, field_var::jfy), cy*v0);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v1);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy), cy*v2);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v3);
//      k_field_scatter_access(ii, field_var::jfy) += cy*v0;
//      k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
//      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
//      k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;
//
//      ACCUMULATE_J( z,x,y );
//      //Kokkos::atomic_add(&k_field(ii, field_var::jfz), cz*v0);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz), cz*v1);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v2);
//      //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v3);
//      k_field_scatter_access(ii, field_var::jfz) += cz*v0;
//      k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
//      k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
//      k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
//// TODO: Make this optimization work with the accumulator bypass
//      }
//
////#ifdef VPIC_ENABLE_TEAM_REDUCTION
////      if(min_inbnds == max_inbnds && min_index == max_index) {
////        ACCUMULATE_J( x,y,z );
////        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jx, v0, v1, v2, v3);
////        ACCUMULATE_J( y,z,x );
////        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jy, v0, v1, v2, v3);
////        ACCUMULATE_J( z,x,y );
////        contribute_current(team_member, k_accumulators_scatter_access, ii, accumulator_var::jz, v0, v1, v2, v3);
////      } else {
////#endif
////        ACCUMULATE_J( x,y,z );
////        k_accumulators_scatter_access(ii, accumulator_var::jx, 0) += v0;
////        k_accumulators_scatter_access(ii, accumulator_var::jx, 1) += v1;
////        k_accumulators_scatter_access(ii, accumulator_var::jx, 2) += v2;
////        k_accumulators_scatter_access(ii, accumulator_var::jx, 3) += v3;
////      
////        ACCUMULATE_J( y,z,x );
////        k_accumulators_scatter_access(ii, accumulator_var::jy, 0) += v0;
////        k_accumulators_scatter_access(ii, accumulator_var::jy, 1) += v1;
////        k_accumulators_scatter_access(ii, accumulator_var::jy, 2) += v2;
////        k_accumulators_scatter_access(ii, accumulator_var::jy, 3) += v3;
////      
////        ACCUMULATE_J( z,x,y );
////        k_accumulators_scatter_access(ii, accumulator_var::jz, 0) += v0;
////        k_accumulators_scatter_access(ii, accumulator_var::jz, 1) += v1;
////        k_accumulators_scatter_access(ii, accumulator_var::jz, 2) += v2;
////        k_accumulators_scatter_access(ii, accumulator_var::jz, 3) += v3;
////#ifdef VPIC_ENABLE_TEAM_REDUCTION
////      }
////#endif
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
//      if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
//<<<<<<< Updated upstream
//                     k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
//=======
//                     k_accumulators_sa, k_neighbors, rangel, rangeh, qsp ) )
////                     k_accumulators_sa, g, k_neighbors, rangel, rangeh, qsp ) )
//>>>>>>> Stashed changes
//      {
//        if( k_nm(0) < max_nm )
//        {
//            const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
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
//
//            // Tag this one as having left
//            //k_particles(p_index, particle_var::pi) = 999999;
//
//            // Copy local local_pm back
//            //local_pm_dispx = local_pm->dispx;
//            //local_pm_dispy = local_pm->dispy;
//            //local_pm_dispz = local_pm->dispz;
//            //local_pm_i = local_pm->i;
//            //printf("rank copying %d to nm %d \n", local_pm_i, nm);
//            //copy_local_to_pm(nm);
//        }
//      }
//    }
//#ifdef VPIC_ENABLE_HIERARCHICAL
//  }
//  });
//#endif
//  });
//  Kokkos::Experimental::contribute(k_field, k_f_sv);
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
//}

void
advance_p( /**/  species_t            * RESTRICT sp,
           interpolator_array_t * RESTRICT ia,
           field_array_t* RESTRICT fa,
           OptimizationSettings* opt_settings ) {
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

  Kokkos::deep_copy(sp->k_nm_d, 0);

  auto rangel = sp->g->rangel;
  auto rangeh = sp->g->rangeh;

  auto particles_d = sp->k_p_d;
  auto particles_indices_d = sp->k_p_i_d;
  auto particles_copy_d = sp->k_pc_d;
  auto particles_copy_indices_d = sp->k_pc_i_d;
  auto particles_movers_d = sp->k_pm_d;
  auto particles_movers_indices_d = sp->k_pm_i_d;
  auto k_field = fa->k_f_d;
  auto field_sa = fa->k_field_sa_d;
  auto interpolators = ia->k_i_d;
  auto num_movers = sp->k_nm_d;
  auto neighbors = sp->g->k_neighbor_d;
  float q = sp->q;
  float cx = 0.25 * sp->g->rdy * sp->g->rdz / sp->g->dt;
  float cy = 0.25 * sp->g->rdz * sp->g->rdx / sp->g->dt;
  float cz = 0.25 * sp->g->rdx * sp->g->rdy / sp->g->dt;
  int np = sp->np;
  int max_nm = sp->max_nm;
  int nx = sp->g->nx;
  int ny = sp->g->ny;
  int nz = sp->g->nz;

  if(opt_settings != NULL && opt_settings->enable_hierarchical) {
    bool enable_team_reduce = opt_settings->enable_team_reduction;
    auto team_policy = Kokkos::TeamPolicy<>(opt_settings->advance_p_league_size, opt_settings->advance_p_team_size);
    int per_league = np/opt_settings->advance_p_league_size;
    if(np%opt_settings->advance_p_league_size > 0)
      per_league++;
    Kokkos::parallel_for("advance_p", team_policy, KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [=] (size_t pindex) {
        int p_index = team_member.league_rank()*per_league + pindex;
        if(p_index < np) {
          advance_p_kokkos(
            p_index,
            particles_d,
            particles_indices_d,
            particles_copy_d,
            particles_copy_indices_d,
            particles_movers_d,
            particles_movers_indices_d,
            field_sa,
            interpolators,
            num_movers,
            neighbors,
            qdt_2mc,
            cdt_dx,
            cdt_dy,
            cdt_dz,
            q,
            cx,
            cy,
            cz,
            np,
            max_nm,
            nx,
            ny,
            nz,
            rangel,
            rangeh,
            enable_team_reduce,
            &team_member
          );
        }
     });
    });
  } else {
    bool enable_team_reduce = false;
    auto range_policy = Kokkos::RangePolicy<>(0,np);
    Kokkos::parallel_for("advance_p", range_policy, KOKKOS_LAMBDA (size_t p_index) {
        advance_p_kokkos(
          p_index,
          particles_d,
          particles_indices_d,
          particles_copy_d,
          particles_copy_indices_d,
          particles_movers_d,
          particles_movers_indices_d,
          field_sa,
          interpolators,
          num_movers,
          neighbors,
          qdt_2mc,
          cdt_dx,
          cdt_dy,
          cdt_dz,
          q,
          cx,
          cy,
          cz,
          np,
          max_nm,
          nx,
          ny,
          nz,
          rangel,
          rangeh,
          enable_team_reduce
        );
    });
  }

//>>>>>>> Stashed changes
  Kokkos::Experimental::contribute(k_field, field_sa);
  fa->k_field_sa_d.reset_except(fa->k_f_d);
  //field_array->k_field_sa_d.reset_except(field_array->k_f_d);
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
