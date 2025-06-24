// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#include "sfa_private.h"

#define F(ind,v) k_field(f##ind##_index, field_var::v)

#define INIT_STENCIL()							\
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);			\
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);			\
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);			\
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);			\
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);			\
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);			\
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);

#define COPY_FD(fd_)                                              \
  Kokkos::parallel_for("hyb_smooth copy_##fd_",                   \
                       Kokkos::RangePolicy<>(0,nv),               \
                       KOKKOS_LAMBDA(const int v) {               \
      k_field(v, field_var::tmpsm) = k_field(v, field_var::fd_);   \
  });

#define SMOOTH_FD(fd_)                                                        \
  Kokkos::parallel_for("hyb_smooth smooth_##fd_",                             \
                       xyz_policy,                                            \
                       KOKKOS_LAMBDA(const int x, const int y, const int z) { \
      INIT_STENCIL();                                                         \
      F(0,fd_) = twelfth*(six*F(0,tmpsm) + F(x,tmpsm) + F(mx,tmpsm)           \
                                         + F(y,tmpsm) + F(my,tmpsm)           \
                                         + F(z,tmpsm) + F(mz,tmpsm));         \
  });

void
hyb_smooth_moments( field_array_t * RESTRICT fa ) {
  if( !fa     ) ERROR(( "Bad args" ));

  k_field_t k_field = fa->k_f_d;
  const grid_t * g = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  const float twelfth = 1./12., six=6.;
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1, 1, 1}, {nx+1, ny+1, nz+1});

  COPY_FD(jfx);  SMOOTH_FD(jfx);
  COPY_FD(jfy);  SMOOTH_FD(jfy);
  COPY_FD(jfz);  SMOOTH_FD(jfz);
  COPY_FD(rhof); SMOOTH_FD(rhof);
}

void
hyb_smooth_b( field_array_t * RESTRICT fa ) {
  if( !fa     ) ERROR(( "Bad args" ));

  k_field_t k_field = fa->k_f_d;
  const grid_t * g = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  const float twelfth = 1./12., six=6.;
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1, 1, 1}, {nx+1, ny+1, nz+1});

  COPY_FD(cbx); SMOOTH_FD(cbx);
  COPY_FD(cby); SMOOTH_FD(cby);
  COPY_FD(cbz); SMOOTH_FD(cbz);

  k_begin_remote_ghost_hyb_b(fa);
  k_end_remote_ghost_hyb_b  (fa);
  k_hyb_local_ghost_b(fa, fa->g );
}

void
hyb_smooth_eb_interp( field_array_t * RESTRICT fa, bool smoothed ) {
  if( !fa     ) ERROR(( "Bad args" ));

  k_field_t k_field = fa->k_f_d;
  const grid_t * g = fa->g;
  size_t nx   = g->nx;
  size_t ny   = g->ny;
  size_t nz   = g->nz;
  size_t nv   = g->nv;
  const float twelfth = 1./12., six=6.;
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1, 1, 1}, {nx+1, ny+1, nz+1});

  //Copy for smoothing
  //Copy is still necessary when nsm=0 because field->particle interpolator
  //coefficients use smoothed field vars.
  Kokkos::parallel_for("hyb_smooth store_eb", Kokkos::RangePolicy<>(0,nv),
                       KOKKOS_LAMBDA(const int v) {
                         k_field(v, field_var::ox) = k_field(v, field_var::cbx);
                         k_field(v, field_var::oy) = k_field(v, field_var::cby);
                         k_field(v, field_var::oz) = k_field(v, field_var::cbz);
                         k_field(v, field_var::tx) = k_field(v, field_var::ex);
                         k_field(v, field_var::ty) = k_field(v, field_var::ey);
                         k_field(v, field_var::tz) = k_field(v, field_var::ez);
                       });

  int ism = g->nsm;
  while(ism>0) {
    if (smoothed) {
Kokkos::Profiling::pushRegion("Smooth_eb_interp::Smoothed::ox,oy,oz,tx,tz,tz");
      COPY_FD(ox); SMOOTH_FD(ox);
      COPY_FD(oy); SMOOTH_FD(oy);
      COPY_FD(oz); SMOOTH_FD(oz);
      COPY_FD(tx); SMOOTH_FD(tx);
      COPY_FD(ty); SMOOTH_FD(ty);
      COPY_FD(tz); SMOOTH_FD(tz);
Kokkos::Profiling::popRegion();
    } else {
Kokkos::Profiling::pushRegion("Smooth_eb_interp::Not smoothed::cbx,cby,cbz,ex,ez,ez");
      COPY_FD(cbx); SMOOTH_FD(cbx);
      COPY_FD(cby); SMOOTH_FD(cby);
      COPY_FD(cbz); SMOOTH_FD(cbz);
      COPY_FD(ex);  SMOOTH_FD(ex);
      COPY_FD(ey);  SMOOTH_FD(ey);
      COPY_FD(ez);  SMOOTH_FD(ez);
Kokkos::Profiling::popRegion();
    }
    // exchange (ox,oy,oz) ghosts
Kokkos::Profiling::pushRegion("Smooth_eb_interp::Exchange ox,oy,oz ghosts");
    k_begin_remote_ghost_hyb_o(fa);
    k_end_remote_ghost_hyb_o  (fa);
Kokkos::Profiling::popRegion();
    // exchange (tx,ty,tz) ghosts
Kokkos::Profiling::pushRegion("Smooth_eb_interp::Exchange tx,ty,tz ghosts");
    k_begin_remote_ghost_hyb_t(fa);
    k_end_remote_ghost_hyb_t  (fa);
Kokkos::Profiling::popRegion();
Kokkos::Profiling::pushRegion("Smooth_eb_interp::Local ghost ot");
    k_hyb_local_ghost_ot(fa, fa->g );
Kokkos::Profiling::popRegion();
    ism--;
  }
}
