/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "vpic.h"
#include <tuple>

# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE { \
	int _ix, _iy, _iz;                                    \
	_ix  = (rank);        /* ix = ix+gpx*( iy+gpy*iz ) */ \
	_iy  = _ix/int(px);   /* iy = iy+gpy*iz */            \
	_ix -= _iy*int(px);   /* ix = ix */                   \
	_iz  = _iy/int(py);   /* iz = iz */                   \
	_iy -= _iz*int(py);   /* iy = iy */                   \
	(ix) = _ix;                                           \
  (iy) = _iy;                                           \
  (iz) = _iz;                                           \
} END_PRIMITIVE

namespace PoyntingVar {
  enum PVars {
    p=0,
    e1=1,
    e2=2,
    cb1=3,
    cb2=4
  };
}

/** Poynting diagnostic
 *  Compute Poynting flux (E x B) for each selected face. 
 *  Flux is normalized by the number of cells in each face. Setting PoyntingSum
 *  flag will also return the integrated Poynting flux on each face. Diagnostic
 *  assumes uniform domains.
 *  On the x faces, e1 = ey, e2 = ez, cb1 = cby, cb2 = cbz
 *  On the y faces, e1 = ez, e2 = ex, cb1 = cbz, cb2 = cbx
 *  On the z faces, e1 = ex, e2 = ey, cb1 = cbx, cb2 = cby
 *
 *  \param face_enum Bitmask marking which faces to compute Poynting flux
 *  \param e0 Peak instantaneous E field in "natural units"
 *
 *  \retval Tuple containing the full Poynting flux data and the integrated 
 *  Poynting flux on each face {-x,+x, -y,+y, -z,+z}
 */
std::tuple<Kokkos::View<double*[5]>, Kokkos::View<double[6]>> 
vpic_simulation::poynting_flux(const int face_enum, const double e0) {
  const int nx = grid->nx, ny = grid->ny, nz = grid->nz;
  const int tx = px, ty = py, tz = pz; // Topology
  const int num_xcells = int(nx*tx);
  const int num_ycells = int(ny*ty);
  const int num_zcells = int(nz*tz);

  const double norm = 1.0 / (grid->cvac*grid->cvac*e0*e0);
  uint64_t skip = 0, stride = 0;
  auto field = field_array->k_f_d;

  // Get position of domain in global topology
  int ix, iy, iz;
  RANK_TO_INDEX( int(rank()), ix, iy, iz );  

  // Calculate space for storing poynting flux
  if(face_enum & NegXFace) stride += num_ycells * num_zcells;
  if(face_enum & PosXFace) stride += num_ycells * num_zcells;
  if(face_enum & NegYFace) stride += num_zcells * num_xcells;
  if(face_enum & PosYFace) stride += num_zcells * num_xcells;
  if(face_enum & NegZFace) stride += num_xcells * num_ycells;
  if(face_enum & PosZFace) stride += num_xcells * num_ycells;

  Kokkos::View<double*[5]> lpview("Local Poynting View",  stride); 
  Kokkos::View<double*[5]> gpview("Global Poynting View", stride); 
  Kokkos::deep_copy(lpview, 0);
  Kokkos::View<double[6], Kokkos::LayoutRight> psum("Local Poynting sum"); 
  Kokkos::View<double[6], Kokkos::LayoutRight> gpsum("Global Poynting sum");
  Kokkos::deep_copy(psum, 0);
  auto psum_sv = Kokkos::Experimental::create_scatter_view(psum);

  using FaceRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

#define xINDEX_FORTRAN_3(F) INDEX_FORTRAN_3(F,j+1,k+1,0,nx+1,0,ny+1,0,nz+1)
#define yINDEX_FORTRAN_3(F) INDEX_FORTRAN_3(k+1,F,j+1,0,nx+1,0,ny+1,0,nz+1)
#define zINDEX_FORTRAN_3(F) INDEX_FORTRAN_3(j+1,k+1,F,0,nx+1,0,ny+1,0,nz+1)
#define xDIM 0
#define yDIM 1
#define zDIM 2

#define COMPUTE_POYNTING_FLUX(X,Y,Z, _direction)                               \
  int face = _direction < 0 ? 1 : n##X-1;                                      \
  int face_id  = _direction < 0 ? 2*X##DIM : 2*X##DIM+1;                       \
  Kokkos::parallel_for("Poynting Flux", FaceRange({0,1},{n##Y+0,n##Z+1}),      \
  KOKKOS_LAMBDA(const int j, const int k) {                                    \
    float e1, e2, cb1, cb2;                                                    \
    auto psum_sa = psum_sv.access();                                           \
    /* In output, the 2D surface arrays A[j,k] are FORTRAN indexed: */         \
    /* The j quantity varyies fastest, k, slowest. */                          \
    int index = int(  ((i##Y*n##Y) + j-0)                                      \
                    + ((i##Z*n##Z) + k-1) * (n##Y*t##Y)                        \
                    + skip);                                                   \
    int k1  = X##INDEX_FORTRAN_3(face);                                        \
    int k2  = X##INDEX_FORTRAN_3(face+1);                                      \
    e1  = field(k2, field_var::e##Y);                                          \
    e2  = field(k2, field_var::e##Z);                                          \
    cb1 = 0.5*(field(k1, field_var::cb##Y)+field(k2, field_var::cb##Y));       \
    cb2 = 0.5*(field(k1, field_var::cb##Z)+field(k2, field_var::cb##Z));       \
    const double flux = ( e1*cb2 - e2*cb1 ) * norm;                            \
    lpview(index, PoyntingVar::p)   = flux;                                    \
    lpview(index, PoyntingVar::e1)  = e1;                                      \
    lpview(index, PoyntingVar::e2)  = e2;                                      \
    lpview(index, PoyntingVar::cb1) = cb1;                                     \
    lpview(index, PoyntingVar::cb2) = cb2;                                     \
    psum_sa(face_id) += flux;                                                  \
  });

  // Compute poynting flux and store E and B components
  skip = 0;
  if ( (face_enum & NegXFace) && (ix == 0) ) {
    COMPUTE_POYNTING_FLUX(x,y,z, -1);
    skip += num_ycells * num_zcells;
  } 
  if ( (face_enum & PosXFace) && (ix == tx-1) ) {
    COMPUTE_POYNTING_FLUX(x,y,z,  1);
    skip += num_ycells * num_zcells;
  } 
  if ( (face_enum & NegYFace) && (iy == 0) ) {
    COMPUTE_POYNTING_FLUX(y,z,x, -1);
    skip += num_zcells * num_xcells;
  } 
  if ( (face_enum & PosYFace) && (iy == ty-1) ) {
    COMPUTE_POYNTING_FLUX(y,z,x,  1);
    skip += num_zcells * num_xcells;
  } 
  if ( (face_enum & NegZFace) && (iz == 0) ) {
    COMPUTE_POYNTING_FLUX(z,x,y, -1);
    skip += num_xcells * num_ycells;
  } 
  if ( (face_enum & PosZFace) && (iz == tz-1) ) {
    COMPUTE_POYNTING_FLUX(z,x,y,  1);
    skip += num_xcells * num_ycells;
  }
  Kokkos::fence();
  Kokkos::Experimental::contribute(psum, psum_sv);

  auto lpview_h = Kokkos::create_mirror_view(lpview);
  auto gpview_h = Kokkos::create_mirror_view(gpview);
  Kokkos::deep_copy(lpview_h, lpview);
  mp_allsum_d(lpview_h.data(), gpview_h.data(), stride);
  Kokkos::deep_copy(gpview, gpview_h);

  skip = 0;
  if ( face_enum & PoyntingSum ) {
    // Sum over all surfaces
    auto lpsum_h = Kokkos::create_mirror_view(psum);
    auto gpsum_h = Kokkos::create_mirror_view(gpsum);
    Kokkos::deep_copy(lpsum_h, psum);

    // Reduce all surfaces into single global view
    mp_allsum_d(lpsum_h.data(), gpsum_h.data(), 6);

    // Divide by number of mesh points summed over
    gpsum_h(0) /= static_cast<double>(num_ycells*num_zcells);
    gpsum_h(1) /= static_cast<double>(num_ycells*num_zcells);
    gpsum_h(2) /= static_cast<double>(num_zcells*num_xcells);
    gpsum_h(3) /= static_cast<double>(num_zcells*num_xcells);
    gpsum_h(4) /= static_cast<double>(num_xcells*num_ycells);
    gpsum_h(5) /= static_cast<double>(num_xcells*num_ycells);
    Kokkos::deep_copy(gpsum, gpsum_h);
  }
  
  return std::make_tuple(gpview, gpsum);
#undef xDIM
#undef yDIM
#undef zDIM
#undef xINDEX_FOTRAN_3
#undef yINDEX_FOTRAN_3
#undef zINDEX_FOTRAN_3
#undef COMPUTE_POYNTING_FLUX
} // poynting_flux

/** Poynting flux tally diagnostic
 *  Sum Poynting flux (E x B) for each selected face. 
 *  On the x faces, e1 = ey, e2 = ez, cb1 = cby, cb2 = cbz
 *  On the y faces, e1 = ez, e2 = ex, cb1 = cbz, cb2 = cbx
 *  On the z faces, e1 = ex, e2 = ey, cb1 = cbx, cb2 = cby
 *
 *  \param face_enum Bitmask marking which faces to compute Poynting flux
 *  \param e0 Peak instantaneous E field in "natural units"
 *
 *  \retval View with Poynting flux sum {-x,+x, -y,+y, -z,+z}
 */
Kokkos::View<double[6]> 
vpic_simulation::poynting_flux_tally(const int face_enum, const double e0) {
  const int nx = grid->nx, ny = grid->ny, nz = grid->nz;
  const int tx = px, ty = py, tz = pz;
  const int num_xcells = int(nx*tx);
  const int num_ycells = int(ny*ty);
  const int num_zcells = int(nz*tz);

  // MKS 
  // n.b. 1/mu0 = c^2 * eps0 and poynting should be dA * dt * ExB * (1/mu0)
  const double norm_xface = grid->dt * grid->dy * grid->dz * e0 * grid->cvac * grid->cvac;
  const double norm_yface = grid->dt * grid->dz * grid->dx * e0 * grid->cvac * grid->cvac;
  const double norm_zface = grid->dt * grid->dx * grid->dy * e0 * grid->cvac * grid->cvac;

  uint64_t skip = 0;
  auto field = field_array->k_f_d;

  // Get position of domain in global topology
  int ix, iy, iz;
  RANK_TO_INDEX( int(rank()), ix, iy, iz );  

  Kokkos::View<double[6]> psum("Local Poynting sum"); 
  Kokkos::deep_copy(psum, 0);
  auto psum_sv = Kokkos::Experimental::create_scatter_view(psum);

  using FaceRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

#define xVOXEL(F,j,k) VOXEL(F,j,k,nx,ny,nz)
#define yVOXEL(F,j,k) VOXEL(k,F,j,nx,ny,nz)
#define zVOXEL(F,j,k) VOXEL(j,k,F,nx,ny,nz)
#define xDIM 0
#define yDIM 1
#define zDIM 2

#define COMPUTE_POYNTING_FLUX_TALLY(X,Y,Z, _direction)                         \
  int face = _direction < 0 ? 1 : n##X;                                        \
  int face_idx  = _direction < 0 ? 2*X##DIM : 2*X##DIM+1;                      \
  Kokkos::parallel_for("Poynting Flux", FaceRange({1,1},{n##Y+1,n##Z+1}),      \
  KOKKOS_LAMBDA(const int j, const int k) {                                    \
    float e1, e2, cb1, cb2;                                                    \
    auto psum_sa = psum_sv.access();                                           \
    e1  = 0.25*(  field(X##VOXEL(face,  j,  k),   field_var::e##Y)             \
                + field(X##VOXEL(face,  j,  k+1), field_var::e##Y)             \
                + field(X##VOXEL(face+1,j,  k),   field_var::e##Y)             \
                + field(X##VOXEL(face+1,j,  k+1), field_var::e##Y) );          \
    e2  = 0.25*(  field(X##VOXEL(face,  j,  k),   field_var::e##Z)             \
                + field(X##VOXEL(face,  j+1,k),   field_var::e##Z)             \
                + field(X##VOXEL(face+1,j,  k),   field_var::e##Z)             \
                + field(X##VOXEL(face+1,j+1,k),   field_var::e##Z) );          \
    cb1 = 0.50*(  field(X##VOXEL(face,  j,  k),   field_var::cb##Y)            \
                + field(X##VOXEL(face,  j+1,k),   field_var::cb##Y) );         \
    cb2 = 0.50*(  field(X##VOXEL(face,  j,  k),   field_var::cb##Z)            \
                + field(X##VOXEL(face,  j,  k+1), field_var::cb##Z) );         \
    double flux = -_direction*( e1*cb2-e2*cb1 )*norm_##X##face;                \
    psum_sa(face_idx) += flux;                                                 \
  });

  // Compute poynting flux sum
  skip = 0;
  if ( (face_enum & NegXFace) && (ix == 0) ) {
    COMPUTE_POYNTING_FLUX_TALLY(x,y,z, -1);
    skip += num_ycells * num_zcells;
  } 
  if ( (face_enum & PosXFace) && (ix == tx-1) ) {
    COMPUTE_POYNTING_FLUX_TALLY(x,y,z,  1);
    skip += num_ycells * num_zcells;
  } 
  if ( (face_enum & NegYFace) && (iy == 0) ) {
    COMPUTE_POYNTING_FLUX_TALLY(y,z,x, -1);
    skip += num_zcells * num_xcells;
  } 
  if ( (face_enum & PosYFace) && (iy == ty-1) ) {
    COMPUTE_POYNTING_FLUX_TALLY(y,z,x,  1);
    skip += num_zcells * num_xcells;
  } 
  if ( (face_enum & NegZFace) && (iz == 0) ) {
    COMPUTE_POYNTING_FLUX_TALLY(z,x,y, -1);
    skip += num_xcells * num_ycells;
  } 
  if ( (face_enum & PosZFace) && (iz == tz-1) ) {
    COMPUTE_POYNTING_FLUX_TALLY(z,x,y,  1);
    skip += num_xcells * num_ycells;
  }
  Kokkos::fence();
  Kokkos::Experimental::contribute(psum, psum_sv);
  
#undef xVOXEL
#undef yVOXEL
#undef zVOXEL
#undef xDIM
#undef yDIM
#undef zDIM
#undef COMPUTE_POYNTING_FLUX
  return psum;
} // poynting_flux

/*------------------------------------------------------------------------------
 * Compute poynting flux sum over all boundaries
 *
 * Inputs:
 *   e0 Peak instantaneous E field in "natural units"
 *----------------------------------------------------------------------------*/
double vpic_simulation::poynting_flux(double e0) {
	double psum=0.0, gpsum=0.0;
  int face_enum = All;
  // Compute Poynting flux sum for all 6 faces
  auto flux_tally = poynting_flux_tally(face_enum, e0);
  auto flux_tally_h = Kokkos::create_mirror_view(flux_tally);
  Kokkos::deep_copy(flux_tally_h, flux_tally);

	// Sum flux contributions
  for(size_t i=0; i<flux_tally_h.size(); i++)
    psum += flux_tally_h(i);

	// Collect sums over all ranks
	mp_allsum_d( &psum, &gpsum, 1 );

	return gpsum;
} // poynting_flux

#undef RANK_TO_INDEX
