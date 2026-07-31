// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define F(ind,v) k_field(f##ind##_index, field_var::v)

#define INIT_STENCIL()                                               \
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);               \
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);               \
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);               \
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);               \
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);               \
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);               \
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);               \
  /* Load curvilinear mesh indices */                                \
  size_t m0_index  = GRID_TO_MESH(x,   y,   z,   nx, ny, nz);       \
  size_t mx_index  = GRID_TO_MESH(x+1, y,   z,   nx, ny, nz);       \
  size_t my_index  = GRID_TO_MESH(x,   y+1, z,   nx, ny, nz);       \
  size_t mz_index  = GRID_TO_MESH(x,   y,   z+1, nx, ny, nz);       \
  size_t mmx_index = GRID_TO_MESH(x-1, y,   z,   nx, ny, nz);       \
  size_t mmy_index = GRID_TO_MESH(x,   y-1, z,   nx, ny, nz);       \
  size_t mmz_index = GRID_TO_MESH(x,   y,   z-1, nx, ny, nz);       \
  /* Load all scale factors needed for curl */                       \
  float h1_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_1);          \
  float h2_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_2);          \
  float h3_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_3);          \
  float h1_x  = k_curv_mesh(mx_index,  curv_mesh_var::h_1);          \
  float h2_x  = k_curv_mesh(mx_index,  curv_mesh_var::h_2);          \
  float h3_x  = k_curv_mesh(mx_index,  curv_mesh_var::h_3);          \
  float h1_mx = k_curv_mesh(mmx_index, curv_mesh_var::h_1);          \
  float h2_mx = k_curv_mesh(mmx_index, curv_mesh_var::h_2);          \
  float h3_mx = k_curv_mesh(mmx_index, curv_mesh_var::h_3);          \
  float h1_y  = k_curv_mesh(my_index,  curv_mesh_var::h_1);          \
  float h2_y  = k_curv_mesh(my_index,  curv_mesh_var::h_2);          \
  float h3_y  = k_curv_mesh(my_index,  curv_mesh_var::h_3);          \
  float h1_my = k_curv_mesh(mmy_index, curv_mesh_var::h_1);          \
  float h2_my = k_curv_mesh(mmy_index, curv_mesh_var::h_2);          \
  float h3_my = k_curv_mesh(mmy_index, curv_mesh_var::h_3);          \
  float h1_z  = k_curv_mesh(mz_index,  curv_mesh_var::h_1);          \
  float h2_z  = k_curv_mesh(mz_index,  curv_mesh_var::h_2);          \
  float h3_z  = k_curv_mesh(mz_index,  curv_mesh_var::h_3);          \
  float h1_mz = k_curv_mesh(mmz_index, curv_mesh_var::h_1);          \
  float h2_mz = k_curv_mesh(mmz_index, curv_mesh_var::h_2);          \
  float h3_mz = k_curv_mesh(mmz_index, curv_mesh_var::h_3);          \
  /* Original fluid quantities */                                    \
  float  rho = half*( (one-hstep)*( F(0,rhof) + F(0,rhofold) ) +    \
                      hstep*( three*F(0,rhof) - F(0,rhofold)) );    \
  rho = (rho > den_floor_ohm) ? rho :  den_floor_ohm;               \
  float  invrho = one/rho;                                           \
  float  ux = invrho*half*( (one-hstep)*( F(0,jfx) + F(0,jfxold) ) + \
                            hstep*( three*F(0,jfx) - F(0,jfxold)) ); \
  float  uy = invrho*half*( (one-hstep)*( F(0,jfy) + F(0,jfyold) ) + \
                            hstep*( three*F(0,jfy) - F(0,jfyold)) ); \
  float  uz = invrho*half*( (one-hstep)*( F(0,jfz) + F(0,jfzold) ) + \
                            hstep*( three*F(0,jfz) - F(0,jfzold)) );

// ---------------------------------------------------------------------------
// Covariant electric field (Ohm's law), fully general for orthogonal
// curvilinear grids. Derived in Curvilinear.pdf; conventions:
//   - E is stored COVARIANT (e_i),   B is CONTRAVARIANT (cb^i = cb + cb0),
//   - u = jf/rho is the CONTRAVARIANT bulk velocity (see advance_p scatter),
//   - the logical-coordinate derivative is P(d)*(F(d,.)-F(md,.)) with
//     P(x)=px, P(y)=py, P(z)=pz (px=0.5*rdx, i.e. 0.5/dref, a LOGICAL deriv),
//   - J = h1 h2 h3, g_ii = h_i^2.
// The macro is written cyclically: E(x,y,z) builds E_x, and the (y,z,x) /
// (z,x,y) rotations build E_y, E_z. Direction-tagged scale factors and the
// P() selector all rotate with the arguments, so it is correct for
// anisotropic reference spacing (rdx!=rdy!=rdz), not just isotropic.
//
// Term by term for E_x (indices 1,2,3 = x_,y_,z_):
//   pressure : -(1/qn) d_1(pe)                         = -invrho*P(x_)*d1(pe)
//   -u x B   : -(u x B)_1 = -J (u^2 B^3 - u^3 B^2)
//   Hall     : +(1/qn)(curlB x B)_1, with (curlB)^j=(1/J)eps^{jab}d_a(h_b^2 B^b);
//              the J cancels ->
//              +invrho*[ ( d3(h1^2 B1) - d1(h3^2 B3) )*B3
//                       -( d1(h2^2 B2) - d2(h1^2 B1) )*B2 ]
//   resistive: +eta*tcay*(curlB)_1, (curlB)_i=g_ii(curlB)^i=(h_i^2/J)*(...)
//              +do_eta*eta*tcay*(h1_0^2/J)*( d2(h3^2 B3) - d3(h2^2 B2) )
//   source   : -invrho*rVt*s_x
// All multiplied by tcaz. On CARTESIAN (h=1,J=1) every term reduces to the
// original Cartesian Ohm's law.
// ---------------------------------------------------------------------------

// P(d): select the logical-derivative coefficient for direction d (x/y/z).
#define P(d) P_##d
#define P_x px
#define P_y py
#define P_z pz

// H(comp, cell): the stored scale factor for component "comp" (x/y/z ->
// h1/h2/h3) evaluated at bare-direction cell "cell" (0, x, mx, y, ...). This
// lets the scale factors rotate with the cyclic macro arguments.
#define HN_x 1
#define HN_y 2
#define HN_z 3
#define HCAT(n,cell) h##n##_##cell
#define HEXP(n,cell) HCAT(n,cell)
#define H(comp,cell) HEXP(HN_##comp, cell)

// HGD(dir): half the logical->physical cell size gd_dir/2 in direction dir
// (x/y/z), for converting contravariant velocity to physical in the motional
// term. Rotates with the cyclic macro arguments.
#define HGD_x hgdx
#define HGD_y hgdy
#define HGD_z hgdz
#define HGD(dir) HGD_##dir

// Bf(cell, comp): contravariant B^comp (incl. external cb0) at bare-direction
// cell.
#define Bf(cell, comp) ( F(cell, cb##comp) + F(cell, cb##comp##0) )

// BL(cell, comp): lowered covariant B_comp = h_comp^2 B^comp at "cell".
#define BL(cell, comp) ( H(comp,cell)*H(comp,cell) * Bf(cell, comp) )

// dBL(a, comp): centered logical derivative d_a( h_comp^2 B^comp ), with a and
// comp direction letters. Uses the +a / -a neighbor cells (a and m##a).
#define dBL(a, comp) ( P(a) * ( BL(a, comp) - BL(m##a, comp) ) )

#define E(x_,y_,z_) \
  F(0,e##x_) =      \
    /* Hall: +invrho*[ (d_z(h_x^2 B_x) - d_x(h_z^2 B_z))*B_z                 \
                      -(d_x(h_y^2 B_y) - d_y(h_x^2 B_x))*B_y ] */            \
    invrho * (                                                              \
      ( dBL(z_, x_) - dBL(x_, z_) ) * Bf(0, z_)                             \
    - ( dBL(x_, y_) - dBL(y_, x_) ) * Bf(0, y_) ) +                         \
    /* Motional -(u x B)_x, covariant. u=jf/rho is CONTRAVARIANT u^i; convert   \
       each component to physical velocity u_phys_i = u^i * H_i, H_i=h_i*gd_i/2 \
       (= H(i,0)*HGD(i)), then take the physical cross product with B (cb).      \
       On CARTESIAN u^i*H_i = u_phys_i and h=1 so this reduces to the working    \
       -u_phys_y*B_z + u_phys_z*B_y. */                                          \
    ( - (u##y_ * H(y_,0)*HGD(y_)) * Bf(0,z_)                                  \
      + (u##z_ * H(z_,0)*HGD(z_)) * Bf(0,y_) ) -                              \
    /* pressure: -(1/qn) d_x(pe) */                                         \
    invrho * P(x_) * ( F(x_,pe) - F(m##x_,pe) ) +                           \
    /* resistive: +do_eta*eta*tcay*(h_x^2/J)*( d_y(h_z^2 B_z)-d_z(h_y^2 B_y) ) */ \
    do_eta*eta*F(0,tcay) * ( (H(x_,0)*H(x_,0))/(h1_0*h2_0*h3_0) ) * (       \
        dBL(y_, z_) - dBL(z_, y_) ) -                                       \
    invrho * rVt * F(0,s##x_); \
  F(0,e##x_) *= F(0,tcaz);

// proposed fix
// #define E(x_,y_,z_) \
//   F(0,e##x_) =      \
//     /* Hall: +invrho*[ (d_z(h_x^2 B_x) - d_x(h_z^2 B_z))*B_z                 \
//                       -(d_x(h_y^2 B_y) - d_y(h_x^2 B_x))*B_y ] */            \
//     invrho * (                                                              \
//       ( dBL(z_, x_) - dBL(x_, z_) ) * Bf(0, z_)                             \
//     - ( dBL(x_, y_) - dBL(y_, x_) ) * Bf(0, y_) ) +                         \
//      /* Motional -(u x B)_x, covariant. u=jf/rho is CONTRAVARIANT u^i; convert   \
//         each component to physical velocity u_phys_i = u^i * H_i, H_i=h_i*gd_i/2 \
//         (= H(i,0)*HGD(i)), then take the physical cross product with B (cb).      \
//         On CARTESIAN u^i*H_i = u_phys_i and h=1 so this reduces to the working    \
//         -u_phys_y*B_z + u_phys_z*B_y. */                                          \
//      ( - (u##y_ * H(y_,0)*HGD(y_)) * (Bf(0,z_) * H(z_,0))                      \
//        + (u##z_ * H(z_,0)*HGD(z_)) * (Bf(0,y_) * H(y_,0)) ) -                  \
//     /* pressure: -(1/qn) d_x(pe) */                                         \
//     invrho * P(x_) * ( F(m##x_,pe) - F(x_,pe) ) +                           \
//     /* resistive: +do_eta*eta*tcay*(h_x^2/J)*( d_y(h_z^2 B_z)-d_z(h_y^2 B_y) ) */ \
//     do_eta*eta*F(0,tcay) * ( (H(x_,0)*H(x_,0))/(h1_0*h2_0*h3_0) ) * (       \
//         dBL(y_, z_) - dBL(z_, y_) ) -                                       \
//     invrho * rVt * F(0,s##x_); \
//   F(0,e##x_) *= F(0,tcaz);
  
  
/*
  #define FIXEDGES()\
  if(ny > 1 || nz > 1) { \
  Kokkos::parallel_for("advance_e_x_pos", x_pos, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(ny > 1 || nz > 1) { \
  Kokkos::parallel_for("advance_e_x_neg", x_neg, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nz > 1 || nx > 1) { \
  Kokkos::parallel_for("advance_e_y_pos", y_pos, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nz > 1 || nx > 1) { \
  Kokkos::parallel_for("advance_e_y_neg", y_neg, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nx > 1 || ny > 1) { \
  Kokkos::parallel_for("advance_e_z_pos", z_pos, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  } \
  if(nx > 1 || ny > 1) { \
  Kokkos::parallel_for("advance_e_z_neg", z_neg, KOKKOS_LAMBDA(const int x, const int y, const int z) {\
    INIT_STENCIL();\
    E(x,y,z);\
    E(y,z,x);\
    E(z,x,y);\
    });\
  }
*/
  
  #define FIXEDGES()\
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> xpolicy({0,1,1}, {2,ny+1,nz+1}); \
    Kokkos::parallel_for("advance_e_x_faces", xpolicy, \
    KOKKOS_LAMBDA(const int face, const int y, const int z) { \
      const int x = face == 0 ? 1 : nx; \
      INIT_STENCIL(); \
      E(x,y,z);\
      E(y,z,x);\
      E(z,x,y);\
    }); \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> ypolicy({1,0,1}, {nx+1,2,nz+1}); \
    Kokkos::parallel_for("advance_e_y_faces", ypolicy, \
    KOKKOS_LAMBDA(const int x, const int face, const int z) { \
      const int y = face == 0 ? 1 : ny; \
      INIT_STENCIL(); \
      E(x,y,z);\
      E(y,z,x);\
      E(z,x,y);\
    }); \
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> zpolicy({1,1,0}, {nz+1,ny+1,2}); \
    Kokkos::parallel_for("advance_e_z_faces", zpolicy, \
    KOKKOS_LAMBDA(const int x, const int y, const int face) { \
      const int z = face == 0 ? 1 : nz; \
      INIT_STENCIL(); \
      E(x,y,z);\
      E(y,z,x);\
      E(z,x,y);\
    }); \



void
hyb_advance_e( field_array_t * RESTRICT fa,
                  float frac ) {
  if( !fa     ) ERROR(( "Bad args" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  k_curvilinear_mesh_t k_curv_mesh = fa->g->k_curvilinear_mesh_d;
  //const material_coefficient_t * ALIGNED(128) m = args->p->mc;
  const grid_t                 *              g = args->g;
  const size_t nx = g->nx, ny = g->ny, nz = g->nz;

  const float px = (nx>1) ? 0.5*g->rdx : 0;
  const float py = (ny>1) ? 0.5*g->rdy : 0;
  const float pz = (nz>1) ? 0.5*g->rdz : 0;
  // Half the logical->physical cell size in each direction, gd_i/2 (needed to
  // convert the contravariant bulk velocity u^i = jf/rho into a physical
  // velocity in the motional -u x B term: u_phys_i = u^i * H_i, H_i=h_i*gd_i/2).
  // Uses g->dx directly (NOT p_i, which is zeroed in singleton directions).
  const float hgdx = 0.5f * g->dx;
  const float hgdy = 0.5f * g->dy;
  const float hgdz = 0.5f * g->dz;
  const float eta = g->eta;
  const float den_floor_ohm = g->den_floor_ohm;
  const float rVt = g->rdx*g->rdy*g->rdz/g->dt;

  const float hstep = abs(frac);
  constexpr float half = 1./2., one = 1., three = 3.;
  constexpr size_t ind2  = 2, ind1 = 1;  
  const float do_eta = (frac>0.) ? 1. : 0.;
  //for interior cells
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1,1,1},{nx+1,ny+1,nz+1});
  
  //for edge faces
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_pos(fa->pos_x_face_space, {nx  ,ind1,ind1},{nx+ind1,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_neg(fa->neg_x_face_space, {ind1,ind1,ind1},{ind2   ,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_pos(fa->pos_y_face_space, {ind1,ny  ,ind1},{nx+ind1,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_neg(fa->neg_y_face_space, {ind1,ind1,ind1},{nx+ind1,ind2   ,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_pos(fa->pos_z_face_space, {ind1,ind1,  nz},{nx+ind1,ny+ind1,nz+ind1});
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_neg(fa->neg_z_face_space, {ind1,ind1,ind1},{nx+ind1,ny+ind1,   ind2});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_pos({nx  ,ind1,ind1},{nx+ind1,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> x_neg({ind1,ind1,ind1},{ind2   ,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_pos({ind1,ny  ,ind1},{nx+ind1,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> y_neg({ind1,ind1,ind1},{nx+ind1,ind2   ,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_pos({ind1,ind1,  nz},{nx+ind1,ny+ind1,nz+ind1});
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> z_neg({ind1,ind1,ind1},{nx+ind1,ny+ind1,   ind2});
  
  
  /***************************************************************************
   * Calculate electron pressure
   ***************************************************************************/
    
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Calculate_Electron_Pressure");
  hyb_epress(fa, frac); // Read te, rhof, rhofold, Write pe
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Begin tangential B ghost setup
   ***************************************************************************/
    
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup");
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::Begin_Remote_Ghost_Hybrid_B");
  k_begin_remote_ghost_hyb_b( fa ); // Read: cbx, cby, cbz
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Calculate electron pressure
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Calculate_Electron_Pressure");
  hyb_epress(fa, frac); // Read te, rhof, rhofold, Write: pe
  Kokkos::Profiling::popRegion();

  //Kokkos::Profiling::pushRegion("HybridAdvanceE::Update_E_Interior::Inner");
  //const int minx = nx > 2 ? 2 : 1;
  //const int miny = ny > 2 ? 2 : 1;
  //const int minz = nz > 2 ? 2 : 1;
  //const int maxx = nx > 2 ? nx : 2;
  //const int maxy = ny > 2 ? ny : 2;
  //const int maxz = nz > 2 ? nz : 2;
  //Kokkos::MDRangePolicy<Kokkos::Rank<3>> inner_zyx_policy(fa->inner_comp_space, {minx, miny, minz}, {maxx, maxy, maxz});
  //// Write: ex,ey,ez 
  //// Read: rhof, rhofold, jfx, jfy, jfz, jfxold, jfyold, jfzold, cbx, cby, cbz, tcax, tcay, tcaz, pe
  //Kokkos::parallel_for("hyb_advance_e_inner", inner_zyx_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
  //  INIT_STENCIL();
  //  E(x,y,z);
  //  E(y,z,x);
  //  E(z,x,y);
  //});
  //Kokkos::Profiling::popRegion();

  /***************************************************************************
   * End tangential B ghost setup
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::End_Remote_Ghost_Hybrid_B");
  k_end_remote_ghost_hyb_b(fa); // Write: cbx, cby, cbz
  Kokkos::Profiling::popRegion();

  /***************************************************************************
   * Apply local hybrid ghost b
   ***************************************************************************/
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Tangential_Ghost_Setup::Hybrid_Local_Ghost_B");
  k_hyb_local_ghost_b( fa, fa->g ); // R/W: cbx, cby, cbz
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
    
  /***************************************************************************
   * Update E fields
   ***************************************************************************/ 
    
  //Compute E. Interior cells correct 
   
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Update_E_Interior");
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_inner_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  // Write: ex,ey,ez 
  // Read: rhof, rhofold, jfx, jfy, jfz, jfxold, jfyold, jfzold, cbx, cby, cbz, tcax, tcay, tcaz, pe
  Kokkos::parallel_for("hyb_advance_e_interior", xyz_inner_policy, KOKKOS_LAMBDA(const int x, const int y, const int z) {
    INIT_STENCIL();
    E(x,y,z);
    E(y,z,x);
    E(z,x,y);
  });
  Kokkos::Profiling::popRegion();

  //Fix edge cells
  
  //k_end_remote_ghost_hyb_b(fa);
  //k_hyb_local_ghost_b( fa, fa->g );
  //FIXEDGES()
  //fa->inner_comp_space.fence();
  //fa->pos_x_face_space.fence();
  //fa->neg_x_face_space.fence();
  //fa->pos_y_face_space.fence();
  //fa->neg_y_face_space.fence();
  //fa->pos_z_face_space.fence();
  //fa->neg_z_face_space.fence();

  //Apply hypereta to E field
   
  Kokkos::Profiling::pushRegion("HybridAdvanceE::Apply_Hyper_Eta");
  // Read: cbx, cby, cbz, tcax, tcay, tcaz, ex, ey, ez
  // Write: pex, pey, pez, ex, ey, ez
  if(fa->g->hypereta>0 && do_eta) hyb_heta(fa);
  Kokkos::Profiling::popRegion();
  Kokkos::fence();
}