// Note: This is similar to vacuum_compute_curl_b

#define IN_sfa
#include "sfa_private.h"

typedef struct pipeline_args {
  /**/  field_t      * ALIGNED(128) f;
  const sfa_params_t *              p;
  const grid_t       *              g;
} pipeline_args_t;

#define F(ind,v) k_field(f##ind##_index, field_var::v)

#define INIT_STENCIL()                                    \
  size_t f0_index  = VOXEL(x,   y,   z,    nx,ny,nz);     \
  size_t fx_index  = VOXEL(x+1, y,   z,    nx,ny,nz);     \
  size_t fy_index  = VOXEL(x,   y+1, z,    nx,ny,nz);     \
  size_t fz_index  = VOXEL(x,   y,   z+1,  nx,ny,nz);     \
  size_t fmx_index = VOXEL(x-1, y,   z,    nx,ny,nz);     \
  size_t fmy_index = VOXEL(x,   y-1, z,    nx,ny,nz);     \
  size_t fmz_index = VOXEL(x,   y,   z-1,  nx,ny,nz);     \
  /* Load curvilinear mesh indices */                      \
  size_t m0_index  = GRID_TO_MESH(x,   y,   z,   nx, ny, nz);  \
  size_t mx_index  = GRID_TO_MESH(x+1, y,   z,   nx, ny, nz);  \
  size_t my_index  = GRID_TO_MESH(x,   y+1, z,   nx, ny, nz);  \
  size_t mz_index  = GRID_TO_MESH(x,   y,   z+1, nx, ny, nz);  \
  size_t mmx_index = GRID_TO_MESH(x-1, y,   z,   nx, ny, nz);  \
  size_t mmy_index = GRID_TO_MESH(x,   y-1, z,   nx, ny, nz);  \
  size_t mmz_index = GRID_TO_MESH(x,   y,   z-1, nx, ny, nz);  \
  /* Load scale factors at all stencil points */           \
  float h1_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_1);  \
  float h2_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_2);  \
  float h3_0  = k_curv_mesh(m0_index,  curv_mesh_var::h_3);  \
  float h1_x  = k_curv_mesh(mx_index,  curv_mesh_var::h_1);  \
  float h2_x  = k_curv_mesh(mx_index,  curv_mesh_var::h_2);  \
  float h3_x  = k_curv_mesh(mx_index,  curv_mesh_var::h_3);  \
  float h1_mx = k_curv_mesh(mmx_index, curv_mesh_var::h_1);  \
  float h2_mx = k_curv_mesh(mmx_index, curv_mesh_var::h_2);  \
  float h3_mx = k_curv_mesh(mmx_index, curv_mesh_var::h_3);  \
  float h1_y  = k_curv_mesh(my_index,  curv_mesh_var::h_1);  \
  float h2_y  = k_curv_mesh(my_index,  curv_mesh_var::h_2);  \
  float h3_y  = k_curv_mesh(my_index,  curv_mesh_var::h_3);  \
  float h1_my = k_curv_mesh(mmy_index, curv_mesh_var::h_1);  \
  float h2_my = k_curv_mesh(mmy_index, curv_mesh_var::h_2);  \
  float h3_my = k_curv_mesh(mmy_index, curv_mesh_var::h_3);  \
  float h1_z  = k_curv_mesh(mz_index,  curv_mesh_var::h_1);  \
  float h2_z  = k_curv_mesh(mz_index,  curv_mesh_var::h_2);  \
  float h3_z  = k_curv_mesh(mz_index,  curv_mesh_var::h_3);  \
  float h1_mz = k_curv_mesh(mmz_index, curv_mesh_var::h_1);  \
  float h2_mz = k_curv_mesh(mmz_index, curv_mesh_var::h_2);  \
  float h3_mz = k_curv_mesh(mmz_index, curv_mesh_var::h_3);  \
  /* Compute Jacobians at all stencil points */             \
  float J_0  = h1_0  * h2_0  * h3_0;                        \
  float J_x  = h1_x  * h2_x  * h3_x;                        \
  float J_mx = h1_mx * h2_mx * h3_mx;                       \
  float J_y  = h1_y  * h2_y  * h3_y;                        \
  float J_my = h1_my * h2_my * h3_my;                       \
  float J_z  = h1_z  * h2_z  * h3_z;                        \
  float J_mz = h1_mz * h2_mz * h3_mz;

// Laplacian in curvilinear coordinates for orthogonal grids:
// laplacianB_x = (1/J) [ d_xi(J/h_xi^2 d_xi B_x) + d_eta(J/h_eta^2 d_eta B_x) + d_mu(J/h_mu^2 d_mu B_x) ]
// Using centered finite differences with logical derivatives (px, py, pz)
// For the xi direction: d_xi(J/h_xi^2 d_xi B_x) = 
//   px * [ (J_x/h1_x^2)*(B_x - B_0) - (J_0/h1_0^2)*(B_0 - B_mx) ]
// And similarly for eta and mu directions
#define LPL_B()                                                             \
  {                                                                         \
    /* xi direction contribution to laplacianB_x */                                \
    float dxi_term_x = px * (                                               \
      (J_x / (h1_x * h1_x)) * (F(x, cbx) - F(0, cbx)) -                    \
      (J_0 / (h1_0 * h1_0)) * (F(0, cbx) - F(mx, cbx))                     \
    );                                                                      \
    /* eta direction contribution */                                          \
    float deta_term_x = py * (                                              \
      (J_y / (h2_y * h2_y)) * (F(y, cbx) - F(0, cbx)) -                    \
      (J_0 / (h2_0 * h2_0)) * (F(0, cbx) - F(my, cbx))                     \
    );                                                                      \
    /* mu direction contribution */                                          \
    float dmu_term_x = pz * (                                               \
      (J_z / (h3_z * h3_z)) * (F(z, cbx) - F(0, cbx)) -                    \
      (J_0 / (h3_0 * h3_0)) * (F(0, cbx) - F(mz, cbx))                     \
    );                                                                      \
    F(0, pex) = (dxi_term_x + deta_term_x + dmu_term_x) / J_0;             \
                                                                            \
    /* xi direction contribution to laplacianB_y */                                \
    float dxi_term_y = px * (                                               \
      (J_x / (h1_x * h1_x)) * (F(x, cby) - F(0, cby)) -                    \
      (J_0 / (h1_0 * h1_0)) * (F(0, cby) - F(mx, cby))                     \
    );                                                                      \
    /* eta direction contribution */                                          \
    float deta_term_y = py * (                                              \
      (J_y / (h2_y * h2_y)) * (F(y, cby) - F(0, cby)) -                    \
      (J_0 / (h2_0 * h2_0)) * (F(0, cby) - F(my, cby))                     \
    );                                                                      \
    /* mu direction contribution */                                          \
    float dmu_term_y = pz * (                                               \
      (J_z / (h3_z * h3_z)) * (F(z, cby) - F(0, cby)) -                    \
      (J_0 / (h3_0 * h3_0)) * (F(0, cby) - F(mz, cby))                     \
    );                                                                      \
    F(0, pey) = (dxi_term_y + deta_term_y + dmu_term_y) / J_0;             \
                                                                            \
    /* xi direction contribution to laplacianB_z */                                \
    float dxi_term_z = px * (                                               \
      (J_x / (h1_x * h1_x)) * (F(x, cbz) - F(0, cbz)) -                    \
      (J_0 / (h1_0 * h1_0)) * (F(0, cbz) - F(mx, cbz))                     \
    );                                                                      \
    /* eta direction contribution */                                          \
    float deta_term_z = py * (                                              \
      (J_y / (h2_y * h2_y)) * (F(y, cbz) - F(0, cbz)) -                    \
      (J_0 / (h2_0 * h2_0)) * (F(0, cbz) - F(my, cbz))                     \
    );                                                                      \
    /* mu direction contribution */                                          \
    float dmu_term_z = pz * (                                               \
      (J_z / (h3_z * h3_z)) * (F(z, cbz) - F(0, cbz)) -                    \
      (J_0 / (h3_0 * h3_0)) * (F(0, cbz) - F(mz, cbz))                     \
    );                                                                      \
    F(0, pez) = (dxi_term_z + deta_term_z + dmu_term_z) / J_0;             \
  }

// Curl in curvilinear coordinates for orthogonal grids:
// (curlV)^i = (1/J) eps^ijk d_j(h_k V_k)
// For the x-component (i=1, cyclic in j,k over 2,3):
// (curllaplacianB)^x = (1/J) [d_eta(h_mu laplacianB_mu) - d_mu(h_eta laplacianB_eta)]
// Using centered differences:
#define CURL_LPL_B(x_,y_,z_)                                                \
  {                                                                         \
    /* Load scale factors at y+- and z+- neighbors */                        \
    float h_y_p = (x_ == x) ? h2_y : ((x_ == y) ? h3_y : h1_y);            \
    float h_y_m = (x_ == x) ? h2_my : ((x_ == y) ? h3_my : h1_my);         \
    float h_z_p = (x_ == x) ? h3_z : ((x_ == y) ? h1_z : h2_z);            \
    float h_z_m = (x_ == x) ? h3_mz : ((x_ == y) ? h1_mz : h2_mz);         \
                                                                            \
    /* Compute h*laplacianB at neighboring points */                              \
    float h_pe_yp = h_y_p * F(y_, pe##z_);                                 \
    float h_pe_ym = h_y_m * F(m##y_, pe##z_);                              \
    float h_pe_zp = h_z_p * F(z_, pe##y_);                                 \
    float h_pe_zm = h_z_m * F(m##z_, pe##y_);                              \
                                                                            \
    /* Compute derivatives */                                              \
    float d_eta_term, d_mu_term;                                           \
    if (x_ == x) {                                                          \
      d_eta_term = py * (h_pe_yp - h_pe_ym);                               \
      d_mu_term  = pz * (h_pe_zp - h_pe_zm);                               \
    } else if (x_ == y) {                                                   \
      d_eta_term = pz * (h_pe_yp - h_pe_ym);                               \
      d_mu_term  = px * (h_pe_zp - h_pe_zm);                               \
    } else { /* x_ == z */                                                  \
      d_eta_term = px * (h_pe_yp - h_pe_ym);                               \
      d_mu_term  = py * (h_pe_zp - h_pe_zm);                               \
    }                                                                       \
                                                                            \
    /* Apply curl and accumulate to E field */                             \
    F(0, e##x_) -= hypereta * F(0, tcax) * F(0, tcaz) *                    \
                   (d_eta_term - d_mu_term) / J_0;                          \
  }

void
hyb_heta( field_array_t * RESTRICT fa ) {
  if( !fa     ) ERROR(( "Bad args" ));

  pipeline_args_t args[1];
  args->f = fa->f;
  args->p = (sfa_params_t *)fa->params;
  args->g = fa->g;
  k_field_t k_field = fa->k_f_d;
  k_curvilinear_mesh_t k_curv_mesh = fa->g->k_curvilinear_mesh_d;
  const grid_t *g = args->g;
  const int nx = g->nx, ny = g->ny, nz = g->nz;

  const float px = (nx>1) ? 0.5*g->rdx : 0;
  const float py = (ny>1) ? 0.5*g->rdy : 0;
  const float pz = (nz>1) ? 0.5*g->rdz : 0;
  const float hypereta = g->hypereta;

  // Laplace B Loop
    
  // Write: pex, pey, pez
  // Read: cbx, cby, cbz, scale factors from k_curv_mesh
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> xyz_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  Kokkos::parallel_for("hyb_hypereta_lpl_b", xyz_policy, 
    KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();
      LPL_B();
    });
    
  // Operations on the ghost cells
  k_begin_remote_ghost_hyb_curl_lpl_b(fa); // Read: pex, pey, pez
  k_end_remote_ghost_hyb_curl_lpl_b(fa);   // Write: pex, pey, pez
  k_hyb_local_ghost_lapl_b(fa, fa->g);     // R/W: pex, pey, pez

  // Curl Laplace B Loop

  Kokkos::MDRangePolicy<Kokkos::Rank<3>> curl_policy({1, 1, 1}, {nx+1, ny+1, nz+1});
  // Read: tcax, tcaz, pex, pey, pez, scale factors from k_curv_mesh
  // Write: ex, ey, ez
  Kokkos::parallel_for("hyb_hypereta_curl_lpl_b", curl_policy, 
    KOKKOS_LAMBDA(const int x, const int y, const int z) {
      INIT_STENCIL();
      CURL_LPL_B(x,y,z);
      CURL_LPL_B(y,z,x);
      CURL_LPL_B(z,x,y); 
    });
}