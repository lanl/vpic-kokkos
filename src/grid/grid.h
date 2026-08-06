/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#ifndef _grid_h_
#define _grid_h_

#include "../util/util.h"
#include "../vpic/kokkos_helpers.h"

#define BOUNDARY(i,j,k) (13+(i)+3*(j)+9*(k)) /* FORTRAN -1:1,-1:1,-1:1 */

enum grid_enums {

  // Phase 2 boundary conditions
  anti_symmetric_fields = -1, // E_tang = 0
  pec_fields            = -1,
  metal_fields          = -1,
  symmetric_fields      = -2, // B_tang = 0, B_norm = 0
  pmc_fields            = -3, // B_tang = 0, B_norm floats
  absorb_fields         = -4, // Gamma = 0
  cylindrical_axis_fields = -5, // R=0 axis: R- and theta- vector components flip
                                // sign across the boundary (theta -> theta+pi),
                                // z-components and scalars unchanged. For the
                                // hybrid solver's cylindrical inner (R) boundary.
  spherical_axis_fields   = -6, // theta=0/pi polar axis: theta- and phi- vector
                                // components flip sign across the boundary
                                // (phi -> phi+pi), r-components and scalars
                                // unchanged. Set on the theta (y) faces.
  spherical_center_fields = -7, // r=0 center (point): r- and phi- vector
                                // components flip sign across the boundary
                                // (antipode theta->pi-theta, phi->phi+pi),
                                // theta-components and scalars unchanged. Set
                                // on the inner-r (x) face.

  // Phase 3 boundary conditions
  reflect_particles = -1, // Cell boundary should reflect particles
  absorb_particles  = -2,  // Cell boundary should absorb particles
  cylindrical_axis_particles     = -3,
  spherical_axis_particles       = -4, // spherical polar axis (theta faces)
  spherical_center_particles     = -5  // spherical center r=0 (inner-r face)


  // Symmetry in the field boundary conditions refers to image charge
  // sign
  //
  // Anti-symmetric -> Image charges are opposite signed (ideal metal)
  //                   Boundary rho/j are accumulated over partial voxel+image
  // Symmetric      -> Image charges are same signed (symmetry plane or pmc)
  //                   Boundary rho/j are accumulated over partial voxel+image
  // Absorbing      -> No image charges
  //                   Boundary rho/j are accumulated over partial voxel only
  //
  // rho     -> Anti-symmetric      | rho     -> Symmetric
  // jf_tang -> Anti-symmetric      | jf_tang -> Symmetric
  // E_tang  -> Anti-symmetric      | E_tang  -> Symmetric
  // B_norm  -> Anti-symmetric + DC | B_norm  -> Symmetric      (see note)
  // B_tang  -> Symmetric           | B_tang  -> Anti-symmetric
  // E_norm  -> Symmetric           | E_norm  -> Anti-symmetric (see note)
  // div B   -> Symmetric           | div B   -> Anti-symmetric
  //
  // Note: B_norm is tricky. For a symmetry plane, B_norm on the
  // boundary must be zero as there are no magnetic charges (a
  // non-zero B_norm would imply an infinitesimal layer of magnetic
  // charge). However, if a symmetric boundary is interpreted as a
  // perfect magnetic conductor, B_norm could be present due to
  // magnetic conduction surface charges. Even though there are no
  // bulk volumetric magnetic charges to induce a surface magnetic
  // charge, I think that radiation/waveguide modes/etc could (the
  // total surface magnetic charge in the simulation would be zero
  // though). As a result, symmetric and pmc boundary conditions are
  // treated separately. Symmetric and pmc boundaries are identical
  // except the symmetric boundaries explicitly zero boundary
  // B_norm. Note: anti-symmetric and pec boundary conditions would
  // have the same issue if norm E was located directly on the
  // boundary. However, it is not so this problem does not arise.
  //
  // Note: Absorbing boundary conditions make no effort to clean
  // divergence errors on them. They assume that the ghost div b is
  // zero and force the surface div e on them to be zero. This means
  // ghost norm e can be set to any value on absorbing boundaries.

};

enum grid_type {
  CARTESIAN = 0,
  CYLINDRICAL = 1,
  SPHERICAL = 2,
  GENERAL = 3,
  STRETCHED_CARTESIAN = 4
};


// Given a voxel mesh coordinates (on 0:nx+1,0:ny+1,0:nz+1) and
// voxel mesh resolution (nx,ny,nz), return the index of that voxel.

#define VOXEL(x,y,z, nx,ny,nz) ((x) + ((nx)+2)*((y) + ((ny)+2)*(z)))

// Convert voxel index back to (i,j,k) indices for the local grid
#define UNVOXEL(v, i, j, k, nx, ny, nz) \
  do { \
    int _stride_y = (nx) + 2; \
    int _stride_z = _stride_y * ((ny) + 2); \
    (k) = (v) / _stride_z; \
    int _rem = (v) % _stride_z; \
    (j) = _rem / _stride_y; \
    (i) = _rem % _stride_y; \
  } while(0)

// Convert grid voxel index to curvilinear mesh index
// Grid has 1 ghost layer: (nx+2) x (ny+2) x (nz+2)
// Mesh has 2 ghost layers: (nx+4) x (ny+4) x (nz+4)
// Mesh cell indices are offset by +1 in each dimension
#define VOXEL_TO_MESH(v, nx, ny, nz) \
  (((v) % ((nx)+2) + 1) + \
   ((nx)+4) * ((((v) / ((nx)+2)) % ((ny)+2) + 1) + \
   ((ny)+4) * ((v) / (((nx)+2) * ((ny)+2)) + 1)))

// Convert grid cell indices (i,j,k) to curvilinear mesh linear index
// Grid: (nx+2) x (ny+2) x (nz+2) with 1 ghost layer
// Mesh: (nx+4) x (ny+4) x (nz+4) with 2 ghost layers
// Mesh indices are shifted by +1 in each dimension
#define GRID_TO_MESH(i, j, k, nx, ny, nz) \
  VOXEL((i)+1, (j)+1, (k)+1, (nx)+2, (ny)+2, (nz)+2)

typedef struct grid {

  // System of units
  float dt, cvac, eps0;

  grid_type type = grid_type::GENERAL;

  // Time stepper.  The simulation time is given by
  // t = g->t0 + (double)g->dt*(double)g->step
  int64_t step;             // Current timestep
  double t0;                // Simulation time corresponding to step 0

  // Phase 2 grid data structures
  float x0, y0, z0;         // Min corner local domain (must be coherent)
  float x1, y1, z1;         // Max corner local domain (must be coherent)
  int   nx, ny, nz;         // Local voxel mesh resolution.  Voxels are
                            // indexed FORTRAN style 0:nx+1,0:ny+1,0:nz+1
                            // with voxels 1:nx,1:ny,1:nz being non-ghost
                            // voxels.
  float dx, dy, dz, dV;     // Cell dimensions and volume (CONVENIENCE ...
                            // USE x0,x1 WHEN DECIDING WHICH NODE TO USE!)
  float rdx, rdy, rdz, r8V; // Inverse voxel dimensions and one over
                            // eight times the voxel volume (CONVENIENCE)
  int   sx, sy, sz, nv;     // Voxel indexing x-, y-,z- strides and the
                            // number of local voxels (including ghosts,
                            // (nx+2)(ny+2)(nz+2)), (CONVENIENCE)
  float eta, kappa;         // For J and Te diffusion
  float hypereta;           //
  float nsub, isub;         // subcycling
  int nsm;                  // smoothing for moments
  int nsmb;                 // smooth B fields every nsmb steps

  float den_floor_ohm;    // Density floor for Ohm's law update
  float den_floor_pe;     // Density floor for electron pressure update
  float eos_gamma, eos_den; // Electron fluid adiabatic index, reference density
  float eos_gamma_0;        //Adiabatic index for initializing profiles. 
  
  float   time_sec, length_m, mass_kg, density_SI; //For converting to physical units
  
  int   bc[27];             // (-1:1,-1:1,-1:1) FORTRAN indexed array of
                            // boundary conditions to apply at domain edge
                            // 0 ... nproc-1 ... comm boundary condition
                            // <0 ... locally applied boundary condition

  int gpx = -1, gpy = -1, gpz = -1;   // Store global processor decomposition to let us figure
                            // out where we are in the global decomposition
  double gx0, gy0, gz0;  // Global domain min
  double gx1, gy1, gz1;  // Global domain max
  int gnx, gny, gnz;
  
  // Phase 3 grid data structures
  // NOTE: VOXEL INDEXING LIMITS NUMBER OF VOXELS TO 2^31 (INCLUDING
  // GHOSTS) PER NODE.  NEIGHBOR INDEXING FURTHER LIMITS TO
  // (2^31)/6.  BOUNDARY CONDITION HANDLING LIMITS TO 2^28 PER NODE
  // EMITTER COMPONENT ID INDEXING FURTHER LIMITS TO 2^26 PER NODE.
  // THE LIMIT IS 2^63 OVER ALL NODES THOUGH.
  int64_t * ALIGNED(16) range;
                          // (0:nproc) indexed array giving range of
                          // global indexes of voxel owned by each
                          // processor.  Replicated on each processor.
                          // (range[rank]:range[rank+1]-1) are global
                          // voxels owned by processor "rank".  Note:
                          // range[rank+1]-range[rank] <~ 2^31 / 6

  int64_t * ALIGNED(128) neighbor;
                          // (0:5,0:local_num_voxel-1) FORTRAN indexed
                          // array neighbor(0:5,lidx) are the global
                          // indexes of neighboring voxels of the
                          // voxel with local index "lidx".  Negative
                          // if neighbor is a boundary condition.

  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);
  //

  int64_t rangel, rangeh; // Redundant for move_p performance reasons:
                          //   rangel = range[rank]
                          //   rangeh = range[rank+1]-1.
                          // Note: rangeh-rangel <~ 2^26

  // Nearest neighbor communications ports
  mp_t * mp;
  mp_t* mp_k;
  //  mp_kokkos_t* mp_k;
  //    int max_ports;
  //    k_mpi_t k_mpi_d;
  //    k_mpi_t::HostMirror k_mpi_h;

  k_neighbor_t k_neighbor_d;                // kokkos neighbor view on device
  k_neighbor_t::HostMirror k_neighbor_h;    // kokkos neighbor view on host

  k_curvilinear_mesh_t k_curvilinear_mesh_d; // kokkos view for curvilinear mesh quantities on device
  k_curvilinear_mesh_t::HostMirror k_curvilinear_mesh_h; // kokkos view for curvilinear mesh quantities on host

  // We want to call this *only* once the neighbor is done
  void init_kokkos_grid(int num_neighbor)
  {
      k_neighbor_d = k_neighbor_t("k_neighbor_d", num_neighbor);
      //k_neighbor_h = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), k_neighbor_d);
      k_neighbor_h = Kokkos::create_mirror_view(k_neighbor_d);

      Kokkos::parallel_for("Copy neighbors to host+device",
              Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,
                  num_neighbor), KOKKOS_CLASS_LAMBDA (const int i)
      {
          k_neighbor_h(i) = neighbor[i];
      });

      Kokkos::deep_copy(k_neighbor_d, k_neighbor_h);

      //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      //k_neighbor_h(neighbor, num_neighbor);

      // Copy data over
      // currently implied by unmanaged view

      //k_neighbor_d = Kokkos::create_mirror_view(k_neighbor_d);

      //        max_ports = 27;
      //      k_mpi_d = k_mpi_t("k_mpi_d");
      //      k_mpi_h = Kokkos::create_mirror_view(k_mpi_d);
  }

  using host_execution_policy_md = Kokkos::MDRangePolicy<
    Kokkos::DefaultHostExecutionSpace,
    Kokkos::Rank<3>,
    static_sched,
    Kokkos::IndexType<int>
  >;

  //Initiates the Curvilinear grid components to a default uniform Cartesian grid format.
  void init_cartesian_grid()
  {
    // Per-rank LOCAL curvilinear mesh (sized with local nx,ny,nz), to match how
    // advance_p/local_to_global_cart index it via GRID_TO_MESH. gnx,gny,gnz are
    // GLOBAL; nx=gnx/gpx etc are local.
    const int ghost_layers_per_side = 2;
    const int nx_total = nx + 2 * ghost_layers_per_side;
    const int ny_total = ny + 2 * ghost_layers_per_side;
    const int nz_total = nz + 2 * ghost_layers_per_side;

    //printf("nv=%d",nv);
    const int nv_cm = nx_total * ny_total * nz_total;
    k_curvilinear_mesh_d = k_curvilinear_mesh_t("k_curvilinear_mesh_d", nv_cm);
    k_curvilinear_mesh_h = Kokkos::create_mirror_view(k_curvilinear_mesh_d);

    // Local (per-rank) uniform cell size. x0,x1 are this rank's bounds and
    // nx is the local cell count, so (x1-x0)/nx is the physical cell size.
    const double dx = (x1 - x0) / nx;
    const double dy = (y1 - y0) / ny;
    const double dz = (z1 - z0) / nz;

    Kokkos::parallel_for(
    "Fill curvilinear mesh view",
    host_execution_policy_md({0, 0, 0}, {nx_total, ny_total, nz_total}),
    KOKKOS_CLASS_LAMBDA (const int i, const int j, const int k) {
      const int idx = i + j * nx_total + k * nx_total * ny_total;

      // Physical position from this rank's local origin using the local mesh
      // index (i - ghost). Ghost cells extrapolate beyond [x0,x1] onto the
      // neighbor rank's interior, which is correct for a smooth global map.
      double x_i = x0 + (i - ghost_layers_per_side + 0.5) * dx;
      double y_j = y0 + (j - ghost_layers_per_side + 0.5) * dy;
      double z_k = z0 + (k - ghost_layers_per_side + 0.5) * dz;

      double x = 0 + x_i;
      double y = 0 + y_j;
      double z = 0 + z_k;

      k_curvilinear_mesh_h(idx, curv_mesh_var::h_1)  = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::h_2)  = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::h_3)  = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::jac) = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_1_u) = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_1_v) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_1_w) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_2_u) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_2_v) = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_2_w) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_3_u) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_3_v) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_3_w) = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::xg)  = x;
      k_curvilinear_mesh_h(idx, curv_mesh_var::yg)  = y;
      k_curvilinear_mesh_h(idx, curv_mesh_var::zg)  = z;
    
    });
    
    type = grid_type::CARTESIAN;
    Kokkos::deep_copy(k_curvilinear_mesh_d, k_curvilinear_mesh_h);
  }

  void init_stretched_cartesian_grid(double beta_x = 1.0, double beta_y = 1.0, double beta_z = 1.0)
{
    // The curvilinear mesh is a PER-RANK LOCAL array, sized with the local cell
    // counts (nx,ny,nz), because advance_p/local_to_global_cart index it with
    // GRID_TO_MESH using the local nx,ny,nz. Physical positions are still made
    // globally consistent via this rank's global index offset. gnx,gny,gnz are
    // the GLOBAL counts; nx=gnx/gpx etc are local.
    const int ghost_layers_per_side = 2;
    const int nx_total = nx + 2 * ghost_layers_per_side;
    const int ny_total = ny + 2 * ghost_layers_per_side;
    const int nz_total = nz + 2 * ghost_layers_per_side;

    const int nv_cm = nx_total * ny_total * nz_total;
    k_curvilinear_mesh_d = k_curvilinear_mesh_t("k_curvilinear_mesh_d", nv_cm);
    k_curvilinear_mesh_h = Kokkos::create_mirror_view(k_curvilinear_mesh_d);

    // Total global grid dimensions (local cells * processor count)
    const int global_nx = nx * gpx;
    const int global_ny = ny * gpy;
    const int global_nz = nz * gpz;

    // Uniform grid spacing (computational space)
    const double dxi = 1.0 / global_nx;
    const double deta = 1.0 / global_ny;
    const double dzeta = 1.0 / global_nz;

    // Compute this rank's low end 3D position in the processor grid
    const int rank_i = world_rank % gpx;
    const int rank_j = (world_rank / gpx) % gpy;
    const int rank_k = world_rank / (gpx * gpy);

    // Compute the low end 3D global index for this rank including ghost cells
    int global_i_base = nx * rank_i - ghost_layers_per_side;
    int global_j_base = ny * rank_j - ghost_layers_per_side;
    int global_k_base = nz * rank_k - ghost_layers_per_side;

    Kokkos::parallel_for(
    "Fill stretched Cartesian mesh view",
    host_execution_policy_md({0, 0, 0}, {nx_total, ny_total, nz_total}),
    KOKKOS_CLASS_LAMBDA (const int i, const int j, const int k) {
      const int idx = i + j * nx_total + k * nx_total * ny_total;

      // Compute global indices for this cell
      int global_i_cell = global_i_base + i;
      int global_j_cell = global_j_base + j;
      int global_k_cell = global_k_base + k;

      // Computational coordinates (uniform [0,1])
      double xi = (global_i_cell + 0.5) * dxi;
      double eta = (global_j_cell + 0.5) * deta;
      double zeta = (global_k_cell + 0.5) * dzeta;

      // Apply stretching transformation: tanh-based stretching
      double x_stretched, y_stretched, z_stretched;
      double dx_dxi, dy_deta, dz_dzeta;

      if (beta_x > 1e-10) {
        x_stretched = Kokkos::tanh(beta_x * (xi - 0.5)) / Kokkos::tanh(beta_x * 0.5);
        dx_dxi = beta_x / (Kokkos::tanh(beta_x * 0.5) * 
                 Kokkos::pow(Kokkos::cosh(beta_x * (xi - 0.5)), 2));
      } else {
        x_stretched = 2.0 * xi - 1.0;
        dx_dxi = 2.0;
      }

      if (beta_y > 1e-10) {
        y_stretched = Kokkos::tanh(beta_y * (eta - 0.5)) / Kokkos::tanh(beta_y * 0.5);
        dy_deta = beta_y / (Kokkos::tanh(beta_y * 0.5) * 
                  Kokkos::pow(Kokkos::cosh(beta_y * (eta - 0.5)), 2));
      } else {
        y_stretched = 2.0 * eta - 1.0;
        dy_deta = 2.0;
      }

      if (beta_z > 1e-10) {
        z_stretched = Kokkos::tanh(beta_z * (zeta - 0.5)) / Kokkos::tanh(beta_z * 0.5);
        dz_dzeta = beta_z / (Kokkos::tanh(beta_z * 0.5) * 
                   Kokkos::pow(Kokkos::cosh(beta_z * (zeta - 0.5)), 2));
      } else {
        z_stretched = 2.0 * zeta - 1.0;
        dz_dzeta = 2.0;
      }

      // Map stretched coordinate (range -1..1) to the GLOBAL physical domain
      // [gx0,gx1]. The stretch and xi are defined globally, so the physical
      // position must use the global bounds, not this rank's local x0/x1.
      double x = gx0 + (gx1 - gx0) * (x_stretched + 1.0) * 0.5;
      double y = gy0 + (gy1 - gy0) * (y_stretched + 1.0) * 0.5;
      double z = gz0 + (gz1 - gz0) * (z_stretched + 1.0) * 0.5;

      // Scale factors: dimensionless ratio of local to base (uniform) cell
      // size, so h==1 is uniform (matches CARTESIAN convention). dx_dxi is the
      // global derivative of the stretch map; 0.5*dx_dxi = 1 for beta=0.
      double h1 = 0.5 * dx_dxi;
      double h2 = 0.5 * dy_deta;
      double h3 = 0.5 * dz_dzeta;

      // Scale factors (metric coefficients)
      k_curvilinear_mesh_h(idx, curv_mesh_var::h_1) = h1;
      k_curvilinear_mesh_h(idx, curv_mesh_var::h_2) = h2;
      k_curvilinear_mesh_h(idx, curv_mesh_var::h_3) = h3;

      // Jacobian (volume element)
      k_curvilinear_mesh_h(idx, curv_mesh_var::jac) = h1 * h2 * h3;

      // Basis vectors (constant - Cartesian aligned)
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_1_u) = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_1_v) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_1_w) = 0.0;

      k_curvilinear_mesh_h(idx, curv_mesh_var::e_2_u) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_2_v) = 1.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_2_w) = 0.0;

      k_curvilinear_mesh_h(idx, curv_mesh_var::e_3_u) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_3_v) = 0.0;
      k_curvilinear_mesh_h(idx, curv_mesh_var::e_3_w) = 1.0;

      // Physical coordinates
      k_curvilinear_mesh_h(idx, curv_mesh_var::xg) = x;
      k_curvilinear_mesh_h(idx, curv_mesh_var::yg) = y;
      k_curvilinear_mesh_h(idx, curv_mesh_var::zg) = z;
    });
    
    type = grid_type::STRETCHED_CARTESIAN;
    Kokkos::deep_copy(k_curvilinear_mesh_d, k_curvilinear_mesh_h);
}

  void init_curvilinear_grid()
  {
    init_cartesian_grid();
  }

  // This function utilizes an existing grid for nx,ny,nz.
  // It does not use the Length, Width, or Height of the grid.
  // Nor does it use the defined grid cells in the grid.
  void init_cylindrical_grid()
  {
    // Per-rank LOCAL mesh (sized with local nx,ny,nz) to match GRID_TO_MESH
    // indexing in advance_p / local_to_global_cart. Positions use this rank's
    // local bounds x0,x1 and the local mesh index, matching the CYLINDRICAL
    // branch of local_to_global_cart exactly.
    const int ghost_layers_per_side = 2;
    const int nx_total = nx + 2 * ghost_layers_per_side;
    const int ny_total = ny + 2 * ghost_layers_per_side;
    const int nz_total = nz + 2 * ghost_layers_per_side;
    //printf("nv=%d",nv);
    const int nv_cm = nx_total * ny_total * nz_total;
    k_curvilinear_mesh_d = k_curvilinear_mesh_t("k_curvilinear_mesh_d", nv_cm);
    k_curvilinear_mesh_h = Kokkos::create_mirror_view(k_curvilinear_mesh_d);

    // Local (per-rank) cell spacing in each curvilinear direction.
    const double dr = (x1 - x0) / nx;
    const double dtheta = (y1 - y0) / ny;
    const double dz_cyl = (z1 - z0) / nz;

    Kokkos::parallel_for(
    "Fill curvilinear mesh view",
    host_execution_policy_md({0, 0, 0}, {nx_total, ny_total, nz_total}),
    KOKKOS_CLASS_LAMBDA (const int i, const int j, const int k) {
      const int idx = i + j * nx_total + k * nx_total * ny_total;

      // Physical (r,theta,z) from this rank's local origin and the local mesh
      // index (i - ghost). Matches local_to_global_cart's CYLINDRICAL branch:
      // r = x0 + (voxel - 0.5)*dr, with voxel = i - ghost + 1 => (i-ghost+0.5).
      double r_i = x0 + (i - ghost_layers_per_side + 0.5) * dr;
      double theta_j = y0 + (j - ghost_layers_per_side + 0.5) * dtheta;
      double z_k = z0 + (k - ghost_layers_per_side + 0.5) * dz_cyl;

      if (r_i < 0.0) {
          r_i = -r_i;              // Reflect radius
          theta_j = theta_j + M_PI; // Rotate by 180
      }

      double cos_theta = Kokkos::cos(theta_j);
      double sin_theta = Kokkos::sin(theta_j);

      double x = 0 + r_i * cos_theta;
      double y = 0 + r_i * sin_theta;
      double z = 0 + z_k;

      k_curvilinear_mesh_h(idx,curv_mesh_var::h_1) = 1.0;
      k_curvilinear_mesh_h(idx,curv_mesh_var::h_2) = r_i;
      k_curvilinear_mesh_h(idx,curv_mesh_var::h_3) = 1.0;

      k_curvilinear_mesh_h(idx,curv_mesh_var::jac) = r_i;

      k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_u) = cos_theta;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_v) = sin_theta;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_w) = 0.0;

      k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_u) = -sin_theta;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_v) = cos_theta;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_w) = 0.0;

      k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_u) = 0.0;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_v) = 0.0;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_w) = 1.0;

      k_curvilinear_mesh_h(idx,curv_mesh_var::xg) = x;
      k_curvilinear_mesh_h(idx,curv_mesh_var::yg) = y;
      k_curvilinear_mesh_h(idx,curv_mesh_var::zg) = z;
    }
    );
    type = grid_type::CYLINDRICAL;
    Kokkos::deep_copy(k_curvilinear_mesh_d, k_curvilinear_mesh_h);
  }

  // This function utilizes an existing grid for nx,ny,nz.
  // It does not use the Length, Width, or Height of the grid.
  // Nor does it use the defined grid cells in the grid.
  void init_spherical_grid()
  {
    // Per-rank LOCAL mesh (sized with local nx,ny,nz) to match GRID_TO_MESH
    // indexing. Positions use this rank's local bounds and the local mesh index.
    const int ghost_layers_per_side = 2;
    const int nx_total = nx + 2 * ghost_layers_per_side;
    const int ny_total = ny + 2 * ghost_layers_per_side;
    const int nz_total = nz + 2 * ghost_layers_per_side;
    //printf("nv=%d",nv);
    const int nv_cm = nx_total * ny_total * nz_total;
    k_curvilinear_mesh_d = k_curvilinear_mesh_t("k_curvilinear_mesh_d", nv_cm);
    k_curvilinear_mesh_h = Kokkos::create_mirror_view(k_curvilinear_mesh_d);

    // Local (per-rank) cell spacing in each curvilinear direction.
    const double dr = (x1 - x0) / nx;
    const double dtheta = (y1 - y0) / ny;
    const double dphi = (z1 - z0) / nz;

    Kokkos::parallel_for(
    "Fill curvilinear mesh view",
    host_execution_policy_md({0, 0, 0}, {nx_total, ny_total, nz_total}),
    KOKKOS_CLASS_LAMBDA (const int i, const int j, const int k) {
      const int idx = i + j * nx_total + k * nx_total * ny_total;

      // Physical (r,theta,phi) from local origin and local mesh index (i-ghost).
      double r_i = x0 + (i - ghost_layers_per_side + 0.5) * dr;
      double theta_j = y0 + (j - ghost_layers_per_side + 0.5) * dtheta;
      double phi_k = z0 + (k - ghost_layers_per_side + 0.5) * dphi;

      if (r_i < 0.0) {
          r_i = -r_i;
          theta_j = theta_j + M_PI;
      }
      // Polar-axis reflection for theta ghost cells: crossing theta=0 or theta=pi
      // maps (theta,phi)->(-theta or 2pi-theta, phi+pi). This keeps sin(theta)>=0
      // so the stored h_3 = r sin(theta) and jac = r^2 sin(theta) are non-negative
      // in the ghosts (a negative jac there flips the sign of the field-solve
      // curl/inv_J and drives an instability).
      if (theta_j < 0.0)   { theta_j = -theta_j;            phi_k += M_PI; }
      if (theta_j > M_PI)  { theta_j = 2.0*M_PI - theta_j;  phi_k += M_PI; }

      double cos_theta = Kokkos::cos(theta_j);
      double sin_theta = Kokkos::sin(theta_j);
      double cos_phi = Kokkos::cos(phi_k);
      double sin_phi = Kokkos::sin(phi_k);

      double x = 0 + r_i * sin_theta * cos_phi;
      double y = 0 + r_i * sin_theta * sin_phi;
      double z = 0 + r_i * cos_theta;

      k_curvilinear_mesh_h(idx,curv_mesh_var::h_1) = 1.0;              // h_r
      k_curvilinear_mesh_h(idx,curv_mesh_var::h_2) = r_i;              // h_theta
      k_curvilinear_mesh_h(idx,curv_mesh_var::h_3) = r_i * sin_theta;  // h_phi

      // Jacobian
      k_curvilinear_mesh_h(idx,curv_mesh_var::jac) = r_i * r_i * sin_theta;

      // Basis vectors in Cartesian components
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_u) = sin_theta * cos_phi;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_v) = sin_theta * sin_phi;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_w) = cos_theta;

      k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_u) = cos_theta * cos_phi;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_v) = cos_theta * sin_phi;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_w) = -sin_theta; 

      k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_u) = -sin_phi;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_v) = cos_phi;
      k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_w) = 0.0;

      k_curvilinear_mesh_h(idx,curv_mesh_var::xg) = x;
      k_curvilinear_mesh_h(idx,curv_mesh_var::yg) = y;
      k_curvilinear_mesh_h(idx,curv_mesh_var::zg) = z;
    }
    );
    type = grid_type::SPHERICAL;
    Kokkos::deep_copy(k_curvilinear_mesh_d, k_curvilinear_mesh_h);
  }

  // A tiny, value-copyable snapshot of the geometry needed on-device.
  // Holds scalars + the DEVICE curvilinear-mesh View. Safe to capture
  // by value into a KOKKOS_LAMBDA.
  struct grid_geom_t {
      int   type;
      int   nx, ny, nz;
      double x0, y0, z0;
      double dx, dy, dz;
      k_curvilinear_mesh_t mesh_d;   // the device View type (k_curvilinear_mesh_d)
      k_curvilinear_mesh_t::HostMirror  mesh_h;

      // Single accessor that resolves to the correct space at compile time.
      // Returns a reference to the element; usable read-only here.
      KOKKOS_INLINE_FUNCTION
      float m(int n, int var) const {
        KOKKOS_IF_ON_HOST(  ( return mesh_h(n, var); ) )
        KOKKOS_IF_ON_DEVICE(( return mesh_d(n, var); ) )
      }

      KOKKOS_INLINE_FUNCTION
      void local_to_global(int voxel_i, float dx_p, float dy_p, float dz_p,
                            double& xi_out, double& eta_out, double& mu_out) const {
        int ix, iy, iz;
        UNVOXEL(voxel_i, ix, iy, iz, nx, ny, nz);
        double xic = x0 + (ix - 0.5) * dx;
        double etc = y0 + (iy - 0.5) * dy;
        double muc = z0 + (iz - 0.5) * dz;
        xi_out  = xic + dx_p * dx / 2.0;
        eta_out = etc + dy_p * dy / 2.0;
        mu_out  = muc + dz_p * dz / 2.0;
      }
      
      /* KOKKOS_INLINE_FUNCTION
      void local_to_global_cart(int voxel_i, float dx_p, float dy_p, float dz_p,
                                double& x_out, double& y_out, double& z_out) const {
        int i, j, k;
        UNVOXEL(voxel_i, i, j, k, nx, ny, nz);
        if (type == grid_type::CARTESIAN) {
            local_to_global(voxel_i, dx_p, dy_p, dz_p, x_out, y_out, z_out);
        } else if (type == grid_type::STRETCHED_CARTESIAN) {
            int node_idx = GRID_TO_MESH(i, j, k, nx, ny, nz);
            x_out = mesh(node_idx, curv_mesh_var::xg) + 0.5*dx_p*mesh(node_idx, curv_mesh_var::h_1);
            y_out = mesh(node_idx, curv_mesh_var::yg) + 0.5*dy_p*mesh(node_idx, curv_mesh_var::h_2);
            z_out = mesh(node_idx, curv_mesh_var::zg) + 0.5*dz_p*mesh(node_idx, curv_mesh_var::h_3);
        } else if (type == grid_type::CYLINDRICAL) {
            double r     = x0 + (i - 0.5)*dx + 0.5*dx_p*dx;
            double theta = y0 + (j - 0.5)*dy + 0.5*dy_p*dy;
            double z     = z0 + (k - 0.5)*dz + 0.5*dz_p*dz;
            x_out = r*cos(theta);  y_out = r*sin(theta);  z_out = z;
        } else if (type == grid_type::SPHERICAL) {
            double r     = x0 + (i - 0.5)*dx + 0.5*dx_p*dx;
            double theta = y0 + (j - 0.5)*dy + 0.5*dy_p*dy;
            double phi   = z0 + (k - 0.5)*dz + 0.5*dz_p*dz;
            x_out = r*sin(theta)*cos(phi);
            y_out = r*sin(theta)*sin(phi);
            z_out = r*cos(theta);
        } else {
            float Sx_m1=0.125f*(1-dx_p)*(1-dx_p), Sx_0=0.25f*(3-dx_p*dx_p), Sx_p1=0.125f*(1+dx_p)*(1+dx_p);
            float Sy_m1=0.125f*(1-dy_p)*(1-dy_p), Sy_0=0.25f*(3-dy_p*dy_p), Sy_p1=0.125f*(1+dy_p)*(1+dy_p);
            float Sz_m1=0.125f*(1-dz_p)*(1-dz_p), Sz_0=0.25f*(3-dz_p*dz_p), Sz_p1=0.125f*(1+dz_p)*(1+dz_p);
            x_out=y_out=z_out=0.0;
            for (int kk=-1;kk<=1;kk++){ float Sz=(kk==-1)?Sz_m1:((kk==0)?Sz_0:Sz_p1);
              for (int jj=-1;jj<=1;jj++){ float Sy=(jj==-1)?Sy_m1:((jj==0)?Sy_0:Sy_p1);
                for (int iio=-1;iio<=1;iio++){ float Sx=(iio==-1)?Sx_m1:((iio==0)?Sx_0:Sx_p1);
                  int node_idx = GRID_TO_MESH(i+iio, j+jj, k+kk, nx, ny, nz);
                  float w = Sx*Sy*Sz;
                  x_out += w * mesh(node_idx, curv_mesh_var::xg);   // was ..._h
                  y_out += w * mesh(node_idx, curv_mesh_var::yg);
                  z_out += w * mesh(node_idx, curv_mesh_var::zg);
                }}}
        }
      } */
      KOKKOS_INLINE_FUNCTION
      void local_to_global_cart(int voxel_i,
              float dx_p, float dy_p, float dz_p,
              double& x_out, double& y_out, double& z_out) const {
          int i, j, k;
          UNVOXEL(voxel_i, i, j, k, nx, ny, nz);

          if (type == grid_type::CARTESIAN) {
              local_to_global(voxel_i, dx_p, dy_p, dz_p, x_out, y_out, z_out);
          } else if (type == grid_type::STRETCHED_CARTESIAN) {
              int node_idx = GRID_TO_MESH(i, j, k, nx, ny, nz);
              double x_center = m(node_idx, curv_mesh_var::xg);   // was ..._h
              double y_center = m(node_idx, curv_mesh_var::yg);
              double z_center = m(node_idx, curv_mesh_var::zg);
              x_out = x_center + 0.5 * dx_p * m(node_idx, curv_mesh_var::h_1);
              y_out = y_center + 0.5 * dy_p * m(node_idx, curv_mesh_var::h_2);
              z_out = z_center + 0.5 * dz_p * m(node_idx, curv_mesh_var::h_3);

          } else if (type == grid_type::CYLINDRICAL) {
              // Cylindrical: (r, theta, z) -> (x, y, z)
              
              double r = x0 + (i - 0.5) * dx;
              double theta = y0 + (j - 0.5) * dy;
              double z = z0 + (k - 0.5) * dz;  // Fixed!
              
              double r_relative = 0.5 * dx_p * dx;
              double theta_relative = 0.5 * dy_p * dy;
              double z_relative = 0.5 * dz_p * dz;  // Fixed!
              
              double r_phys = r + r_relative;
              double theta_phys = theta + theta_relative;
              double z_phys = z + z_relative;  // Fixed!
              
              x_out = r_phys * cosf(theta_phys);
              y_out = r_phys * sinf(theta_phys);
              z_out = z_phys;
              
          } else if (type == grid_type::SPHERICAL) {
              // Spherical: (r, theta, phi) -> (x, y, z)
              
              double r = x0 + (i - 0.5) * dx;
              double theta = y0 + (j - 0.5) * dy;
              double phi = z0 + (k - 0.5) * dz;
              
              double r_relative = 0.5 * dx_p * dx;
              double theta_relative = 0.5 * dy_p * dy;
              double phi_relative = 0.5 * dz_p * dz;
              
              double r_phys = r + r_relative;
              double theta_phys = theta + theta_relative;
              double phi_phys = phi + phi_relative;
              
              x_out = r_phys * sinf(theta_phys) * cosf(phi_phys);
              y_out = r_phys * sinf(theta_phys) * sinf(phi_phys);
              z_out = r_phys * cosf(theta_phys);
              
          } else {
              // Use B-spline interpolation from stored mesh data
              // Quadratic B-spline basis (must match compute_bspline_basis in advance_p.cc)
              float Sx_m1 = 0.125f * (1.0f - dx_p) * (1.0f - dx_p);
              float Sx_0  = 0.25f * (3.0f - dx_p * dx_p);
              float Sx_p1 = 0.125f * (1.0f + dx_p) * (1.0f + dx_p);

              float Sy_m1 = 0.125f * (1.0f - dy_p) * (1.0f - dy_p);
              float Sy_0  = 0.25f * (3.0f - dy_p * dy_p);
              float Sy_p1 = 0.125f * (1.0f + dy_p) * (1.0f + dy_p);

              float Sz_m1 = 0.125f * (1.0f - dz_p) * (1.0f - dz_p);
              float Sz_0  = 0.25f * (3.0f - dz_p * dz_p);
              float Sz_p1 = 0.125f * (1.0f + dz_p) * (1.0f + dz_p);
              
              x_out = 0.0;
              y_out = 0.0;
              z_out = 0.0;
              
              // 3x3x3 stencil interpolation
              for (int kk = -1; kk <= 1; kk++) {
                  float Sz = (kk == -1) ? Sz_m1 : ((kk == 0) ? Sz_0 : Sz_p1);
                  
                  for (int jj = -1; jj <= 1; jj++) {
                      float Sy = (jj == -1) ? Sy_m1 : ((jj == 0) ? Sy_0 : Sy_p1);
                      
                      for (int ii = -1; ii <= 1; ii++) {
                          float Sx = (ii == -1) ? Sx_m1 : ((ii == 0) ? Sx_0 : Sx_p1);
                          
                          int node_idx = GRID_TO_MESH(i + ii, j + jj, k + kk, nx, ny, nz);
                          float weight = Sx * Sy * Sz;
                          
                          x_out += weight * m(node_idx, curv_mesh_var::xg);
                          y_out += weight * m(node_idx, curv_mesh_var::yg);
                          z_out += weight * m(node_idx, curv_mesh_var::zg);
                      }
                  }
              }
          }
      }
  };

  // member of grid_t, host-side
  grid_geom_t geom() const {
      return grid_geom_t{ type, nx, ny, nz, x0, y0, z0,
                          dx, dy, dz, k_curvilinear_mesh_d, k_curvilinear_mesh_h };
  }

} grid_t;





// Advance the voxel mesh index (v) and corresponding voxel mesh
// coordinates (x,y,z) in a region with min- and max-corners of
// (xl,yl,zl) and (xh,yh,zh) of a (nx,ny,nz) resolution voxel mesh in
// FORTRAN ordering.  Results will not be valid (v,x,y,z) are not in
// the region or if (v,x,y,z) is the last voxel in that region.
//
// This macro is not robust.  Macro arguments should be safe against
// multiple evaluation.  Further, this macro is not semantically a
// single statement.  (It is meant for use in high performance stencil
// inner loops.)
//
// This is written with seeming extraneously if tests in order to get
// the compiler to generate branceless conditional move and add
// instructions (none of the branches below are actual branches in
// assembly).

#define NEXT_VOXEL(v,x,y,z, xl,xh, yl,yh, zl,zh, nx,ny,nz) \
  (v)++;                                                   \
  (x)++;                                                   \
  if( (x)>(xh) ) (v) +=  (nx)-(xh)+(xl)+1;                 \
  if( (x)>(xh) ) (y)++;                                    \
  if( (x)>(xh) ) (x) = (xl);                               \
  if( (y)>(yh) ) (v) += ((ny)-(yh)+(yl)+1)*((nx)+2);       \
  if( (y)>(yh) ) (z)++;                                    \
  if( (y)>(yh) ) (y) = (yl)



// Quadratic B-spline basis functions and derivatives for the curvilinear mesh.
// Input:  xi in logical coordinate ([-1,1] within a cell)
// Output: basis functions S and derivatives dS for three nodes (i-1, i, i+1)
KOKKOS_INLINE_FUNCTION
void compute_bspline_basis(float xi,
                           float& S_m1, float& S_0, float& S_p1,
                           float& dS_m1, float& dS_0, float& dS_p1) {
  // Basis functions
  S_m1 = 0.125f * (1.0f - xi) * (1.0f - xi);
  S_0  = 0.25f * (3.0f - xi * xi);
  S_p1 = 0.125f * (1.0f + xi) * (1.0f + xi);

  // Derivatives
  dS_m1 = 0.25f * (xi - 1.0f);
  dS_0  = -0.5f * xi;
  dS_p1 = 0.25f * (xi + 1.0f);
}

// Reciprocal basis vectors grad(xi^a) and the Jacobian at a particle position,
// for each supported grid geometry. Used by both advance_p and move_p so the
// current/charge deposit uses coordinate-consistent (contravariant) components.
KOKKOS_INLINE_FUNCTION
void compute_reciprocal_basis(
    const grid::grid_geom_t& geom,
    float dx, float dy, float dz,  // Particle position in logical coords
    int ii,                         //Base voxel index
    int nx, int ny, int nz,
    float gdx, float gdy, float gdz,
    float& grad_xi_x, float& grad_xi_y, float& grad_xi_z,
    float& grad_eta_x, float& grad_eta_y, float& grad_eta_z,
    float& grad_mu_x, float& grad_mu_y, float& grad_mu_z,
    float& jac)
{
  if (geom.type == grid_type::CARTESIAN) {
        grad_xi_x = 2.0f / gdx;
        grad_xi_y = 0.0f;
        grad_xi_z = 0.0f;
        grad_eta_x = 0.0f;
        grad_eta_y = 2.0f / gdy;
        grad_eta_z = 0.0f;
        grad_mu_x = 0.0f;
        grad_mu_y = 0.0f;
        grad_mu_z = 2.0f / gdz;
        jac = gdx * gdy * gdz / 8.0f;

    } else if (geom.type == grid_type::CYLINDRICAL) {
        [[maybe_unused]] int i, j, k;
        UNVOXEL(ii, i, j, k, nx, ny, nz);
        float r = geom.x0 + (i - 0.5) * gdx;
        float theta = geom.y0 + (j - 0.5) * gdy;
        
        float r_relative = 0.5 * dx * gdx;
        float theta_relative = 0.5 * dy * gdy;

        float r_phys = r + r_relative;
        float theta_phys = theta + theta_relative;

        // Axis (r=0) regularization. grad_eta ~ 1/r_phys and inv_jac ~ 1/r_phys
        // both diverge at the axis, so a particle whose sub-cell r_phys lands near
        // the inner face of the first cell (r_phys->0) would dump a huge
        // inv_jac-weighted current/charge and blow up theta. Reflect across the
        // axis (r->-r, theta->theta+pi), matching init_cylindrical_grid, then
        // floor |r| to half a cell (the first cell-center radius) so 1/r_phys is
        // bounded by 2/gdx -- a half-cell buffer that regularizes only the
        // innermost half-cell and leaves larger r untouched.
        if (r_phys < 0.0f) { r_phys = -r_phys; theta_phys += (float)M_PI; }
        const float r_axis_min = 0.5f * gdx;
        if (r_phys < r_axis_min) r_phys = r_axis_min;

        float cos_th = cosf(theta_phys);
        float sin_th = sinf(theta_phys);

        grad_xi_x = (2.0f / gdx) * cos_th;
        grad_xi_y = (2.0f / gdx) * sin_th;
        grad_xi_z = 0.0f;
        grad_eta_x = (-2.0f / gdy) * sin_th / r_phys;
        grad_eta_y = (2.0f / gdy) * cos_th / r_phys;
        grad_eta_z = 0.0f;
        grad_mu_x = 0.0f;
        grad_mu_y = 0.0f;
        grad_mu_z = 2.0f / gdz;
        jac = r_phys * gdx * gdy * gdz / 8.0f;

    } else if (geom.type == grid_type::SPHERICAL) {
        double x_cart, y_cart, z_cart;
        geom.local_to_global_cart(ii, dx, dy, dz, x_cart, y_cart, z_cart);

        float r_phys = sqrtf(x_cart*x_cart + y_cart*y_cart + z_cart*z_cart);

        // Axis regularization. The reciprocal basis has 1/r_phys (center) and
        // 1/(r_phys*sin_theta) (polar axis) factors that diverge at the r=0
        // center and the theta=0,pi poles. Floor r_phys to half a cell FIRST --
        // before it is used in acosf below -- so that at r=0 we do not evaluate
        // acosf(z/0)=acosf(0/0)=NaN, and clamp the acosf argument into [-1,1] to
        // guard against roundoff pushing it slightly out of domain.
        const float r_axis_min = 0.5f * gdx;
        if (r_phys < r_axis_min) r_phys = r_axis_min;

        float cos_arg = z_cart / r_phys;
        cos_arg = fmaxf(-1.0f, fminf(1.0f, cos_arg));
        float theta_phys = acosf(cos_arg);
        float phi_phys = atan2f(y_cart, x_cart);
        float sin_theta = sinf(theta_phys);
        float cos_theta = cosf(theta_phys);
        float sin_phi = sinf(phi_phys);
        float cos_phi = cosf(phi_phys);

        // Floor |sin(theta)| near the poles so the 1/(r sin theta) terms stay finite.
        const float sin_theta_min = 0.5f * gdy; // gdy = dtheta
        float sin_theta_reg = (fabsf(sin_theta) < sin_theta_min)
                            ? (sin_theta < 0.0f ? -sin_theta_min : sin_theta_min)
                            : sin_theta;

        grad_xi_x = (2.0f / gdx) * sin_theta * cos_phi;
        grad_xi_y = (2.0f / gdx) * sin_theta * sin_phi;
        grad_xi_z = (2.0f / gdx) * cos_theta;
        grad_eta_x = (2.0f / gdy) * cos_theta * cos_phi / r_phys;
        grad_eta_y = (2.0f / gdy) * cos_theta * sin_phi / r_phys;
        grad_eta_z = (2.0f / gdy) * (-sin_theta) / r_phys;
        grad_mu_x = (2.0f / gdz) * (-sin_phi) / (r_phys * sin_theta_reg);
        grad_mu_y = (2.0f / gdz) * cos_phi / (r_phys * sin_theta_reg);
        grad_mu_z = 0.0f;
        jac = r_phys * r_phys * sin_theta_reg * gdx * gdy * gdz / 8.0f;

    } else if (geom.type == grid_type::STRETCHED_CARTESIAN) {
        // For stretched Cartesian, basis vectors remain Cartesian-aligned,
        // but scale factors vary with position. Reciprocal basis is simply
        // the inverse of the scale factors.
        int i, j, k;
        UNVOXEL(ii, i, j, k, nx, ny, nz);
        int node_idx = GRID_TO_MESH(i, j, k, nx, ny, nz);
        
        // Get local scale factors at this cell
        float h1 = geom.m(node_idx, curv_mesh_var::h_1);
        float h2 = geom.m(node_idx, curv_mesh_var::h_2);
        float h3 = geom.m(node_idx, curv_mesh_var::h_3);
        
        // Reciprocal basis vectors: grad(xi^alpha) = ehat^alpha / h_alpha
        // Since basis vectors are Cartesian-aligned: ehat^1 = xhat, ehat^2 = yhat, ehat^3 = zhat
        grad_xi_x = 2.0f / (h1 * gdx);
        grad_xi_y = 0.0f;
        grad_xi_z = 0.0f;
        
        grad_eta_x = 0.0f;
        grad_eta_y = 2.0f / (h2 * gdy);
        grad_eta_z = 0.0f;
        
        grad_mu_x = 0.0f;
        grad_mu_y = 0.0f;
        grad_mu_z = 2.0f / (h3 * gdz);
        
        // Jacobian (already stored, but can compute as product of scale factors)
        jac = h1 * h2 * h3 * gdx * gdy * gdz / 8.0f;

    } else {
      //Compute B-spline basis functions and derivatives
      float Sx_m1, Sx_0, Sx_p1, dSx_m1, dSx_0, dSx_p1;
      float Sy_m1, Sy_0, Sy_p1, dSy_m1, dSy_0, dSy_p1;
      float Sz_m1, Sz_0, Sz_p1, dSz_m1, dSz_0, dSz_p1;

      compute_bspline_basis(dx, Sx_m1, Sx_0, Sx_p1, dSx_m1, dSx_0, dSx_p1);
      compute_bspline_basis(dy, Sy_m1, Sy_0, Sy_p1, dSy_m1, dSy_0, dSy_p1);
      compute_bspline_basis(dz, Sz_m1, Sz_0, Sz_p1, dSz_m1, dSz_0, dSz_p1);

      // Get voxel coordinates from linear index
      int xi, yi, zi;
      UNVOXEL(ii,xi, yi, zi,nx,ny,nz);

      // Initialize Jacobian matrix elements
      float dx_dxi = 0.0f, dy_dxi = 0.0f, dz_dxi = 0.0f;
      float dx_deta = 0.0f, dy_deta = 0.0f, dz_deta = 0.0f;
      float dx_dmu = 0.0f, dy_dmu = 0.0f, dz_dmu = 0.0f;

      // Tri-linear interpolation using tensor product of B-splines
      // Loop over 3x3x3 neighboring ndes
      for (int kk = -1; kk <= 1; kk++) {
        float Sz = (kk == -1) ? Sz_m1 : ((kk == 0) ? Sz_0 : Sz_p1);
        float dSz = (kk == -1) ? dSz_m1 : ((kk == 0) ? dSz_0 : dSz_p1);

        for (int jj = -1; jj <= 1; jj++) {
          float Sy = (jj == -1) ? Sy_m1 : ((jj == 0) ? Sy_0 : Sy_p1);
          float dSy = (jj == -1) ? dSy_m1 : ((jj == 0) ? dSy_0 : dSy_p1);

          for (int ii_offset = -1; ii_offset <= 1; ii_offset++) {
            float Sx = (ii_offset == -1) ? Sx_m1 : ((ii_offset == 0) ? Sx_0 : Sx_p1);
            float dSx = (ii_offset == -1) ? dSx_m1 : ((ii_offset == 0) ? dSx_0 : dSx_p1);

            // Compute node index in the curvilinear mesh array
            int node_idx = GRID_TO_MESH(xi+ii_offset,yi+jj,zi+kk,nx,ny,nz);

            // Get Cartesian positions at this node
            float xg = geom.m(node_idx, curv_mesh_var::xg);
            float yg = geom.m(node_idx, curv_mesh_var::yg);
            float zg = geom.m(node_idx, curv_mesh_var::zg);

            // Accumulate Jacobian matrix elements
            dx_dxi += xg * dSx * Sy * Sz;
            dy_dxi += yg * dSx * Sy * Sz;
            dz_dxi += zg * dSx * Sy * Sz;

            dx_deta += xg * Sx * dSy * Sz;
            dy_deta += yg * Sx * dSy * Sz;
            dz_deta += zg * Sx * dSy * Sz;

            dx_dmu += xg * Sx * Sy * dSz;
            dy_dmu += yg * Sx * Sy * dSz;
            dz_dmu += zg * Sx * Sy * dSz;
          }
        }
      }

      // Compute Jacobian determinant
      jac = dx_dxi * (dy_deta * dz_dmu - dy_dmu * dz_deta)
              - dx_deta * (dy_dxi * dz_dmu - dy_dmu * dz_dxi)
              + dx_dmu * (dy_dxi * dz_deta - dy_deta * dz_dxi);

      float inv_jac = 1.0f / jac;

      // Compute reciprocal basis vectors
      grad_xi_x = inv_jac * (dy_deta * dz_dmu - dy_dmu * dz_deta);
      grad_xi_y = inv_jac * (dx_dmu * dz_deta - dx_deta * dz_dmu);
      grad_xi_z = inv_jac * (dx_deta * dy_dmu - dx_dmu * dy_deta);

      grad_eta_x = inv_jac * (dy_dmu * dz_dxi - dy_dxi * dz_dmu);
      grad_eta_y = inv_jac * (dx_dxi * dz_dmu - dx_dmu * dz_dxi);
      grad_eta_z = inv_jac * (dx_dmu * dy_dxi - dx_dxi * dy_dmu);

      grad_mu_x = inv_jac * (dy_dxi * dz_deta - dy_deta * dz_dxi);
      grad_mu_y = inv_jac * (dx_deta * dz_dxi - dx_dxi * dz_deta);
      grad_mu_z = inv_jac * (dx_dxi * dy_deta - dx_deta * dy_dxi);
    }
}

// In grid_structors.c

grid_t *
new_grid( void );

void
delete_grid( grid_t * g );

// In ops.c

void
size_grid( grid_t * g, int lnx, int lny, int lnz );

void
join_grid( grid_t * g, int bound, int rank );

void
set_fbc( grid_t *g, int bound, int fbc );

void
set_pbc( grid_t *g, int bound, int pbc );

// In partition.c

// g->{n,d}{x,y,z} is _coherent_ on all nodes in the domain after
// these calls as are g->{x,y,z}{0,1}.  Due to the vagaries of
// floating point, though g->nx*g->dx may not be the exactly the same
// as g->x1-g->x0 though.  Thus matters when doing things like
// robustly converting global position coordinates to/from local index
// + offset position coordinates.
//
// The robust procedure to convert _from_ a global coordinate to a
// local coordinate is:
//
// (1) Test if this node has ownership of the point using
// g->{x,y,z}{0,1}.  Points with x==g->x1 exactly boundaries should be
// considered part of the local domain only if the corresponding
// x-boundary condition is local.  Similarly for y- and z-.
//
// (2) If this node has ownership of the point, compute the relative
// voxel and offset of the x-coordinate via
// g->nx*((x-g->x0)/(g->x1-g->x0)), _NOT_ (x-g->x0)/g->dx and _NOT_
// (x-g->x0)*(1/g->dx)!  Similarly for y and z.
//
// (3) Break the voxel and offsets into integer and fractional parts.
// Particles exactly on the far wall should have their fractional
// particles set to 1 and their integer parts subtracted by 1.  Double
// the fractional part and subtract by one to get the voxel centered
// offset.  Convert the local voxel coordinates into a local voxel index
// using VOXEL above.
//
// Reverse this protocol to robustly convert from voxel+offset to
// global coordinates.  Due to the vagaries of floating point, the
// inverse process may not be exact.

void
partition_periodic_box( grid_t *g,
			double gx0, double gy0, double gz0,
			double gx1, double gy1, double gz1,
                        int gnx, int gny, int gnz,
                        int gpx, int gpy, int gpz );

void
partition_absorbing_box( grid_t *g,
                         double gx0, double gy0, double gz0,
                         double gx1, double gy1, double gz1,
                         int gnx, int gny, int gnz,
                         int gpx, int gpy, int gpz,
                         int pbc );

void
partition_metal_box( grid_t *g,
                     double gx0, double gy0, double gz0,
                     double gx1, double gy1, double gz1,
                     int gnx, int gny, int gnz,
                     int gpx, int gpy, int gpz );

// In grid_comm.c

// FIXME: SHOULD TAKE A RAW PORT INDEX INSTEAD OF A PORT COORDS

// Start receiving a message from the node.
// Only one message recv may be pending at a time on a given port.

void
begin_recv_port( int i,    // x port coord ([-1,0,1])
                 int j,    // y port coord ([-1,0,1])
                 int k,    // z port coord ([-1,0,1])
                 int size, // Expected size in bytes
                 const grid_t * g );

// Returns pointer to the buffer that begin send will use for the next
// send on the given port.  The buffer is guaranteed to have enough
// room for size bytes.  This is only valid to call if no sends on
// that port are pending.

void * ALIGNED(128)
size_send_port( int i,    // x port coord ([-1,0,1])
                int j,    // y port coord ([-1,0,1])
                int k,    // z port coord ([-1,0,1])
                int size, // Needed send size in bytes
                const grid_t * g );

// Begin sending size bytes of the buffer out the given port.  Only
// one message send may be pending at a time on a given port.  (FIXME:
// WHAT HAPPENS IF SIZE_SEND_PORT size < begin_send_port
// size??)

void
begin_send_port( int i,    // x port coord ([-1,0,1])
                 int j,    // y port coord ([-1,0,1])
                 int k,    // z port coord ([-1,0,1])
                 int size, // Number of bytes to send (in bytes)
                 const grid_t * g );

// Complete the pending recv on the given port.  Only valid to call if
// there is a pending recv.  Returns pointer to a buffer containing
// the received data.  (FIXME: WHAT HAPPENS IF EXPECTED RECV SIZE
// GIVEN IN BEGIN_RECV DOES NOT MATCH END_RECV??)

void * ALIGNED(128)
end_recv_port( int i, // x port coord ([-1,0,1])
               int j, // y port coord ([-1,0,1])
               int k, // z port coord ([-1,0,1])
               const grid_t * g );

// Kokkos versions

// Complete the pending send on the given port.  Only valid to call if
// there is a pending send on the port.  Note that this guarantees
// that send port is available to the caller for additional use, not
// necessarily that the message has arrived at the destination of the
// port.

void
end_send_port( int i, // x port coord ([-1,0,1])
               int j, // y port coord ([-1,0,1])
               int k, // z port coord ([-1,0,1])
               const grid_t * g );

// Star receiving a message from the node.
// Only one message recv may be pending at a time on a given port.
// Must pass receive buffer of necessary size/
void begin_recv_port_kokkos(const grid_t* g, int port, int size, int tag, char* ALIGNED(128) recv_buf);
void begin_recv_port_k(int i, int j, int k, int size, const grid_t* g, char* recv_buf);

// Begin sending size bytes of the buffer out the given port.  Only
// one message send may be pending at a time on a given port.  (FIXME:
// WHAT HAPPENS IF SIZE_SEND_PORT size < begin_send_port
// size??)
void begin_send_port_kokkos(const grid_t* g, int port, int size, int tag, char* ALIGNED(128) send_buf);
void begin_send_port_k(int i, int j, int k, int size, const grid_t* g, char* send_buf);

// Complete the pending recv on the given port.  Only valid to call if
// there is a pending recv.  Received data put into original receive buffer
// from begin_recv.   (FIXME: WHAT HAPPENS IF EXPECTED RECV SIZE
// GIVEN IN BEGIN_RECV DOES NOT MATCH END_RECV??)
void end_recv_port_kokkos(const grid_t* g, int port);
void* end_recv_port_k(int i, int j, int k, const grid_t* g);

// Complete the pending send on the given port.  Only valid to call if
// there is a pending send on the port.  Note that this guarantees
// that send port is available to the caller for additional use, not
// necessarily that the message has arrived at the destination of the
// port.
void end_send_port_kokkos(const grid_t* g, int port);
void end_send_port_k(int i, int j, int k, const grid_t* g);

// In distribute_voxels.c

// Given a block of voxels to be processed, determine the number of
// the first voxel (v,x,y,z) a particular job assigned to a pipeline
// should process and return the number of voxels to process.
//
// It is assumed that the pipelines will process voxels in FORTRAN
// ordering (e.g. inner loop increments x-index).
//
// jobs are indexed from 0 to n_job-1.  jobs are _always_ have the
// number of voxels an integer multiple of the bundle size.  If job
// is set to n_job, this function will determine the parameters of
// the final incomplete bundle.

#define DISTRIBUTE_VOXELS( x0,x1, y0,y1, z0,z1, b, p,P, x,y,z,nv ) do { \
    int _x0=(x0), _y0=(y0), _z0=(z0), _b=(b), _p=(p), _P=(P);           \
    int _nx = (x1)-_x0+1, _ny = (y1)-_y0+1, _nv = _nx*_ny*((z1)-_z0+1); \
    double _t = (double)( _nv/_b ) / (double)_P;                        \
    int          _x=_b*(int)( _t*(double)(_p  ) + 0.5 ), _y, _z;        \
    if( _p<_P ) _nv=_b*(int)( _t*(double)(_p+1) + 0.5 );                \
    _nv -= _x;                 /* x = (x-x0) + nx*((y-y0) + ny*(z-z0)) */ \
    _y   = _nx ? (_x/_nx) : 0; /* y =              (y-y0) + ny*(z-z0)  */ \
    _z   = _ny ? (_y/_ny) : 0; /* z =                          (z-z0)  */ \
    _x  -= _y*_nx;             /* x = (x-x0)                           */ \
    _y  -= _z*_ny;             /* y =              (y-y0)              */ \
    (x)  = _x+_x0;                                                      \
    (y)  = _y+_y0;                                                      \
    (z)  = _z+_z0;                                                      \
    (nv) = _nv;                                                         \
  } while(0)

#endif
