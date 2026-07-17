// VPIC headers
#include "dump_strategy.h"
#include "vpic.h"

/* -1 means no ranks talk */
#define VERBOSE_rank -1

const int max_filename_bytes = 256;

inline void transform_J_to_physical(
    const grid_t* grid,
    int voxel_index,
    float J_xi, float J_eta, float J_zeta,
    float& J_phys_xi, float& J_phys_eta, float& J_phys_zeta)
{
    // J is CONTRAVARIANT, physical = (h_i / 2) * J^i
    auto& k_cmesh_h = grid->k_curvilinear_mesh_h;
    int mesh_index = VOXEL_TO_MESH(voxel_index, grid->nx, grid->ny, grid->nz);
    
    float h1 = k_cmesh_h(mesh_index, curv_mesh_var::h_1);
    float h2 = k_cmesh_h(mesh_index, curv_mesh_var::h_2);
    float h3 = k_cmesh_h(mesh_index, curv_mesh_var::h_3);
    
    J_phys_xi   = (h1 * 0.5f) * J_xi;
    J_phys_eta  = (h2 * 0.5f) * J_eta;
    J_phys_zeta = (h3 * 0.5f) * J_zeta;
}

inline void transform_B_to_physical(
    const grid_t* grid,
    int voxel_index,
    float B_xi, float B_eta, float B_zeta,
    float& B_phys_xi, float& B_phys_eta, float& B_phys_zeta)
{
    // B is CONTRAVARIANT (like J), physical = (h_i / 2) * B^i
    auto& k_cmesh_h = grid->k_curvilinear_mesh_h;
    int mesh_index = VOXEL_TO_MESH(voxel_index, grid->nx, grid->ny, grid->nz);
    
    float h1 = k_cmesh_h(mesh_index, curv_mesh_var::h_1);
    float h2 = k_cmesh_h(mesh_index, curv_mesh_var::h_2);
    float h3 = k_cmesh_h(mesh_index, curv_mesh_var::h_3);
    
    B_phys_xi   = B_xi*h1;
    B_phys_eta  = B_eta*h2;
    B_phys_zeta = B_zeta*h3;
}

inline void transform_E_to_physical(
    const grid_t* grid,
    int voxel_index,
    float E_xi, float E_eta, float E_zeta,
    float& E_phys_xi, float& E_phys_eta, float& E_phys_zeta)
{
    // E is COVARIANT, physical = (2 / h_i) * E_i
    auto& k_cmesh_h = grid->k_curvilinear_mesh_h;
    int mesh_index = VOXEL_TO_MESH(voxel_index, grid->nx, grid->ny, grid->nz);
    
    float h1 = k_cmesh_h(mesh_index, curv_mesh_var::h_1);
    float h2 = k_cmesh_h(mesh_index, curv_mesh_var::h_2);
    float h3 = k_cmesh_h(mesh_index, curv_mesh_var::h_3);
    
    E_phys_xi   = E_xi/h1;
    E_phys_eta  = E_eta/h2;
    E_phys_zeta = E_zeta/h3;
}

// Create a new dump strategy
Dump_Strategy *
new_dump_strategy(DumpStrategyID dump_strategy_id,
                  vpic_simulation *vpic_simu)
{
  Dump_Strategy *ds;
  MALLOC(ds, 1);
  //CLEAR(ds, 1);

  // Do any post init/restore simulation modifications
  switch (dump_strategy_id)
  {
    case DUMP_STRATEGY_BINARY:
      if (vpic_simu->rank() == 0)
        std::cout << "# DUMP_STRATEGY_BINARY  enabled \n";
      ds = new BinaryDump(vpic_simu->rank(), vpic_simu->nproc());
      break;
    case DUMP_STRATEGY_HDF5:
#ifdef VPIC_ENABLE_HDF5
      if (vpic_simu->rank() == 0)
        std::cout << "DUMP_STRATEGY_HDF5 enabled \n";
      ds = new HDF5Dump(vpic_simu->rank(), vpic_simu->nproc(),
          vpic_simu->num_step, vpic_simu->field_interval,
          vpic_simu->hydro_interval, vpic_simu->fluid_interval);
#else
      std::cout << "HDF5Dump is not enabled \n";
#endif
      break;
    default:
      break;
  }

  return ds;
}

// Delete a dump strategy
void delete_dump_strategy(Dump_Strategy *ds)
{
  if (!ds)
    return;
  UNREGISTER_OBJECT(ds);
  FREE(ds);
}

/*****************************************************************************
 * Binary dump IO
 *****************************************************************************/

// Dump fields in binary format
void BinaryDump::dump_fields(
    const char *fbase,
    int step,
    grid_t *grid,
    field_array_t *field_array,
    int ftag)
{
  if (step > field_array->last_copied)
      field_array->copy_to_host();

  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[3];
  if (rank == 0) printf("Calling Binary dump_fields ... \n");

  if (!fbase) ERROR(("Invalid filename"));

  if (rank == 0) MESSAGE(("Dumping fields to \"%s\"", fbase));

  if (ftag)
    snprintf(fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step, rank);
  else
    snprintf(fname, max_filename_bytes, "%s.%i", fbase, rank);

  FileIOStatus status = fileIO.open(fname, io_write);
  if (status == fail)
    ERROR(("Could not open \"%s\".", fname));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = grid->nx;
  size_t nyout = grid->ny;
  size_t nzout = grid->nz;
  float dxout = grid->dx;
  float dyout = grid->dy;
  float dzout = grid->dz;

  WRITE_HEADER_V0(dump_type::field_dump, -1, 0, fileIO, step, rank, nproc);

  dim[0] = grid->nx + 2;
  dim[1] = grid->ny + 2;
  dim[2] = grid->nz + 2;
  
  // CREATE A COPY TO TRANSFORM
  field_t *f_transformed;
  MALLOC(f_transformed, dim[0] * dim[1] * dim[2]);
  COPY(f_transformed, field_array->f, dim[0] * dim[1] * dim[2]);
  
  // APPLY TRANSFORMATION TO CURRENT DENSITIES, ELECTRIC AND MAGNETIC FIELDS
for(int k = 0; k < dim[2]; k++) {
  for(int j = 0; j < dim[1]; j++) {
    for(int i = 0; i < dim[0]; i++) {
      int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
      
      // Transform electric field (COVARIANT: E)
      float E_xi   = field_array->f[idx].ex;
      float E_eta  = field_array->f[idx].ey;
      float E_zeta = field_array->f[idx].ez;
      
      float E_phys_xi, E_phys_eta, E_phys_zeta;
      transform_E_to_physical(grid, idx, E_xi, E_eta, E_zeta,
                             E_phys_xi, E_phys_eta, E_phys_zeta);
      
      f_transformed[idx].ex = E_phys_xi;
      f_transformed[idx].ey = E_phys_eta;
      f_transformed[idx].ez = E_phys_zeta;
      
      // Transform magnetic field (CONTRAVARIANT: B)
      float B_xi   = field_array->f[idx].cbx;
      float B_eta  = field_array->f[idx].cby;
      float B_zeta = field_array->f[idx].cbz;
      
      float B_phys_xi, B_phys_eta, B_phys_zeta;
      transform_B_to_physical(grid, idx, B_xi, B_eta, B_zeta,
                             B_phys_xi, B_phys_eta, B_phys_zeta);
      
      f_transformed[idx].cbx = B_phys_xi;
      f_transformed[idx].cby = B_phys_eta;
      f_transformed[idx].cbz = B_phys_zeta;
      
      // Transform old magnetic field (CONTRAVARIANT: B0)
      B_xi   = field_array->f[idx].cbx0;
      B_eta  = field_array->f[idx].cby0;
      B_zeta = field_array->f[idx].cbz0;
      
      transform_B_to_physical(grid, idx, B_xi, B_eta, B_zeta,
                             B_phys_xi, B_phys_eta, B_phys_zeta);
      
      f_transformed[idx].cbx0 = B_phys_xi;
      f_transformed[idx].cby0 = B_phys_eta;
      f_transformed[idx].cbz0 = B_phys_zeta;
      
      // Transform current currents (CONTRAVARIANT: J) - existing code
      float J_xi   = field_array->f[idx].jfx;
      float J_eta  = field_array->f[idx].jfy;
      float J_zeta = field_array->f[idx].jfz;
      
      float J_phys_xi, J_phys_eta, J_phys_zeta;
      transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                             J_phys_xi, J_phys_eta, J_phys_zeta);
      
      f_transformed[idx].jfx = J_phys_xi;
      f_transformed[idx].jfy = J_phys_eta;
      f_transformed[idx].jfz = J_phys_zeta;
      
      // Transform old currents (CONTRAVARIANT: Jold) - existing code
      J_xi   = field_array->f[idx].jfxold;
      J_eta  = field_array->f[idx].jfyold;
      J_zeta = field_array->f[idx].jfzold;
      
      transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                             J_phys_xi, J_phys_eta, J_phys_zeta);
      
      f_transformed[idx].jfxold = J_phys_xi;
      f_transformed[idx].jfyold = J_phys_eta;
      f_transformed[idx].jfzold = J_phys_zeta;
    }
  }
}
  
  WRITE_ARRAY_HEADER(f_transformed, 3, dim, fileIO);
  fileIO.write(f_transformed, dim[0] * dim[1] * dim[2]);
  
  FREE(f_transformed);
  

  if (fileIO.close())
    ERROR(("File close failed on dump fields!!!"));
}

// Dump hydro in binary format
void BinaryDump::dump_hydro(
    const char *fbase,
    int step,
    species_t *sp,
    grid_t *grid,
    hydro_array_t *hydro_array,
    interpolator_array_t *interpolator_array,
    int ftag)
{
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[3];

  if (!sp) ERROR(("Invalid species \"%s\"", sp->name));
  if ( rank==0 ) log_printf("Dumping hydro for %s using Binary\n", sp->name);

  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;

  Kokkos::deep_copy(hydro_array->k_h_d, 0.0);
  accumulate_hydro_p_kokkos(
      particles,
      particles_i,
      hydro_array->k_h_d,
      interpolators_k,
      sp
  );

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_array->copy_to_host();
  synchronize_hydro_array( hydro_array );
#else
  // This does not give consistent results
  synchronize_hydro_array_kokkos(hydro_array);
  hydro_array->copy_to_host();
#endif

  if (!fbase)
    ERROR(("Invalid filename"));

  if (rank == 0)
    MESSAGE(("Dumping \"%s\" hydro fields to \"%s\"", sp->name, fbase));

  if (ftag)
    snprintf(fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step, rank);
  else
    snprintf(fname, max_filename_bytes, "%s.%i", fbase, rank);
  FileIOStatus status = fileIO.open(fname, io_write);
  if (status == fail)
    ERROR(("Could not open \"%s\".", fname));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = grid->nx;
  size_t nyout = grid->ny;
  size_t nzout = grid->nz;
  float dxout = grid->dx;
  float dyout = grid->dy;
  float dzout = grid->dz;

  WRITE_HEADER_V0(dump_type::hydro_dump, sp->id, sp->q / sp->m, fileIO, step, rank, nproc);

  dim[0] = grid->nx + 2;
  dim[1] = grid->ny + 2;
  dim[2] = grid->nz + 2;
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_t *h_transformed;
  MALLOC(h_transformed, dim[0] * dim[1] * dim[2]);
  COPY(h_transformed, hydro_array->h, dim[0] * dim[1] * dim[2]);
  
  // APPLY YOUR TRANSFORMATION
  for(int k = 0; k < dim[2]; k++) {
    for(int j = 0; j < dim[1]; j++) {
      for(int i = 0; i < dim[0]; i++) {
        int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
        
        float J_xi   = hydro_array->h[idx].jx;
        float J_eta  = hydro_array->h[idx].jy;
        float J_zeta = hydro_array->h[idx].jz;
        
        float J_phys_xi, J_phys_eta, J_phys_zeta;
        transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                               J_phys_xi, J_phys_eta, J_phys_zeta);
        
        h_transformed[idx].jx = J_phys_xi;
        h_transformed[idx].jy = J_phys_eta;
        h_transformed[idx].jz = J_phys_zeta;

      }
    }
  }
  
  WRITE_ARRAY_HEADER(h_transformed, 3, dim, fileIO);
  fileIO.write(h_transformed, dim[0] * dim[1] * dim[2]);
  
  FREE(h_transformed);
#else
  hydro_t h[1];
  WRITE_ARRAY_HEADER(h, 3, dim, fileIO);
  for(int i=0; i<dim[0]*dim[1]*dim[2]; i++) {
    for(int v=0; v<HYDRO_VAR_COUNT; v++) {
      fileIO.write(&hydro_array->k_h_h(i, v), 1);
    }
    // Additional padding to match legacy structures
    double _pad = 0;
    fileIO.write(&_pad, 2);
  }
#endif
  if (fileIO.close())
    ERROR(("File close failed on dump hydro!!!"));
}

// Dump particles in binary format
void BinaryDump::dump_particles(
    const char *fbase,
    int step,
    species_t *sp,
    grid_t *grid,
    interpolator_array_t *interpolator_array,
    int ftag)
{
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[1];
  size_t buf_start;
  static particle_t *ALIGNED(128) p_buf = NULL;

  // TODO: reconcile this with MAX_IO_CHUNK, and update Cmake option
  // description to explain what backends use it
#define PBUF_SIZE 32768 // 1MB of particles

  if (!sp)
    ERROR(("Invalid species name \"%s\".", sp->name));

  if (!fbase)
    ERROR(("Invalid filename"));

  // Update the particles on the host only if they haven't been recently
  if (step > sp->last_copied)
    sp->copy_to_host();

  // Update interpolators on host
  interpolator_array->copy_to_host();

  if (!p_buf)
    MALLOC_ALIGNED(p_buf, PBUF_SIZE, 128);

  if (rank == 0)
    MESSAGE(("Dumping \"%s\" particles to \"%s\"", sp->name, fbase));

  if (ftag)
    snprintf(fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step, rank);
  else
    snprintf(fname, max_filename_bytes, "%s.%i", fbase, rank);
  FileIOStatus status = fileIO.open(fname, io_write);
  if (status == fail)
    ERROR(("Could not open \"%s\"", fname));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = grid->nx;
  size_t nyout = grid->ny;
  size_t nzout = grid->nz;
  float dxout = grid->dx;
  float dyout = grid->dy;
  float dzout = grid->dz;

  WRITE_HEADER_V0(dump_type::particle_dump, sp->id, sp->q / sp->m, fileIO, step, rank, nproc);

  dim[0] = sp->np;
  WRITE_ARRAY_HEADER(p_buf, 1, dim, fileIO);

  // Copy a PBUF_SIZE hunk of the particle list into the particle
  // buffer, timecenter it and write it out. This is done this way to
  // guarantee the particle list unchanged while not requiring too
  // much memory.

  // FIXME: WITH A PIPELINED CENTER_P, PBUF NOMINALLY SHOULD BE QUITE
  // LARGE.

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  particle_t *sp_p = sp->p;
  sp->p = p_buf;
  size_t sp_np = sp->np;
  sp->np = 0;
  size_t sp_max_np = sp->max_np;
  sp->max_np = PBUF_SIZE;
  for (buf_start = 0; buf_start < sp_np; buf_start += PBUF_SIZE)
  {
    sp->np = sp_np - buf_start;
    if (sp->np > PBUF_SIZE)
        sp->np = PBUF_SIZE;
    //COPY(sp->p, &sp_p[buf_start], sp->np);
    Kokkos::parallel_for("Copy particles to write buffer", 
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, size_t>(static_cast<size_t>(0), sp->np), 
      KOKKOS_LAMBDA(const size_t idx) {
      sp->p[idx].dx = sp->k_p_h(idx+buf_start, particle_var::dx);
      sp->p[idx].dy = sp->k_p_h(idx+buf_start, particle_var::dy);
      sp->p[idx].dz = sp->k_p_h(idx+buf_start, particle_var::dz);
      sp->p[idx].i  = sp->k_p_i_h(idx+buf_start);
      sp->p[idx].ux = sp->k_p_h(idx+buf_start, particle_var::ux);
      sp->p[idx].uy = sp->k_p_h(idx+buf_start, particle_var::uy);
      sp->p[idx].uz = sp->k_p_h(idx+buf_start, particle_var::uz);
      sp->p[idx].w  = sp->k_p_h(idx+buf_start, particle_var::w);
#ifdef VARIABLE_CHARGE
      sp->p[idx].qp = sp->k_p_h(idx+buf_start, particle_var::qp);
#endif
    });
    center_p(sp, interpolator_array);
    fileIO.write(sp->p, sp->np);
  }
  sp->p = sp_p;
  sp->np = sp_np;
  sp->max_np = sp_max_np;
#else
  size_t p_buf_np = 0;
  center_p(sp, interpolator_array);
  Kokkos::fence();
  Kokkos::View<particle_t*, Kokkos::DefaultHostExecutionSpace, 
               Kokkos::MemoryTraits<Kokkos::Unmanaged> > p_buffer(p_buf, PBUF_SIZE);
  for (buf_start = 0; buf_start < sp->np; buf_start += PBUF_SIZE)
  {
    const size_t p_buf_np = std::min(sp->np - buf_start, static_cast<size_t>(PBUF_SIZE));
    Kokkos::parallel_for("Copy particles to write buffer", 
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, size_t>(static_cast<size_t>(0), p_buf_np), 
      KOKKOS_LAMBDA(const size_t idx) {
      p_buffer(idx).dx = sp->k_p_h(idx+buf_start, particle_var::dx);
      p_buffer(idx).dy = sp->k_p_h(idx+buf_start, particle_var::dy);
      p_buffer(idx).dz = sp->k_p_h(idx+buf_start, particle_var::dz);
      p_buffer(idx).i  = sp->k_p_i_h(idx+buf_start);
      p_buffer(idx).ux = sp->k_p_h(idx+buf_start, particle_var::ux);
      p_buffer(idx).uy = sp->k_p_h(idx+buf_start, particle_var::uy);
      p_buffer(idx).uz = sp->k_p_h(idx+buf_start, particle_var::uz);
      p_buffer(idx).w  = sp->k_p_h(idx+buf_start, particle_var::w);
#ifdef VARIABLE_CHARGE
      p_buffer(idx).qp = sp->k_p_h(idx+buf_start, particle_var::qp);
#endif
    });
    Kokkos::fence();
    fileIO.write(p_buffer.data(), p_buf_np);
  }
  uncenter_p(sp, interpolator_array);
#endif

  if(p_buf)
    FREE_ALIGNED(p_buf);

  if (fileIO.close())
    ERROR(("File close failed on dump particles!!!"));
}

// Dump fluids in binary format
void BinaryDump::dump_fluids(
    const char *fbase,
    int step,
    fluid_species_t *fsp,
    grid_t *grid,
    int ftag)
{
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[3];

  if( !fsp ) ERROR(( "Invalid fluid species \"%s\"", fsp->name ));

  if (step > fsp->last_copied)  fsp->copy_to_host();


  if( !fbase ) ERROR(( "Invalid filename" ));

  if( rank==0 )
    MESSAGE(("Dumping \"%s\" fluid species to \"%s\"",fsp->name,fbase));

  if( ftag ) {
      snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step, rank );
  }
  else {
      snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank );
  }

  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail) ERROR(( "Could not open \"%s\".", fname ));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = grid->nx;
  size_t nyout = grid->ny;
  size_t nzout = grid->nz;
  float dxout = grid->dx;
  float dyout = grid->dy;
  float dzout = grid->dz;

  WRITE_HEADER_V0(dump_type::hydro_dump, fsp->id, fsp->q/fsp->m,
      fileIO, step, rank, nproc); // To-do: Needs changing?

  dim[0] = grid->nx+2; // To-do: Change if changing ghost cell #.
  dim[1] = grid->ny+2;
  dim[2] = grid->nz+2;
  WRITE_ARRAY_HEADER( fsp->fl, 3, dim, fileIO );
  fileIO.write( fsp->fl, dim[0]*dim[1]*dim[2] );
  if( fileIO.close() ) ERROR(( "File close failed on dump fluids!!!" ));
}

// Field dump in binary format
// Field dump in binary format
void BinaryDump::field_dump(
    DumpParameters & dumpParams,
    int step,
    grid_t *grid,
    field_array_t *field_array)
{

  // Update the fields if necessary
  if (step > field_array->last_copied)
    field_array->copy_to_host();

  // Create directory for this time step
  char timeDir[max_filename_bytes];
  int ret = snprintf(timeDir, max_filename_bytes, "%s/T.%ld", dumpParams.baseDir, (long)step);
  if (ret < 0) {
      ERROR(("snprintf failed"));
  }
  FileUtils::makeDirectory(timeDir);

  // Open the file for output
  char filename[max_filename_bytes];
  ret = snprintf(filename, max_filename_bytes, "%s/T.%ld/%s.%ld.%d", dumpParams.baseDir, (long)step,
          dumpParams.baseFileName, (long)step, rank);
  if (ret < 0) {
      ERROR(("snprintf failed"));
  }

  FileIO fileIO;
  FileIOStatus status;

  status = fileIO.open(filename, io_write);
  if( status==fail ) ERROR(( "Failed opening file: %s", filename ));

  // convenience
  const size_t istride(dumpParams.stride_x);
  const size_t jstride(dumpParams.stride_y);
  const size_t kstride(dumpParams.stride_z);

  // Check stride values.
  if(remainder(grid->nx, istride) != 0)
    ERROR(("x stride must be an integer factor of nx"));
  if(remainder(grid->ny, jstride) != 0)
    ERROR(("y stride must be an integer factor of ny"));
  if(remainder(grid->nz, kstride) != 0)
    ERROR(("z stride must be an integer factor of nz"));

  int dim[3];

  /* define to do C-style indexing */
# define f(x,y,z) f[ VOXEL(x,y,z, grid->nx,grid->ny,grid->nz) ]

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = (grid->nx)/istride;
  size_t nyout = (grid->ny)/jstride;
  size_t nzout = (grid->nz)/kstride;
  float dxout = (grid->dx)*istride;
  float dyout = (grid->dy)*jstride;
  float dzout = (grid->dz)*kstride;

  /* CREATE TRANSFORMED COPY OF FIELD ARRAY */
  dim[0] = grid->nx + 2;
  dim[1] = grid->ny + 2;
  dim[2] = grid->nz + 2;
  
  field_t *f_transformed;
  MALLOC(f_transformed, dim[0] * dim[1] * dim[2]);
  COPY(f_transformed, field_array->f, dim[0] * dim[1] * dim[2]);
  
  // APPLY TRANSFORMATION TO CURRENT DENSITIES
  // APPLY TRANSFORMATION TO CURRENT DENSITIES, ELECTRIC AND MAGNETIC FIELDS
for(int k = 0; k < dim[2]; k++) {
  for(int j = 0; j < dim[1]; j++) {
    for(int i = 0; i < dim[0]; i++) {
      int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
      
      // Transform electric field (COVARIANT: E)
      float E_xi   = field_array->f[idx].ex;
      float E_eta  = field_array->f[idx].ey;
      float E_zeta = field_array->f[idx].ez;
      
      float E_phys_xi, E_phys_eta, E_phys_zeta;
      transform_E_to_physical(grid, idx, E_xi, E_eta, E_zeta,
                             E_phys_xi, E_phys_eta, E_phys_zeta);
      
      f_transformed[idx].ex = E_phys_xi;
      f_transformed[idx].ey = E_phys_eta;
      f_transformed[idx].ez = E_phys_zeta;
      
      // Transform magnetic field (CONTRAVARIANT: B)
      float B_xi   = field_array->f[idx].cbx;
      float B_eta  = field_array->f[idx].cby;
      float B_zeta = field_array->f[idx].cbz;
      
      float B_phys_xi, B_phys_eta, B_phys_zeta;
      transform_B_to_physical(grid, idx, B_xi, B_eta, B_zeta,
                             B_phys_xi, B_phys_eta, B_phys_zeta);
      
      f_transformed[idx].cbx = B_phys_xi;
      f_transformed[idx].cby = B_phys_eta;
      f_transformed[idx].cbz = B_phys_zeta;
      
      // Transform old magnetic field (CONTRAVARIANT: B0)
      B_xi   = field_array->f[idx].cbx0;
      B_eta  = field_array->f[idx].cby0;
      B_zeta = field_array->f[idx].cbz0;
      
      transform_B_to_physical(grid, idx, B_xi, B_eta, B_zeta,
                             B_phys_xi, B_phys_eta, B_phys_zeta);
      
      f_transformed[idx].cbx0 = B_phys_xi;
      f_transformed[idx].cby0 = B_phys_eta;
      f_transformed[idx].cbz0 = B_phys_zeta;
      
      // Transform current currents (CONTRAVARIANT: J) - existing code
      float J_xi   = field_array->f[idx].jfx;
      float J_eta  = field_array->f[idx].jfy;
      float J_zeta = field_array->f[idx].jfz;
      
      float J_phys_xi, J_phys_eta, J_phys_zeta;
      transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                             J_phys_xi, J_phys_eta, J_phys_zeta);
      
      f_transformed[idx].jfx = J_phys_xi;
      f_transformed[idx].jfy = J_phys_eta;
      f_transformed[idx].jfz = J_phys_zeta;
      
      // Transform old currents (CONTRAVARIANT: Jold) - existing code
      J_xi   = field_array->f[idx].jfxold;
      J_eta  = field_array->f[idx].jfyold;
      J_zeta = field_array->f[idx].jfzold;
      
      transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                             J_phys_xi, J_phys_eta, J_phys_zeta);
      
      f_transformed[idx].jfxold = J_phys_xi;
      f_transformed[idx].jfyold = J_phys_eta;
      f_transformed[idx].jfzold = J_phys_zeta;
    }
  }
}

  /* Banded output will write data as a single block-array as opposed to
   * the Array-of-Structure format that is used for native storage.
   *
   * Additionally, the user can specify a stride pattern to reduce
   * the resolution of the data that are output.  If a stride is
   * specified for a particular dimension, VPIC will write the boundary
   * plus every "stride" elements in that dimension. */

  if(dumpParams.format == band) {

    WRITE_HEADER_V0(dump_type::field_dump, -1, 0, fileIO, step, rank, nproc);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

    if( rank==VERBOSE_rank ) {
      std::cerr << "nxout: " << nxout << std::endl;
      std::cerr << "nyout: " << nyout << std::endl;
      std::cerr << "nzout: " << nzout << std::endl;
      std::cerr << "nx: " << grid->nx << std::endl;
      std::cerr << "ny: " << grid->ny << std::endl;
      std::cerr << "nz: " << grid->nz << std::endl;
    }

    WRITE_ARRAY_HEADER(f_transformed, 3, dim, fileIO);

    // Create a variable list of field values to output.
    size_t numvars = std::min(dumpParams.output_vars.bitsum(),
                              total_field_variables);
    size_t * varlist = new size_t[numvars];

    for(size_t i(0), c(0); i<total_field_variables; i++)
      if(dumpParams.output_vars.bitset(i)) varlist[c++] = i;

    if( rank==VERBOSE_rank ) printf("\nBEGIN_OUTPUT\n");

    // more efficient for standard case
    if(istride == 1 && jstride == 1 && kstride == 1)
      for(size_t v(0); v<numvars; v++) {
      for(size_t k(0); k<nzout+2; k++) {
      for(size_t j(0); j<nyout+2; j++) {
      for(size_t i(0); i<nxout+2; i++) {
              const uint32_t * fref = reinterpret_cast<uint32_t *>(&f_transformed[VOXEL(i,j,k,grid->nx,grid->ny,grid->nz)]);
              fileIO.write(&fref[varlist[v]], 1);
              if(rank==VERBOSE_rank) printf("%f ", f_transformed[VOXEL(i,j,k,grid->nx,grid->ny,grid->nz)].ex);
              if(rank==VERBOSE_rank) std::cout << "(" << i << " " << j << " " << k << ")" << std::endl;
      } if(rank==VERBOSE_rank) std::cout << std::endl << "ROW_BREAK " << j << " " << k << std::endl;
      } if(rank==VERBOSE_rank) std::cout << std::endl << "PLANE_BREAK " << k << std::endl;
      } if(rank==VERBOSE_rank) std::cout << std::endl << "BLOCK_BREAK" << std::endl;
      }

    else

      for(size_t v(0); v<numvars; v++) {
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
              const uint32_t * fref = reinterpret_cast<uint32_t *>(&f_transformed[VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz)]);
              fileIO.write(&fref[varlist[v]], 1);
              if(rank==VERBOSE_rank) printf("%f ", f_transformed[VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz)].ex);
              if(rank==VERBOSE_rank) std::cout << "(" << ioff << " " << joff << " " << koff << ")" << std::endl;
      } if(rank==VERBOSE_rank) std::cout << std::endl << "ROW_BREAK " << joff << " " << koff << std::endl;
      } if(rank==VERBOSE_rank) std::cout << std::endl << "PLANE_BREAK " << koff << std::endl;
      } if(rank==VERBOSE_rank) std::cout << std::endl << "BLOCK_BREAK" << std::endl;
      }

    delete[] varlist;

  } else { // band_interleave

    WRITE_HEADER_V0(dump_type::field_dump, -1, 0, fileIO, step, rank, nproc);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

    WRITE_ARRAY_HEADER(f_transformed, 3, dim, fileIO);

    if(istride == 1 && jstride == 1 && kstride == 1)
      fileIO.write(f_transformed, dim[0]*dim[1]*dim[2]);
    else
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
            fileIO.write(&f_transformed[VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz)], 1);
      }
      }
      }
  }

# undef f

  FREE(f_transformed);

  if( fileIO.close() ) ERROR(( "File close failed on field dump!!!" ));
}

// Hydro dump in binary format
// Hydro dump in binary format
void BinaryDump::hydro_dump(
    DumpParameters& dumpParams,
    int step,
    species_t *sp,
    grid_t *grid,
    hydro_array_t *hydro_array,
    interpolator_array_t *interpolator_array)
{
  using hydro_scalar_t = k_hydro_t::non_const_value_type;
  // Create directory for this time step
  char timeDir[max_filename_bytes];
  snprintf(timeDir, max_filename_bytes, "%s/T.%ld", dumpParams.baseDir, (long)step);
  FileUtils::makeDirectory(timeDir);

  // Open the file for output
  char filename[max_filename_bytes];
  int ret = snprintf( filename, max_filename_bytes, "%s/T.%ld/%s.%ld.%d", dumpParams.baseDir, (long)step,
           dumpParams.baseFileName, (long)step, rank );
  if (ret < 0) {
      ERROR(("snprintf failed"));
  }

  FileIO fileIO;
  FileIOStatus status;

  status = fileIO.open(filename, io_write);
  if(status == fail) ERROR(("Failed opening file: %s", filename));

  if( !sp ) ERROR(( "Invalid species name: %s", sp->name ));

  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;

  Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos(
      particles,
      particles_i,
      hydro_array->k_h_d,
      interpolators_k,
      sp
  );

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_array->copy_to_host();
  synchronize_hydro_array( hydro_array );
#else
  // This does not give consistent results
  synchronize_hydro_array_kokkos(hydro_array);
  hydro_array->copy_to_host();
#endif

  // convenience
  const size_t istride(dumpParams.stride_x);
  const size_t jstride(dumpParams.stride_y);
  const size_t kstride(dumpParams.stride_z);

  // Check stride values.
  if(remainder(grid->nx, istride) != 0)
    ERROR(("x stride must be an integer factor of nx"));
  if(remainder(grid->ny, jstride) != 0)
    ERROR(("y stride must be an integer factor of ny"));
  if(remainder(grid->nz, kstride) != 0)
    ERROR(("z stride must be an integer factor of nz"));

  int dim[3];

  /* define to do C-style indexing */
# define hydro(x,y,z) hydro_array->h[VOXEL(x,y,z, grid->nx,grid->ny,grid->nz)]

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = (grid->nx)/istride;
  size_t nyout = (grid->ny)/jstride;
  size_t nzout = (grid->nz)/kstride;
  float dxout = (grid->dx)*istride;
  float dyout = (grid->dy)*jstride;
  float dzout = (grid->dz)*kstride;

  /* CREATE TRANSFORMED COPY OF HYDRO ARRAY */
  dim[0] = grid->nx + 2;
  dim[1] = grid->ny + 2;
  dim[2] = grid->nz + 2;

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_t *h_transformed;
  MALLOC(h_transformed, dim[0] * dim[1] * dim[2]);
  COPY(h_transformed, hydro_array->h, dim[0] * dim[1] * dim[2]);
  
  // APPLY TRANSFORMATION TO CURRENT DENSITIES
  for(int k = 0; k < dim[2]; k++) {
    for(int j = 0; j < dim[1]; j++) {
      for(int i = 0; i < dim[0]; i++) {
        int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
        
        float J_xi   = hydro_array->h[idx].jx;
        float J_eta  = hydro_array->h[idx].jy;
        float J_zeta = hydro_array->h[idx].jz;
        
        float J_phys_xi, J_phys_eta, J_phys_zeta;
        transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                               J_phys_xi, J_phys_eta, J_phys_zeta);
        
        h_transformed[idx].jx = J_phys_xi;
        h_transformed[idx].jy = J_phys_eta;
        h_transformed[idx].jz = J_phys_zeta;
      }
    }
  }
#else
  // For non-legacy, create a transformed view
  k_hydro_t h_transformed("h_transformed", dim[0] * dim[1] * dim[2], HYDRO_VAR_COUNT);
  Kokkos::deep_copy(h_transformed, hydro_array->k_h_h);
  
  // Transform on host
  for(int k = 0; k < dim[2]; k++) {
    for(int j = 0; j < dim[1]; j++) {
      for(int i = 0; i < dim[0]; i++) {
        int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
        
        float J_xi   = hydro_array->k_h_h(idx, hydro_var::jx);
        float J_eta  = hydro_array->k_h_h(idx, hydro_var::jy);
        float J_zeta = hydro_array->k_h_h(idx, hydro_var::jz);
        
        float J_phys_xi, J_phys_eta, J_phys_zeta;
        transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                               J_phys_xi, J_phys_eta, J_phys_zeta);
        
        h_transformed(idx, hydro_var::jx) = J_phys_xi;
        h_transformed(idx, hydro_var::jy) = J_phys_eta;
        h_transformed(idx, hydro_var::jz) = J_phys_zeta;
      }
    }
  }
#endif

  /* Banded output will write data as a single block-array as opposed to
   * the Array-of-Structure format that is used for native storage.
   *
   * Additionally, the user can specify a stride pattern to reduce
   * the resolution of the data that are output.  If a stride is
   * specified for a particular dimension, VPIC will write the boundary
   * plus every "stride" elements in that dimension.
   */
  if(dumpParams.format == band) {

    WRITE_HEADER_V0(dump_type::hydro_dump, sp->id, sp->q / sp->m, fileIO, step, rank, nproc);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
    WRITE_ARRAY_HEADER(h_transformed, 3, dim, fileIO);
#else
    hydro_t h[1];
    WRITE_ARRAY_HEADER(h, 3, dim, fileIO);
#endif

    /*
     * Create a variable list of hydro values to output.
     */
    size_t numvars = std::min(dumpParams.output_vars.bitsum(),
                              total_hydro_variables);
    size_t * varlist = new size_t[numvars];
    for(size_t i(0), c(0); i<total_hydro_variables; i++)
      if( dumpParams.output_vars.bitset(i) ) varlist[c++] = i;

    // More efficient for standard case
    if(istride == 1 && jstride == 1 && kstride == 1)

      for(size_t v(0); v<numvars; v++)
      for(size_t k(0); k<nzout+2; k++)
      for(size_t j(0); j<nyout+2; j++)
      for(size_t i(0); i<nxout+2; i++) {
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
              const uint32_t * href = reinterpret_cast<uint32_t *>(&h_transformed[VOXEL(i,j,k,grid->nx,grid->ny,grid->nz)]);
              fileIO.write(&href[varlist[v]], 1);
#else
              fileIO.write(&(h_transformed(VOXEL(i,j,k,grid->nx,grid->ny,grid->nz), varlist[v])), 1);
#endif
      }

    else

      for(size_t v(0); v<numvars; v++)
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
              const uint32_t * href = reinterpret_cast<uint32_t *>(&h_transformed[VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz)]);
              fileIO.write(&href[varlist[v]], 1);
#else
              fileIO.write(&(h_transformed(VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz), varlist[v])), 1);
#endif
      }
      }
      }

    delete[] varlist;

  } else { // band_interleave

    WRITE_HEADER_V0(dump_type::hydro_dump, sp->id, sp->q / sp->m, fileIO, step, rank, nproc);

    dim[0] = nxout;
    dim[1] = nyout;
    dim[2] = nzout;

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
    WRITE_ARRAY_HEADER(h_transformed, 3, dim, fileIO);
#else
    hydro_t h[1];
    WRITE_ARRAY_HEADER(h, 3, dim, fileIO);
#endif

    if(istride == 1 && jstride == 1 && kstride == 1) {

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
      fileIO.write(h_transformed, dim[0] * dim[1] * dim[2]);
#else
      if( std::is_same<Kokkos::LayoutRight, k_hydro_t::array_layout>::value ) {
        fileIO.write(h_transformed.data(), dim[0] * dim[1] * dim[2]);
      } else {
        for(int i=0; i<dim[0]; i++) {
          for(int j=0; j<dim[1]; j++) {
            for(int k=0; k<dim[2]; k++) {
              for(size_t v=0; v<HYDRO_VAR_COUNT; v++) {
                fileIO.write(&h_transformed(VOXEL(i,j,k,grid->nx,grid->ny,grid->nz), v), 1);
              }
              hydro_scalar_t _pad = 0;
              fileIO.write(&_pad, 1);
              fileIO.write(&_pad, 1);
            }
          }
        }
      }
#endif

    } else {

      for(size_t k(0); k<nzout; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
            fileIO.write(&h_transformed[VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz)], 1);
#else
        for(size_t v=0; v<HYDRO_VAR_COUNT; v++) 
          fileIO.write(&(h_transformed(VOXEL(ioff,joff,koff,grid->nx,grid->ny,grid->nz), v)), 1);
#endif
      }
      }
      }
    }
  }

# undef hydro

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  FREE(h_transformed);
#endif

  if( fileIO.close() ) ERROR(( "File close failed on hydro dump!!!" ));
}


// Fluid dump in binary format
void BinaryDump::fluid_dump(
    DumpParameters& dumpParams,
    int step,
    fluid_species_t *fsp,
    grid_t *grid)
{

  // Create directory for this time step
  char timeDir[max_filename_bytes];
  snprintf(timeDir, max_filename_bytes, "%s/T.%ld", dumpParams.baseDir, (long)step);
  FileUtils::makeDirectory(timeDir);

  // Open the file for output
  char filename[max_filename_bytes];
  int ret = snprintf( filename, max_filename_bytes, "%s/T.%ld/%s.%ld.%d", dumpParams.baseDir, (long)step,
           dumpParams.baseFileName, (long)step, rank );
  if (ret < 0) {
      ERROR(("snprintf failed"));
  }

  FileIO fileIO;
  FileIOStatus status;

  status = fileIO.open(filename, io_write);
  if(status == fail) ERROR(("Failed opening file: %s", filename));

  if( !fsp ) ERROR(( "Invalid fluid species name: %s", fsp->name ));

  // This does not give consistent results
  //synchronize_hydro_array_kokkos(hydro_array);

  if (step > fsp->last_copied)
    fsp->copy_to_host();

  //synchronize_hydro_array( hydro_array );

  // convenience
  const size_t istride(dumpParams.stride_x);
  const size_t jstride(dumpParams.stride_y);
  const size_t kstride(dumpParams.stride_z);

  // Check stride values.
  if(remainder(grid->nx, istride) != 0)
    ERROR(("x stride must be an integer factor of nx"));
  if(remainder(grid->ny, jstride) != 0)
    ERROR(("y stride must be an integer factor of ny"));
  if(remainder(grid->nz, kstride) != 0)
    ERROR(("z stride must be an integer factor of nz"));

  int dim[3];

  /* define to do C-style indexing */
# define fluid(x,y,z) fsp->fl[VOXEL(x,y,z, grid->nx,grid->ny,grid->nz)]

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  size_t nxout = (grid->nx)/istride;
  size_t nyout = (grid->ny)/jstride;
  size_t nzout = (grid->nz)/kstride;
  float dxout = (grid->dx)*istride;
  float dyout = (grid->dy)*jstride;
  float dzout = (grid->dz)*kstride;

  /* Banded output will write data as a single block-array as opposed to
   * the Array-of-Structure format that is used for native storage.
   *
   * Additionally, the user can specify a stride pattern to reduce
   * the resolution of the data that are output.  If a stride is
   * specified for a particular dimension, VPIC will write the boundary
   * plus every "stride" elements in that dimension.
   */
  if(dumpParams.format == band) {

    WRITE_HEADER_V0(dump_type::hydro_dump, fsp->id, fsp->q / fsp->m, fileIO, step, rank, nproc);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

    WRITE_ARRAY_HEADER(fsp->fl, 3, dim, fileIO);

    /*
     * Create a variable list of hydro values to output.
     */
    size_t numvars = std::min(dumpParams.output_vars.bitsum(),
                              total_fluid_variables);  // To-do: Define this
    size_t * varlist = new size_t[numvars];
    for(size_t i(0), c(0); i<total_fluid_variables; i++)
      if( dumpParams.output_vars.bitset(i) ) varlist[c++] = i;

    // More efficient for standard case
    if(istride == 1 && jstride == 1 && kstride == 1)

      for(size_t v(0); v<numvars; v++)
      for(size_t k(0); k<nzout+2; k++)
      for(size_t j(0); j<nyout+2; j++)
      for(size_t i(0); i<nxout+2; i++) {
              const uint32_t * flref = reinterpret_cast<uint32_t *>(&fluid(i,j,k));
              fileIO.write(&flref[varlist[v]], 1);
      }

    else

      for(size_t v(0); v<numvars; v++)
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
              const uint32_t * flref = reinterpret_cast<uint32_t *>(&fluid(ioff,joff,koff));
              fileIO.write(&flref[varlist[v]], 1);
      }
      }
      }

    delete[] varlist;

  } else { // band_interleave

    WRITE_HEADER_V0(dump_type::hydro_dump, fsp->id, fsp->q / fsp->m, fileIO, step, rank, nproc);

    dim[0] = nxout;
    dim[1] = nyout;
    dim[2] = nzout;

    WRITE_ARRAY_HEADER(fsp->fl, 3, dim, fileIO);

    if(istride == 1 && jstride == 1 && kstride == 1)

      fileIO.write(fsp->fl, dim[0]*dim[1]*dim[2]);

    else

      for(size_t k(0); k<nzout; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
            fileIO.write(&fluid(ioff,joff,koff), 1);
      }
      }
      }
  }

# undef fluid

  if( fileIO.close() ) ERROR(( "File close failed on fluid dump!!!" ));
}

#ifdef VPIC_ENABLE_HDF5

/*****************************************************************************
 * HDF5 dump IO
 *****************************************************************************/

// Dump fields in HDF5 format
void HDF5Dump::dump_fields(
    const char *fbase,
    int step,
    grid_t *grid,
    field_array_t *field_array,
    int ftag)
{
  if ( rank==0 ) log_printf("Dumping fields using HDF5\n");

  // Update the fields if necessary
  if (step > field_array->last_copied)
    field_array->copy_to_host();

  /* CREATE TRANSFORMED COPY OF FIELD ARRAY */
  int dim[3];
  dim[0] = grid->nx + 2;
  dim[1] = grid->ny + 2;
  dim[2] = grid->nz + 2;
  
  field_t *f_transformed;
  MALLOC(f_transformed, dim[0] * dim[1] * dim[2]);
  COPY(f_transformed, field_array->f, dim[0] * dim[1] * dim[2]);
  
  // APPLY TRANSFORMATION TO CURRENT DENSITIES
  // APPLY TRANSFORMATION TO CURRENT DENSITIES, ELECTRIC AND MAGNETIC FIELDS
for(int k = 0; k < dim[2]; k++) {
  for(int j = 0; j < dim[1]; j++) {
    for(int i = 0; i < dim[0]; i++) {
      int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
      
      // Transform electric field (COVARIANT: E)
      float E_xi   = field_array->f[idx].ex;
      float E_eta  = field_array->f[idx].ey;
      float E_zeta = field_array->f[idx].ez;
      
      float E_phys_xi, E_phys_eta, E_phys_zeta;
      transform_E_to_physical(grid, idx, E_xi, E_eta, E_zeta,
                             E_phys_xi, E_phys_eta, E_phys_zeta);
      
      f_transformed[idx].ex = E_phys_xi;
      f_transformed[idx].ey = E_phys_eta;
      f_transformed[idx].ez = E_phys_zeta;
      
      // Transform magnetic field (CONTRAVARIANT: B)
      float B_xi   = field_array->f[idx].cbx;
      float B_eta  = field_array->f[idx].cby;
      float B_zeta = field_array->f[idx].cbz;
      
      float B_phys_xi, B_phys_eta, B_phys_zeta;
      transform_B_to_physical(grid, idx, B_xi, B_eta, B_zeta,
                             B_phys_xi, B_phys_eta, B_phys_zeta);
      
      f_transformed[idx].cbx = B_phys_xi;
      f_transformed[idx].cby = B_phys_eta;
      f_transformed[idx].cbz = B_phys_zeta;
      
      // Transform old magnetic field (CONTRAVARIANT: B0)
      B_xi   = field_array->f[idx].cbx0;
      B_eta  = field_array->f[idx].cby0;
      B_zeta = field_array->f[idx].cbz0;
      
      transform_B_to_physical(grid, idx, B_xi, B_eta, B_zeta,
                             B_phys_xi, B_phys_eta, B_phys_zeta);
      
      f_transformed[idx].cbx0 = B_phys_xi;
      f_transformed[idx].cby0 = B_phys_eta;
      f_transformed[idx].cbz0 = B_phys_zeta;
      
      // Transform current currents (CONTRAVARIANT: J)
      float J_xi   = field_array->f[idx].jfx;
      float J_eta  = field_array->f[idx].jfy;
      float J_zeta = field_array->f[idx].jfz;
      
      float J_phys_xi, J_phys_eta, J_phys_zeta;
      transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                             J_phys_xi, J_phys_eta, J_phys_zeta);
      
      f_transformed[idx].jfx = J_phys_xi;
      f_transformed[idx].jfy = J_phys_eta;
      f_transformed[idx].jfz = J_phys_zeta;
      
      // Transform old currents (CONTRAVARIANT: Jold)
      J_xi   = field_array->f[idx].jfxold;
      J_eta  = field_array->f[idx].jfyold;
      J_zeta = field_array->f[idx].jfzold;
      
      transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                             J_phys_xi, J_phys_eta, J_phys_zeta);
      
      f_transformed[idx].jfxold = J_phys_xi;
      f_transformed[idx].jfyold = J_phys_eta;
      f_transformed[idx].jfzold = J_phys_zeta;
    }
  }
}

#define fpp(x, y, z) f_transformed[VOXEL(x, y, z, grid->nx, grid->ny, grid->nz)]

#define DUMP_FIELD_TO_HDF5(DSET_NAME, ATTRIBUTE_NAME, ELEMENT_TYPE)                                         \
{                                                                                                           \
  dset_id = H5Dcreate(group_id, DSET_NAME, ELEMENT_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
  temp_buf_index = 0;                                                                                       \
  for (int i(stride_x); i < grid->nx + 1; i += stride_x)                                                    \
  {                                                                                                         \
    for (int j(stride_y); j < grid->ny + 1; j += stride_y)                                                  \
    {                                                                                                       \
      for (int k(stride_z); k < grid->nz + 1; k += stride_z)                                                \
      {                                                                                                     \
        temp_buf[temp_buf_index] = fpp(i, j, k).ATTRIBUTE_NAME;                                             \
        temp_buf_index = temp_buf_index + 1;                                                                \
      }                                                                                                     \
    }                                                                                                       \
  }                                                                                                         \
  dataspace_id = H5Dget_space(dset_id);                                                                     \
  H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, global_offset, NULL, field_local_size, NULL);           \
  H5Dwrite(dset_id, ELEMENT_TYPE, memspace, dataspace_id, plist_id, temp_buf);                              \
  H5Sclose(dataspace_id);                                                                                   \
  H5Dclose(dset_id);                                                                                        \
}

  //char fname[256];
  //char field_scratch[128];
  //char subfield_scratch[256];

  // create the directory and sub-directory
  std::string field_dir = "./fields_hdf5";
  FileUtils::makeDirectory(field_dir.c_str());
  std::string subfield_dir = field_dir + "/T." + std::to_string(step) + "/";
  FileUtils::makeDirectory(subfield_dir.c_str());

  // create the file
  std::string filename = subfield_dir + "/fields_" + std::to_string(step) + ".h5";
  double el1 = uptime();
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  // create the group for the time step
  std::string group_name = "Timestep_" + std::to_string(step);
  hid_t group_id = H5Gcreate(file_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  el1 = uptime() - el1;
  if ( rank==0 ) log_printf("TimeHDF5Open: %.2f s\n", el1);
  double el2 = uptime();

  // prepare for writing the data
  float *temp_buf = (float *)malloc(sizeof(float) * (grid->nx / stride_x) *
                                                    (grid->ny / stride_y) *
                                                    (grid->nz / stride_z));
  hsize_t temp_buf_index;
  hid_t dset_id;
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // data topology
  hsize_t field_global_size[3], field_local_size[3], global_offset[3];
  field_global_size[0] = (grid->nx * grid->gpx) / stride_x;
  field_global_size[1] = (grid->ny * grid->gpy) / stride_y;
  field_global_size[2] = (grid->nz * grid->gpz) / stride_z;

  field_local_size[0] = grid->nx / stride_x;
  field_local_size[1] = grid->ny / stride_y;
  field_local_size[2] = grid->nz / stride_z;

  int _ix, _iy, _iz;
  _ix = rank;
  _iy = _ix / grid->gpx;
  _ix -= _iy * grid->gpx;
  _iz = _iy / grid->gpy;
  _iy -= _iz * grid->gpy;

  global_offset[0] = grid->nx * _ix / stride_x;
  global_offset[1] = grid->ny * _iy / stride_y;
  global_offset[2] = grid->nz * _iz / stride_z;

  // prepare the spaces for parallel writing
  hid_t filespace = H5Screate_simple(3, field_global_size, NULL);
  hid_t memspace = H5Screate_simple(3, field_local_size, NULL);
  hid_t dataspace_id;

  // write the data
  if (field_dump_flag.flags["ex"]) DUMP_FIELD_TO_HDF5("ex", ex, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["ey"]) DUMP_FIELD_TO_HDF5("ey", ey, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["ez"]) DUMP_FIELD_TO_HDF5("ez", ez, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["div_e_err"]) DUMP_FIELD_TO_HDF5("div_e_err", div_e_err, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["cbx"]) DUMP_FIELD_TO_HDF5("cbx", cbx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["cby"]) DUMP_FIELD_TO_HDF5("cby", cby, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["cbz"]) DUMP_FIELD_TO_HDF5("cbz", cbz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["pe"]) DUMP_FIELD_TO_HDF5("pe", pe, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["cbx0"]) DUMP_FIELD_TO_HDF5("cbx0", cbx0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["cby0"]) DUMP_FIELD_TO_HDF5("cby0", cby0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["cbz0"]) DUMP_FIELD_TO_HDF5("cbz0", cbz0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["te0"]) DUMP_FIELD_TO_HDF5("te0", te0, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["tcax"]) DUMP_FIELD_TO_HDF5("tcax", tcax, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["tcay"]) DUMP_FIELD_TO_HDF5("tcay", tcay, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["tcaz"]) DUMP_FIELD_TO_HDF5("tcaz", tcaz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["rhob"]) DUMP_FIELD_TO_HDF5("rhob", rhob, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["jfx"]) DUMP_FIELD_TO_HDF5("jfx", jfx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["jfy"]) DUMP_FIELD_TO_HDF5("jfy", jfy, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["jfz"]) DUMP_FIELD_TO_HDF5("jfz", jfz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["rhof"]) DUMP_FIELD_TO_HDF5("rhof", rhof, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["jfxold"]) DUMP_FIELD_TO_HDF5("jfxold", jfxold, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["jfyold"]) DUMP_FIELD_TO_HDF5("jfyold", jfyold, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["jfzold"]) DUMP_FIELD_TO_HDF5("jfzold", jfzold, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["rhofold"]) DUMP_FIELD_TO_HDF5("rhofold", rhofold, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["tx"]) DUMP_FIELD_TO_HDF5("tx", tx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["ty"]) DUMP_FIELD_TO_HDF5("ty", ty, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["tz"]) DUMP_FIELD_TO_HDF5("tz", tz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["te"]) DUMP_FIELD_TO_HDF5("te", te, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["ox"]) DUMP_FIELD_TO_HDF5("ox", ox, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["oy"]) DUMP_FIELD_TO_HDF5("oy", oy, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["oz"]) DUMP_FIELD_TO_HDF5("oz", oz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["oe"]) DUMP_FIELD_TO_HDF5("oe", oe, H5T_NATIVE_FLOAT);
  
  if (field_dump_flag.flags["pex"]) DUMP_FIELD_TO_HDF5("pex", pex, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["pey"]) DUMP_FIELD_TO_HDF5("pey", pey, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["pez"]) DUMP_FIELD_TO_HDF5("pez", pez, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["div_b_err"]) DUMP_FIELD_TO_HDF5("div_b_err", div_e_err, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["ux"]) DUMP_FIELD_TO_HDF5("ux", ux, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["uy"]) DUMP_FIELD_TO_HDF5("uy", uy, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["uz"]) DUMP_FIELD_TO_HDF5("uz", uz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["ue"]) DUMP_FIELD_TO_HDF5("ue", ue, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["sx"]) DUMP_FIELD_TO_HDF5("sx", sx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["sy"]) DUMP_FIELD_TO_HDF5("sy", sy, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["sz"]) DUMP_FIELD_TO_HDF5("sz", sz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["se"]) DUMP_FIELD_TO_HDF5("se", se, H5T_NATIVE_FLOAT);

#ifdef EXTERNAL_FORCE
  if (field_dump_flag.flags["Ex0"]) DUMP_FIELD_TO_HDF5("Ex0", Ex0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["Ey0"]) DUMP_FIELD_TO_HDF5("Ey0", Ey0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["Ez0"]) DUMP_FIELD_TO_HDF5("Ez0", Ez0, H5T_NATIVE_FLOAT);

  if (field_dump_flag.flags["Gx0"]) DUMP_FIELD_TO_HDF5("Gx0", Gx0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["Gy0"]) DUMP_FIELD_TO_HDF5("Gy0", Gy0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.flags["Gz0"]) DUMP_FIELD_TO_HDF5("Gz0", Gz0, H5T_NATIVE_FLOAT);
#endif

  el2 = uptime() - el2;
  if ( rank==0 ) log_printf("TimeHDF5Write: %.2f s\n", el2);

  double el3 = uptime();

  free(temp_buf);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

# undef fpp
  FREE(f_transformed);

  el3 = uptime() - el3;
  if ( rank==0 ) log_printf("TimeHDF5Close: %.2f s\n", el3);

  if (rank == 0) {
    char const *output_xml_file = "./fields_hdf5/hdf5_field.xdmf";
    char dimensions_3d[128];
    sprintf(dimensions_3d, "%lld %lld %lld", field_global_size[0], field_global_size[1], field_global_size[2]);
    char dimensions_4d[128];
    sprintf(dimensions_4d, "%lld %lld %lld %d", field_global_size[0], field_global_size[1], field_global_size[2], 3);
    char orignal[128];
    float dx = stride_x * grid->dx;
    float dy = stride_y * grid->dy;
    float dz = stride_z * grid->dz;
    float x0 = grid->x0 + dx;
    float y0 = grid->y0 + dy;
    float z0 = grid->z0 + dz;
    sprintf(orignal, "%f %f %f", x0, y0, z0);
    char dxdydz[128];
    sprintf(dxdydz, "%f %f %f", dx, dy, dz);

    int nframes = num_step / field_interval + 1;
    static int field_tframe = 0;
    if (field_tframe >= 1) {
      if (field_tframe == (nframes - 1)) {
        invert_field_xml_item(output_xml_file, "fields", step, dimensions_4d, dimensions_3d, 1);
      } else {
        invert_field_xml_item(output_xml_file, "fields", step, dimensions_4d, dimensions_3d, 0);
      }
    } else {
      create_file_with_header(output_xml_file, dimensions_3d, orignal, dxdydz, nframes, field_interval);
      if (field_tframe == (nframes - 1)) {
        invert_field_xml_item(output_xml_file, "fields", step, dimensions_4d, dimensions_3d, 1);
      } else {
        invert_field_xml_item(output_xml_file, "fields", step, dimensions_4d, dimensions_3d, 0);
      }
    }
    field_tframe++;
  }
}

// Dump hydro in HDF5 format
// Dump hydro in HDF5 format
void HDF5Dump::dump_hydro(
    const char *fbase,
    int step,
    species_t *sp,
    grid_t *grid,
    hydro_array_t *hydro_array,
    interpolator_array_t *interpolator_array,
    int ftag)
{
  // prepare the data
  if (!sp) ERROR(("Invalid species name: %s", sp->name));
  if ( rank==0 ) log_printf("Dumping hydro for %s using HDF5\n", sp->name);

  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;

  Kokkos::deep_copy(hydro_array->k_h_d, 0.0);
  accumulate_hydro_p_kokkos(
      particles,
      particles_i,
      hydro_array->k_h_d,
      interpolators_k,
      sp
  );

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_array->copy_to_host();
  synchronize_hydro_array( hydro_array );
#else
  synchronize_hydro_array_kokkos(hydro_array);
  hydro_array->copy_to_host();
#endif

  /* CREATE TRANSFORMED COPY OF HYDRO ARRAY */
  /* CREATE TRANSFORMED COPY OF HYDRO ARRAY */
  int dim[3];
  dim[0] = grid->nx + 2;
  dim[1] = grid->ny + 2;
  dim[2] = grid->nz + 2;

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  hydro_t *h_transformed;
  MALLOC(h_transformed, dim[0] * dim[1] * dim[2]);
  COPY(h_transformed, hydro_array->h, dim[0] * dim[1] * dim[2]);
  
  // APPLY TRANSFORMATION TO CURRENT DENSITIES
  for(int k = 0; k < dim[2]; k++) {
    for(int j = 0; j < dim[1]; j++) {
      for(int i = 0; i < dim[0]; i++) {
        int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
        
        float J_xi   = hydro_array->h[idx].jx;
        float J_eta  = hydro_array->h[idx].jy;
        float J_zeta = hydro_array->h[idx].jz;
        
        float J_phys_xi, J_phys_eta, J_phys_zeta;
        transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                               J_phys_xi, J_phys_eta, J_phys_zeta);
        
        h_transformed[idx].jx = J_phys_xi;
        h_transformed[idx].jy = J_phys_eta;
        h_transformed[idx].jz = J_phys_zeta;
      }
    }
  }
#define GET_HYDRO_VAR(HYDRO, VOXEL, VAR) h_transformed[VOXEL].VAR
#else
  // For non-legacy, create a transformed view
  k_hydro_t h_transformed("h_transformed", dim[0] * dim[1] * dim[2], HYDRO_VAR_COUNT);
  Kokkos::deep_copy(h_transformed, hydro_array->k_h_h);
  
  // Transform on host
  for(int k = 0; k < dim[2]; k++) {
    for(int j = 0; j < dim[1]; j++) {
      for(int i = 0; i < dim[0]; i++) {
        int idx = VOXEL(i, j, k, grid->nx, grid->ny, grid->nz);
        
        float J_xi   = hydro_array->k_h_h(idx, hydro_var::jx);
        float J_eta  = hydro_array->k_h_h(idx, hydro_var::jy);
        float J_zeta = hydro_array->k_h_h(idx, hydro_var::jz);
        
        float J_phys_xi, J_phys_eta, J_phys_zeta;
        transform_J_to_physical(grid, idx, J_xi, J_eta, J_zeta,
                               J_phys_xi, J_phys_eta, J_phys_zeta);
        
        h_transformed(idx, hydro_var::jx) = J_phys_xi;
        h_transformed(idx, hydro_var::jy) = J_phys_eta;
        h_transformed(idx, hydro_var::jz) = J_phys_zeta;
      }
    }
  }
#define GET_HYDRO_VAR(HYDRO, VOXEL, VAR) h_transformed(VOXEL, hydro_var::VAR)
#endif

#define DUMP_HYDRO_TO_HDF5(DSET_NAME, ATTRIBUTE_NAME, ELEMENT_TYPE)                                         \
{                                                                                                           \
  dset_id = H5Dcreate(group_id, DSET_NAME, ELEMENT_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
  temp_buf_index = 0;                                                                                       \
  for (int i(stride_x); i < grid->nx + 1; i += stride_x)                                                    \
  {                                                                                                         \
    for (int j(stride_y); j < grid->ny + 1; j += stride_y)                                                  \
    {                                                                                                       \
      for (int k(stride_z); k < grid->nz + 1; k += stride_z)                                                \
      {                                                                                                     \
        auto voxel = VOXEL(i,j,k,grid->nx, grid->ny, grid->nz);                                             \
        temp_buf[temp_buf_index] = GET_HYDRO_VAR(hydro_array, voxel, ATTRIBUTE_NAME);                       \
        temp_buf_index = temp_buf_index + 1;                                                                \
      }                                                                                                     \
    }                                                                                                       \
  }                                                                                                         \
  dataspace_id = H5Dget_space(dset_id);                                                                     \
  H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, global_offset, NULL, hydro_local_size, NULL);           \
  H5Dwrite(dset_id, ELEMENT_TYPE, memspace, dataspace_id, plist_id, temp_buf);                              \
  H5Sclose(dataspace_id);                                                                                   \
  H5Dclose(dset_id);                                                                                        \
}

  // create the directory and sub-directory
  std::string hydro_dir = "./hydro_hdf5";
  std::string subhydro_dir = hydro_dir + "/T." + std::to_string(step) + "/";
  FileUtils::makeDirectory(hydro_dir.c_str());
  FileUtils::makeDirectory(subhydro_dir.c_str());

  std::string hydro_fname = subhydro_dir + "/hydro_" + std::string(sp->name) + "_" + std::to_string(step) + ".h5";
  double el1 = uptime();
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(hydro_fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  std::string hydro_gname = "Timestep_" + std::to_string(step);
  hid_t group_id = H5Gcreate(file_id, hydro_gname.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  el1 = uptime() - el1;
  if ( rank==0 ) log_printf("TimeHDF5Open: %.2f s\n", el1);
  double el2 = uptime();

  // prepare for writing the data
  using val_type = k_hydro_t::non_const_value_type;
  val_type *temp_buf = (val_type *)malloc(sizeof(val_type) * 
                                          (grid->nx / stride_x) *
                                          (grid->ny / stride_y) *
                                          (grid->nz / stride_z));
  hsize_t temp_buf_index;
  hid_t dset_id;
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // data topology
  hsize_t hydro_global_size[3], hydro_local_size[3], global_offset[3];
  hydro_global_size[0] = (grid->nx * grid->gpx) / stride_x;
  hydro_global_size[1] = (grid->ny * grid->gpy) / stride_y;
  hydro_global_size[2] = (grid->nz * grid->gpz) / stride_z;

  hydro_local_size[0] = grid->nx / stride_x;
  hydro_local_size[1] = grid->ny / stride_y;
  hydro_local_size[2] = grid->nz / stride_z;

  int _ix, _iy, _iz;
  _ix = rank;
  _iy = _ix / grid->gpx;
  _ix -= _iy * grid->gpx;
  _iz = _iy / grid->gpy;
  _iy -= _iz * grid->gpy;

  global_offset[0] = (grid->nx) * _ix / stride_x;
  global_offset[1] = (grid->ny) * _iy / stride_y;
  global_offset[2] = (grid->nz) * _iz / stride_z;

  // prepare the spaces for parallel writing
  hid_t filespace = H5Screate_simple(3, hydro_global_size, NULL);
  hid_t memspace = H5Screate_simple(3, hydro_local_size, NULL);
  hid_t dataspace_id;

  // write the data
  if (hydro_dump_flag.flags["jx"]) DUMP_HYDRO_TO_HDF5("jx", jx, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["jy"]) DUMP_HYDRO_TO_HDF5("jy", jy, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["jz"]) DUMP_HYDRO_TO_HDF5("jz", jz, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["rho"]) DUMP_HYDRO_TO_HDF5("rho", rho, H5T_NATIVE_DOUBLE);

  if (hydro_dump_flag.flags["px"]) DUMP_HYDRO_TO_HDF5("px", px, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["py"]) DUMP_HYDRO_TO_HDF5("py", py, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["pz"]) DUMP_HYDRO_TO_HDF5("pz", pz, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["rho_m"]) DUMP_HYDRO_TO_HDF5("rho_m", rho_m, H5T_NATIVE_DOUBLE);

  if (hydro_dump_flag.flags["txx"]) DUMP_HYDRO_TO_HDF5("txx", txx, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["tyy"]) DUMP_HYDRO_TO_HDF5("tyy", tyy, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["tzz"]) DUMP_HYDRO_TO_HDF5("tzz", tzz, H5T_NATIVE_DOUBLE);

  if (hydro_dump_flag.flags["tyz"]) DUMP_HYDRO_TO_HDF5("tyz", tyz, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["tzx"]) DUMP_HYDRO_TO_HDF5("tzx", tzx, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["txy"]) DUMP_HYDRO_TO_HDF5("txy", txy, H5T_NATIVE_DOUBLE);

#ifdef VARIABLE_CHARGE
  if (hydro_dump_flag.flags["qmin"]) DUMP_HYDRO_TO_HDF5("qmin", qmin, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["qmax"]) DUMP_HYDRO_TO_HDF5("qmax", qmax, H5T_NATIVE_DOUBLE);

  if (hydro_dump_flag.flags["n_q0"]) DUMP_HYDRO_TO_HDF5("n_q0", n_q0, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["n_q1"]) DUMP_HYDRO_TO_HDF5("n_q1", n_q1, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["n_q2"]) DUMP_HYDRO_TO_HDF5("n_q2", n_q2, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["n_q3"]) DUMP_HYDRO_TO_HDF5("n_q3", n_q3, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["n_q4"]) DUMP_HYDRO_TO_HDF5("n_q4", n_q4, H5T_NATIVE_DOUBLE);
  if (hydro_dump_flag.flags["n_q5"]) DUMP_HYDRO_TO_HDF5("n_q5", n_q5, H5T_NATIVE_DOUBLE);
#endif

  el2 = uptime() - el2;
  if ( rank==0 ) log_printf("TimeHDF5Write: %.2f s\n", el2);

  double el3 = uptime();

  free(temp_buf);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

#undef GET_HYDRO_VAR
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  FREE(h_transformed);
#endif

  el3 = uptime() - el3;
  if ( rank==0 ) log_printf("TimeHDF5Close: %.2f s\n", el3);

  if (rank == 0) {
    char output_xml_file[128];
    sprintf(output_xml_file, "./%s/%s%s%s", "hydro_hdf5", "hydro-", sp->name, ".xdmf");
    char dimensions_3d[128];
    sprintf(dimensions_3d, "%lld %lld %lld", hydro_global_size[0], hydro_global_size[1], hydro_global_size[2]);
    char dimensions_4d[128];
    sprintf(dimensions_4d, "%lld %lld %lld %d", hydro_global_size[0], hydro_global_size[1], hydro_global_size[2], 3);
    char orignal[128];
    float dx = stride_x * grid->dx;
    float dy = stride_y * grid->dy;
    float dz = stride_z * grid->dz;
    float x0 = grid->x0 + dx;
    float y0 = grid->y0 + dy;
    float z0 = grid->z0 + dz;
    sprintf(orignal, "%f %f %f", x0, y0, z0);
    char dxdydz[128];
    sprintf(dxdydz, "%f %f %f", dx, dy, dz);

    int nframes = num_step / hydro_interval + 1;

    const int tframe = tframe_map[sp->id];

    char speciesname_new[128];
    sprintf(speciesname_new, "hydro_%s", sp->name);
    if (tframe >= 1)
    {
      if (tframe == (nframes - 1)) {
        invert_hydro_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 1);
      } else {
        invert_hydro_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 0);
      }
    } else {
      create_file_with_header(output_xml_file, dimensions_3d, orignal, dxdydz, nframes, hydro_interval);
      if (tframe == (nframes - 1)) {
        invert_hydro_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 1);
      } else {
        invert_hydro_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 0);
      }
    }
    tframe_map[sp->id]++;
  }
}

// Dump particles in HDF5 format
void HDF5Dump::dump_particles(
    const char *fbase,
    int step,
    species_t *sp,
    grid_t *grid,
    interpolator_array_t *interpolator_array,
    int ftag)
{
  //char fname[256];
  char group_name[256];
  //char particle_scratch[128];
  char subparticle_scratch[128];

  if( !sp ) ERROR(( "Invalid species name \"%s\".", sp->name ));
  if ( rank==0 ) log_printf("Dumping %s particles using HDF5\n", sp->name);

  // Update the particles on the host only if they haven't been recently
  if (step > sp->last_copied)
    sp->copy_to_host();

  // Update interpolators on host
  interpolator_array->copy_to_host();

  const long long np_local = (sp->np + stride_particle - 1) / stride_particle;

  // TODO: Allow the user to toggle the timing output
  const int print_timing = 0;

  // make a copy of the part of particle data to be dumped
  double ec1 = uptime();

  // prepare the data
  int sp_np = sp->np;
  int sp_max_np = sp->max_np;
  particle_t *ALIGNED(128) p_buf = NULL;
  if (!p_buf)
    MALLOC_ALIGNED(p_buf, np_local, 128);

  center_p(sp, interpolator_array);

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  particle_t *sp_p = sp->p;
  sp->p = p_buf;
  sp->np = np_local;
  sp->max_np = np_local;

  for (long long iptl = 0, i = 0; iptl < sp_np; iptl += stride_particle, ++i) {
    //COPY(&sp->p[i], &sp_p[iptl], 1);

    sp->p[i].dx = sp->k_p_h(iptl, particle_var::dx);
    sp->p[i].dy = sp->k_p_h(iptl, particle_var::dy);
    sp->p[i].dz = sp->k_p_h(iptl, particle_var::dz);
    sp->p[i].i  = sp->k_p_i_h(iptl);
    sp->p[i].ux = sp->k_p_h(iptl, particle_var::ux);
    sp->p[i].uy = sp->k_p_h(iptl, particle_var::uy);
    sp->p[i].uz = sp->k_p_h(iptl, particle_var::uz);
    sp->p[i].w  = sp->k_p_h(iptl, particle_var::w);
#ifdef VARIABLE_CHARGE
    sp->p[i].qp = sp->k_p_h(iptl, particle_var::qp);
#endif
  }

  //extract float and int data out of particle struct. This is a bit silly and looses type safety
  float * Pf = (float *)sp->p;
  int *   Pi = (int *)sp->p;
#else
  sp->np = np_local;
  sp->max_np = np_local;
  for (long long iptl = 0, i = 0; iptl < sp_np; iptl += stride_particle, ++i) {
    p_buf[i].dx = sp->k_p_h(iptl, particle_var::dx);
    p_buf[i].dy = sp->k_p_h(iptl, particle_var::dy);
    p_buf[i].dz = sp->k_p_h(iptl, particle_var::dz);
    p_buf[i].i  = sp->k_p_i_h(iptl);
    p_buf[i].ux = sp->k_p_h(iptl, particle_var::ux);
    p_buf[i].uy = sp->k_p_h(iptl, particle_var::uy);
    p_buf[i].uz = sp->k_p_h(iptl, particle_var::uz);
    p_buf[i].w  = sp->k_p_h(iptl, particle_var::w );
#ifdef VARIABLE_CHARGE
    p_buf[i].qp = sp->k_p_h(iptl, particle_var::qp);
#endif
  }
  //extract float and int data out of particle struct. This is a bit silly and looses type safety
  float * Pf = (float *)p_buf;
  int *   Pi = (int *)p_buf;
#endif

  ec1 = uptime() - ec1;
  if(print_timing)
    MESSAGE(("time in copying particle data: %fs, np_local = %lld", ec1, np_local));

  // Create target directory and subdirectory for the timestep
  std::string particle_dir = "./particle_hdf5";
  std::string subparticle_dir = particle_dir + "/T." + std::to_string(step) + "/";
  FileUtils::makeDirectory(particle_dir.c_str());
  FileUtils::makeDirectory(subparticle_dir.c_str());
  //sprintf(particle_scratch, "./%s", "particle_hdf5");
  //FileUtils::makeDirectory(particle_scratch);
  //sprintf(subparticle_scratch, "%s/T.%d/", particle_scratch, step);
  //FileUtils::makeDirectory(subparticle_scratch);

  // open HDF5 file for species
  std::string particle_fname = subparticle_dir + "/" + std::string(sp->name) + "_" + std::to_string(step) + ".h5";
  //sprintf(fname, "%s/%s_%d.h5", subparticle_scratch, sp->name, step);
  sprintf(group_name, "/Timestep_%d", step);
  double el1 = uptime();

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(particle_fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Pclose(plist_id);

  long long total_particles, offset;
  long long numparticles = np_local;
  MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  offset -= numparticles;

  hid_t filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);

#ifdef VARIABLE_CHARGE
  hsize_t n_p_vars = 9;
#else 
  hsize_t n_p_vars = 8;
#endif

  hsize_t memspace_count_temp = numparticles * n_p_vars;
  hid_t memspace = H5Screate_simple(1, &memspace_count_temp, NULL);

  // The converted global_ids are stored compact, not strided
  hsize_t linearspace_count_temp = numparticles;
  hid_t linearspace = H5Screate_simple(1, &linearspace_count_temp, NULL);

  plist_id = H5Pcreate(H5P_DATASET_XFER);

  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (hsize_t *)&offset, NULL, (hsize_t *)&numparticles, NULL);

  hsize_t memspace_start = 0, memspace_stride = n_p_vars, memspace_count = np_local;
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, &memspace_start, &memspace_stride, &memspace_count, NULL);

  el1 = uptime() - el1;
  if(print_timing) MESSAGE(("Particle TimeHDF5Open: %f s\n", el1));

  double el2 = uptime();

  hid_t dset_id = H5Dcreate(group_id, "dX", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  int ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "dY", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 1);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "dZ", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 2);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "Ux", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 4);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "Uy", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 5);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "Uz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 6);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "q", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 7);

#ifdef VARIABLE_CHARGE
  dset_id = H5Dcreate(group_id, "qp", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 8);
#endif
  H5Dclose(dset_id);

#define OUTPUT_CONVERT_GLOBAL_ID 1
#ifdef OUTPUT_CONVERT_GLOBAL_ID
# define UNVOXEL(rank, ix, iy, iz, nx, ny, nz) BEGIN_PRIMITIVE {   \
  int _ix, _iy, _iz;                                               \
  _ix  = (rank);        /* ix = ix+gpx*( iy+gpy*iz ) */            \
  _iy  = _ix/int(nx);   /* iy = iy+gpy*iz */                       \
  _ix -= _iy*int(nx);   /* ix = ix */                              \
  _iz  = _iy/int(ny);   /* iz = iz */                              \
  _iy -= _iz*int(ny);   /* iy = iy */                              \
  (ix) = _ix;                                                      \
  (iy) = _iy;                                                      \
  (iz) = _iz;                                                      \
} END_PRIMITIVE

  Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace> global_pi("Global IDs", numparticles);
  const int mpi_rank = rank;

  Kokkos::parallel_for("Convert Global ID", 
  Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, numparticles), 
  KOKKOS_LAMBDA(const int i) {
    int local_i = sp->k_p_i_h(i);

    int ix, iy, iz, rx, ry, rz;
    // Convert rank to local x/y/z
    UNVOXEL(mpi_rank, rx, ry, rz, grid->gpx, grid->gpy, grid->gpz);

    // Calculate local ix/iy/iz
    UNVOXEL(local_i, ix, iy, iz, grid->nx+2, grid->ny+2, grid->nz+2);

    // Account for the "first" ghost cell
    ix = ix - 1;
    iy = iy - 1;
    iz = iz - 1;

    // Convert ix/iy/iz to global
    int gix = ix + (grid->nx * rx);
    int giy = iy + (grid->ny * ry);
    int giz = iz + (grid->nz * rz);

    // calculate global grid sizes
    int gnx = grid->nx * grid->gpx;
    int gny = grid->ny * grid->gpy;
    int gnz = grid->nz * grid->gpz;

    // TODO: find a better way to account for the hard coded ghosts in VOXEL
    int global_i = VOXEL(gix, giy, giz, gnx-2, gny-2, gnz-2);

    //std::cout << mpi_rank << " local i " << local_i << " becomes " << global_i << std::endl;
    global_pi(i) = global_i;
  });
#undef UNVOXEL

  dset_id = H5Dcreate(group_id, "i", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_INT, linearspace, filespace, plist_id, global_pi.data());
  H5Dclose(dset_id);

#else
  dset_id = H5Dcreate(group_id, "i", H5T_NATIVE_INT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace, plist_id, Pi + 3);
  H5Dclose(dset_id);
#endif

  dset_id = H5Dcreate(group_id, "Ux", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 4);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "Uy", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 5);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "Uz", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 6);
  H5Dclose(dset_id);

  dset_id = H5Dcreate(group_id, "w", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 7);
  H5Dclose(dset_id);

#ifdef VARIABLE_CHARGE
  dset_id = H5Dcreate(group_id, "q", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 8);
  H5Dclose(dset_id);
#endif

  el2 = uptime() - el2;
  if(print_timing) MESSAGE(("Particle TimeHDF5Write: %f s \n", el2));

  double el3 = uptime();
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);
  el3 = uptime() - el3;
  if(print_timing) MESSAGE(("Particle TimeHDF5Close: %f s\n", el3));

  uncenter_p( sp, interpolator_array );
#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  sp->p = sp_p;
#endif
  FREE_ALIGNED(p_buf);
  sp->np = sp_np;
  sp->max_np = sp_max_np;

  // Write metadata
  // Note that these are all "local" metadata for each rank. Global metadata
  // such as total number of cells per dimension or box size still need to be
  // computed. For now that is left to postprocessing, but it could be done
  // inside the code using a bunch of MPI calls or by making the assumption
  // of equal sized local domains (which is currently satisfied by
  // partition_periodic_box and friends).

  char meta_fname[256];

  sprintf(meta_fname, "%s/grid_metadata_%s_%d.h5", subparticle_scratch, sp->name, step);

  double meta_el1 = uptime();

  hid_t meta_plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(meta_plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t meta_file_id = H5Fcreate(meta_fname, H5F_ACC_TRUNC, H5P_DEFAULT, meta_plist_id);
  hid_t meta_group_id = H5Gcreate(meta_file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Pclose(meta_plist_id);

  long long meta_total_particles, meta_offset;
  long long meta_numparticles = 1;
  MPI_Allreduce(&meta_numparticles, &meta_total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Scan(&meta_numparticles, &meta_offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  meta_offset -= meta_numparticles;

  hid_t meta_filespace = H5Screate_simple(1, (hsize_t *)&meta_total_particles, NULL);
  hid_t meta_memspace = H5Screate_simple(1, (hsize_t *)&meta_numparticles, NULL);
  meta_plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(meta_plist_id, H5FD_MPIO_COLLECTIVE);
  H5Sselect_hyperslab(meta_filespace, H5S_SELECT_SET, (hsize_t *)&meta_offset, NULL, (hsize_t *)&meta_numparticles, NULL);
  meta_el1 = uptime() - meta_el1;
  if(print_timing) MESSAGE(("Metafile TimeHDF5Open: %f s\n", meta_el1));

  double meta_el2 = uptime();

  hid_t meta_dset_id = H5Dcreate(meta_group_id, "np_local", H5T_NATIVE_INT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_INT, meta_memspace, meta_filespace, meta_plist_id, (int32_t *)&np_local);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "nx", H5T_NATIVE_INT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_INT, meta_memspace, meta_filespace, meta_plist_id, &grid->nx);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "ny", H5T_NATIVE_INT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_INT, meta_memspace, meta_filespace, meta_plist_id, &grid->ny);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "nz", H5T_NATIVE_INT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_INT, meta_memspace, meta_filespace, meta_plist_id, &grid->nz);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "x0", H5T_NATIVE_FLOAT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_FLOAT, meta_memspace, meta_filespace, meta_plist_id, &grid->x0);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "y0", H5T_NATIVE_FLOAT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_FLOAT, meta_memspace, meta_filespace, meta_plist_id, &grid->y0);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "z0", H5T_NATIVE_FLOAT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_FLOAT, meta_memspace, meta_filespace, meta_plist_id, &grid->z0);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "dx", H5T_NATIVE_FLOAT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_FLOAT, meta_memspace, meta_filespace, meta_plist_id, &grid->dx);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "dy", H5T_NATIVE_FLOAT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_FLOAT, meta_memspace, meta_filespace, meta_plist_id, &grid->dy);
  H5Dclose(meta_dset_id);

  meta_dset_id = H5Dcreate(meta_group_id, "dz", H5T_NATIVE_FLOAT, meta_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(meta_dset_id, H5T_NATIVE_FLOAT, meta_memspace, meta_filespace, meta_plist_id, &grid->dz);
  H5Dclose(meta_dset_id);

  meta_el2 = uptime() - meta_el2;
  if(print_timing) MESSAGE(("Metafile TimeHDF5Write: %f s\n", meta_el2));
  double meta_el3 = uptime();
  H5Sclose(meta_memspace);
  H5Sclose(meta_filespace);
  H5Pclose(meta_plist_id);
  H5Gclose(meta_group_id);
  H5Fclose(meta_file_id);
  meta_el3 = uptime() - meta_el3;
  if(print_timing) MESSAGE(("Metafile TimeHDF5Close: %f s\n", meta_el3));
}

// Dump fluids in HDF5 format
void HDF5Dump::dump_fluids(
    const char *fbase,
    int step,
    fluid_species_t *fsp,
    grid_t *grid,
    int ftag)
{
  // prepare the data
  if( !fsp ) ERROR(( "Invalid fluid species \"%s\"", fsp->name ));
  if ( rank==0 ) log_printf("Dumping %s fluid using HDF5\n", fsp->name);

  if (step > fsp->last_copied)  fsp->copy_to_host();

#define DUMP_FLUID_TO_HDF5(DSET_NAME, ATTRIBUTE_NAME, ELEMENT_TYPE)                                         \
{                                                                                                           \
  dset_id = H5Dcreate(group_id, DSET_NAME, ELEMENT_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
  temp_buf_index = 0;                                                                                       \
  for (int i(stride_x); i < grid->nx + 1; i += stride_x)                                                    \
  {                                                                                                         \
    for (int j(stride_y); j < grid->ny + 1; j += stride_y)                                                  \
    {                                                                                                       \
      for (int k(stride_z); k < grid->nz + 1; k += stride_z)                                                \
      {                                                                                                     \
        temp_buf[temp_buf_index] = fsp->fl[VOXEL(i,j,k, grid->nx,grid->ny,grid->nz)].ATTRIBUTE_NAME;        \
        temp_buf_index = temp_buf_index + 1;                                                                \
      }                                                                                                     \
    }                                                                                                       \
  }                                                                                                         \
  dataspace_id = H5Dget_space(dset_id);                                                                     \
  H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, global_offset, NULL, fluid_local_size, NULL);           \
  H5Dwrite(dset_id, ELEMENT_TYPE, memspace, dataspace_id, plist_id, temp_buf);                              \
  H5Sclose(dataspace_id);                                                                                   \
  H5Dclose(dset_id);                                                                                        \
}
  char hname[256];
  //char fluid_scratch[128];
  //char subfluid_scratch[128];

  // create the directory and sub-directory
  std::string fluid_dir = "./fluid_hdf5";
  std::string subfluid_dir = fluid_dir + "/T." + std::to_string(step) + "/";
  FileUtils::makeDirectory(fluid_dir.c_str());
  FileUtils::makeDirectory(subfluid_dir.c_str());
  //sprintf(fluid_scratch, "./%s", "fluid_hdf5");
  //FileUtils::makeDirectory(fluid_scratch);
  //sprintf(subfluid_scratch, "%s/T.%d/", fluid_scratch, step);
  //FileUtils::makeDirectory(subfluid_scratch);

  std::string fluid_fname = subfluid_dir + "/fluid_" + std::string(fsp->name) + "_" + std::to_string(step) + ".h5";
  //sprintf(hname, "%s/fluid_%s_%d.h5", subfluid_scratch, fsp->name, step);
  double el1 = uptime();
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(fluid_fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  sprintf(hname, "Timestep_%d", step);
  hid_t group_id = H5Gcreate(file_id, hname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  el1 = uptime() - el1;
  if ( rank==0 ) log_printf("TimeHDF5Open: %.2f s\n", el1);
  double el2 = uptime();

  // prepare for writing the data
  float *temp_buf = (float *)malloc(sizeof(float) * (grid->nx / stride_x) *
                                                    (grid->ny / stride_y) *
                                                    (grid->nz / stride_z));
  hsize_t temp_buf_index;
  hid_t dset_id;
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // data topology
  hsize_t fluid_global_size[3], fluid_local_size[3], global_offset[3];
  fluid_global_size[0] = (grid->nx * grid->gpx) / stride_x;
  fluid_global_size[1] = (grid->ny * grid->gpy) / stride_y;
  fluid_global_size[2] = (grid->nz * grid->gpz) / stride_z;

  fluid_local_size[0] = grid->nx / stride_x;
  fluid_local_size[1] = grid->ny / stride_y;
  fluid_local_size[2] = grid->nz / stride_z;

  int _ix, _iy, _iz;
  _ix = rank;
  _iy = _ix / grid->gpx;
  _ix -= _iy * grid->gpx;
  _iz = _iy / grid->gpy;
  _iy -= _iz * grid->gpy;

  global_offset[0] = (grid->nx) * _ix / stride_x;
  global_offset[1] = (grid->ny) * _iy / stride_y;
  global_offset[2] = (grid->nz) * _iz / stride_z;

  // prepare the spaces for parallel writing
  hid_t filespace = H5Screate_simple(3, fluid_global_size, NULL);
  hid_t memspace = H5Screate_simple(3, fluid_local_size, NULL);
  hid_t dataspace_id;

  // write the data
  DUMP_FLUID_TO_HDF5("den", den, H5T_NATIVE_FLOAT);
  DUMP_FLUID_TO_HDF5("tmp", tmp, H5T_NATIVE_FLOAT);
  DUMP_FLUID_TO_HDF5("prs", prs, H5T_NATIVE_FLOAT);
  DUMP_FLUID_TO_HDF5("ux", ux, H5T_NATIVE_FLOAT);
  DUMP_FLUID_TO_HDF5("uy", uy, H5T_NATIVE_FLOAT);
  DUMP_FLUID_TO_HDF5("uz", uz, H5T_NATIVE_FLOAT);

  el2 = uptime() - el2;
  if ( rank==0 ) log_printf("TimeHDF5Write: %.2f s\n", el2);

  double el3 = uptime();

  free(temp_buf);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
  H5Gclose(group_id);
  H5Fclose(file_id);

  el3 = uptime() - el3;
  if ( rank==0 ) log_printf("TimeHDF5Close: %.2f s\n", el3);

  if (rank == 0) {
    char output_xml_file[128];
    sprintf(output_xml_file, "./%s/%s%s%s", "fluid_hdf5", "fluid-", fsp->name, ".xdmf");
    char dimensions_3d[128];
    sprintf(dimensions_3d, "%lld %lld %lld", fluid_global_size[0], fluid_global_size[1], fluid_global_size[2]);
    char dimensions_4d[128];
    sprintf(dimensions_4d, "%lld %lld %lld %d", fluid_global_size[0], fluid_global_size[1], fluid_global_size[2], 3);
    char orignal[128];
    float dx = stride_x * grid->dx;
    float dy = stride_y * grid->dy;
    float dz = stride_z * grid->dz;
    float x0 = grid->x0 + dx;
    float y0 = grid->y0 + dy;
    float z0 = grid->z0 + dz;
    sprintf(orignal, "%f %f %f", x0, y0, z0);
    char dxdydz[128];
    sprintf(dxdydz, "%f %f %f", dx, dy, dz);

    int nframes = num_step / fluid_interval + 1;

    const int tframe = tframe_map[fsp->id];

    char speciesname_new[128];
    sprintf(speciesname_new, "fluid_%s", fsp->name);
    if (tframe >= 1)
    {
      if (tframe == (nframes - 1)) {
        invert_fluid_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 1);
      } else {
        invert_fluid_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 0);
      }
    } else {
      create_file_with_header(output_xml_file, dimensions_3d, orignal, dxdydz, nframes, fluid_interval);
      if (tframe == (nframes - 1)) {
        invert_fluid_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 1);
      } else {
        invert_fluid_xml_item(output_xml_file, speciesname_new, step, dimensions_4d, dimensions_3d, 0);
      }
    }
    tframe_map[fsp->id]++;
  }
}

// field_dump in HDF5 format
void HDF5Dump::field_dump(
      DumpParameters& dumpParams,
      int step,
      grid_t *grid,
      field_array_t *field_array)
{
  // Create a variable list of field values to output.
  //size_t numvars = std::min(dumpParams.output_vars.bitsum(),
  //                          total_field_variables);

  for(size_t i(0); i<total_field_variables; i++) {
    if(dumpParams.output_vars.bitset(i))
      field_dump_flag.flags[field_dump_flag.flag_keys[i]] = true;
    else
      field_dump_flag.flags[field_dump_flag.flag_keys[i]] = false;
  }

  // convenience
  const size_t istride(dumpParams.stride_x);
  const size_t jstride(dumpParams.stride_y);
  const size_t kstride(dumpParams.stride_z);

  // Check stride values.
  if(remainder(grid->nx, istride) != 0)
    ERROR(("x stride must be an integer factor of nx"));
  if(remainder(grid->ny, jstride) != 0)
    ERROR(("y stride must be an integer factor of ny"));
  if(remainder(grid->nz, kstride) != 0)
    ERROR(("z stride must be an integer factor of nz"));

  set_strides(istride, jstride, kstride);
  dump_fields(dumpParams.baseFileName, step, grid, field_array, 1);
}

// hydro_dump in HDF5 format
void HDF5Dump::hydro_dump(
    DumpParameters& dumpParams,
    int step,
    species_t *sp,
    grid_t *grid,
    hydro_array_t *hydro_array,
    interpolator_array_t *interpolator_array)
{
  // Create a variable list of field values to output.
  size_t numvars = std::min(dumpParams.output_vars.bitsum(),
                            total_hydro_variables);

  for(size_t i(0); i<total_hydro_variables; i++) {
    if(dumpParams.output_vars.bitset(i))
      hydro_dump_flag.flags[hydro_dump_flag.flag_keys[i]] = true;
    else
      hydro_dump_flag.flags[hydro_dump_flag.flag_keys[i]] = false;
  }
  // convenience
  const size_t istride(dumpParams.stride_x);
  const size_t jstride(dumpParams.stride_y);
  const size_t kstride(dumpParams.stride_z);

  // Check stride values.
  if(remainder(grid->nx, istride) != 0)
    ERROR(("x stride must be an integer factor of nx"));
  if(remainder(grid->ny, jstride) != 0)
    ERROR(("y stride must be an integer factor of ny"));
  if(remainder(grid->nz, kstride) != 0)
    ERROR(("z stride must be an integer factor of nz"));

  set_strides(istride, jstride, kstride);
  dump_hydro(dumpParams.baseFileName, step, sp, grid, hydro_array,
      interpolator_array, 1);
}

// fluid_dump in HDF5 format
void HDF5Dump::fluid_dump(
    DumpParameters& dumpParams,
    int step,
    fluid_species_t *fsp,
    grid_t *grid)
{
  // convenience
  const size_t istride(dumpParams.stride_x);
  const size_t jstride(dumpParams.stride_y);
  const size_t kstride(dumpParams.stride_z);

  // Check stride values.
  if(remainder(grid->nx, istride) != 0)
    ERROR(("x stride must be an integer factor of nx"));
  if(remainder(grid->ny, jstride) != 0)
    ERROR(("y stride must be an integer factor of ny"));
  if(remainder(grid->nz, kstride) != 0)
    ERROR(("z stride must be an integer factor of nz"));

  set_strides(istride, jstride, kstride);
  dump_fluids(dumpParams.baseFileName, step, fsp, grid, 1);
}

#endif // #ifdef VPIC_ENABLE_HDF5

