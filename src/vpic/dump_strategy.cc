// VPIC headers
#include "dump_strategy.h"
#include "vpic.h"

const int max_filename_bytes = 256;

// Create a new dump strategy
Dump_Strategy *
new_dump_strategy(DumpStrategyID dump_strategy_id,
                  vpic_simulation *vpic_simu)
{
  Dump_Strategy *ds;
  MALLOC(ds, 1);
  CLEAR(ds, 1);

  // Do any post init/restore simulation modifications
  switch (dump_strategy_id)
  {
    case DUMP_STRATEGY_BINARY:
      if (vpic_simu->rank() == 0)
        std::cout << "DUMP_STRATEGY_BINARY  enabled \n";
      ds = new BinaryDump(vpic_simu->rank(), vpic_simu->nproc());
      break;
    case DUMP_STRATEGY_HDF5:
#ifdef VPIC_ENABLE_HDF5
      if (vpic_simu->rank() == 0)
        std::cout << "DUMP_STRATEGY_HDF5 enabled \n";
      ds = new HDF5Dump(vpic_simu->rank(), vpic_simu->nproc(),
          vpic_simu->num_step, vpic_simu->field_interval,
          vpic_simu->hydro_interval);
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
  printf("Calling Binary dump_fields ... \n");

  if (!fbase)
    ERROR(("Invalid filename"));

  if (rank == 0)
    MESSAGE(("Dumping fields to \"%s\"", fbase));

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
  WRITE_ARRAY_HEADER(field_array->f, 3, dim, fileIO);
  fileIO.write(field_array->f, dim[0] * dim[1] * dim[2]);
  if (fileIO.close())
    ERROR(("File close failed on dump fields!!!"));
}

// Dump hydro in binary format
void BinaryDump::dump_hydro(
    const char *fbase,
    int step,
    hydro_array_t *hydro_array,
    species_t *sp,
    interpolator_array_t *interpolator_array,
    grid_t *grid,
    int ftag)
{
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[3];
  printf("Calling Binary dump_hydro ... \n");

  if (!sp)
    ERROR(("Invalid species \"%s\"", sp->name));

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

  // This is slower in my tests
  //synchronize_hydro_array_kokkos(hydro_array);

  hydro_array->copy_to_host();

  synchronize_hydro_array( hydro_array );

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
  WRITE_ARRAY_HEADER(hydro_array->h, 3, dim, fileIO);
  fileIO.write(hydro_array->h, dim[0] * dim[1] * dim[2]);
  if (fileIO.close())
    ERROR(("File close failed on dump hydro!!!"));
}

// Dump particles in binary format
void BinaryDump::dump_particles(
    const char *fbase,
    species_t *sp,
    grid_t *grid,
    int step,
    interpolator_array_t *interpolator_array,
    int ftag)
{
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[1], buf_start;
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

  particle_t *sp_p = sp->p;
  sp->p = p_buf;
  int sp_np = sp->np;
  sp->np = 0;
  int sp_max_np = sp->max_np;
  sp->max_np = PBUF_SIZE;
  for (buf_start = 0; buf_start < sp_np; buf_start += PBUF_SIZE)
  {
    sp->np = sp_np - buf_start;
    if (sp->np > PBUF_SIZE)
        sp->np = PBUF_SIZE;
    COPY(sp->p, &sp_p[buf_start], sp->np);
    center_p(sp, interpolator_array);
    fileIO.write(sp->p, sp->np);
  }
  sp->p = sp_p;
  sp->np = sp_np;
  sp->max_np = sp_max_np;

  if (fileIO.close())
    ERROR(("File close failed on dump particles!!!"));
}

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

#define fpp(x, y, z) f[VOXEL(x, y, z, grid->nx, grid->ny, grid->nz)]

#define DUMP_FIELD_TO_HDF5(DSET_NAME, ATTRIBUTE_NAME, ELEMENT_TYPE)                                           \
{                                                                                                           \
  dset_id = H5Dcreate(group_id, DSET_NAME, ELEMENT_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
  temp_buf_index = 0;                                                                                       \
  for (size_t i(1); i < grid->nx + 1; i++)                                                                  \
  {                                                                                                         \
    for (size_t j(1); j < grid->ny + 1; j++)                                                                \
    {                                                                                                       \
      for (size_t k(1); k < grid->nz + 1; k++)                                                              \
      {                                                                                                     \
        temp_buf[temp_buf_index] = field_array->fpp(i, j, k).ATTRIBUTE_NAME;                                \
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

  char fname[256];
  char field_scratch[128];
  char subfield_scratch[128];

  // create the directory and sub-directory
  sprintf(field_scratch, "./%s", "fields_hdf5");
  FileUtils::makeDirectory(field_scratch);
  sprintf(subfield_scratch, "%s/T.%zu/", field_scratch, step);
  FileUtils::makeDirectory(subfield_scratch);

  // create the file
  sprintf(fname, "%s/%s_%zu.h5", subfield_scratch, "fields", step);
  double el1 = uptime();
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  // create the group for the time step
  sprintf(fname, "Timestep_%zu", step);
  hid_t group_id = H5Gcreate(file_id, fname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  el1 = uptime() - el1;
  if ( rank==0 ) log_printf("TimeHDF5Open: %.2f s\n", el1);
  double el2 = uptime();

  // prepare for writing the data
  float *temp_buf = (float *)malloc(sizeof(float) * (grid->nx) * (grid->ny) * (grid->nz));
  hsize_t temp_buf_index;
  hid_t dset_id;
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // data topology
  hsize_t field_global_size[3], field_local_size[3], global_offset[3];
  field_global_size[0] = (grid->nx * grid->gpx);
  field_global_size[1] = (grid->ny * grid->gpy);
  field_global_size[2] = (grid->nz * grid->gpz);

  field_local_size[0] = grid->nx;
  field_local_size[1] = grid->ny;
  field_local_size[2] = grid->nz;

  int _ix, _iy, _iz;
  _ix = rank;
  _iy = _ix / grid->gpx;
  _ix -= _iy * grid->gpx;
  _iz = _iy / grid->gpy;
  _iy -= _iz * grid->gpy;

  global_offset[0] = grid->nx * _ix;
  global_offset[1] = grid->ny * _iy;
  global_offset[2] = grid->nz * _iz;

  // prepare the spaces for parallel writing
  hid_t filespace = H5Screate_simple(3, field_global_size, NULL);
  hid_t memspace = H5Screate_simple(3, field_local_size, NULL);
  hid_t dataspace_id;

  // write the data
  if (field_dump_flag.ex) DUMP_FIELD_TO_HDF5("ex", ex, H5T_NATIVE_FLOAT);
  if (field_dump_flag.ey) DUMP_FIELD_TO_HDF5("ey", ey, H5T_NATIVE_FLOAT);
  if (field_dump_flag.ez) DUMP_FIELD_TO_HDF5("ez", ez, H5T_NATIVE_FLOAT);
  if (field_dump_flag.div_e_err) DUMP_FIELD_TO_HDF5("div_e_err", div_e_err, H5T_NATIVE_FLOAT);

  if (field_dump_flag.cbx) DUMP_FIELD_TO_HDF5("cbx", cbx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.cby) DUMP_FIELD_TO_HDF5("cby", cby, H5T_NATIVE_FLOAT);
  if (field_dump_flag.cbz) DUMP_FIELD_TO_HDF5("cbz", cbz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.div_b_err) DUMP_FIELD_TO_HDF5("div_b_err", div_b_err, H5T_NATIVE_FLOAT);

  if (field_dump_flag.cbx0) DUMP_FIELD_TO_HDF5("cbx0", cbx0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.cby0) DUMP_FIELD_TO_HDF5("cby0", cby0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.cbz0) DUMP_FIELD_TO_HDF5("cbz0", cbz0, H5T_NATIVE_FLOAT);
  if (field_dump_flag.tmpsm) DUMP_FIELD_TO_HDF5("tmpsm", tmpsm, H5T_NATIVE_FLOAT);

  if (field_dump_flag.tcax) DUMP_FIELD_TO_HDF5("tcax", tcax, H5T_NATIVE_FLOAT);
  if (field_dump_flag.tcay) DUMP_FIELD_TO_HDF5("tcay", tcay, H5T_NATIVE_FLOAT);
  if (field_dump_flag.tcaz) DUMP_FIELD_TO_HDF5("tcaz", tcaz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.rhob) DUMP_FIELD_TO_HDF5("rhob", rhob, H5T_NATIVE_FLOAT);

  if (field_dump_flag.jfx) DUMP_FIELD_TO_HDF5("jfx", jfx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.jfy) DUMP_FIELD_TO_HDF5("jfy", jfy, H5T_NATIVE_FLOAT);
  if (field_dump_flag.jfz) DUMP_FIELD_TO_HDF5("jfz", jfz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.rhof) DUMP_FIELD_TO_HDF5("rhof", rhof, H5T_NATIVE_FLOAT);

  if (field_dump_flag.jfxold) DUMP_FIELD_TO_HDF5("jfxold", jfxold, H5T_NATIVE_FLOAT);
  if (field_dump_flag.jfyold) DUMP_FIELD_TO_HDF5("jfyold", jfyold, H5T_NATIVE_FLOAT);
  if (field_dump_flag.jfzold) DUMP_FIELD_TO_HDF5("jfzold", jfzold, H5T_NATIVE_FLOAT);
  if (field_dump_flag.rhofold) DUMP_FIELD_TO_HDF5("rhofold", rhofold, H5T_NATIVE_FLOAT);

  if (field_dump_flag.tx) DUMP_FIELD_TO_HDF5("tx", tx, H5T_NATIVE_FLOAT);
  if (field_dump_flag.ty) DUMP_FIELD_TO_HDF5("ty", ty, H5T_NATIVE_FLOAT);
  if (field_dump_flag.tz) DUMP_FIELD_TO_HDF5("tz", tz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.te) DUMP_FIELD_TO_HDF5("te", te, H5T_NATIVE_FLOAT);

  if (field_dump_flag.ox) DUMP_FIELD_TO_HDF5("ox", ox, H5T_NATIVE_FLOAT);
  if (field_dump_flag.oy) DUMP_FIELD_TO_HDF5("oy", oy, H5T_NATIVE_FLOAT);
  if (field_dump_flag.oz) DUMP_FIELD_TO_HDF5("oz", oz, H5T_NATIVE_FLOAT);
  if (field_dump_flag.oe) DUMP_FIELD_TO_HDF5("oe", oe, H5T_NATIVE_FLOAT);

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
    char const *output_xml_file = "./fields_hdf5/hdf5_field.xdmf";
    char dimensions_3d[128];
    sprintf(dimensions_3d, "%lld %lld %lld", field_global_size[0], field_global_size[1], field_global_size[2]);
    char dimensions_4d[128];
    sprintf(dimensions_4d, "%lld %lld %lld %d", field_global_size[0], field_global_size[1], field_global_size[2], 3);
    char orignal[128];
    sprintf(orignal, "%f %f %f", grid->x0, grid->y0, grid->z0);
    char dxdydz[128];
    sprintf(dxdydz, "%f %f %f", grid->dx, grid->dy, grid->dz);

    int nframes = num_step / field_interval + 1;
    static int field_tframe = 0;
    // TODO: this footer dumping is more likely better done in a
    // destructor, rather than hoping a multiple division works out
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
void HDF5Dump::dump_hydro(
    const char *fbase,
    int step,
    hydro_array_t *hydro_array,
    species_t *sp,
    interpolator_array_t *interpolator_array,
    grid_t *grid,
    int ftag)
{
#define DUMP_HYDRO_TO_HDF5(DSET_NAME, ATTRIBUTE_NAME, ELEMENT_TYPE)                                           \
{                                                                                                           \
  dset_id = H5Dcreate(group_id, DSET_NAME, ELEMENT_TYPE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
  temp_buf_index = 0;                                                                                       \
  for (size_t i(1); i < grid->nx + 1; i++)                                                                  \
  {                                                                                                         \
    for (size_t j(1); j < grid->ny + 1; j++)                                                                \
    {                                                                                                       \
      for (size_t k(1); k < grid->nz + 1; k++)                                                              \
      {                                                                                                     \
        temp_buf[temp_buf_index] = hydro_array->h[VOXEL(i,j,k, grid->nx,grid->ny,grid->nz)].ATTRIBUTE_NAME; \
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
  // prepare the data
  if (!sp) ERROR(("Invalid species name: %s", sp->name));

  if ( rank==0 ) log_printf("Dumping hydro for %s using HDF5\n", sp->name);

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

  // The legacy synchronize is actually a bit faster
  //synchronize_hydro_array_kokkos(hydro_array);

  hydro_array->copy_to_host();
  synchronize_hydro_array( hydro_array );

  char hname[256];
  char hydro_scratch[128];
  char subhydro_scratch[128];

  // create the directory and sub-directory
  sprintf(hydro_scratch, "./%s", "hydro_hdf5");
  FileUtils::makeDirectory(hydro_scratch);
  sprintf(subhydro_scratch, "%s/T.%zu/", hydro_scratch, step);
  FileUtils::makeDirectory(subhydro_scratch);

  sprintf(hname, "%s/hydro_%s_%zu.h5", subhydro_scratch, sp->name, step);
  double el1 = uptime();
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(hname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  sprintf(hname, "Timestep_%zu", step);
  hid_t group_id = H5Gcreate(file_id, hname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  el1 = uptime() - el1;
  if ( rank==0 ) log_printf("TimeHDF5Open: %.2f s\n", el1);
  double el2 = uptime();

  // prepare for writing the data
  float *temp_buf = (float *)malloc(sizeof(float) * (grid->nx) * (grid->ny) * (grid->nz));
  hsize_t temp_buf_index;
  hid_t dset_id;
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // data topology
  hsize_t hydro_global_size[3], hydro_local_size[3], global_offset[3];
  hydro_global_size[0] = (grid->nx * grid->gpx);
  hydro_global_size[1] = (grid->ny * grid->gpy);
  hydro_global_size[2] = (grid->nz * grid->gpz);

  hydro_local_size[0] = grid->nx;
  hydro_local_size[1] = grid->ny;
  hydro_local_size[2] = grid->nz;

  int _ix, _iy, _iz;
  _ix = rank;
  _iy = _ix / grid->gpx;
  _ix -= _iy * grid->gpx;
  _iz = _iy / grid->gpy;
  _iy -= _iz * grid->gpy;

  global_offset[0] = (grid->nx) * _ix;
  global_offset[1] = (grid->ny) * _iy;
  global_offset[2] = (grid->nz) * _iz;

  // prepare the spaces for parallel writing
  hid_t filespace = H5Screate_simple(3, hydro_global_size, NULL);
  hid_t memspace = H5Screate_simple(3, hydro_local_size, NULL);
  hid_t dataspace_id;

  // write the data
  if (hydro_dump_flag.jx) DUMP_HYDRO_TO_HDF5("jx", jx, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.jy) DUMP_HYDRO_TO_HDF5("jy", jy, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.jz) DUMP_HYDRO_TO_HDF5("jz", jz, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.rho) DUMP_HYDRO_TO_HDF5("rho", rho, H5T_NATIVE_FLOAT);

  if (hydro_dump_flag.px) DUMP_HYDRO_TO_HDF5("px", px, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.py) DUMP_HYDRO_TO_HDF5("py", py, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.pz) DUMP_HYDRO_TO_HDF5("pz", pz, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.ke) DUMP_HYDRO_TO_HDF5("ke", ke, H5T_NATIVE_FLOAT);

  if (hydro_dump_flag.txx) DUMP_HYDRO_TO_HDF5("txx", txx, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.tyy) DUMP_HYDRO_TO_HDF5("tyy", tyy, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.tzz) DUMP_HYDRO_TO_HDF5("tzz", tzz, H5T_NATIVE_FLOAT);

  if (hydro_dump_flag.tyz) DUMP_HYDRO_TO_HDF5("tyz", tyz, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.tzx) DUMP_HYDRO_TO_HDF5("tzx", tzx, H5T_NATIVE_FLOAT);
  if (hydro_dump_flag.txy) DUMP_HYDRO_TO_HDF5("txy", txy, H5T_NATIVE_FLOAT);

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
    sprintf(output_xml_file, "./%s/%s%s%s", "hydro_hdf5", "hydro-", sp->name, ".xdmf");
    char dimensions_3d[128];
    sprintf(dimensions_3d, "%lld %lld %lld", hydro_global_size[0], hydro_global_size[1], hydro_global_size[2]);
    char dimensions_4d[128];
    sprintf(dimensions_4d, "%lld %lld %lld %d", hydro_global_size[0], hydro_global_size[1], hydro_global_size[2], 3);
    char orignal[128];
    sprintf(orignal, "%f %f %f", grid->x0, grid->y0, grid->z0);
    char dxdydz[128];
    sprintf(dxdydz, "%f %f %f", grid->dx, grid->dy, grid->dz);

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
    species_t *sp,
    grid_t *grid,
    int step,
    interpolator_array_t *interpolator_array,
    int ftag)
{
  char fname[256];
  char group_name[256];
  char particle_scratch[128];
  char subparticle_scratch[128];

  if( !sp ) ERROR(( "Invalid species name \"%s\".", sp->name ));
  if ( rank==0 ) log_printf("Dumping %s particles using HDF5\n", sp->name);

  // Update the particles on the host only if they haven't been recently
  if (step > sp->last_copied)
    sp->copy_to_host();

  // TODO: Allow the user to set this
  const int stride_particle_dump = 1;
  const long long np_local = (sp->np + stride_particle_dump - 1) / stride_particle_dump;

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
  particle_t *sp_p = sp->p;
  sp->p = p_buf;
  sp->np = np_local;
  sp->max_np = np_local;

  for (long long iptl = 0, i = 0; iptl < sp_np; iptl += stride_particle_dump, ++i) {
    COPY(&sp->p[i], &sp_p[iptl], 1);
  }

  center_p(sp, interpolator_array);

  ec1 = uptime() - ec1;
  if(print_timing)
    MESSAGE(("time in copying particle data: %fs, np_local = %lld", ec1, np_local));

  //extract float and int data out of particle struct. This is a bit silly and looses type safety
  float * Pf = (float *)sp->p;
  int *   Pi = (int *)sp->p;

  // Create target directory and subdirectory for the timestep
  sprintf(particle_scratch, "./%s", "particle_hdf5");
  FileUtils::makeDirectory(particle_scratch);
  sprintf(subparticle_scratch, "%s/T.%ld/", particle_scratch, step);
  FileUtils::makeDirectory(subparticle_scratch);

  // open HDF5 file for species
  sprintf(fname, "%s/%s_%ld.h5", subparticle_scratch, sp->name, step);
  sprintf(group_name, "/Timestep_%ld", step);
  double el1 = uptime();

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  H5Pclose(plist_id);

  long long total_particles, offset;
  long long numparticles = np_local;
  MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  offset -= numparticles;

  hid_t filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);

  hsize_t memspace_count_temp = numparticles * 8;
  hid_t memspace = H5Screate_simple(1, &memspace_count_temp, NULL);

  // The converted global_ids are stored compact, not strided
  hsize_t linearspace_count_temp = numparticles;
  hid_t linearspace = H5Screate_simple(1, &linearspace_count_temp, NULL);

  plist_id = H5Pcreate(H5P_DATASET_XFER);

  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, (hsize_t *)&offset, NULL, (hsize_t *)&numparticles, NULL);

  hsize_t memspace_start = 0, memspace_stride = 8, memspace_count = np_local;
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

  std::vector<int> global_pi;
  global_pi.reserve(numparticles);
  const int mpi_rank = rank;

  // TODO: this could be parallel
  for (int i = 0; i < numparticles; i++) {
    int local_i = sp->p[i].i;

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
    global_pi[i] = global_i;
  }
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

  dset_id = H5Dcreate(group_id, "q", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  ierr = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Pf + 7);
  H5Dclose(dset_id);

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

  sp->p = sp_p;
  sp->np = sp_np;
  sp->max_np = sp_max_np;
  FREE_ALIGNED(p_buf);

  // Write metadata
  // Note that these are all "local" metadata for each rank. Global metadata
  // such as total number of cells per dimension or box size still need to be
  // computed. For now that is left to postprocessing, but it could be done
  // inside the code using a bunch of MPI calls or by making the assumption
  // of equal sized local domains (which is currently satisfied by
  // partition_periodic_box and friends).

  char meta_fname[256];

  sprintf(meta_fname, "%s/grid_metadata_%s_%ld.h5", subparticle_scratch, sp->name, step);

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
