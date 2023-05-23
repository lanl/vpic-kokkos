/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Heavily revised and extended from earlier V4PIC versions
 *
 * snell - revised to add strided dumps, time history dumps, others  20080404
 */

#include <cassert>

#include "vpic.h"
#include "dumpmacros.h"
#include "../util/io/FileUtils.h"

/* -1 means no ranks talk */
#define VERBOSE_rank -1

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"
#endif

// FIXME: NEW FIELDS IN THE GRID READ/WRITE WAS HACKED UP TO BE BACKWARD
// COMPATIBLE WITH EXISTING EXTERNAL 3RD PARTY VISUALIZATION SOFTWARE.
// IN THE LONG RUN, THIS EXTERNAL SOFTWARE WILL NEED TO BE UPDATED.

const int max_filename_bytes = 256;

int vpic_simulation::dump_mkdir(const char * dname) {
	return FileUtils::makeDirectory(dname);
} // dump_mkdir

int vpic_simulation::dump_cwd(char * dname, size_t size) {
	return FileUtils::getCurrentWorkingDirectory(dname, size);
} // dump_mkdir

/*****************************************************************************
 * ASCII dump IO
 *****************************************************************************/
void
vpic_simulation::dump_energies( const char *fname,
                                bool tracers,
                                int append ) {
  double en_f[6], en_p;
  species_t *sp;
  FileIO fileIO;
  FileIOStatus status(fail);

  if( !fname ) ERROR(("Invalid file name"));

  if( rank()==0 ) {
    status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\".", fname ));
    else {
      if( append==0 ) {
        fileIO.print( "%% Layout\n%% step ex ey ez bx by bz" );
        LIST_FOR_EACH(sp,species_list)
          fileIO.print( " \"%s\"", sp->name );
        fileIO.print( "\n" );
        fileIO.print( "%% timestep = %e\n", grid->dt );
      }
      fileIO.print( "%li", (long)step() );
    }
  }

//  field_array->kernel->energy_f( en_f, field_array );
  field_array->kernel->energy_f_kokkos( en_f, field_array );
  if( rank()==0 && status!=fail )
    fileIO.print( " %e %e %e %e %e %e",
                  en_f[0], en_f[1], en_f[2],
                  en_f[3], en_f[4], en_f[5] );

  LIST_FOR_EACH(sp,species_list) {
    en_p = energy_p_kokkos( sp, interpolator_array );
    if( rank()==0 && status!=fail ) fileIO.print( " %e", en_p );
  }

  if( rank()==0 && status!=fail ) {
    fileIO.print( "\n" );
    if( fileIO.close() ) ERROR(("File close failed on dump energies!!!"));
  }
}

void
vpic_simulation::dump_energies( const char *fname,
                                int append ) {
  dump_energies(fname, false, append);
}

// Note: dump_species/materials assume that names do not contain any \n!

void
vpic_simulation::dump_species( const char *fname ) {
  species_t *sp;
  FileIO fileIO;

  if( rank() ) return;
  if( !fname ) ERROR(( "Invalid file name" ));
  MESSAGE(( "Dumping species to \"%s\"", fname ));
  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail ) ERROR(( "Could not open \"%s\".", fname ));
  LIST_FOR_EACH( sp, species_list )
    fileIO.print( "%s %i %e %e", sp->name, sp->id, sp->q, sp->m );
  if( fileIO.close() ) ERROR(( "File close failed on dump species!!!" ));
}

void
vpic_simulation::dump_species( const char *fname, bool tracers ) {
  dump_species(fname, false);
}

void
vpic_simulation::dump_materials( const char *fname ) {
  FileIO fileIO;
  material_t *m;
  if( rank() ) return;
  if( !fname ) ERROR(( "Invalid file name" ));
  MESSAGE(( "Dumping materials to \"%s\"", fname ));
  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));
  LIST_FOR_EACH( m, material_list )
    fileIO.print( "%s\n%i\n%e %e %e\n%e %e %e\n%e %e %e\n",
                  m->name, m->id,
                  m->epsx,   m->epsy,   m->epsz,
                  m->mux,    m->muy,    m->muz,
                  m->sigmax, m->sigmay, m->sigmaz );
  if( fileIO.close() ) ERROR(( "File close failed on dump materials!!!" ));
}

void
vpic_simulation::dump_tracers_csv( const char *sp_name,
                                   uint32_t dump_vars,
                                   const char *fbase,
                                   const int append,
                                   int ftag )
{

    species_t *sp;
    char fname[max_filename_bytes];
    FileIO fileIO;
    int dim[1], buf_start;
    static particle_t * ALIGNED(128) p_buf = NULL;

    sp = find_species_name( sp_name, tracers_list );
    if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name ));

    if( !fbase ) ERROR(( "Invalid filename" ));

    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( rank()==0 )
        MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,fbase));

    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
    }
    else {
        snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
    }

    FileIOStatus status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    std::string header_str = std::string("step,rank,tracer_id,cell_id,dx,dy,dz,ux,uy,uz,w");
    if(dump_vars & DumpVar::GlobalPos) {
      header_str += ",posx,posy,posz";
    }
    if(dump_vars & DumpVar::Efield) {
      header_str += ",ex,ey,ez";
    }
    if(dump_vars & DumpVar::Bfield) {
      header_str += ",bx,by,bz";
    }
    if(dump_vars & DumpVar::CurrentDensity) {
      header_str += ",jx,jy,jz";
    }
    if(dump_vars & DumpVar::ChargeDensity) {
      header_str += ",rho";
    }
    if(dump_vars & DumpVar::MomentumDensity) {
      header_str += ",px,py,pz";
    }
    if(dump_vars & DumpVar::KEDensity) {
      header_str += ",ke";
    }
    if(dump_vars & DumpVar::StressTensor) {
      header_str += ",txx,tyy,tzz,tyz,tzx,txy";
    }

    if( append==0 ) {
      fileIO.print( "%s\n", header_str.c_str() );
    }

    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->num_annotations.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }

    if(static_cast<uint32_t>(dump_vars) > DumpVar::Bfield) {
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
    }

    int tracer_idx = sp->num_annotations.get_annotation_index<int64_t>("TracerID");
    auto& interp = interpolator_array->k_i_h;

#define _nxg (grid->nx + 2)
#define _nyg (grid->ny + 2)
#define _nzg (grid->nz + 2)
#define i0 (ii%_nxg)
#define j0 ((ii/_nxg)%_nyg)
#define k0 (ii/(_nxg*_nyg))
#define tracer_x ((i0 + (dx0-1)*0.5) * grid->dx + grid->x0)
#define tracer_y ((j0 + (dy0-1)*0.5) * grid->dy + grid->y0)
#define tracer_z ((k0 + (dz0-1)*0.5) * grid->dz + grid->z0)
    for(uint32_t i=0; i<sp->np; i++) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      float ux0 = sp->k_p_h(i, particle_var::ux);
      float uy0 = sp->k_p_h(i, particle_var::uy);
      float uz0 = sp->k_p_h(i, particle_var::uz);
      float w0  = sp->k_p_h(i, particle_var::w);
      int   ii  = sp->k_p_i_h(i);
      fileIO.print("%ld,%d,%ld,%d,%e,%e,%e,%e,%e,%e,%e", 
        step(), rank(), sp->annotations_h.get<int64_t>(i,tracer_idx), ii,
        dx0, dy0, dz0, ux0, uy0, uz0, w0);
      if(dump_vars & DumpVar::GlobalPos) {
        fileIO.print(",%e,%e,%e", tracer_x, tracer_y, tracer_z);
      }
      if(dump_vars & DumpVar::Efield) {
        float ex  = interp(ii,interpolator_var::ex ) + dy0*interp(ii,interpolator_var::dexdy) + dz0*(interp(ii,interpolator_var::dexdz) + dy0*interp(ii,interpolator_var::d2exdydz)); 
        float ey  = interp(ii,interpolator_var::ey ) + dz0*interp(ii,interpolator_var::deydz) + dx0*(interp(ii,interpolator_var::deydx) + dz0*interp(ii,interpolator_var::d2eydzdx)); 
        float ez  = interp(ii,interpolator_var::ez ) + dx0*interp(ii,interpolator_var::dezdx) + dy0*(interp(ii,interpolator_var::dezdy) + dx0*interp(ii,interpolator_var::d2ezdxdy)); 
        fileIO.print(",%e,%e,%e", ex, ey, ez);
      }
      if(dump_vars & DumpVar::Bfield) {
        float bx  = interp(ii,interpolator_var::cbx) + dx0*interp(ii,interpolator_var::dcbxdx); 
        float by  = interp(ii,interpolator_var::cby) + dy0*interp(ii,interpolator_var::dcbydy); 
        float bz  = interp(ii,interpolator_var::cbz) + dz0*interp(ii,interpolator_var::dcbzdz); 
        fileIO.print(",%e,%e,%e", bx, by, bz);
      }
      if(dump_vars & DumpVar::CurrentDensity) {
        float jx  = hydro_array->k_h_h(ii, hydro_var::jx);
        float jy  = hydro_array->k_h_h(ii, hydro_var::jy);
        float jz  = hydro_array->k_h_h(ii, hydro_var::jz);
        fileIO.print(",%e,%e,%e", jx, jy, jz);
      }
      if(dump_vars & DumpVar::ChargeDensity) {
        float rho = hydro_array->k_h_h(ii, hydro_var::rho);
        fileIO.print(",%e", rho);
      }
      if(dump_vars & DumpVar::MomentumDensity) {
        float px  = hydro_array->k_h_h(ii, hydro_var::px);
        float py  = hydro_array->k_h_h(ii, hydro_var::py);
        float pz  = hydro_array->k_h_h(ii, hydro_var::pz);
        fileIO.print(",%e,%e,%e", px, py, pz);
      }
      if(dump_vars & DumpVar::KEDensity) {
        float ke = hydro_array->k_h_h(ii, hydro_var::ke);
        fileIO.print(",%e", ke);
      }
      if(dump_vars & DumpVar::StressTensor) {
        float txx = hydro_array->k_h_h(ii, hydro_var::txx);
        float tyy = hydro_array->k_h_h(ii, hydro_var::tyy);
        float tzz = hydro_array->k_h_h(ii, hydro_var::tzz);
        float tyz = hydro_array->k_h_h(ii, hydro_var::tyz);
        float tzx = hydro_array->k_h_h(ii, hydro_var::tzx);
        float txy = hydro_array->k_h_h(ii, hydro_var::txy);
        fileIO.print(",%e,%e,%e,%e,%e,%e", txx, tyy, tzz, tyz, tzx, txy);
      }
      fileIO.print( "\n" );
    }
#undef nxg 
#undef nyg 
#undef nzg 
#undef i0 
#undef j0 
#undef k0 
#undef tracer_x 
#undef tracer_y 
#undef tracer_z 

    if( fileIO.close() ) ERROR(("File close failed on dump particles!!!"));
}

/*****************************************************************************
 * Binary dump IO
 *****************************************************************************/

/*
enum dump_types {
  grid_dump = 0,
  field_dump = 1,
  hydro_dump = 2,
  particle_dump = 3,
  restart_dump = 4
};
*/

namespace dump_type {
  const int grid_dump = 0;
  const int field_dump = 1;
  const int hydro_dump = 2;
  const int particle_dump = 3;
  const int restart_dump = 4;
  const int history_dump = 5;
} // namespace

void
vpic_simulation::dump_grid( const char *fbase ) {
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[4];

  if( !fbase ) ERROR(( "Invalid filename" ));
  if( rank()==0 ) MESSAGE(( "Dumping grid to \"%s\"", fbase ));

  snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail ) ERROR(( "Could not open \"%s\".", fname ));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  nxout = grid->nx;
  nyout = grid->ny;
  nzout = grid->nz;
  dxout = grid->dx;
  dyout = grid->dy;
  dzout = grid->dz;

  WRITE_HEADER_V0( dump_type::grid_dump, -1, 0, fileIO );

  dim[0] = 3;
  dim[1] = 3;
  dim[2] = 3;
  WRITE_ARRAY_HEADER( grid->bc, 3, dim, fileIO );
  fileIO.write( grid->bc, dim[0]*dim[1]*dim[2] );

  dim[0] = nproc()+1;
  WRITE_ARRAY_HEADER( grid->range, 1, dim, fileIO );
  fileIO.write( grid->range, dim[0] );

  dim[0] = 6;
  dim[1] = grid->nx+2;
  dim[2] = grid->ny+2;
  dim[3] = grid->nz+2;
  WRITE_ARRAY_HEADER( grid->neighbor, 4, dim, fileIO );
  fileIO.write( grid->neighbor, dim[0]*dim[1]*dim[2]*dim[3] );

  if( fileIO.close() ) ERROR(( "File close failed on dump grid!!!" ));
}

void
vpic_simulation::dump_fields( const char *fbase, int ftag ) {
    // Update the fields if necessary
    if (step() > field_array->last_copied)
        field_array->copy_to_host();

  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[3];

  if( !fbase ) ERROR(( "Invalid filename" ));

  if( rank()==0 ) MESSAGE(( "Dumping fields to \"%s\"", fbase ));

  if( ftag ) snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
  else       snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );

  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail ) ERROR(( "Could not open \"%s\".", fname ));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  nxout = grid->nx;
  nyout = grid->ny;
  nzout = grid->nz;
  dxout = grid->dx;
  dyout = grid->dy;
  dzout = grid->dz;

  WRITE_HEADER_V0( dump_type::field_dump, -1, 0, fileIO );

  dim[0] = grid->nx+2;
  dim[1] = grid->ny+2;
  dim[2] = grid->nz+2;
  WRITE_ARRAY_HEADER( field_array->f, 3, dim, fileIO );
  fileIO.write( field_array->f, dim[0]*dim[1]*dim[2] );
  if( fileIO.close() ) ERROR(( "File close failed on dump fields!!!" ));
}

void
vpic_simulation::dump_hydro( const char *sp_name,
                             const char *fbase,
                             bool tracers,
                             int ftag ) {


  species_t *sp;
  char fname[max_filename_bytes];
  FileIO fileIO;
  int dim[3];

  sp = find_species_name( sp_name, species_list );
  if( !sp ) ERROR(( "Invalid species \"%s\"", sp_name ));

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

  if( !fbase ) ERROR(( "Invalid filename" ));

  if( rank()==0 )
    MESSAGE(("Dumping \"%s\" hydro fields to \"%s\"",sp->name,fbase));

  if( ftag ) {
      snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
  }
  else {
      snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
  }

  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail) ERROR(( "Could not open \"%s\".", fname ));

  /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
  nxout = grid->nx;
  nyout = grid->ny;
  nzout = grid->nz;
  dxout = grid->dx;
  dyout = grid->dy;
  dzout = grid->dz;

  WRITE_HEADER_V0( dump_type::hydro_dump,sp->id,sp->q/sp->m,fileIO);

  dim[0] = grid->nx+2;
  dim[1] = grid->ny+2;
  dim[2] = grid->nz+2;
  WRITE_ARRAY_HEADER( hydro_array->h, 3, dim, fileIO );
  fileIO.write( hydro_array->h, dim[0]*dim[1]*dim[2] );
  if( fileIO.close() ) ERROR(( "File close failed on dump hydro!!!" ));
}

void
vpic_simulation::dump_hydro( const char *sp_name,
                             const char *fbase,
                             int ftag ) {
  dump_hydro(sp_name, fbase, false, ftag);
}

void
vpic_simulation::dump_particles( const char *sp_name,
                                 const char *fbase,
                                 int ftag )
{

    species_t *sp;
    char fname[max_filename_bytes];
    FileIO fileIO;
    int dim[1], buf_start;
    static particle_t * ALIGNED(128) p_buf = NULL;
# define PBUF_SIZE 32768 // 1MB of particles

    sp = find_species_name( sp_name, species_list );
    if( !sp ) ERROR(( "Invalid species name \"%s\".", sp_name ));

    if( !fbase ) ERROR(( "Invalid filename" ));

    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( !p_buf ) MALLOC_ALIGNED( p_buf, PBUF_SIZE, 128 );

    if( rank()==0 )
        MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,fbase));

    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
    }
    else {
        snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
    }

    FileIOStatus status = fileIO.open(fname, io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    /* IMPORTANT: these values are written in WRITE_HEADER_V0 */
    nxout = grid->nx;
    nyout = grid->ny;
    nzout = grid->nz;
    dxout = grid->dx;
    dyout = grid->dy;
    dzout = grid->dz;

    WRITE_HEADER_V0( dump_type::particle_dump, sp->id, sp->q/sp->m, fileIO );

    dim[0] = sp->np;
    WRITE_ARRAY_HEADER( p_buf, 1, dim, fileIO );

    // Copy a PBUF_SIZE hunk of the particle list into the particle
    // buffer, timecenter it and write it out. This is done this way to
    // guarantee the particle list unchanged while not requiring too
    // much memory.

    // FIXME: WITH A PIPELINED CENTER_P, PBUF NOMINALLY SHOULD BE QUITE
    // LARGE.

    particle_t * sp_p = sp->p;      sp->p      = p_buf;
    int sp_np         = sp->np;     sp->np     = 0;
    int sp_max_np     = sp->max_np; sp->max_np = PBUF_SIZE;
    for( buf_start=0; buf_start<sp_np; buf_start += PBUF_SIZE ) {
        sp->np = sp_np-buf_start; if( sp->np > PBUF_SIZE ) sp->np = PBUF_SIZE;
        COPY( sp->p, &sp_p[buf_start], sp->np );
        center_p( sp, interpolator_array );
        fileIO.write( sp->p, sp->np );
    }
    sp->p      = sp_p;
    sp->np     = sp_np;
    sp->max_np = sp_max_np;

    if( fileIO.close() ) ERROR(("File close failed on dump particles!!!"));
}

/*------------------------------------------------------------------------------
 * HDF5 Dumps
 *---------------------------------------------------------------------------*/
void vpic_simulation::dump_tracers_hdf5(const char* sp_name, 
                                        const char*fbase, 
                                        int append, 
                                        int fname_tag) {
//  char fname[256];
//  char group_name[256];
//  char particle_scratch[128];
//  char subparticle_scratch[128];
//    
//  species_t* sp = find_species_name( sp_name, tracers_list );
//  if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name));
//
//  // Update the particles on the host only if they haven't been recently
//  if (step() > sp->last_copied)
//    sp->copy_to_host();
//
//  // Control amount of data dumped with stride TODO: Make user adjustable
//  const int stride_particle_dump = 1;
//  const long long np_local = (sp->np + stride_particle_dump - 1) / stride_particle_dump;
//  
////  // Timing measurement flag
////  bool print_timing = false;
////  double ec1 = uptime();
//
//  int sp_np = sp->np;
//  int sp_max_np = sp->max_np;
//
//  // Copy particles?
//  
//  // Center particles?
//
////  ec1 = uptime() - ec1;
////  if(print_timing) MESSAGE(("Time in copying particle data: %fs, np_local = %lld", ec1, np_local));
//
//  // Create target directory and subdirectory for the timestep
//  sprintf(particle_scratch, "./%s", "particle_hdf5");
//  sprintf(subparticle_scratch, "%s/T.%ld/", particle_scratch, step());
//  dump_mkdir(particle_scratch);
//  dump_mkdir(subparticle_scratch);
//
//  // Open HDF5 file for species
//  sprintf(fname, "%s/%s_%ld.h5", subparticle_scratch, sp->name, step());
//  sprintf(group_name, "/Timestep_%ld", step());
//  double el1 = uptime();
//  
//  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS); // Create property list with file access property
//  h5Pset_fapl_mpio(plist_d, MPI_COMM_WORLD, MPI_INFO_NULL); // Store MPI IO communication info to file access property list (MPI_INFO_NULL == No hints)
//  hid_t file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id); // Create HDF5 file, truncate if it already exists. Use default file creation property list
//  hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); // Create group with default link, group creation, and group access property lists
//
//  H5Pclose(plist_id); // Close property list
//
//  //Calculate the total number of particles and the offsets for each rank
//  long long total_particles, offset;
//  long long numparticles = np_local;
//  MPI_Allreduce(&numparticles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); 
//  MPI_Scan(&numparticles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
//  offset -= numparticles; 
//
//  // Create new simple dataspace and open
//  hid_t filespace = H5Screate_simple(1, (hsize_t *)&total_particles, NULL);
//
//  hsize_t memspace_count_temp = numparticles * 8;
//  hid_t memspace = H5Screate_simple(1, &memspace_count_temp, NULL);
//
}

/*------------------------------------------------------------------------------
 * New dump logic
 *---------------------------------------------------------------------------*/

#include <iostream>

static FieldInfo fieldInfo[12] = {
	{ "Electric Field", "VECTOR", "3", "FLOATING_POINT", sizeof(float) },
	{ "Electric Field Divergence Error", "SCALAR", "1", "FLOATING_POINT",
		sizeof(float) },
	{ "Magnetic Field", "VECTOR", "3", "FLOATING_POINT", sizeof(float) },
	{ "Magnetic Field Divergence Error", "SCALAR", "1", "FLOATING_POINT",
		sizeof(float) },
	{ "TCA Field", "VECTOR", "3", "FLOATING_POINT", sizeof(float) },
	{ "Bound Charge Density", "SCALAR", "1", "FLOATING_POINT", sizeof(float) },
	{ "Free Current Field", "VECTOR", "3", "FLOATING_POINT", sizeof(float) },
	{ "Charge Density", "SCALAR", "1", "FLOATING_POINT", sizeof(float) },
	{ "Edge Material", "VECTOR", "3", "INTEGER", sizeof(material_id) },
	{ "Node Material", "SCALAR", "1", "INTEGER", sizeof(material_id) },
	{ "Face Material", "VECTOR", "3", "INTEGER", sizeof(material_id) },
	{ "Cell Material", "SCALAR", "1", "INTEGER", sizeof(material_id) }
}; // fieldInfo

static HydroInfo hydroInfo[5] = {
	{ "Current Density", "VECTOR", "3", "FLOATING_POINT", sizeof(float) },
	{ "Charge Density", "SCALAR", "1", "FLOATING_POINT", sizeof(float) },
	{ "Momentum Density", "VECTOR", "3", "FLOATING_POINT", sizeof(float) },
	{ "Kinetic Energy Density", "SCALAR", "1", "FLOATING_POINT",
		sizeof(float) },
	{ "Stress Tensor", "TENSOR", "6", "FLOATING_POINT", sizeof(float) }
	/*
	{ "STRESS_DIAGONAL", "VECTOR", "3", "FLOATING_POINT", sizeof(float) }
	{ "STRESS_OFFDIAGONAL", "VECTOR", "3", "FLOATING_POINT", sizeof(float) }
	*/
}; // hydroInfo

void
vpic_simulation::create_field_list( char * strlist,
                                    DumpParameters & dumpParams ) {
  strcpy(strlist, "");
  for(size_t i(0), pass(0); i<total_field_groups; i++)
    if(dumpParams.output_vars.bitset(field_indeces[i])) {
      if(i>0 && pass) strcat(strlist, ", ");
      else pass = 1;
      strcat(strlist, fieldInfo[i].name);
    }
}

void
vpic_simulation::create_hydro_list( char * strlist,
                                    DumpParameters & dumpParams ) {
  strcpy(strlist, "");
  for(size_t i(0), pass(0); i<total_hydro_groups; i++)
    if(dumpParams.output_vars.bitset(hydro_indeces[i])) {
      if(i>0 && pass) strcat(strlist, ", ");
      else pass = 1;
      strcat(strlist, hydroInfo[i].name);
    }
}

void
vpic_simulation::print_hashed_comment( FileIO & fileIO,
                                       const char * comment) {
  fileIO.print("################################################################################\n");
  fileIO.print("# %s\n", comment);
  fileIO.print("################################################################################\n");
}

void
vpic_simulation::global_header( const char * base,
                                std::vector<DumpParameters *> dumpParams ) {
  if( rank() ) return;

  // Open the file for output
  char filename[max_filename_bytes];
  snprintf(filename, max_filename_bytes, "%s.vpc", base);

  FileIO fileIO;
  FileIOStatus status;

  status = fileIO.open(filename, io_write);
  if(status == fail) ERROR(("Failed opening file: %s", filename));

  print_hashed_comment(fileIO, "Header version information");
  fileIO.print("VPIC_HEADER_VERSION 1.0.0\n\n");

  print_hashed_comment(fileIO,
                       "Header size for data file headers in bytes");
  fileIO.print("DATA_HEADER_SIZE 123\n\n");

  // Global grid inforation
  print_hashed_comment(fileIO, "Time step increment");
  fileIO.print("GRID_DELTA_T %f\n\n", grid->dt);

  print_hashed_comment(fileIO, "GRID_CVAC");
  fileIO.print("GRID_CVAC %f\n\n", grid->cvac);

  print_hashed_comment(fileIO, "GRID_EPS0");
  fileIO.print("GRID_EPS0 %f\n\n", grid->eps0);

  print_hashed_comment(fileIO, "Grid extents in the x-dimension");
  fileIO.print("GRID_EXTENTS_X %f %f\n\n", grid->x0, grid->x1);

  print_hashed_comment(fileIO, "Grid extents in the y-dimension");
  fileIO.print("GRID_EXTENTS_Y %f %f\n\n", grid->y0, grid->y1);

  print_hashed_comment(fileIO, "Grid extents in the z-dimension");
  fileIO.print("GRID_EXTENTS_Z %f %f\n\n", grid->z0, grid->z1);

  print_hashed_comment(fileIO, "Spatial step increment in x-dimension");
  fileIO.print("GRID_DELTA_X %f\n\n", grid->dx);

  print_hashed_comment(fileIO, "Spatial step increment in y-dimension");
  fileIO.print("GRID_DELTA_Y %f\n\n", grid->dy);

  print_hashed_comment(fileIO, "Spatial step increment in z-dimension");
  fileIO.print("GRID_DELTA_Z %f\n\n", grid->dz);

  print_hashed_comment(fileIO, "Domain partitions in x-dimension");
  fileIO.print("GRID_TOPOLOGY_X %d\n\n", px);

  print_hashed_comment(fileIO, "Domain partitions in y-dimension");
  fileIO.print("GRID_TOPOLOGY_Y %d\n\n", py);

  print_hashed_comment(fileIO, "Domain partitions in z-dimension");
  fileIO.print("GRID_TOPOLOGY_Z %d\n\n", pz);

  // Global data inforation
  assert(dumpParams.size() >= 2);

  print_hashed_comment(fileIO, "Field data information");
  fileIO.print("FIELD_DATA_DIRECTORY %s\n", dumpParams[0]->baseDir);
  fileIO.print("FIELD_DATA_BASE_FILENAME %s\n",
               dumpParams[0]->baseFileName);

  // Create a variable list of field values to output.
  size_t numvars = std::min(dumpParams[0]->output_vars.bitsum(field_indeces,
                                                              total_field_groups),
                            total_field_groups);
  size_t * varlist = new size_t[numvars];
  for(size_t v(0), c(0); v<total_field_groups; v++)
    if(dumpParams[0]->output_vars.bitset(field_indeces[v]))
      varlist[c++] = v;

  // output variable list
  fileIO.print("FIELD_DATA_VARIABLES %d\n", numvars);

  for(size_t v(0); v<numvars; v++)
    fileIO.print("\"%s\" %s %s %s %d\n", fieldInfo[varlist[v]].name,
                 fieldInfo[varlist[v]].degree, fieldInfo[varlist[v]].elements,
                 fieldInfo[varlist[v]].type, fieldInfo[varlist[v]].size);

  fileIO.print("\n");

  delete[] varlist;
  varlist = NULL;

  // Create a variable list for each species to output
  print_hashed_comment(fileIO, "Number of species with output data");
  fileIO.print("NUM_OUTPUT_SPECIES %d\n\n", dumpParams.size()-1);
  char species_comment[128];
  for(size_t i(1); i<dumpParams.size(); i++) {
    numvars = std::min(dumpParams[i]->output_vars.bitsum(hydro_indeces,
                                                         total_hydro_groups),
                       total_hydro_groups);

    snprintf(species_comment, max_filename_bytes, "Species(%d) data information", (int)i);
    print_hashed_comment(fileIO, species_comment);
    fileIO.print("SPECIES_DATA_DIRECTORY %s\n",
                 dumpParams[i]->baseDir);
    fileIO.print("SPECIES_DATA_BASE_FILENAME %s\n",
                 dumpParams[i]->baseFileName);

    fileIO.print("HYDRO_DATA_VARIABLES %d\n", numvars);

    varlist = new size_t[numvars];
    for(size_t v(0), c(0); v<total_hydro_groups; v++)
      if(dumpParams[i]->output_vars.bitset(hydro_indeces[v]))
        varlist[c++] = v;

    for(size_t v(0); v<numvars; v++)
      fileIO.print("\"%s\" %s %s %s %d\n", hydroInfo[varlist[v]].name,
                   hydroInfo[varlist[v]].degree, hydroInfo[varlist[v]].elements,
                   hydroInfo[varlist[v]].type, hydroInfo[varlist[v]].size);


    delete[] varlist;
    varlist = NULL;

    if(i<dumpParams.size()-1) fileIO.print("\n");
  }


  if( fileIO.close() ) ERROR(( "File close failed on global header!!!" ));
}

void
vpic_simulation::field_dump( DumpParameters & dumpParams ) {

    // Update the fields if necessary
    if (step() > field_array->last_copied)
      field_array->copy_to_host();

  // Create directory for this time step
  char timeDir[max_filename_bytes];
  int ret = snprintf(timeDir, max_filename_bytes, "%s/T.%ld", dumpParams.baseDir, (long)step());
  if (ret < 0) {
      ERROR(("snprintf failed"));
  }
  dump_mkdir(timeDir);

  // Open the file for output
  char filename[max_filename_bytes];
  ret = snprintf(filename, max_filename_bytes, "%s/T.%ld/%s.%ld.%d", dumpParams.baseDir, (long)step(),
          dumpParams.baseFileName, (long)step(), rank());
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
  nxout = (grid->nx)/istride;
  nyout = (grid->ny)/jstride;
  nzout = (grid->nz)/kstride;
  dxout = (grid->dx)*istride;
  dyout = (grid->dy)*jstride;
  dzout = (grid->dz)*kstride;

  /* Banded output will write data as a single block-array as opposed to
   * the Array-of-Structure format that is used for native storage.
   *
   * Additionally, the user can specify a stride pattern to reduce
   * the resolution of the data that are output.  If a stride is
   * specified for a particular dimension, VPIC will write the boundary
   * plus every "stride" elements in that dimension. */

  if(dumpParams.format == band) {

    WRITE_HEADER_V0(dump_type::field_dump, -1, 0, fileIO);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

    if( rank()==VERBOSE_rank ) {
      std::cerr << "nxout: " << nxout << std::endl;
      std::cerr << "nyout: " << nyout << std::endl;
      std::cerr << "nzout: " << nzout << std::endl;
      std::cerr << "nx: " << grid->nx << std::endl;
      std::cerr << "ny: " << grid->ny << std::endl;
      std::cerr << "nz: " << grid->nz << std::endl;
    }

    WRITE_ARRAY_HEADER(field_array->f, 3, dim, fileIO);

    // Create a variable list of field values to output.
    size_t numvars = std::min(dumpParams.output_vars.bitsum(),
                              total_field_variables);
    size_t * varlist = new size_t[numvars];

    for(size_t i(0), c(0); i<total_field_variables; i++)
      if(dumpParams.output_vars.bitset(i)) varlist[c++] = i;

    if( rank()==VERBOSE_rank ) printf("\nBEGIN_OUTPUT\n");

    // more efficient for standard case
    if(istride == 1 && jstride == 1 && kstride == 1)
      for(size_t v(0); v<numvars; v++) {
      for(size_t k(0); k<nzout+2; k++) {
      for(size_t j(0); j<nyout+2; j++) {
      for(size_t i(0); i<nxout+2; i++) {
              const uint32_t * fref = reinterpret_cast<uint32_t *>(&field_array->f(i,j,k));
              fileIO.write(&fref[varlist[v]], 1);
              if(rank()==VERBOSE_rank) printf("%f ", field_array->f(i,j,k).ex);
              if(rank()==VERBOSE_rank) std::cout << "(" << i << " " << j << " " << k << ")" << std::endl;
      } if(rank()==VERBOSE_rank) std::cout << std::endl << "ROW_BREAK " << j << " " << k << std::endl;
      } if(rank()==VERBOSE_rank) std::cout << std::endl << "PLANE_BREAK " << k << std::endl;
      } if(rank()==VERBOSE_rank) std::cout << std::endl << "BLOCK_BREAK" << std::endl;
      }

    else

      for(size_t v(0); v<numvars; v++) {
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
              const uint32_t * fref = reinterpret_cast<uint32_t *>(&field_array->f(ioff,joff,koff));
              fileIO.write(&fref[varlist[v]], 1);
              if(rank()==VERBOSE_rank) printf("%f ", field_array->f(ioff,joff,koff).ex);
              if(rank()==VERBOSE_rank) std::cout << "(" << ioff << " " << joff << " " << koff << ")" << std::endl;
      } if(rank()==VERBOSE_rank) std::cout << std::endl << "ROW_BREAK " << joff << " " << koff << std::endl;
      } if(rank()==VERBOSE_rank) std::cout << std::endl << "PLANE_BREAK " << koff << std::endl;
      } if(rank()==VERBOSE_rank) std::cout << std::endl << "BLOCK_BREAK" << std::endl;
      }

    delete[] varlist;

  } else { // band_interleave

    WRITE_HEADER_V0(dump_type::field_dump, -1, 0, fileIO);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

    WRITE_ARRAY_HEADER(field_array->f, 3, dim, fileIO);

    if(istride == 1 && jstride == 1 && kstride == 1)
      fileIO.write(field_array->f, dim[0]*dim[1]*dim[2]);
    else
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
            fileIO.write(&field_array->f(ioff,joff,koff), 1);
      }
      }
      }
  }

# undef f

  if( fileIO.close() ) ERROR(( "File close failed on field dump!!!" ));
}

void
vpic_simulation::hydro_dump( const char * speciesname,
                             DumpParameters & dumpParams ) {

  // Create directory for this time step
  char timeDir[max_filename_bytes];
  snprintf(timeDir, max_filename_bytes, "%s/T.%ld", dumpParams.baseDir, (long)step());
  dump_mkdir(timeDir);

  // Open the file for output
  char filename[max_filename_bytes];
  int ret = snprintf( filename, max_filename_bytes, "%s/T.%ld/%s.%ld.%d", dumpParams.baseDir, (long)step(),
           dumpParams.baseFileName, (long)step(), rank() );
  if (ret < 0) {
      ERROR(("snprintf failed"));
  }

  FileIO fileIO;
  FileIOStatus status;

  status = fileIO.open(filename, io_write);
  if(status == fail) ERROR(("Failed opening file: %s", filename));

  species_t * sp = find_species_name(speciesname, species_list);
  if( !sp ) ERROR(( "Invalid species name: %s", speciesname ));

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
  nxout = (grid->nx)/istride;
  nyout = (grid->ny)/jstride;
  nzout = (grid->nz)/kstride;
  dxout = (grid->dx)*istride;
  dyout = (grid->dy)*jstride;
  dzout = (grid->dz)*kstride;

  /* Banded output will write data as a single block-array as opposed to
   * the Array-of-Structure format that is used for native storage.
   *
   * Additionally, the user can specify a stride pattern to reduce
   * the resolution of the data that are output.  If a stride is
   * specified for a particular dimension, VPIC will write the boundary
   * plus every "stride" elements in that dimension.
   */
  if(dumpParams.format == band) {

    WRITE_HEADER_V0(dump_type::hydro_dump, sp->id, sp->q/sp->m, fileIO);

    dim[0] = nxout+2;
    dim[1] = nyout+2;
    dim[2] = nzout+2;

    WRITE_ARRAY_HEADER(hydro_array->h, 3, dim, fileIO);

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
              const uint32_t * href = reinterpret_cast<uint32_t *>(&hydro(i,j,k));
              fileIO.write(&href[varlist[v]], 1);
      }

    else

      for(size_t v(0); v<numvars; v++)
      for(size_t k(0); k<nzout+2; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout+2; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout+2; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
              const uint32_t * href = reinterpret_cast<uint32_t *>(&hydro(ioff,joff,koff));
              fileIO.write(&href[varlist[v]], 1);
      }
      }
      }

    delete[] varlist;

  } else { // band_interleave

    WRITE_HEADER_V0(dump_type::hydro_dump, sp->id, sp->q/sp->m, fileIO);

    dim[0] = nxout;
    dim[1] = nyout;
    dim[2] = nzout;

    WRITE_ARRAY_HEADER(hydro_array->h, 3, dim, fileIO);

    if(istride == 1 && jstride == 1 && kstride == 1)

      fileIO.write(hydro_array->h, dim[0]*dim[1]*dim[2]);

    else

      for(size_t k(0); k<nzout; k++) { const size_t koff = (k == 0) ? 0 : (k == nzout+1) ? grid->nz+1 : k*kstride;
      for(size_t j(0); j<nyout; j++) { const size_t joff = (j == 0) ? 0 : (j == nyout+1) ? grid->ny+1 : j*jstride;
      for(size_t i(0); i<nxout; i++) { const size_t ioff = (i == 0) ? 0 : (i == nxout+1) ? grid->nx+1 : i*istride;
            fileIO.write(&hydro(ioff,joff,koff), 1);
      }
      }
      }
  }

# undef hydro

  if( fileIO.close() ) ERROR(( "File close failed on hydro dump!!!" ));
}
