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

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  std::map<std::string, double> energy_map;
//  LIST_FOR_EACH(sp, species_list) {
//    en_p = energy_p_kokkos( sp, interpolator_array );
//    if(rank() == 0 && status!=fail) {
//      std::string sp_name = std::string(sp->name);
//      if(sp->is_tracer) {
//        sp_name = std::string(sp->parent_species->name);
//      }
//      if(energy_map.find(sp_name) == energy_map.end()) {
//        energy_map[sp_name] = en_p;
//      } else {
//        energy_map[sp_name] += en_p;
//      }
//    }
//  }
//  LIST_FOR_EACH(sp,species_list) {
//    if( rank()==0 && !(sp->is_tracer) ) fileIO.print( " %e", energy_map[sp->name] );
//  }

  LIST_FOR_EACH(sp,species_list) {
    en_p = energy_p_kokkos( sp, interpolator_array );
    if(rank() == 0 && status!=fail && !sp->is_tracer) {
      std::string sp_name = std::string(sp->name);
      energy_map[sp_name] = en_p;
    }
  }
  LIST_FOR_EACH(sp,tracers_list) {
    en_p = energy_p_kokkos( sp, interpolator_array );
    if(rank() == 0 && status!=fail && (sp->parent_species == NULL)) {
      std::string sp_name = std::string(sp->name);
      energy_map[sp_name] = en_p;
    }
    if(rank() == 0 && status!=fail && sp->is_tracer) {
      std::string sp_name = std::string(sp->parent_species->name);
      energy_map[sp_name] += en_p;
    }
  }
  LIST_FOR_EACH(sp,species_list) {
    if( rank()==0 && !sp->is_tracer ) fileIO.print( " %e", energy_map[sp->name] );
  }
  LIST_FOR_EACH(sp,tracers_list) {
    if( rank()==0 && (sp->parent_species == NULL) ) fileIO.print( " %e", energy_map[sp->name] );
  }
#else
  LIST_FOR_EACH(sp,species_list) {
    en_p = energy_p_kokkos( sp, interpolator_array );
    if( rank()==0 && status!=fail ) fileIO.print( " %e", en_p );
  }
#endif

  if( rank()==0 && status!=fail ) {
    fileIO.print( "\n" );
    if( fileIO.close() ) ERROR(("File close failed on dump energies!!!"));
  }
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

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
void
vpic_simulation::dump_particles_csv( const char *sp_name,
                                     uint32_t dump_vars,
                                     const char *fbase,
                                     const int append,
                                     int ftag )
{
    species_t *sp;
    char fname[max_filename_bytes];
    FileIO fileIO;

    // Get species
    sp = find_species_name( sp_name, species_list );
    if(sp == NULL)
      sp = find_species_name(sp_name, tracers_list);
    if( !sp ) ERROR(( "Invalid species name \"%s\".", sp_name ));

    if( !fbase ) ERROR(( "Invalid filename" ));

    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( rank()==0 )
        MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,fbase));

    // Create output filename
    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
    }
    else {
        snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
    }

    FileIOStatus status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    // Create header string
    std::string header_str = std::string("Timestep,rank,cell_id,dx,dy,dz,ux,uy,uz,w");
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
      header_str += ",ke_dens";
    }
    if(dump_vars & DumpVar::StressTensor) {
      header_str += ",txx,tyy,tzz,tyz,tzx,txy";
    }
    if(dump_vars & DumpVar::ParticleKE) {
      header_str += ",ke";
    }
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    if(sp->using_annotations) {
      for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
        header_str += ",";
        header_str += sp->annotation_vars.i32_vars[j].c_str();
      }
      for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
        header_str += ",";
        header_str += sp->annotation_vars.i64_vars[j].c_str();
      }
      for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
        header_str += ",";
        header_str += sp->annotation_vars.f32_vars[j].c_str();
      }
      for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
        header_str += ",";
        header_str += sp->annotation_vars.f64_vars[j].c_str();
      }
    }
#endif

    // Write header string
    if( append==0 ) {
      fileIO.print( "%s\n", header_str.c_str() );
    }

    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, copy weight back to particles for hydro quantities
    if(sp->using_annotations && sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }
#endif

    // Compute hydro quantities
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

//    int tracer_idx = sp->annotation_vars.get_annotation_index<int64_t>("TracerID");
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
    // Write tracer data
    for(int32_t i=0; i<sp->np; i++) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      float ux0 = sp->k_p_h(i, particle_var::ux);
      float uy0 = sp->k_p_h(i, particle_var::uy);
      float uz0 = sp->k_p_h(i, particle_var::uz);
      float w0  = sp->k_p_h(i, particle_var::w);
      int   ii  = sp->k_p_i_h(i);
      fileIO.print("%ld,%d,%d,%e,%e,%e,%e,%e,%e,%e", 
        step(), rank(), ii,
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
      if(dump_vars & DumpVar::ParticleKE) {
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        fileIO.print(",%e", v0);
      }
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
      // Print annotations
      if(sp->using_annotations) {
        for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
          fileIO.print(",%d", sp->annotations_h.get<int>(i, j));
        }
        for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
          fileIO.print(",%ld", sp->annotations_h.get<int64_t>(i, j));
        }
        for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
          fileIO.print(",%e", sp->annotations_h.get<float>(i, j));
        }
        for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
          fileIO.print(",%e", sp->annotations_h.get<double>(i, j));
        }
      }
#endif
      fileIO.print( "\n" );
    }
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, copy weight back to particles for hydro quantities
    if(sp->using_annotations && sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }
#endif
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

void
vpic_simulation::dump_tracers_buffered_csv( const char *sp_name,
                                            uint32_t dump_vars,
                                            const char *fbase,
                                            const int append,
                                            int ftag )
{

  species_t *sp;
  char fname[max_filename_bytes];
  FileIO fileIO;

  // Get species
  sp = find_species_name( sp_name, tracers_list );
  if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name ));

  if( !fbase ) ERROR(( "Invalid filename" ));

  // Create and write header string
  if( append==0 ) {
    // Create output filename
    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
    } else {
        snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
    }

    FileIOStatus status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    std::string header_str = std::string("Timestep,rank,tracer_id,cell_id,dx,dy,dz,ux,uy,uz,w");
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
      header_str += ",ke_dens";
    }
    if(dump_vars & DumpVar::StressTensor) {
      header_str += ",txx,tyy,tzz,tyz,tzx,txy";
    }
    if(dump_vars & DumpVar::ParticleKE) {
      header_str += ",ke";
    }
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.i32_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.i64_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.f32_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.f64_vars[j].c_str();
    }
#endif
    fileIO.print( "%s\n", header_str.c_str() );
    if( fileIO.close() ) ERROR(("File close failed on writing header for dump particles!!!"));
  }

  // Buffer tracers
  if(sp->nparticles_buffered+sp->np < sp->particle_io_buffer.extent(0) && step() != num_step) {
    printf("Buffering %d particles (%d already buffered, %lu max)\n", sp->np, sp->nparticles_buffered, sp->particle_io_buffer.extent(0));
    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( rank()==0 )
        MESSAGE(("Buffering \"%s\" particles",sp->name));

    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }
#endif

    // Compute hydro quantities
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

    // Get index of tracer ID
//    int tracer_idx = sp->annotation_vars.get_annotation_index<int64_t>("TracerID");
    auto& interp = interpolator_array->k_i_h;

    auto particle_slice = Kokkos::make_pair(0, sp->np);
    auto buffer_slice = Kokkos::make_pair(sp->nparticles_buffered,sp->nparticles_buffered+sp->np);

    // Copy particles into buffer
    auto particle_subview = Kokkos::subview(sp->k_p_h, particle_slice, Kokkos::ALL());
    auto particle_buffer_subview = Kokkos::subview(sp->particle_io_buffer, buffer_slice, Kokkos::ALL());
    auto particle_cell_subview = Kokkos::subview(sp->k_p_i_h, particle_slice);
    auto particle_cell_buffer_subview = Kokkos::subview(sp->particle_cell_io_buffer, buffer_slice);
    Kokkos::deep_copy(particle_buffer_subview, particle_subview); 
    Kokkos::deep_copy(particle_cell_buffer_subview, particle_cell_subview); 

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // Copy annotations into buffers
    for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.i32, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.i32, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
    for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.i64, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.i64, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
    for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.f32, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.f32, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
    for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.f64, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.f64, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
#endif

    // Buffer tracer data
    sp->np_per_ts_io_buffer.push_back(std::make_pair(sp->np, step()));
    for(uint32_t i=0; i<sp->np; i++) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      float ux0 = sp->k_p_h(i, particle_var::ux);
      float uy0 = sp->k_p_h(i, particle_var::uy);
      float uz0 = sp->k_p_h(i, particle_var::uz);
      float w0  = sp->k_p_h(i, particle_var::w);
      int   ii  = sp->k_p_i_h(i);
      if(dump_vars & DumpVar::Efield) {
        float ex  = interp(ii,interpolator_var::ex ) + dy0*interp(ii,interpolator_var::dexdy) + dz0*(interp(ii,interpolator_var::dexdz) + dy0*interp(ii,interpolator_var::d2exdydz)); 
        float ey  = interp(ii,interpolator_var::ey ) + dz0*interp(ii,interpolator_var::deydz) + dx0*(interp(ii,interpolator_var::deydx) + dz0*interp(ii,interpolator_var::d2eydzdx)); 
        float ez  = interp(ii,interpolator_var::ez ) + dx0*interp(ii,interpolator_var::dezdx) + dy0*(interp(ii,interpolator_var::dezdy) + dx0*interp(ii,interpolator_var::d2ezdxdy)); 
        sp->efields_io_buffer(sp->nparticles_buffered+i, 0) = ex;
        sp->efields_io_buffer(sp->nparticles_buffered+i, 1) = ey;
        sp->efields_io_buffer(sp->nparticles_buffered+i, 2) = ez;
      }
      if(dump_vars & DumpVar::Bfield) {
        float bx  = interp(ii,interpolator_var::cbx) + dx0*interp(ii,interpolator_var::dcbxdx); 
        float by  = interp(ii,interpolator_var::cby) + dy0*interp(ii,interpolator_var::dcbydy); 
        float bz  = interp(ii,interpolator_var::cbz) + dz0*interp(ii,interpolator_var::dcbzdz); 
        sp->bfields_io_buffer(sp->nparticles_buffered+i, 0) = bx;
        sp->bfields_io_buffer(sp->nparticles_buffered+i, 1) = by;
        sp->bfields_io_buffer(sp->nparticles_buffered+i, 2) = bz;
      }
      if(dump_vars & DumpVar::CurrentDensity) {
        float jx  = hydro_array->k_h_h(ii, hydro_var::jx);
        float jy  = hydro_array->k_h_h(ii, hydro_var::jy);
        float jz  = hydro_array->k_h_h(ii, hydro_var::jz);
        sp->current_dens_io_buffer(sp->nparticles_buffered+i, 0) = jx;
        sp->current_dens_io_buffer(sp->nparticles_buffered+i, 1) = jy;
        sp->current_dens_io_buffer(sp->nparticles_buffered+i, 2) = jz;
      }
      if(dump_vars & DumpVar::ChargeDensity) {
        float rho = hydro_array->k_h_h(ii, hydro_var::rho);
        sp->charge_dens_io_buffer(sp->nparticles_buffered+i) = rho;
      }
      if(dump_vars & DumpVar::MomentumDensity) {
        float px  = hydro_array->k_h_h(ii, hydro_var::px);
        float py  = hydro_array->k_h_h(ii, hydro_var::py);
        float pz  = hydro_array->k_h_h(ii, hydro_var::pz);
        sp->momentum_dens_io_buffer(sp->nparticles_buffered+i, 0) = px;
        sp->momentum_dens_io_buffer(sp->nparticles_buffered+i, 1) = py;
        sp->momentum_dens_io_buffer(sp->nparticles_buffered+i, 2) = pz;
      }
      if(dump_vars & DumpVar::KEDensity) {
        float ke = hydro_array->k_h_h(ii, hydro_var::ke);
        sp->ke_dens_io_buffer(sp->nparticles_buffered+i) = ke;
      }
      if(dump_vars & DumpVar::StressTensor) {
        float txx = hydro_array->k_h_h(ii, hydro_var::txx);
        float tyy = hydro_array->k_h_h(ii, hydro_var::tyy);
        float tzz = hydro_array->k_h_h(ii, hydro_var::tzz);
        float tyz = hydro_array->k_h_h(ii, hydro_var::tyz);
        float tzx = hydro_array->k_h_h(ii, hydro_var::tzx);
        float txy = hydro_array->k_h_h(ii, hydro_var::txy);
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 0) = txx;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 1) = tyy;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 2) = tzz;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 3) = tyz;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 4) = tzx;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 5) = txy;
      }
      if(dump_vars & DumpVar::ParticleKE) {
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        sp->particle_ke_io_buffer(sp->nparticles_buffered+i) = v0;
      }
    }
    sp->nparticles_buffered += sp->np;     
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }
#endif
  } else { // Dump buffered tracers
    printf("Writing %d particles and %d buffered particles\n", sp->np, sp->nparticles_buffered);
    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( rank()==0 )
        MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,fbase));

    // Create output filename
    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
    }
    else {
        snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
    }

    FileIOStatus status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    // Get references to necessary data structures
    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }
#endif

    // Compute hydro quantities
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

    int tracer_idx = sp->annotation_vars.get_annotation_index<int64_t>("TracerID");
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
    // Write buffered tracer data
    uint32_t particle_idx = 0;
    // Iterate through each timestep
    for(uint32_t ts_idx=0; ts_idx < sp->np_per_ts_io_buffer.size(); ts_idx++) {
      int64_t time_step = sp->np_per_ts_io_buffer[ts_idx].second;
      // Iterate through each particle in the timestep
      for(uint32_t p_idx=0; p_idx<sp->np_per_ts_io_buffer[ts_idx].first; p_idx++) {
        uint32_t i = particle_idx + p_idx;
        float dx0 = sp->particle_io_buffer(i, particle_var::dx);
        float dy0 = sp->particle_io_buffer(i, particle_var::dy);
        float dz0 = sp->particle_io_buffer(i, particle_var::dz);
        float ux0 = sp->particle_io_buffer(i, particle_var::ux);
        float uy0 = sp->particle_io_buffer(i, particle_var::uy);
        float uz0 = sp->particle_io_buffer(i, particle_var::uz);
        float w0  = sp->particle_io_buffer(i, particle_var::w);
        int   ii  = sp->particle_cell_io_buffer(i);
        fileIO.print("%ld,%d,%ld,%d,%e,%e,%e,%e,%e,%e,%e", 
          time_step, rank(), sp->annotations_io_buffer.get<int64_t>(i,tracer_idx), ii,
          dx0, dy0, dz0, ux0, uy0, uz0, w0);
        if(dump_vars & DumpVar::GlobalPos) {
          fileIO.print(",%e,%e,%e", tracer_x, tracer_y, tracer_z);
        }
        if(dump_vars & DumpVar::Efield) {
          float ex  = sp->efields_io_buffer(i, 0);
          float ey  = sp->efields_io_buffer(i, 1);
          float ez  = sp->efields_io_buffer(i, 2);
          fileIO.print(",%e,%e,%e", ex, ey, ez);
        }
        if(dump_vars & DumpVar::Bfield) {
          float bx  = sp->bfields_io_buffer(i,0); 
          float by  = sp->bfields_io_buffer(i,1); 
          float bz  = sp->bfields_io_buffer(i,2); 
          fileIO.print(",%e,%e,%e", bx, by, bz);
        }
        if(dump_vars & DumpVar::CurrentDensity) {
          float jx  = sp->current_dens_io_buffer(i,0);
          float jy  = sp->current_dens_io_buffer(i,1);
          float jz  = sp->current_dens_io_buffer(i,2);
          fileIO.print(",%e,%e,%e", jx, jy, jz);
        }
        if(dump_vars & DumpVar::ChargeDensity) {
          float rho = sp->charge_dens_io_buffer(i);
          fileIO.print(",%e", rho);
        }
        if(dump_vars & DumpVar::MomentumDensity) {
          float px  = sp->momentum_dens_io_buffer(i,0);
          float py  = sp->momentum_dens_io_buffer(i,1);
          float pz  = sp->momentum_dens_io_buffer(i,2);
          fileIO.print(",%e,%e,%e", px, py, pz);
        }
        if(dump_vars & DumpVar::KEDensity) {
          float ke = sp->ke_dens_io_buffer(i);
          fileIO.print(",%e", ke);
        }
        if(dump_vars & DumpVar::StressTensor) {
          float txx = sp->stress_tensor_io_buffer(i,0);
          float tyy = sp->stress_tensor_io_buffer(i,1);
          float tzz = sp->stress_tensor_io_buffer(i,2);
          float tyz = sp->stress_tensor_io_buffer(i,3);
          float tzx = sp->stress_tensor_io_buffer(i,4);
          float txy = sp->stress_tensor_io_buffer(i,5);
          fileIO.print(",%e,%e,%e,%e,%e,%e", txx, tyy, tzz, tyz, tzx, txy);
        }
        if(dump_vars & DumpVar::ParticleKE) {
          float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
          float msp = sp->m;
          float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                                 dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
          float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                                 dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
          float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                                 dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
          v0 = v0*v0 + v1*v1 + v2*v2;
          v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
          fileIO.print(",%e", v0);
        }
        // Print annotations
        for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
          fileIO.print(",%d", sp->annotations_io_buffer.get<int>(i, j));
        }
        for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
          fileIO.print(",%ld", sp->annotations_io_buffer.get<int64_t>(i, j));
        }
        for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
          fileIO.print(",%e", sp->annotations_io_buffer.get<float>(i, j));
        }
        for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
          fileIO.print(",%e", sp->annotations_io_buffer.get<double>(i, j));
        }
        fileIO.print( "\n" );
      }
      particle_idx += sp->np_per_ts_io_buffer[ts_idx].first;
    }

    // Write tracer data
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
      if(dump_vars & DumpVar::ParticleKE) {
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        fileIO.print(",%e", v0);
      }
      // Print annotations
      for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
        fileIO.print(",%d", sp->annotations_h.get<int>(i, j));
      }
      for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
        fileIO.print(",%ld", sp->annotations_h.get<int64_t>(i, j));
      }
      for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
        fileIO.print(",%e", sp->annotations_h.get<float>(i, j));
      }
      for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
        fileIO.print(",%e", sp->annotations_h.get<double>(i, j));
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
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, reset weight to 0 for copied particles
    if(sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }
#endif
    sp->nparticles_buffered = 0;
    sp->np_per_ts_io_buffer.clear();
  }
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
//    int dim[1], buf_start;
//    static particle_t * ALIGNED(128) p_buf = NULL;

    // Get species
    sp = find_species_name( sp_name, tracers_list );
    if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name ));

    if( !fbase ) ERROR(( "Invalid filename" ));

    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( rank()==0 )
        MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,fbase));

    // Create output filename
    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i", fbase, (long)step(), rank() );
    }
    else {
        snprintf( fname, max_filename_bytes, "%s.%i", fbase, rank() );
    }

    FileIOStatus status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    // Create header string
    std::string header_str = std::string("Timestep,rank,tracer_id,cell_id,dx,dy,dz,ux,uy,uz,w");
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
      header_str += ",ke_dens";
    }
    if(dump_vars & DumpVar::StressTensor) {
      header_str += ",txx,tyy,tzz,tyz,tzx,txy";
    }
    if(dump_vars & DumpVar::ParticleKE) {
      header_str += ",ke";
    }
    for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.i32_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.i64_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.f32_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.f64_vars[j].c_str();
    }

    // Write header string
    if( append==0 ) {
      fileIO.print( "%s\n", header_str.c_str() );
    }

    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }

    // Compute hydro quantities
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

    int tracer_idx = sp->annotation_vars.get_annotation_index<int64_t>("TracerID");
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
    // Write tracer data
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
      if(dump_vars & DumpVar::ParticleKE) {
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        fileIO.print(",%e", v0);
      }
      // Print annotations
      for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
        fileIO.print(",%d", sp->annotations_h.get<int>(i, j));
      }
      for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
        fileIO.print(",%ld", sp->annotations_h.get<int64_t>(i, j));
      }
      for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
        fileIO.print(",%e", sp->annotations_h.get<float>(i, j));
      }
      for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
        fileIO.print(",%e", sp->annotations_h.get<double>(i, j));
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
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, reset weight to 0 for copied particles
    if(sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }
#endif

    if( fileIO.close() ) ERROR(("File close failed on dump particles!!!"));
}
#endif

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
#ifdef VPIC_ENABLE_HDF5
void
vpic_simulation::dump_tracers_buffered_hdf5( const char *sp_name,
                                            uint32_t dump_vars,
                                            const char *fbase,
                                            const int append )
{
  species_t *sp;
  char fname[max_filename_bytes];
  FileIO fileIO;
  char group_name[256];

  // Get species
  sp = find_species_name( sp_name, tracers_list );
  if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name ));

  if( !fbase ) ERROR(( "Invalid filename" ));

  sprintf(fname, "%s.h5", fbase);

  // Create file access template with parallel IO access
  herr_t status;
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  
  // Try to create species HDF5 file with default file creation/access property lists
  hid_t file_id;
  if(append == 0) {
    file_id = H5Fcreate(fname, H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    status = H5Fclose(file_id);
  }

  // Check if any buffers are filled. If any process needs to dump then all must do it together
  int dump_flag = sp->nparticles_buffered+sp->np < sp->particle_io_buffer.extent(0);
  MPI_Allreduce(MPI_IN_PLACE, &dump_flag, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

  // Buffer tracers
  if(dump_flag && step() != num_step) {
    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    if( rank()==0 )
        MESSAGE(("Buffering \"%s\" particles",sp->name));

    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }

    // Compute hydro quantities
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

    // Get index of tracer ID
    auto& interp = interpolator_array->k_i_h;

    auto particle_slice = Kokkos::make_pair(0, sp->np);
    auto buffer_slice = Kokkos::make_pair(sp->nparticles_buffered,sp->nparticles_buffered+sp->np);

    // Copy particles into buffer
    auto particle_subview = Kokkos::subview(sp->k_p_h, particle_slice, Kokkos::ALL());
    auto particle_buffer_subview = Kokkos::subview(sp->particle_io_buffer, buffer_slice, Kokkos::ALL());
    auto particle_cell_subview = Kokkos::subview(sp->k_p_i_h, particle_slice);
    auto particle_cell_buffer_subview = Kokkos::subview(sp->particle_cell_io_buffer, buffer_slice);
    Kokkos::deep_copy(particle_buffer_subview, particle_subview); 
    Kokkos::deep_copy(particle_cell_buffer_subview, particle_cell_subview); 

    // Copy annotations into buffers
    for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.i32, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.i32, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
    for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.i64, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.i64, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
    for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.f32, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.f32, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }
    for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
      auto annote_subview = Kokkos::subview(sp->annotations_h.f64, particle_slice, j);
      auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer.f64, buffer_slice, j);
      Kokkos::deep_copy(buffer_subview, annote_subview);
    }

    // Buffer tracer data
    sp->np_per_ts_io_buffer.push_back(std::make_pair(sp->np, step()));
    Kokkos::parallel_for("Buffer data", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      int   ii  = sp->k_p_i_h(i);
      float ux0 = sp->k_p_h(i, particle_var::ux);
      float uy0 = sp->k_p_h(i, particle_var::uy);
      float uz0 = sp->k_p_h(i, particle_var::uz);
      float w0  = sp->k_p_h(i, particle_var::w);
      if(dump_vars & DumpVar::Efield) {
        float ex  = interp(ii,interpolator_var::ex ) + dy0*interp(ii,interpolator_var::dexdy) + dz0*(interp(ii,interpolator_var::dexdz) + dy0*interp(ii,interpolator_var::d2exdydz)); 
        float ey  = interp(ii,interpolator_var::ey ) + dz0*interp(ii,interpolator_var::deydz) + dx0*(interp(ii,interpolator_var::deydx) + dz0*interp(ii,interpolator_var::d2eydzdx)); 
        float ez  = interp(ii,interpolator_var::ez ) + dx0*interp(ii,interpolator_var::dezdx) + dy0*(interp(ii,interpolator_var::dezdy) + dx0*interp(ii,interpolator_var::d2ezdxdy)); 
        sp->efields_io_buffer(sp->nparticles_buffered+i, 0) = ex;
        sp->efields_io_buffer(sp->nparticles_buffered+i, 1) = ey;
        sp->efields_io_buffer(sp->nparticles_buffered+i, 2) = ez;
      }
      if(dump_vars & DumpVar::Bfield) {
        float bx  = interp(ii,interpolator_var::cbx) + dx0*interp(ii,interpolator_var::dcbxdx); 
        float by  = interp(ii,interpolator_var::cby) + dy0*interp(ii,interpolator_var::dcbydy); 
        float bz  = interp(ii,interpolator_var::cbz) + dz0*interp(ii,interpolator_var::dcbzdz); 
        sp->bfields_io_buffer(sp->nparticles_buffered+i, 0) = bx;
        sp->bfields_io_buffer(sp->nparticles_buffered+i, 1) = by;
        sp->bfields_io_buffer(sp->nparticles_buffered+i, 2) = bz;
      }
      if(dump_vars & DumpVar::CurrentDensity) {
        float jx  = hydro_array->k_h_h(ii, hydro_var::jx);
        float jy  = hydro_array->k_h_h(ii, hydro_var::jy);
        float jz  = hydro_array->k_h_h(ii, hydro_var::jz);
        sp->current_dens_io_buffer(sp->nparticles_buffered+i, 0) = jx;
        sp->current_dens_io_buffer(sp->nparticles_buffered+i, 1) = jy;
        sp->current_dens_io_buffer(sp->nparticles_buffered+i, 2) = jz;
      }
      if(dump_vars & DumpVar::ChargeDensity) {
        float rho = hydro_array->k_h_h(ii, hydro_var::rho);
        sp->charge_dens_io_buffer(sp->nparticles_buffered+i) = rho;
      }
      if(dump_vars & DumpVar::MomentumDensity) {
        float px  = hydro_array->k_h_h(ii, hydro_var::px);
        float py  = hydro_array->k_h_h(ii, hydro_var::py);
        float pz  = hydro_array->k_h_h(ii, hydro_var::pz);
        sp->momentum_dens_io_buffer(sp->nparticles_buffered+i, 0) = px;
        sp->momentum_dens_io_buffer(sp->nparticles_buffered+i, 1) = py;
        sp->momentum_dens_io_buffer(sp->nparticles_buffered+i, 2) = pz;
      }
      if(dump_vars & DumpVar::KEDensity) {
        float ke = hydro_array->k_h_h(ii, hydro_var::ke);
        sp->ke_dens_io_buffer(sp->nparticles_buffered+i) = ke;
      }
      if(dump_vars & DumpVar::StressTensor) {
        float txx = hydro_array->k_h_h(ii, hydro_var::txx);
        float tyy = hydro_array->k_h_h(ii, hydro_var::tyy);
        float tzz = hydro_array->k_h_h(ii, hydro_var::tzz);
        float tyz = hydro_array->k_h_h(ii, hydro_var::tyz);
        float tzx = hydro_array->k_h_h(ii, hydro_var::tzx);
        float txy = hydro_array->k_h_h(ii, hydro_var::txy);
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 0) = txx;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 1) = tyy;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 2) = tzz;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 3) = tyz;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 4) = tzx;
        sp->stress_tensor_io_buffer(sp->nparticles_buffered+i, 5) = txy;
      }
      if(dump_vars & DumpVar::ParticleKE) {
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        sp->particle_ke_io_buffer(sp->nparticles_buffered+i) = v0;
      }
    });
    sp->nparticles_buffered += sp->np;     
    // If needed, reset weight to 0 for copied particles
    if(sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }
  } else { // Dump buffered tracers
    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied)
      sp->copy_to_host();

    // Calculate total number of tracers and per rank offsets
    uint64_t total_particles, offset;
    uint64_t num_particles;

    if( rank()==0 )
        MESSAGE(("Dumping %ld \"%s\" particles to \"%s\"", sp->particle_cell_io_buffer.extent(0), sp->name,fbase));

    // Get references to necessary data structures
    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }

    // Compute hydro quantities
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

    auto& interp = interpolator_array->k_i_h;

    // Open file
    file_id = H5Fopen(fname, H5F_ACC_RDWR, plist_id);

    // Write buffered tracer data
    uint32_t particle_idx = 0;
    // Iterate through each timestep
    for(uint32_t ts_idx=0; ts_idx < sp->np_per_ts_io_buffer.size(); ts_idx++) {
      int64_t time_step = sp->np_per_ts_io_buffer[ts_idx].second;

      // Get # of local particles to write for this timestep
      num_particles = sp->np_per_ts_io_buffer[ts_idx].first;
      // Calculate the total number of particles for this timestep
      MPI_Allreduce(&num_particles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      // Calcualte the offset for each rank
      MPI_Scan(&num_particles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      offset -= num_particles;

      if(total_particles > 0) {
        // Create group for time step
        sprintf(group_name, "/Timestep_%ld", time_step);
        hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Create dataspace describing dims for particle datasets
        hid_t dataspace_id = H5Screate_simple(1, (hsize_t*)(&total_particles), NULL);

        // Set MPIO to collective mode
        hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

        // Select slab of dataset for each rank
        hsize_t stride = 1;
        hsize_t block = 1;
        status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, (hsize_t*)(&offset), &stride, (hsize_t*)(&num_particles), &block);
    
        // Create memspace
        hid_t memspace_id = H5Screate_simple(1, (hsize_t*)(&num_particles), NULL);

        // Slice of buffered particles
        auto slice = Kokkos::make_pair(static_cast<uint64_t>(particle_idx), static_cast<uint64_t>(particle_idx) + num_particles);

        // Create subviews for data
        auto dx_subview = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::dx));
        auto dy_subview = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::dy));
        auto dz_subview = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::dz));
        auto ux_subview = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::ux));
        auto uy_subview = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::uy));
        auto uz_subview = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::uz));
        auto w_subview  = Kokkos::subview(sp->particle_io_buffer, slice, (int)(particle_var::w));
        auto i_subview  = Kokkos::subview(sp->particle_cell_io_buffer, slice);

        // Create datasets, one for each variable, using dataspace and default property lists
        hid_t dataset_dx_id = H5Dcreate(group_id, "dx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_dy_id = H5Dcreate(group_id, "dy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_dz_id = H5Dcreate(group_id, "dz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_ux_id = H5Dcreate(group_id, "ux", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_uy_id = H5Dcreate(group_id, "uy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_uz_id = H5Dcreate(group_id, "uz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_w_id  = H5Dcreate(group_id, "w",  H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_i_id  = H5Dcreate(group_id, "i",  H5T_STD_I32LE,    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // Write data to slab
        status = H5Dwrite(dataset_dx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, dx_subview.data());
        status = H5Dwrite(dataset_dy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, dy_subview.data());
        status = H5Dwrite(dataset_dz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, dz_subview.data());
        status = H5Dwrite(dataset_ux_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, ux_subview.data());
        status = H5Dwrite(dataset_uy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, uy_subview.data());
        status = H5Dwrite(dataset_uz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, uz_subview.data());
        status = H5Dwrite(dataset_w_id,  H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, w_subview.data());
        status = H5Dwrite(dataset_i_id,  H5T_STD_I32LE,  memspace_id, dataspace_id, dxpl_id, i_subview.data());

        using host_memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;
  
        // Dump Global position if specified
        if(dump_vars & DumpVar::GlobalPos) {
          auto pos_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Pos Host View", num_particles);
          Kokkos::parallel_for("Calculate global position", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, num_particles), KOKKOS_LAMBDA(const uint32_t i) {
            float dx0 = dx_subview(i);
            float dy0 = dy_subview(i);
            float dz0 = dz_subview(i);
            int   ii  = i_subview(i);
            
            // Compute global position of particle
            if(dump_vars & DumpVar::GlobalPos) {
              int nxg_ = grid->nx + 2;
              int nyg_ = grid->ny + 2;
              int i0 = ii % nxg_;
              int j0 = (ii/nxg_) % nyg_;
              int k0 = ii/(nxg_*nyg_);
              float tracer_x = (i0 + (dx0-1)*0.5) * grid->dx + grid->x0;
              float tracer_y = (j0 + (dy0-1)*0.5) * grid->dy + grid->y0;
              float tracer_z = (k0 + (dz0-1)*0.5) * grid->dz + grid->z0;
              pos_view(i, 0) = tracer_x;
              pos_view(i, 1) = tracer_y;
              pos_view(i, 2) = tracer_z;
            }
          });
          auto posx_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 0);
          auto posy_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 1);
          auto posz_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 2);
          hid_t dataset_posx_id = H5Dcreate(group_id, "posx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_posy_id = H5Dcreate(group_id, "posy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_posz_id = H5Dcreate(group_id, "posz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_posx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, posx_subview.data());
          status = H5Dwrite(dataset_posy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, posy_subview.data());
          status = H5Dwrite(dataset_posz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, posz_subview.data());
          H5Dclose(dataset_posx_id);
          H5Dclose(dataset_posy_id);
          H5Dclose(dataset_posz_id);
        }
  
        // Dump E field if specified
        if(dump_vars & DumpVar::Efield) {
          auto efieldx_subview = Kokkos::subview(sp->efields_io_buffer, slice, 0);
          auto efieldy_subview = Kokkos::subview(sp->efields_io_buffer, slice, 1);
          auto efieldz_subview = Kokkos::subview(sp->efields_io_buffer, slice, 2);
          hid_t dataset_efieldx_id = H5Dcreate(group_id, "ex", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_efieldy_id = H5Dcreate(group_id, "ey", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_efieldz_id = H5Dcreate(group_id, "ez", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_efieldx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, efieldx_subview.data());
          status = H5Dwrite(dataset_efieldy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, efieldy_subview.data());
          status = H5Dwrite(dataset_efieldz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, efieldz_subview.data());
          H5Dclose(dataset_efieldx_id);
          H5Dclose(dataset_efieldy_id);
          H5Dclose(dataset_efieldz_id);
        }
  
        // Dump B field if specified
        if(dump_vars & DumpVar::Bfield) {
          auto bfieldx_subview = Kokkos::subview(sp->bfields_io_buffer, slice, 0);
          auto bfieldy_subview = Kokkos::subview(sp->bfields_io_buffer, slice, 1);
          auto bfieldz_subview = Kokkos::subview(sp->bfields_io_buffer, slice, 2);
          hid_t dataset_bfieldx_id = H5Dcreate(group_id, "bx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_bfieldy_id = H5Dcreate(group_id, "by", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_bfieldz_id = H5Dcreate(group_id, "bz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_bfieldx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, bfieldx_subview.data());
          status = H5Dwrite(dataset_bfieldy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, bfieldy_subview.data());
          status = H5Dwrite(dataset_bfieldz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, bfieldz_subview.data());
          H5Dclose(dataset_bfieldx_id);
          H5Dclose(dataset_bfieldy_id);
          H5Dclose(dataset_bfieldz_id);
        }
  
        // Dump current density if specified
        if(dump_vars & DumpVar::CurrentDensity) {
          auto jx_subview = Kokkos::subview(sp->current_dens_io_buffer, slice, 0);
          auto jy_subview = Kokkos::subview(sp->current_dens_io_buffer, slice, 1);
          auto jz_subview = Kokkos::subview(sp->current_dens_io_buffer, slice, 2);
          hid_t dataset_jx_id = H5Dcreate(group_id, "jx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_jy_id = H5Dcreate(group_id, "jy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_jz_id = H5Dcreate(group_id, "jz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_jx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, jx_subview.data());
          status = H5Dwrite(dataset_jy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, jy_subview.data());
          status = H5Dwrite(dataset_jz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, jz_subview.data());
          H5Dclose(dataset_jx_id);
          H5Dclose(dataset_jy_id);
          H5Dclose(dataset_jz_id);
        }
  
        // Dump charge density if specified
        if(dump_vars & DumpVar::ChargeDensity) {
          auto charge_view = Kokkos::subview(sp->charge_dens_io_buffer, slice);
          hid_t dataset_rho_id = H5Dcreate(group_id, "rho", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_rho_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, charge_view.data());
          H5Dclose(dataset_rho_id);
        }
  
        // Dump momentum density if specified
        if(dump_vars & DumpVar::MomentumDensity) {
          auto px_subview = Kokkos::subview(sp->momentum_dens_io_buffer, slice, 0);
          auto py_subview = Kokkos::subview(sp->momentum_dens_io_buffer, slice, 1);
          auto pz_subview = Kokkos::subview(sp->momentum_dens_io_buffer, slice, 2);
          hid_t dataset_px_id = H5Dcreate(group_id, "px", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_py_id = H5Dcreate(group_id, "py", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_pz_id = H5Dcreate(group_id, "pz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_px_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, px_subview.data());
          status = H5Dwrite(dataset_py_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, py_subview.data());
          status = H5Dwrite(dataset_pz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, pz_subview.data());
          H5Dclose(dataset_px_id);
          H5Dclose(dataset_py_id);
          H5Dclose(dataset_pz_id);
        }
  
        // Dump kinetic energy density if specified
        if(dump_vars & DumpVar::KEDensity) {
          auto ke_view = Kokkos::subview(sp->ke_dens_io_buffer, slice);
          hid_t dataset_ke_id = H5Dcreate(group_id, "ke_dens", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_ke_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, ke_view.data());
          H5Dclose(dataset_ke_id);
        }
  
        // Dump stress tensor if specified
        if(dump_vars & DumpVar::StressTensor) {
          auto txx_subview = Kokkos::subview(sp->stress_tensor_io_buffer, slice, 0);
          auto tyy_subview = Kokkos::subview(sp->stress_tensor_io_buffer, slice, 1);
          auto tzz_subview = Kokkos::subview(sp->stress_tensor_io_buffer, slice, 2);
          auto tyz_subview = Kokkos::subview(sp->stress_tensor_io_buffer, slice, 3);
          auto tzx_subview = Kokkos::subview(sp->stress_tensor_io_buffer, slice, 4);
          auto txy_subview = Kokkos::subview(sp->stress_tensor_io_buffer, slice, 5);
          hid_t dataset_txx_id = H5Dcreate(group_id, "txx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_tyy_id = H5Dcreate(group_id, "tyy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_tzz_id = H5Dcreate(group_id, "tzz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_tyz_id = H5Dcreate(group_id, "tyz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_tzx_id = H5Dcreate(group_id, "tzx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t dataset_txy_id = H5Dcreate(group_id, "txy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_txx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, txx_subview.data());
          status = H5Dwrite(dataset_tyy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, tyy_subview.data());
          status = H5Dwrite(dataset_tzz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, tzz_subview.data());
          status = H5Dwrite(dataset_tyz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, tyz_subview.data());
          status = H5Dwrite(dataset_tzx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, tzx_subview.data());
          status = H5Dwrite(dataset_txy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, txy_subview.data());
          H5Dclose(dataset_txx_id);
          H5Dclose(dataset_tyy_id);
          H5Dclose(dataset_tzz_id);
          H5Dclose(dataset_tyz_id);
          H5Dclose(dataset_tzx_id);
          H5Dclose(dataset_txy_id);
        }

        // Dump kinetic energy of particle if specified
        if(dump_vars & DumpVar::ParticleKE) {
          auto ke_view = Kokkos::subview(sp->particle_ke_io_buffer, slice);
          hid_t dataset_ke_id = H5Dcreate(group_id, "ke", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_ke_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, ke_view.data());
          H5Dclose(dataset_ke_id);
        }
  
        // Dump int annotations
        for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
          auto i32_subview = Kokkos::subview(sp->annotations_io_buffer.i32, slice, j);
          hid_t dataset_i32_annote_id = H5Dcreate(group_id, sp->annotation_vars.i32_vars[j].c_str(), H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_i32_annote_id, H5T_STD_I32LE, memspace_id, dataspace_id, dxpl_id, i32_subview.data());
          H5Dclose(dataset_i32_annote_id);
        }
        // Dump 64-bit integer annotations
        for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
          auto i64_subview = Kokkos::subview(sp->annotations_io_buffer.i64, slice, j);
          hid_t dataset_i64_annote_id = H5Dcreate(group_id, sp->annotation_vars.i64_vars[j].c_str(), H5T_STD_I64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_i64_annote_id, H5T_STD_I64LE, memspace_id, dataspace_id, dxpl_id, i64_subview.data());
          H5Dclose(dataset_i64_annote_id);
        }
        // Dump 32-bit floating-point annotations
        for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
          auto f32_subview = Kokkos::subview(sp->annotations_io_buffer.f32, slice, j);
          hid_t dataset_f32_annote_id = H5Dcreate(group_id, sp->annotation_vars.f32_vars[j].c_str(), H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_f32_annote_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, dxpl_id, f32_subview.data());
          H5Dclose(dataset_f32_annote_id);
        }
        // Dump 64-bit floating-point annotations
        for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
          auto f64_subview = Kokkos::subview(sp->annotations_io_buffer.f64, slice, j);
          hid_t dataset_f64_annote_id = H5Dcreate(group_id, sp->annotation_vars.f64_vars[j].c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          status = H5Dwrite(dataset_f64_annote_id, H5T_IEEE_F64LE, memspace_id, dataspace_id, dxpl_id, f64_subview.data());
          H5Dclose(dataset_f64_annote_id);
        }

        // Move to next time step
        particle_idx += sp->np_per_ts_io_buffer[ts_idx].first;

        status = H5Dclose(dataset_dx_id);
        status = H5Dclose(dataset_dy_id);
        status = H5Dclose(dataset_dz_id);
        status = H5Dclose(dataset_ux_id);
        status = H5Dclose(dataset_uy_id);
        status = H5Dclose(dataset_uz_id);
        status = H5Dclose(dataset_w_id);
        status = H5Dclose(dataset_i_id);
        status = H5Sclose(memspace_id);
        status = H5Sclose(dataspace_id);
        status = H5Gclose(group_id);
        status = H5Pclose(dxpl_id);
      }
    }

    // Write current tracer data
    sprintf(group_name, "/Timestep_%ld", step());

    // Create group step
    hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Close plist_id
    H5Pclose(plist_id);

    // Calculate offsets and # of particles to write for each rank
    num_particles = sp->np;
    MPI_Allreduce(&num_particles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&num_particles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    offset -= num_particles;

    if(total_particles > 0) {

      // Create dataspace describing dims for particle datasets
      hid_t dataspace_id = H5Screate_simple(1, (hsize_t*)(&total_particles), NULL);

      // Set MPIO to collective mode
      plist_id = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

      // Select slab of dataset for each rank
      herr_t status;
      hsize_t stride = 1;
      hsize_t block = 1;
      status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, (hsize_t*)(&offset), &stride, (hsize_t*)(&num_particles), &block);

      // Create memspace
      hid_t memspace_id = H5Screate_simple(1, (hsize_t*)(&num_particles), NULL);

      // Create subviews for data
      auto dx_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::dx));
      auto dy_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::dy));
      auto dz_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::dz));
      auto ux_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::ux));
      auto uy_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::uy));
      auto uz_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::uz));
      auto w_subview  = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::w));

      // Create datasets, one for each variable, using dataspace and default property lists
      hid_t dataset_dx_id = H5Dcreate(group_id, "dx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_dy_id = H5Dcreate(group_id, "dy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_dz_id = H5Dcreate(group_id, "dz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_ux_id = H5Dcreate(group_id, "ux", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_uy_id = H5Dcreate(group_id, "uy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_uz_id = H5Dcreate(group_id, "uz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_w_id  = H5Dcreate(group_id, "w",  H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      hid_t dataset_i_id  = H5Dcreate(group_id, "i",  H5T_STD_I32LE,    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      // Write data to slab
      status = H5Dwrite(dataset_dx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, dx_subview.data());
      status = H5Dwrite(dataset_dy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, dy_subview.data());
      status = H5Dwrite(dataset_dz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, dz_subview.data());
      status = H5Dwrite(dataset_ux_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, ux_subview.data());
      status = H5Dwrite(dataset_uy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, uy_subview.data());
      status = H5Dwrite(dataset_uz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, uz_subview.data());
      status = H5Dwrite(dataset_w_id,  H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, w_subview.data());
      status = H5Dwrite(dataset_i_id,  H5T_STD_I32LE,  memspace_id, dataspace_id, H5P_DEFAULT, sp->k_p_i_h.data());

      using host_memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;

      // Dump Global position if specified
      if(dump_vars & DumpVar::GlobalPos) {
        auto pos_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Pos Host View", sp->np);
        Kokkos::parallel_for("Calculate global position", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          float dx0 = sp->k_p_h(i, particle_var::dx);
          float dy0 = sp->k_p_h(i, particle_var::dy);
          float dz0 = sp->k_p_h(i, particle_var::dz);
          int   ii  = sp->k_p_i_h(i);
          
          // Compute global position of particle
          if(dump_vars & DumpVar::GlobalPos) {
            int nxg_ = grid->nx + 2;
            int nyg_ = grid->ny + 2;
            int i0 = ii % nxg_;
            int j0 = (ii/nxg_) % nyg_;
            int k0 = ii/(nxg_*nyg_);
            float tracer_x = (i0 + (dx0-1)*0.5) * grid->dx + grid->x0;
            float tracer_y = (j0 + (dy0-1)*0.5) * grid->dy + grid->y0;
            float tracer_z = (k0 + (dz0-1)*0.5) * grid->dz + grid->z0;
            pos_view(i, 0) = tracer_x;
            pos_view(i, 1) = tracer_y;
            pos_view(i, 2) = tracer_z;
          }
        });
        auto posx_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 0);
        auto posy_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 1);
        auto posz_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 2);
        hid_t dataset_posx_id = H5Dcreate(group_id, "posx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_posy_id = H5Dcreate(group_id, "posy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_posz_id = H5Dcreate(group_id, "posz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_posx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, posx_subview.data());
        status = H5Dwrite(dataset_posy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, posy_subview.data());
        status = H5Dwrite(dataset_posz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, posz_subview.data());
        H5Dclose(dataset_posx_id);
        H5Dclose(dataset_posy_id);
        H5Dclose(dataset_posz_id);
      }

      // Dump E field if specified
      if(dump_vars & DumpVar::Efield) {
        auto efield_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("E field Host View", sp->np);
        Kokkos::parallel_for("Calculate E field", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          float dx0 = sp->k_p_h(i, particle_var::dx);
          float dy0 = sp->k_p_h(i, particle_var::dy);
          float dz0 = sp->k_p_h(i, particle_var::dz);
          int   ii  = sp->k_p_i_h(i);
          efield_view(i,0) = interp(ii,interpolator_var::ex ) + dy0*interp(ii,interpolator_var::dexdy) + dz0*(interp(ii,interpolator_var::dexdz) + dy0*interp(ii,interpolator_var::d2exdydz)); 
          efield_view(i,1) = interp(ii,interpolator_var::ey ) + dz0*interp(ii,interpolator_var::deydz) + dx0*(interp(ii,interpolator_var::deydx) + dz0*interp(ii,interpolator_var::d2eydzdx)); 
          efield_view(i,2) = interp(ii,interpolator_var::ez ) + dx0*interp(ii,interpolator_var::dezdx) + dy0*(interp(ii,interpolator_var::dezdy) + dx0*interp(ii,interpolator_var::d2ezdxdy)); 
        });
        auto efieldx_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 0);
        auto efieldy_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 1);
        auto efieldz_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 2);
        hid_t dataset_efieldx_id = H5Dcreate(group_id, "ex", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_efieldy_id = H5Dcreate(group_id, "ey", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_efieldz_id = H5Dcreate(group_id, "ez", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_efieldx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, efieldx_subview.data());
        status = H5Dwrite(dataset_efieldy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, efieldy_subview.data());
        status = H5Dwrite(dataset_efieldz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, efieldz_subview.data());
        H5Dclose(dataset_efieldx_id);
        H5Dclose(dataset_efieldy_id);
        H5Dclose(dataset_efieldz_id);
      }

      // Dump B field if specified
      if(dump_vars & DumpVar::Bfield) {
        auto bfield_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("B field Host View", sp->np);
        Kokkos::parallel_for("Calculate B field", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          float dx0 = sp->k_p_h(i, particle_var::dx);
          float dy0 = sp->k_p_h(i, particle_var::dy);
          float dz0 = sp->k_p_h(i, particle_var::dz);
          int   ii  = sp->k_p_i_h(i);
          bfield_view(i,0)  = interp(ii,interpolator_var::cbx) + dx0*interp(ii,interpolator_var::dcbxdx); 
          bfield_view(i,1)  = interp(ii,interpolator_var::cby) + dy0*interp(ii,interpolator_var::dcbydy); 
          bfield_view(i,2)  = interp(ii,interpolator_var::cbz) + dz0*interp(ii,interpolator_var::dcbzdz); 
        });
        auto bfieldx_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 0);
        auto bfieldy_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 1);
        auto bfieldz_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 2);
        hid_t dataset_bfieldx_id = H5Dcreate(group_id, "bx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_bfieldy_id = H5Dcreate(group_id, "by", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_bfieldz_id = H5Dcreate(group_id, "bz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_bfieldx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, bfieldx_subview.data());
        status = H5Dwrite(dataset_bfieldy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, bfieldy_subview.data());
        status = H5Dwrite(dataset_bfieldz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, bfieldz_subview.data());
        H5Dclose(dataset_bfieldx_id);
        H5Dclose(dataset_bfieldy_id);
        H5Dclose(dataset_bfieldz_id);
      }

      // Dump current density if specified
      if(dump_vars & DumpVar::CurrentDensity) {
        auto current_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Current density Host View", sp->np);
        Kokkos::parallel_for("Collect current density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          int   ii  = sp->k_p_i_h(i);
          current_view(i, 0) = hydro_array->k_h_h(ii, hydro_var::jx);
          current_view(i, 1) = hydro_array->k_h_h(ii, hydro_var::jy);
          current_view(i, 2) = hydro_array->k_h_h(ii, hydro_var::jz);
        });
        auto jx_subview = Kokkos::subview(current_view, Kokkos::ALL(), 0);
        auto jy_subview = Kokkos::subview(current_view, Kokkos::ALL(), 1);
        auto jz_subview = Kokkos::subview(current_view, Kokkos::ALL(), 2);
        hid_t dataset_jx_id = H5Dcreate(group_id, "jx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_jy_id = H5Dcreate(group_id, "jy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_jz_id = H5Dcreate(group_id, "jz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_jx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, jx_subview.data());
        status = H5Dwrite(dataset_jy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, jy_subview.data());
        status = H5Dwrite(dataset_jz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, jz_subview.data());
        H5Dclose(dataset_jx_id);
        H5Dclose(dataset_jy_id);
        H5Dclose(dataset_jz_id);
      }

      // Dump charge density if specified
      if(dump_vars & DumpVar::ChargeDensity) {
        auto charge_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("Charge density Host View", sp->np);
        Kokkos::parallel_for("Calculate charge density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          int   ii  = sp->k_p_i_h(i);
          charge_view(i) = hydro_array->k_h_h(ii, hydro_var::rho);
        });
        hid_t dataset_rho_id = H5Dcreate(group_id, "rho", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_rho_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, charge_view.data());
        H5Dclose(dataset_rho_id);
      }

      // Dump momentum density if specified
      if(dump_vars & DumpVar::MomentumDensity) {
        auto momentum_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Momentum Host View", sp->np);
        Kokkos::parallel_for("Collect momentum density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          int   ii  = sp->k_p_i_h(i);
          momentum_view(i, 0) = hydro_array->k_h_h(ii, hydro_var::px);
          momentum_view(i, 1) = hydro_array->k_h_h(ii, hydro_var::py);
          momentum_view(i, 2) = hydro_array->k_h_h(ii, hydro_var::pz);
        });
        auto px_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 0);
        auto py_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 1);
        auto pz_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 2);
        hid_t dataset_px_id = H5Dcreate(group_id, "px", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_py_id = H5Dcreate(group_id, "py", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_pz_id = H5Dcreate(group_id, "pz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_px_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, px_subview.data());
        status = H5Dwrite(dataset_py_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, py_subview.data());
        status = H5Dwrite(dataset_pz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, pz_subview.data());
        H5Dclose(dataset_px_id);
        H5Dclose(dataset_py_id);
        H5Dclose(dataset_pz_id);
      }

      // Dump kinetic energy density if specified
      if(dump_vars & DumpVar::KEDensity) {
        auto ke_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("KE Host View", sp->np);
        Kokkos::parallel_for("Collect KE density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          int   ii  = sp->k_p_i_h(i);
          ke_view(i) = hydro_array->k_h_h(ii, hydro_var::ke);
        });
        hid_t dataset_ke_id = H5Dcreate(group_id, "ke_dens", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_ke_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, ke_view.data());
        H5Dclose(dataset_ke_id);
      }

      // Dump stress tensor if specified
      if(dump_vars & DumpVar::StressTensor) {
        auto stress_view = Kokkos::View<float*[6], Kokkos::LayoutLeft, host_memory_space>("Stress tensor Host View", sp->np);
        Kokkos::parallel_for("Collect stress tensor", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          int   ii  = sp->k_p_i_h(i);
          stress_view(i, 0) = hydro_array->k_h_h(ii, hydro_var::txx);
          stress_view(i, 1) = hydro_array->k_h_h(ii, hydro_var::tyy);
          stress_view(i, 2) = hydro_array->k_h_h(ii, hydro_var::tzz);
          stress_view(i, 3) = hydro_array->k_h_h(ii, hydro_var::tyz);
          stress_view(i, 4) = hydro_array->k_h_h(ii, hydro_var::tzx);
          stress_view(i, 5) = hydro_array->k_h_h(ii, hydro_var::txy);
        });
        auto txx_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 0);
        auto tyy_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 1);
        auto tzz_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 2);
        auto tyz_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 3);
        auto tzx_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 4);
        auto txy_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 5);
        hid_t dataset_txx_id = H5Dcreate(group_id, "txx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_tyy_id = H5Dcreate(group_id, "tyy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_tzz_id = H5Dcreate(group_id, "tzz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_tyz_id = H5Dcreate(group_id, "tyz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_tzx_id = H5Dcreate(group_id, "tzx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t dataset_txy_id = H5Dcreate(group_id, "txy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_txx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, txx_subview.data());
        status = H5Dwrite(dataset_tyy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tyy_subview.data());
        status = H5Dwrite(dataset_tzz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tzz_subview.data());
        status = H5Dwrite(dataset_tyz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tyz_subview.data());
        status = H5Dwrite(dataset_tzx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tzx_subview.data());
        status = H5Dwrite(dataset_txy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, txy_subview.data());
        H5Dclose(dataset_txx_id);
        H5Dclose(dataset_tyy_id);
        H5Dclose(dataset_tzz_id);
        H5Dclose(dataset_tyz_id);
        H5Dclose(dataset_tzx_id);
        H5Dclose(dataset_txy_id);
      }

      // Dump kinetic energy of particle if specified
      if(dump_vars & DumpVar::ParticleKE) {
        auto ke_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("KE Host View", sp->np);
        Kokkos::parallel_for("Calculate KE", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
          float dx0 = sp->k_p_h(i, particle_var::dx);
          float dy0 = sp->k_p_h(i, particle_var::dy);
          float dz0 = sp->k_p_h(i, particle_var::dz);
          int   ii  = sp->k_p_i_h(i);
          float ux0 = sp->k_p_h(i, particle_var::ux);
          float uy0 = sp->k_p_h(i, particle_var::uy);
          float uz0 = sp->k_p_h(i, particle_var::uz);
          float w0  = sp->k_p_h(i, particle_var::w);
          float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
          float msp = sp->m;
          float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                                 dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
          float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                                 dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
          float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                                 dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
          v0 = v0*v0 + v1*v1 + v2*v2;
          v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
          ke_view(i) = v0;
        });
        hid_t dataset_ke_id = H5Dcreate(group_id, "ke", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_ke_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, ke_view.data());
        H5Dclose(dataset_ke_id);
      }

      // Dump int annotations
      for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
        auto i32_subview = Kokkos::subview(sp->annotations_h.i32, Kokkos::ALL, j);
        hid_t dataset_i32_annote_id = H5Dcreate(group_id, sp->annotation_vars.i32_vars[j].c_str(), H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_i32_annote_id, H5T_STD_I32LE, memspace_id, dataspace_id, H5P_DEFAULT, i32_subview.data());
        H5Dclose(dataset_i32_annote_id);
      }
      // Dump 64-bit integer annotations
      for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
        auto i64_subview = Kokkos::subview(sp->annotations_h.i64, Kokkos::ALL, j);
        hid_t dataset_i64_annote_id = H5Dcreate(group_id, sp->annotation_vars.i64_vars[j].c_str(), H5T_STD_I64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_i64_annote_id, H5T_STD_I64LE, memspace_id, dataspace_id, H5P_DEFAULT, i64_subview.data());
        H5Dclose(dataset_i64_annote_id);
      }
      // Dump 32-bit floating-point annotations
      for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
        auto f32_subview = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL, j);
        hid_t dataset_f32_annote_id = H5Dcreate(group_id, sp->annotation_vars.f32_vars[j].c_str(), H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_f32_annote_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, f32_subview.data());
        H5Dclose(dataset_f32_annote_id);
      }
      // Dump 64-bit floating-point annotations
      for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
        auto f64_subview = Kokkos::subview(sp->annotations_h.f64, Kokkos::ALL, j);
        hid_t dataset_f64_annote_id = H5Dcreate(group_id, sp->annotation_vars.f64_vars[j].c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset_f64_annote_id, H5T_IEEE_F64LE, memspace_id, dataspace_id, H5P_DEFAULT, f64_subview.data());
        H5Dclose(dataset_f64_annote_id);
      }

      // Close HDF5 objects 
      status = H5Dclose(dataset_dx_id);
      status = H5Dclose(dataset_dy_id);
      status = H5Dclose(dataset_dz_id);
      status = H5Dclose(dataset_ux_id);
      status = H5Dclose(dataset_uy_id);
      status = H5Dclose(dataset_uz_id);
      status = H5Dclose(dataset_w_id);
      status = H5Dclose(dataset_i_id);
      status = H5Sclose(memspace_id);
      status = H5Sclose(dataspace_id);
    }
    status = H5Gclose(group_id);
    status = H5Fclose(file_id);

    // If needed, reset weight to 0 for copied particles
    if(sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }

    // Clear buffers
    sp->nparticles_buffered = 0;
    sp->np_per_ts_io_buffer.clear();
  }
  H5Pclose(plist_id);
}

void vpic_simulation::dump_tracers_hdf5(const char* sp_name, 
                                        const uint32_t dump_vars,
                                        const char*fbase, 
                                        int append) {
  char fname[256];
  char group_name[256];
//  char particle_scratch[128];
//  char subparticle_scratch[128];
    
  // Get species
  species_t* sp = find_species_name( sp_name, tracers_list );
  if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name));

  // Update the particles on the host only if they haven't been recently
  if (step() > sp->last_copied)
    sp->copy_to_host();

  // Calculate total number of tracers and per rank offsets
  const long long np_local = sp->np;
  uint64_t total_particles, offset;
  uint64_t num_particles = np_local;
  MPI_Allreduce(&num_particles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Scan(&num_particles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  offset -= num_particles;
  
//  int sp_np = sp->np;
//  int sp_max_np = sp->max_np;

  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;

  // Copy weight to particles in the copy case for any hydro calculations
  if(sp->tracer_type == TracerType::Copy) {
    int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
    auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
    auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
    auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
    Kokkos::deep_copy(w_subview_h, w_annote);
    Kokkos::deep_copy(w_subview_d, w_subview_h);
  }

  // Compute hydro quantities
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
//  int tracer_idx = sp->annotation_vars.get_annotation_index<int64_t>("TracerID");
  auto& interp = interpolator_array->k_i_h;

  // Set output file name and group name
  sprintf(fname, "%s.h5", sp_name);
  sprintf(group_name, "/Timestep_%ld", step());

  if( rank()==0 )
      MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,fname));

  // Create file access template with parallel IO access
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  // Try to create species HDF5 file with default file creation/access property lists
  hid_t file_id;
  if(append == 0) {
    file_id = H5Fcreate(fname, H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
  } else {
    file_id = H5Fopen(fname, H5F_ACC_RDWR, plist_id);
  }

  // Create group for each step
  hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Close plist_id
  H5Pclose(plist_id);

  // Calculate total number of tracers and per rank offsets
  num_particles = sp->np;
  MPI_Allreduce(&num_particles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Scan(&num_particles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  offset -= num_particles;

  // Create dataspace describing dims for particle datasets
  hid_t dataspace_id = H5Screate_simple(1, (hsize_t*)(&total_particles), NULL);

  // Set MPIO to collective mode
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // Select slab of dataset for each rank
  herr_t status;
  hsize_t stride = 1;
  hsize_t block = 1;
  status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, (hsize_t*)(&offset), &stride, (hsize_t*)(&num_particles), &block);

  // Create memspace
  hid_t memspace_id = H5Screate_simple(1, (hsize_t*)(&num_particles), NULL);

  // Create subviews for data
  auto dx_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::dx));
  auto dy_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::dy));
  auto dz_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::dz));
  auto ux_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::ux));
  auto uy_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::uy));
  auto uz_subview = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::uz));
  auto w_subview  = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), (int)(particle_var::w));

  // Create datasets, one for each variable, using dataspace and default property lists
  hid_t dataset_dx_id = H5Dcreate(group_id, "dx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_dy_id = H5Dcreate(group_id, "dy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_dz_id = H5Dcreate(group_id, "dz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_ux_id = H5Dcreate(group_id, "ux", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_uy_id = H5Dcreate(group_id, "uy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_uz_id = H5Dcreate(group_id, "uz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_w_id  = H5Dcreate(group_id, "w",  H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_i_id  = H5Dcreate(group_id, "i",  H5T_STD_I32LE,    dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // Write data to slab
  status = H5Dwrite(dataset_dx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, dx_subview.data());
  status = H5Dwrite(dataset_dy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, dy_subview.data());
  status = H5Dwrite(dataset_dz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, dz_subview.data());
  status = H5Dwrite(dataset_ux_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, ux_subview.data());
  status = H5Dwrite(dataset_uy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, uy_subview.data());
  status = H5Dwrite(dataset_uz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, uz_subview.data());
  status = H5Dwrite(dataset_w_id,  H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, w_subview.data());
  status = H5Dwrite(dataset_i_id,  H5T_STD_I32LE,  memspace_id, dataspace_id, H5P_DEFAULT, sp->k_p_i_h.data());

  using host_memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;

  // Dump Global position if specified
  if(dump_vars & DumpVar::GlobalPos) {
    auto pos_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Pos Host View", sp->np);
    Kokkos::parallel_for("Calculate global position", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
//      float ux0 = sp->k_p_h(i, particle_var::ux);
//      float uy0 = sp->k_p_h(i, particle_var::uy);
//      float uz0 = sp->k_p_h(i, particle_var::uz);
//      float w0  = sp->k_p_h(i, particle_var::w);
      int   ii  = sp->k_p_i_h(i);
      
      // Compute global position of particle
      if(dump_vars & DumpVar::GlobalPos) {
        int nxg_ = grid->nx + 2;
        int nyg_ = grid->ny + 2;
//        int nzg_ = grid->nz + 2;
        int i0 = ii % nxg_;
        int j0 = (ii/nxg_) % nyg_;
        int k0 = ii/(nxg_*nyg_);
        float tracer_x = (i0 + (dx0-1)*0.5) * grid->dx + grid->x0;
        float tracer_y = (j0 + (dy0-1)*0.5) * grid->dy + grid->y0;
        float tracer_z = (k0 + (dz0-1)*0.5) * grid->dz + grid->z0;
        pos_view(i, 0) = tracer_x;
        pos_view(i, 1) = tracer_y;
        pos_view(i, 2) = tracer_z;
      }
    });
    auto posx_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 0);
    auto posy_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 1);
    auto posz_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 2);
    hid_t dataset_posx_id = H5Dcreate(group_id, "posx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_posy_id = H5Dcreate(group_id, "posy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_posz_id = H5Dcreate(group_id, "posz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_posx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, posx_subview.data());
    status = H5Dwrite(dataset_posy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, posy_subview.data());
    status = H5Dwrite(dataset_posz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, posz_subview.data());
    H5Dclose(dataset_posx_id);
    H5Dclose(dataset_posy_id);
    H5Dclose(dataset_posz_id);
  }

  // Dump E field if specified
  if(dump_vars & DumpVar::Efield) {
    auto efield_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("E field Host View", sp->np);
    Kokkos::parallel_for("Calculate E field", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      int   ii  = sp->k_p_i_h(i);
      efield_view(i,0) = interp(ii,interpolator_var::ex ) + dy0*interp(ii,interpolator_var::dexdy) + dz0*(interp(ii,interpolator_var::dexdz) + dy0*interp(ii,interpolator_var::d2exdydz)); 
      efield_view(i,1) = interp(ii,interpolator_var::ey ) + dz0*interp(ii,interpolator_var::deydz) + dx0*(interp(ii,interpolator_var::deydx) + dz0*interp(ii,interpolator_var::d2eydzdx)); 
      efield_view(i,2) = interp(ii,interpolator_var::ez ) + dx0*interp(ii,interpolator_var::dezdx) + dy0*(interp(ii,interpolator_var::dezdy) + dx0*interp(ii,interpolator_var::d2ezdxdy)); 
    });
    auto efieldx_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 0);
    auto efieldy_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 1);
    auto efieldz_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 2);
    hid_t dataset_efieldx_id = H5Dcreate(group_id, "ex", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_efieldy_id = H5Dcreate(group_id, "ey", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_efieldz_id = H5Dcreate(group_id, "ez", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_efieldx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, efieldx_subview.data());
    status = H5Dwrite(dataset_efieldy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, efieldy_subview.data());
    status = H5Dwrite(dataset_efieldz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, efieldz_subview.data());
    H5Dclose(dataset_efieldx_id);
    H5Dclose(dataset_efieldy_id);
    H5Dclose(dataset_efieldz_id);
  }

  // Dump B field if specified
  if(dump_vars & DumpVar::Bfield) {
    auto bfield_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("B field Host View", sp->np);
    Kokkos::parallel_for("Calculate B field", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      int   ii  = sp->k_p_i_h(i);
      bfield_view(i,0)  = interp(ii,interpolator_var::cbx) + dx0*interp(ii,interpolator_var::dcbxdx); 
      bfield_view(i,1)  = interp(ii,interpolator_var::cby) + dy0*interp(ii,interpolator_var::dcbydy); 
      bfield_view(i,2)  = interp(ii,interpolator_var::cbz) + dz0*interp(ii,interpolator_var::dcbzdz); 
    });
    auto bfieldx_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 0);
    auto bfieldy_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 1);
    auto bfieldz_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 2);
    hid_t dataset_bfieldx_id = H5Dcreate(group_id, "bx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_bfieldy_id = H5Dcreate(group_id, "by", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_bfieldz_id = H5Dcreate(group_id, "bz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_bfieldx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, bfieldx_subview.data());
    status = H5Dwrite(dataset_bfieldy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, bfieldy_subview.data());
    status = H5Dwrite(dataset_bfieldz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, bfieldz_subview.data());
    H5Dclose(dataset_bfieldx_id);
    H5Dclose(dataset_bfieldy_id);
    H5Dclose(dataset_bfieldz_id);
  }

  // Dump current density if specified
  if(dump_vars & DumpVar::CurrentDensity) {
    auto current_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Current density Host View", sp->np);
    Kokkos::parallel_for("Collect current density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      int   ii  = sp->k_p_i_h(i);
      current_view(i, 0) = hydro_array->k_h_h(ii, hydro_var::jx);
      current_view(i, 1) = hydro_array->k_h_h(ii, hydro_var::jy);
      current_view(i, 2) = hydro_array->k_h_h(ii, hydro_var::jz);
    });
    auto jx_subview = Kokkos::subview(current_view, Kokkos::ALL(), 0);
    auto jy_subview = Kokkos::subview(current_view, Kokkos::ALL(), 1);
    auto jz_subview = Kokkos::subview(current_view, Kokkos::ALL(), 2);
    hid_t dataset_jx_id = H5Dcreate(group_id, "jx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_jy_id = H5Dcreate(group_id, "jy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_jz_id = H5Dcreate(group_id, "jz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_jx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, jx_subview.data());
    status = H5Dwrite(dataset_jy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, jy_subview.data());
    status = H5Dwrite(dataset_jz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, jz_subview.data());
    H5Dclose(dataset_jx_id);
    H5Dclose(dataset_jy_id);
    H5Dclose(dataset_jz_id);
  }

  // Dump charge density if specified
  if(dump_vars & DumpVar::ChargeDensity) {
    auto charge_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("Charge density Host View", sp->np);
    Kokkos::parallel_for("Calculate charge density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      int   ii  = sp->k_p_i_h(i);
      charge_view(i) = hydro_array->k_h_h(ii, hydro_var::rho);
    });
    hid_t dataset_rho_id = H5Dcreate(group_id, "rho", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_rho_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, charge_view.data());
    H5Dclose(dataset_rho_id);
  }

  // Dump momentum density if specified
  if(dump_vars & DumpVar::MomentumDensity) {
    auto momentum_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Momentum Host View", sp->np);
    Kokkos::parallel_for("Collect momentum density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      int   ii  = sp->k_p_i_h(i);
      momentum_view(i, 0) = hydro_array->k_h_h(ii, hydro_var::px);
      momentum_view(i, 1) = hydro_array->k_h_h(ii, hydro_var::py);
      momentum_view(i, 2) = hydro_array->k_h_h(ii, hydro_var::pz);
    });
    auto px_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 0);
    auto py_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 1);
    auto pz_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 2);
    hid_t dataset_px_id = H5Dcreate(group_id, "px", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_py_id = H5Dcreate(group_id, "py", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_pz_id = H5Dcreate(group_id, "pz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_px_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, px_subview.data());
    status = H5Dwrite(dataset_py_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, py_subview.data());
    status = H5Dwrite(dataset_pz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, pz_subview.data());
    H5Dclose(dataset_px_id);
    H5Dclose(dataset_py_id);
    H5Dclose(dataset_pz_id);
  }

  // Dump kinetic energy density if specified
  if(dump_vars & DumpVar::KEDensity) {
    auto ke_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("KE Host View", sp->np);
    auto momentum_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Momentum Host View", sp->np);
    Kokkos::parallel_for("Collect KE density", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      int   ii  = sp->k_p_i_h(i);
      ke_view(i) = hydro_array->k_h_h(ii, hydro_var::ke);
    });
    hid_t dataset_ke_id = H5Dcreate(group_id, "ke_dens", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_ke_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, ke_view.data());
    H5Dclose(dataset_ke_id);
  }

  // Dump stress tensor if specified
  if(dump_vars & DumpVar::StressTensor) {
    auto stress_view = Kokkos::View<float*[6], Kokkos::LayoutLeft, host_memory_space>("Stress tensor Host View", sp->np);
    Kokkos::parallel_for("Collect stress tensor", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      int   ii  = sp->k_p_i_h(i);
      stress_view(i, 0) = hydro_array->k_h_h(ii, hydro_var::txx);
      stress_view(i, 1) = hydro_array->k_h_h(ii, hydro_var::tyy);
      stress_view(i, 2) = hydro_array->k_h_h(ii, hydro_var::tzz);
      stress_view(i, 3) = hydro_array->k_h_h(ii, hydro_var::tyz);
      stress_view(i, 4) = hydro_array->k_h_h(ii, hydro_var::tzx);
      stress_view(i, 5) = hydro_array->k_h_h(ii, hydro_var::txy);
    });
    auto txx_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 0);
    auto tyy_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 1);
    auto tzz_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 2);
    auto tyz_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 3);
    auto tzx_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 4);
    auto txy_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 5);
    hid_t dataset_txx_id = H5Dcreate(group_id, "txx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_tyy_id = H5Dcreate(group_id, "tyy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_tzz_id = H5Dcreate(group_id, "tzz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_tyz_id = H5Dcreate(group_id, "tyz", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_tzx_id = H5Dcreate(group_id, "tzx", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_txy_id = H5Dcreate(group_id, "txy", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_txx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, txx_subview.data());
    status = H5Dwrite(dataset_tyy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tyy_subview.data());
    status = H5Dwrite(dataset_tzz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tzz_subview.data());
    status = H5Dwrite(dataset_tyz_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tyz_subview.data());
    status = H5Dwrite(dataset_tzx_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, tzx_subview.data());
    status = H5Dwrite(dataset_txy_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, txy_subview.data());
    H5Dclose(dataset_txx_id);
    H5Dclose(dataset_tyy_id);
    H5Dclose(dataset_tzz_id);
    H5Dclose(dataset_tyz_id);
    H5Dclose(dataset_tzx_id);
    H5Dclose(dataset_txy_id);
  }

  // Dump kinetic energy of particle if specified
  if(dump_vars & DumpVar::ParticleKE) {
    auto ke_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("KE Host View", sp->np);
    Kokkos::parallel_for("Calculate KE", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np), KOKKOS_LAMBDA(const uint32_t i) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      int   ii  = sp->k_p_i_h(i);
      float ux0 = sp->k_p_h(i, particle_var::ux);
      float uy0 = sp->k_p_h(i, particle_var::uy);
      float uz0 = sp->k_p_h(i, particle_var::uz);
      float w0  = sp->k_p_h(i, particle_var::w);
      float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
      float msp = sp->m;
      float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                             dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
      float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                             dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
      float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                             dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
      v0 = v0*v0 + v1*v1 + v2*v2;
      v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
      ke_view(i) = v0;
    });
    hid_t dataset_ke_id = H5Dcreate(group_id, "ke", H5T_NATIVE_FLOAT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_ke_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, ke_view.data());
    H5Dclose(dataset_ke_id);
  }

  // Dump int annotations
  for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
    auto i32_subview = Kokkos::subview(sp->annotations_h.i32, Kokkos::ALL, j);
    hid_t dataset_i32_annote_id = H5Dcreate(group_id, sp->annotation_vars.i32_vars[j].c_str(), H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_i32_annote_id, H5T_STD_I32LE, memspace_id, dataspace_id, H5P_DEFAULT, i32_subview.data());
    H5Dclose(dataset_i32_annote_id);
  }
  // Dump 64-bit integer annotations
  for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
    auto i64_subview = Kokkos::subview(sp->annotations_h.i64, Kokkos::ALL, j);
    hid_t dataset_i64_annote_id = H5Dcreate(group_id, sp->annotation_vars.i64_vars[j].c_str(), H5T_STD_I64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_i64_annote_id, H5T_STD_I64LE, memspace_id, dataspace_id, H5P_DEFAULT, i64_subview.data());
    H5Dclose(dataset_i64_annote_id);
  }
  // Dump 32-bit floating-point annotations
  for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
    auto f32_subview = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL, j);
    hid_t dataset_f32_annote_id = H5Dcreate(group_id, sp->annotation_vars.f32_vars[j].c_str(), H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_f32_annote_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, f32_subview.data());
    H5Dclose(dataset_f32_annote_id);
  }
  // Dump 64-bit floating-point annotations
  for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
    auto f64_subview = Kokkos::subview(sp->annotations_h.f64, Kokkos::ALL, j);
    hid_t dataset_f64_annote_id = H5Dcreate(group_id, sp->annotation_vars.f64_vars[j].c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_f64_annote_id, H5T_IEEE_F64LE, memspace_id, dataspace_id, H5P_DEFAULT, f64_subview.data());
    H5Dclose(dataset_f64_annote_id);
  }

  // If needed, reset weight to 0 for copied particles
  if(sp->tracer_type == TracerType::Copy) {
    auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
    auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
    Kokkos::deep_copy(w_subview_h, 0.0);
    Kokkos::deep_copy(w_subview_d, 0.0);
  }

  // Close 
  status = H5Dclose(dataset_dx_id);
  status = H5Dclose(dataset_dy_id);
  status = H5Dclose(dataset_dz_id);
  status = H5Dclose(dataset_ux_id);
  status = H5Dclose(dataset_uy_id);
  status = H5Dclose(dataset_uz_id);
  status = H5Dclose(dataset_w_id);
  status = H5Dclose(dataset_i_id);
  status = H5Sclose(memspace_id);
  status = H5Sclose(dataspace_id);
  status = H5Pclose(plist_id);
  status = H5Gclose(group_id);
  status = H5Fclose(file_id);
}
#endif

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
  char species_comment[256];
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
