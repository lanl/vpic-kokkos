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

// C++ headers
#include <cassert>
#include <iostream>

// VPIC headers
#include "vpic.h"
#include "dumpmacros.h"
#include "../util/io/FileUtils.h"
#include <filesystem>

/* -1 means no ranks talk */
#define VERBOSE_rank -1

// TODO: this should live somewhere more sensible
std::unordered_map<species_id, size_t> tframe_map;
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

void vpic_simulation::enable_binary_dump()
{
  //    dump_strategy = std::unique_ptr<Dump_Strategy>(new BinaryDump( rank(), nproc() ));
  // dump_strategy = new BinaryDump(rank(), nproc());
  dump_strategy_id = DUMP_STRATEGY_BINARY;
}

#ifdef VPIC_ENABLE_HDF5
void vpic_simulation::enable_hdf5_dump()
{
  if (rank() == 0)
    std::cout << "Enabling HDF5 IO backend" << std::endl;
  // dump_strategy = std::unique_ptr<Dump_Strategy>(new HDF5Dump(rank(), nproc()));
  // dump_strategy = new HDF5Dump(rank(), nproc());
  dump_strategy_id = DUMP_STRATEGY_HDF5;
}
#endif

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

  // Piggybacking on existing allreduce to do global NaN check
  // and terminate simulation early if needed
  bool hasnan_f = false;
  bool hasnan_p = false;
  hasnan_f = hasnan_f || (en_f[0] != en_f[0]);
  hasnan_f = hasnan_f || (en_f[1] != en_f[1]);
  hasnan_f = hasnan_f || (en_f[2] != en_f[2]);
  hasnan_f = hasnan_f || (en_f[3] != en_f[3]);
  hasnan_f = hasnan_f || (en_f[4] != en_f[4]);
  hasnan_f = hasnan_f || (en_f[5] != en_f[5]);

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  std::map<std::string, double> energy_map;
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
    } else if(rank() == 0 && status!=fail && sp->is_tracer) {
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
    hasnan_p = hasnan_p || (en_p != en_p);
  }
#endif

  if( rank()==0 && status!=fail ) {
    fileIO.print( "\n" );
    if( fileIO.close() ) ERROR(("File close failed on dump energies!!!"));
  }

  // NaN check should come after file write closes cleanly
  if (hasnan_f) ERROR(("NaN found in field energies, terminating early"));
  if (hasnan_p) ERROR(("NaN found in particle energies, terminating early"));
}

void
vpic_simulation::dump_particles_count( const char *fname,
                                       int append ) {
  species_t *sp;
  FileIO fileIO;
  FileIOStatus status(fail);

  if( !fname ) ERROR(("Invalid file name"));

  if( rank()==0 ) {
    status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\".", fname ));
    else {
      if( append==0 ) {
        fileIO.print( "%% Layout\n%% step" );
        LIST_FOR_EACH(sp,species_list)
          fileIO.print( " \"%s\"", sp->name );
        fileIO.print( "\n" );
        fileIO.print( "%% timestep = %e\n", grid->dt );
      }
      fileIO.print( "%li", (long)step() );
    }
  }

  LIST_FOR_EACH(sp,species_list) {
    int64_t local = (int64_t)sp->np;
    int64_t global;
    mp_allsum_li( &local, &global, 1 );
    if( rank()==0 && status!=fail ) fileIO.print( " %li", global );
  }

  if( rank()==0 && status!=fail ) {
    fileIO.print( "\n" );
    if( fileIO.close() ) ERROR(("File close failed on dump count particles!!!"));
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
vpic_simulation::dump_fluid_species( const char *fname ) {
  fluid_species_t *fsp;
  FileIO fileIO;

  if( rank() ) return;
  if( !fname ) ERROR(( "Invalid file name" ));
  MESSAGE(( "Dumping fluid species to \"%s\"", fname ));
  FileIOStatus status = fileIO.open(fname, io_write);
  if( status==fail ) ERROR(( "Could not open \"%s\".", fname ));
  LIST_FOR_EACH( fsp, fluid_species_list )
    fileIO.print( "%s %i %e %e", fsp->name, fsp->id, fsp->q, fsp->m );
  if( fileIO.close() ) ERROR(( "File close failed on dump fluid species!!!" ));
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

/*****************************************************************************
 * Binary dump IO
 *****************************************************************************/

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

  WRITE_HEADER_V0(dump_type::grid_dump, -1, 0, fileIO, step(), rank(), nproc());

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
  dump_strategy->dump_fields(
      fbase,
      step(),
      grid,
      field_array,
      ftag);
}

void
vpic_simulation::dump_hydro( const char *sp_name,
                             const char *fbase,
                             int ftag ) {
  species_t *sp = find_species_name(sp_name, species_list);
  dump_strategy->dump_hydro(
      fbase,
      step(),
      sp,
      grid,
      hydro_array,
      interpolator_array,
      ftag);
}

void
vpic_simulation::dump_particles( const char *sp_name,
                                 const char *fbase,
                                 int ftag ) {

  species_t *sp = find_species_name(sp_name, species_list);
  dump_strategy->dump_particles(
      fbase,
      step(),
      sp,
      grid,
      interpolator_array,
      ftag);
}

void
vpic_simulation::dump_fluids( const char *fsp_name,
            const char *fbase,
            int ftag ) {

  fluid_species_t *fsp = find_fluid_species_name( fsp_name, fluid_species_list );
  dump_strategy->dump_fluids(
      fbase,
      step(),
      fsp,
      grid,
      ftag);
}

/*------------------------------------------------------------------------------
 * New dump logic
 *---------------------------------------------------------------------------*/

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

/*void
vpic_simulation::create_fluid_list( char * strlist,
                                    DumpParameters & dumpParams ) {
  strcpy(strlist, "");
  for(size_t i(0), pass(0); i<total_fluid_groups; i++)
    if(dumpParams.output_vars.bitset(fluid_indeces[i])) {
      if(i>0 && pass) strcat(strlist, ", ");
      else pass = 1;
      strcat(strlist, fluidInfo[i].name);
    }
    }*/

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
  char species_comment[max_filename_bytes];
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

  dump_strategy->field_dump(
      dumpParams,
      step(),
      grid,
      field_array);
}

void
vpic_simulation::hydro_dump( const char * speciesname,
                             DumpParameters & dumpParams ) {

  species_t * sp = find_species_name(speciesname, species_list);
  dump_strategy->hydro_dump(
      dumpParams,
      step(),
      sp,
      grid,
      hydro_array,
      interpolator_array);
}

void
vpic_simulation::fluid_dump( const char * speciesname,
                             DumpParameters & dumpParams ) {

  fluid_species_t *fsp = find_fluid_species_name( speciesname, fluid_species_list );
  dump_strategy->fluid_dump(
      dumpParams,
      step(),
      fsp,
      grid);
}

#if defined( VPIC_ENABLE_HDF5 ) && defined( VPIC_ENABLE_TRACER_PARTICLES )
void 
vpic_simulation::tracer_dump(const char* species_name, 
                             DumpParameters& dumpParams)
{
  std::string filename = std::string(dumpParams.baseDir) + std::string("/") 
                       + std::string(dumpParams.baseFileName);
  uint32_t dump_vars = 0;
  for(uint32_t i=0; i<32; i++) {
    if(dumpParams.output_vars.bitset(i)) {
      dump_vars |= (1<<i);
    }
  }
  species_t* sp = find_species_name( species_name, tracers_list );
  // Dump tracers to buffer or file
  if(sp->np_buffered_max > 0) {
    dump_tracers_buffered_hdf5(species_name, dump_vars, filename.c_str());
  } else {
    dump_tracers_hdf5(species_name, dump_vars, filename.c_str());
  }
}
#endif
