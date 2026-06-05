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

// FIXME: MOVE THIS INTO VPIC.HXX TO BE TRULY INLINE

void
vpic_simulation::inject_particle( species_t * sp,
                                  double x,  double y,  double z,
                                  double ux, double uy, double uz,
                                  double w,  double age,
                                  int update_rhob,
                                  double qp) {
  int ix, iy, iz;

  // Check input parameters
  if( !sp                ) ERROR(( "Invalid species" ));
  if( w < 0              ) ERROR(( "inject_particle: w < 0" ));

  const double x0 = (double)grid->x0, y0 = (double)grid->y0, z0 = (double)grid->z0;
  const double x1 = (double)grid->x1, y1 = (double)grid->y1, z1 = (double)grid->z1;
  const int    nx = grid->nx,         ny = grid->ny,         nz = grid->nz;
  
  // Do not inject if the particle is strictly outside the local domain
  // or if a far wall of local domain shared with a neighbor
  // FIXME: DO THIS THE PHASE-3 WAY WITH GRID->NEIGHBOR
  // NOT THE PHASE-2 WAY WITH GRID->BC

  if( (x<x0) | (x>x1) | ( (x==x1) & (grid->bc[BOUNDARY(1,0,0)]>=0 ) ) ) return;
  if( (y<y0) | (y>y1) | ( (y==y1) & (grid->bc[BOUNDARY(0,1,0)]>=0 ) ) ) return;
  if( (z<z0) | (z>z1) | ( (z==z1) & (grid->bc[BOUNDARY(0,0,1)]>=0 ) ) ) return;

  // This node should inject the particle
    
  if( sp->np>=sp->max_np ) ERROR(( "No room to inject particle" ));

  // Compute the injection cell and coordinate in cell coordinate system
  // BJA:  Note the use of double precision here for accurate particle 
  //       placement on large meshes. 
 
  // The ifs allow for injection on the far walls of the local computational
  // domain when necessary
 
  x  = ((double)nx)*((x-x0)/(x1-x0)); // x is rigorously on [0,nx]
  ix = (int)x;                        // ix is rigorously on [0,nx]
  x -= (double)ix;                    // x is rigorously on [0,1)
  x  = (x+x)-1;                       // x is rigorously on [-1,1)
  if( ix==nx ) x = 1;                 // On far wall ... conditional move
  if( ix==nx ) ix = nx-1;             // On far wall ... conditional move
  ix++;                               // Adjust for mesh indexing

  y  = ((double)ny)*((y-y0)/(y1-y0)); // y is rigorously on [0,ny]
  iy = (int)y;                        // iy is rigorously on [0,ny]
  y -= (double)iy;                    // y is rigorously on [0,1)
  y  = (y+y)-1;                       // y is rigorously on [-1,1)
  if( iy==ny ) y = 1;                 // On far wall ... conditional move
  if( iy==ny ) iy = ny-1;             // On far wall ... conditional move
  iy++;                               // Adjust for mesh indexing

  z  = ((double)nz)*((z-z0)/(z1-z0)); // z is rigorously on [0,nz]
  iz = (int)z;                        // iz is rigorously on [0,nz]
  z -= (double)iz;                    // z is rigorously on [0,1)
  z  = (z+z)-1;                       // z is rigorously on [-1,1)
  if( iz==nz ) z = 1;                 // On far wall ... conditional move
  if( iz==nz ) iz = nz-1;             // On far wall ... conditional move
  iz++;                               // Adjust for mesh indexing

#ifdef VPIC_ENABLE_LEGACY_DATA_STRUCTURES
  size_t p_index = Kokkos::atomic_fetch_inc(&(sp->np));
  particle_t * p = sp->p + p_index;
  p->dx = (float)x; // Note: Might be rounded to be on [-1,1]
  p->dy = (float)y; // Note: Might be rounded to be on [-1,1]
  p->dz = (float)z; // Note: Might be rounded to be on [-1,1]
  p->i  = VOXEL(ix,iy,iz, nx,ny,nz);
  p->ux = (float)ux;
  p->uy = (float)uy;
  p->uz = (float)uz;
  p->w  = w;
#ifdef VARIABLE_CHARGE
  if(qp == std::numeric_limits<double>::infinity()) {
    p->qp = sp->q;
  } else {
    p->qp = (float)qp;
  }
#endif
#ifdef VPIC_ENABLE_TRACER_PARTICLES
  if(sp->is_tracer) {
    int tracer_idx = sp->annotation_vars.get_annotation_index<int>(std::string("TracerID"));
    sp->annotations_h.set<int>(p_index, tracer_idx, rank()*sp->max_np + p_index); 
  }
#endif

  if( update_rhob ) accumulate_rhob( field_array->f, p, grid, -sp->q );

  if( age!=0 ) {
    if( sp->nm>=sp->max_nm )
      WARNING(( "No movers available to age injected  particle" ));
    particle_mover_t * pm = sp->pm + sp->nm;
    age *= grid->cvac*grid->dt/sqrt( ux*ux + uy*uy + uz*uz + 1 );
    pm->dispx = ux*age*grid->rdx;
    pm->dispy = uy*age*grid->rdy;
    pm->dispz = uz*age*grid->rdz;
    pm->i     = sp->np-1;
    sp->nm += move_p( sp->p, pm, field_array->k_jf_accum_h, grid, sp->q );
  }
#else
  size_t idx = Kokkos::atomic_fetch_inc(&(sp->np));
  sp->k_p_h(idx, particle_var::dx) = static_cast<float>(x);
  sp->k_p_h(idx, particle_var::dy) = static_cast<float>(y);
  sp->k_p_h(idx, particle_var::dz) = static_cast<float>(z);
  sp->k_p_h(idx, particle_var::ux) = static_cast<float>(ux);
  sp->k_p_h(idx, particle_var::uy) = static_cast<float>(uy);
  sp->k_p_h(idx, particle_var::uz) = static_cast<float>(uz);
  sp->k_p_h(idx, particle_var::w)  = w;
  sp->k_p_i_h(idx) = VOXEL(ix,iy,iz,nx,ny,nz);
#ifdef VARIABLE_CHARGE
  if(qp == std::numeric_limits<double>::infinity()) {
    sp->k_p_h(idx, particle_var::qp) = sp->q;
  } else {
    sp->k_p_h(idx, particle_var::qp) = static_cast<float>(qp);
  }
#endif

  if( update_rhob ) k_accumulate_rhob_single_cpu( field_array->k_f_rhob_accum_h, sp->k_p_h, sp->k_p_i_h, idx, grid, -sp->q);

  if( age!=0 ) {
    if( sp->nm >= sp->max_nm )
      WARNING(( "No movers available to age injected particle" ));
    particle_mover_t * pm = sp->pm + sp->nm;
    age *= grid->cvac*grid->dt/sqrt( ux*ux + uy*uy + uz*uz + 1 );
    pm->dispx = ux*age*grid->rdx;
    pm->dispy = uy*age*grid->rdy;
    pm->dispz = uz*age*grid->rdz;
    pm->i     = idx;

    move_p_kokkos_host_serial( sp->k_p_h, sp->k_p_i_h, pm, 
                               field_array->k_jf_accum_h, 
                               grid, grid->k_neighbor_h, 
                               grid->rangel, grid->rangeh, 
                               sp->q );
  }
#endif
}

void
vpic_simulation::inject_particle_r( species_t * sp,
                                    double x,  double y,  double z,
                                    double ux, double uy, double uz,
                                    double w,  double age,
                                    int update_rhob,
                                    double qp ) {
  int ix, iy, iz;

  // Check input parameters
  if( !sp                ) ERROR(( "Invalid species" ));
  if( w < 0              ) ERROR(( "inject_particle: w < 0" ));

  const double x0 = (double)grid->x0, y0 = (double)grid->y0, z0 = (double)grid->z0;
  const double x1 = (double)grid->x1, y1 = (double)grid->y1, z1 = (double)grid->z1;
  const int    nx = grid->nx,         ny = grid->ny,         nz = grid->nz;
  
  // Do not inject if the particle is strictly outside the local domain
  // or if a far wall of local domain shared with a neighbor
  // FIXME: DO THIS THE PHASE-3 WAY WITH GRID->NEIGHBOR
  // NOT THE PHASE-2 WAY WITH GRID->BC

  if( (x<x0) | (x>x1) | ( (x==x1) & (grid->bc[BOUNDARY(1,0,0)]>=0 ) ) ) return;
  if( (y<y0) | (y>y1) | ( (y==y1) & (grid->bc[BOUNDARY(0,1,0)]>=0 ) ) ) return;
  if( (z<z0) | (z>z1) | ( (z==z1) & (grid->bc[BOUNDARY(0,0,1)]>=0 ) ) ) return;

  // This node should inject the particle
    
  if( sp->np>=sp->max_np ) ERROR(( "No room to inject particle" )); // Remove?

  // Compute the injection cell and coordinate in cell coordinate system
  // BJA:  Note the use of double precision here for accurate particle 
  //       placement on large meshes. 
 
  // The ifs allow for injection on the far walls of the local computational
  // domain when necessary
 
  x  = ((double)nx)*((x-x0)/(x1-x0)); // x is rigorously on [0,nx]
  ix = (int)x;                        // ix is rigorously on [0,nx]
  x -= (double)ix;                    // x is rigorously on [0,1)
  x  = (x+x)-1;                       // x is rigorously on [-1,1)
  if( ix==nx ) x = 1;                 // On far wall ... conditional move
  if( ix==nx ) ix = nx-1;             // On far wall ... conditional move
  ix++;                               // Adjust for mesh indexing

  y  = ((double)ny)*((y-y0)/(y1-y0)); // y is rigorously on [0,ny]
  iy = (int)y;                        // iy is rigorously on [0,ny]
  y -= (double)iy;                    // y is rigorously on [0,1)
  y  = (y+y)-1;                       // y is rigorously on [-1,1)
  if( iy==ny ) y = 1;                 // On far wall ... conditional move
  if( iy==ny ) iy = ny-1;             // On far wall ... conditional move
  iy++;                               // Adjust for mesh indexing

  z  = ((double)nz)*((z-z0)/(z1-z0)); // z is rigorously on [0,nz]
  iz = (int)z;                        // iz is rigorously on [0,nz]
  z -= (double)iz;                    // z is rigorously on [0,1)
  z  = (z+z)-1;                       // z is rigorously on [-1,1)
  if( iz==nz ) z = 1;                 // On far wall ... conditional move
  if( iz==nz ) iz = nz-1;             // On far wall ... conditional move
  iz++;                               // Adjust for mesh indexing

  size_t p_index = Kokkos::atomic_fetch_inc(&(sp->np));

  // Add particle to receive list (on host), so it will be copied to
  // device along with the boundary_p particles.
  auto& particle_recv = sp->k_pr_h;
  auto& particle_recv_i = sp->k_pr_i_h;
  int write_index = sp->num_to_copy;

  particle_recv(write_index, particle_var::dx) = (float)x;
  particle_recv(write_index, particle_var::dy) = (float)y;
  particle_recv(write_index, particle_var::dz) = (float)z;
  particle_recv(write_index, particle_var::ux) = (float)ux;
  particle_recv(write_index, particle_var::uy) = (float)uy;
  particle_recv(write_index, particle_var::uz) = (float)uz;
  particle_recv(write_index, particle_var::w)  = w;
#ifdef VARIABLE_CHARGE
  if(qp == std::numeric_limits<double>::infinity()) {
    particle_recv(write_index, particle_var::qp) = sp->q;
  } else {
    particle_recv(write_index, particle_var::qp) = (float)qp;
  }
#endif
  
  int pii = VOXEL(ix,iy,iz, nx,ny,nz);
  particle_recv_i(write_index) = pii;

  // track how many particles we buffer up here
  sp->num_to_copy++;

  if( update_rhob ) ERROR(( "rho_b update not yet implemented." ));

  if( age!=0 ) ERROR(( "Aging of injected particles not yet implemented." ));

}


void
vpic_simulation::apply_artificial_loss_cone( species_t * sp,
                                             float tan2_alpha_lc,
                                             float dt_lc,
                                             float ML ) {
  if( !sp ) ERROR(( "apply_artificial_loss_cone: Invalid species" ));
  if( tan2_alpha_lc < 0 ) ERROR(( "apply_artificial_loss_cone: tan2_alpha_lc < 0" ));

  const int np = sp->np;
  if( np <= 0 ) return;

  auto kp  = sp->k_p_d;
  auto kpi = sp->k_p_i_d;
  auto kf  = field_array->k_f_d;

  const int nv = grid->nv;

  Kokkos::parallel_for(
    "apply_artificial_loss_cone",
    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, np),
    KOKKOS_LAMBDA (const int n) {

      const float w = kp(n, particle_var::w);
      if( w == 0.f ) return;

      const int cell = kpi(n);
      if( cell < 0 || cell >= nv ) return;

      const float ux = kp(n, particle_var::ux);
      const float uy = kp(n, particle_var::uy);
      const float uz = kp(n, particle_var::uz);

      const float Bx = kf(cell, field_var::cbx) + kf(cell, field_var::cbx0);
      const float By = kf(cell, field_var::cby) + kf(cell, field_var::cby0);
      const float Bz = kf(cell, field_var::cbz) + kf(cell, field_var::cbz0);

      const float B2 = Bx*Bx + By*By + Bz*Bz;
      if( B2 <= 0.f ) return;

      const float invB = 1.f / sqrtf(B2);
      const float bhx  = Bx * invB;
      const float bhy  = By * invB;
      const float bhz  = Bz * invB;

      const float upar  = ux*bhx + uy*bhy + uz*bhz;
      const float upar2 = upar*upar;
      if( upar2 <= 0.f ) return;

      const float u2 = ux*ux + uy*uy + uz*uz;
      float uperp2 = u2 - upar2;
      if( uperp2 < 0.f ) uperp2 = 0.f;

      if( uperp2 < upar2 * tan2_alpha_lc ) {
        // hard cut model
        //kp(n, particle_var::w)  = 0.f;

        //decay model
        float wnew = w * Kokkos::exp( -dt_lc * Kokkos::abs(upar) / ML );
        // if( wnew < 1e-12f ) {
        //   wnew = 0.f;
        // }
        kp(n, particle_var::w) = wnew;
      }
    });

  Kokkos::fence();
}


// Add capability to modify certain fields "on the fly" so that one
// can, e.g., extend a run, change a quota, or modify a dump interval
// without having to rerun from the start.
//
// File is specified in arg 3 slot in command line inputs.  File is in
// ASCII format with each field in the form: field val [newline].
//
// Allowable values of field variables are: num_steps, quota,
// checkpt_interval, hydro_interval, field_interval, particle_interval
// ndfld, ndhyd, ndpar, ndhis, ndgrd, head_option,
// istride, jstride, kstride, stride_option, pstride
//
// [x]_interval sets interval value for dump type [x].  Set interval
// to zero to turn off dump type.
//
// FIXME-KJB: STRIP_CMDLINE ALLOWS SOMETHING CLEANER AND MORE POWERFUL
 
#define SETIVAR( V, A, S ) do {                                          \
    V=(A);                                                               \
    if ( rank()==0 ) log_printf( "*** Modifying %s to value %d", S, A ); \
  } while(0)

#define SETDVAR( V, A, S ) do {                                           \
    V=(A);                                                                \
    if ( rank()==0 ) log_printf( "*** Modifying %s to value %le", S, A ); \
  } while(0)

#define ITEST( V, N, A ) \
  if( sscanf( line, N " %d",  &iarg )==1 ) SETIVAR( V, A, N )

#define DTEST( V, N, A ) \
  if( sscanf( line, N " %le", &darg )==1 ) SETDVAR( V, A, N )
 
void
vpic_simulation::modify( const char *fname ) {
  FILE *handle=NULL;
  char line[128];
  int iarg=0;
  double darg=0;
 
  // Open the modfile
  handle = fopen( fname, "r" );
  if( !handle ) ERROR(( "Modfile read failed" ));
  // Parse modfile
  while( fgets( line, 127, handle ) ) {
    DTEST( quota,             "quota",             darg );
    ITEST( num_step,          "num_step",          iarg );
    ITEST( checkpt_interval,  "checkpt_interval",  (iarg<0 ? 0 : iarg) );
    ITEST( hydro_interval,    "hydro_interval",    (iarg<0 ? 0 : iarg) );
    ITEST( field_interval,    "field_interval",    (iarg<0 ? 0 : iarg) );
    ITEST( particle_interval, "particle_interval", (iarg<0 ? 0 : iarg) );
    ITEST( ndfld, "ndfld", (iarg<0 ? 0 : iarg) );
    ITEST( ndhyd, "ndhyd", (iarg<0 ? 0 : iarg) );
    ITEST( ndpar, "ndpar", (iarg<0 ? 0 : iarg) );
    ITEST( ndhis, "ndhis", (iarg<0 ? 0 : iarg) );
    ITEST( ndgrd, "ndgrd", (iarg<0 ? 0 : iarg) );
    ITEST( head_option, "head_option", (iarg<0 ? 0 : iarg) );
    ITEST( istride, "istride", (iarg<1 ? 1 : iarg) );
    ITEST( jstride, "jstride", (iarg<1 ? 1 : iarg) );
    ITEST( kstride, "kstride", (iarg<1 ? 1 : iarg) );
    ITEST( stride_option, "stride_option", (iarg<1 ? 1 : iarg) );
    ITEST( pstride, "pstride", (iarg<1 ? 1 : iarg) );
    ITEST( stepdigit, "stepdigit", (iarg<0 ? 0 : iarg) );
    ITEST( rankdigit, "rankdigit", (iarg<0 ? 0 : iarg) );
  }
}

#undef SETIVAR
#undef SETDVAR
#undef ITEST
#undef DTEST

#if defined(ENABLE_OPENSSL)
#include "../util/checksum.h"

void vpic_simulation::checksum_fields(CheckSum & cs) {
  checkSumBuffer<field_array_t>(field_array, grid->nv, cs, "sha1");

  if(nproc() > 1) {
    const unsigned int csels = cs.length*nproc();
    unsigned char * sums(NULL);

    if(rank() == 0) {
      sums = new unsigned char[csels];
    } // if

    // gather sums from all ranks
    mp_gather_uc(cs.value, sums, cs.length);

    if(rank() == 0) {
      checkSumBuffer<unsigned char>(sums, csels, cs, "sha1");
      delete[] sums;
    } // if
  } // if
} // vpic_simulation::output_checksum_fields

void vpic_simulation::output_checksum_fields() {
  CheckSum cs;
  checkSumBuffer<field_array_t>(field_array, grid->nv, cs, "sha1");

  if(nproc() > 1) {
    const unsigned int csels = cs.length*nproc();
    unsigned char * sums(NULL);

    if( rank() == 0) {
      sums = new unsigned char[csels];
    } // if

    // gather sums from all ranks
    mp_gather_uc(cs.value, sums, cs.length);

    if( rank() == 0) {
      checkSumBuffer<unsigned char>(sums, csels, cs, "sha1");
      MESSAGE(("FIELDS SHA1CHECKSUM: %s", cs.strvalue));
      delete[] sums;
    } // if
  }
  else {
    MESSAGE(("FIELDS SHA1CHECKSUM: %s", cs.strvalue));
  } // if
} // vpic_simulation::output_checksum_fields

void vpic_simulation::checksum_species(const char * species, CheckSum & cs) {
  species_t * sp = find_species_name(species, species_list);
  if(sp == NULL) {
    ERROR(("Invalid species name \"%s\".", species));
  } // if
  
  checkSumBuffer<particle_t>(sp->p, sp->np, cs, "sha1");

  if(nproc() > 1) {
    const unsigned int csels = cs.length*nproc();
    unsigned char * sums(NULL);

    if(rank() == 0) {
      sums = new unsigned char[csels];
    } // if

    // gather sums from all ranks
    mp_gather_uc(cs.value, sums, cs.length);

    if(rank() == 0) {
      checkSumBuffer<unsigned char>(sums, csels, cs, "sha1");
      MESSAGE(("SPECIES \"%s\" SHA1CHECKSUM: %s", species, cs.strvalue));
      delete[] sums;
    } // if
  } // if
} // vpic_simulation::checksum_species

void vpic_simulation::output_checksum_species(const char * species) {
  species_t * sp = find_species_name(species, species_list);
  if(sp == NULL) {
    ERROR(("Invalid species name \"%s\".", species));
  } // if
  
  CheckSum cs;
  checkSumBuffer<particle_t>(sp->p, sp->np, cs, "sha1");

  if(nproc() > 1) {
    const unsigned int csels = cs.length*nproc();
    unsigned char * sums(NULL);

    if( rank() == 0) {
      sums = new unsigned char[csels];
    } // if

    // gather sums from all ranks
    mp_gather_uc(cs.value, sums, cs.length);

    if( rank() == 0) {
      checkSumBuffer<unsigned char>(sums, csels, cs, "sha1");
      MESSAGE(("SPECIES \"%s\" SHA1CHECKSUM: %s", species, cs.strvalue));
      delete[] sums;
    } // if

    // gather sums from all ranks
    mp_gather_uc(cs.value, sums, cs.length);

    if( rank() == 0) {
      checkSumBuffer<unsigned char>(sums, csels, cs, "sha1");
      MESSAGE(("SPECIES \"%s\" SHA1CHECKSUM: %s", species, cs.strvalue));
      delete[] sums;
    } // if
  } else {
    MESSAGE(("SPECIES \"%s\" SHA1CHECKSUM: %s", species, cs.strvalue));
  } // if
} // vpic_simulation::output_checksum_species

#endif // ENABLE_OPENSSL
