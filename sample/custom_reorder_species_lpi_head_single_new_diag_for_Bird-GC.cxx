//========================================================================
//
// LPI N-D SRS and SBS 
//
// modified 05/04/05 by Brian Albright
// modified 03/05/06 by Lin Yin
// modified 02/18/07 add laser chirp
// modified 4/6/18 add summed poynting and tallied k.e. pinhole diagnostics
// includes tally maxwellian boundary condition, which tallies energies of outgoing particles
//
// includes new diag. pushed from secure May 2018
//
// Single speckle deck for Ari, Feb. 2019
// Poynting diag. turned on
// FFT diag. turned on
// velocity diag. turned on
// no particle collisions 
// no time-ave dumps
//========================================================================

// ASCII Logging of IO writes to disk.  Since wall clock elapsed time is written,
// and these require an MPI_Allreduce, don't include this macro inside code
// which only executes on a single processor, else deadlock will ensue. 
// Turn off logging by setting "#if 0"

#if 0
#  define DIAG_LOG(MSG)                                          \
   {                                                             \
     FILE *fptmp=NULL;                                           \
     char fnametmp[BUFLEN];                                      \
     sprintf( fnametmp, "log/diag_log.%i", mp_rank(grid->mp) );  \
     if ( !(fptmp=fopen(fnametmp, "a")) ) ERROR(("Cannot open file %s", fnametmp));        \
     fprintf( fptmp, "At time %e (time step %i): %s\n", mp_elapsed(grid->mp), step, MSG ); \
     fclose( fptmp );                                            \
   }
#else
#  define DIAG_LOG(MSG) 
#endif

// Flag to put barriers in the begin_diagnostics{} stub in order to help
// debug I/0
#define DEBUG_SYNCHRONIZE_IO 0

// Employ turnstiles to partially serialize the high-volume file writes. 
// In this case, the restart dumps.  Set NUM_TURNSTILES to be the desired
// number of simultaneous writes. 
#define NUM_TURNSTILES 256 


// Needed for implementing 2D MPI domain decomposition

# define RANK_TO_INDEX(rank,ix,iy,iz) BEGIN_PRIMITIVE {                   \
    int _ix, _iy, _iz;                                                    \
    _ix  = (rank);                        /* ix = ix+gpx*( iy+gpy*iz ) */ \
    _iy  = _ix/int(global->topology_x);   /* iy = iy+gpy*iz */            \
    _ix -= _iy*int(global->topology_x);   /* ix = ix */                   \
    _iz  = _iy/int(global->topology_y);   /* iz = iz */                   \
    _iy -= _iz*int(global->topology_y);   /* iy = iy */                   \
    (ix) = _ix;                                                           \
    (iy) = _iy;                                                           \
    (iz) = _iz;                                                           \
  } END_PRIMITIVE 

// General purpose memory allocation macro 

#define ALLOCATE(A,LEN,TYPE)                                             \
  if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) ) ERROR(("Cannot allocate.")); 

//---------------------------------------------------------------------------
// User defined boundary handler. 
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
//
// Begin histogram user boundary handler with maxwellian reflux 
//
//---------------------------------------------------------------------------
//
// Notes:  1) Silently assumes maximum species 
//         2) MAKE SURE THAT WE SIZE THE BOUNDARY DATA IN VPIC SO THAT IT IS 
//            LARGE ENOUGH TO HOLD THE HISTOGRAMS!  (Runtime check in deck is
//            recommended). 
//         3) Upon initialization, user is responsible for setting: 
//              - kemax for each species 
//              - initted=0 for handler
//              - desired write_interval
//              - use_pinhole flag (whether to use pinhole)
//              - pinhole angles in yx and zx (if needed)
//         4) No need for user to set dke values; they will be done automatically
//         5) Algorithm assumes that c=1 in the simulation units.  If not, then 
//            we probably have to fix the calculation of gamma (see FIXME). 
//         6) Boundary data, like all custom boundary handlers, is saved upon 
//            restart.

// Total number of species to tally
// NSPEC now done automatically
// #define NSPEC 2

// Number of desired bins in the histograms
#define NBINS 300

// Macro to index the ke histogram
#define KEINDEX(ISPEC,IBIN) ( (ISPEC)*NBINS + (IBIN) )



// Try to copy/paste boundary handler from maxwellian

// Refluxing boundary condition on particles.  Calculate normalized
// momenta from the various perp and para temperatures.  Then, sample
// Maxwellian specra from Maxwellian distributsions that have the
// given normalized momenta.
//
// The perpendicular spectra are sampled from a bi-Maxwellian.  The
// parallel spectra are sampled from vpara = sqrt(2) vth sqrt(-log h)
// where h is uniformly distributed over range (0,1).  Strictly
// speaking, this method only works for non-relativistic distributions
// of edge particles.
//
// This routine requires that in the input deck one has defined arrays
// of id[], ut_para[], ut_perp[] for each particle species.  The
// routine creates a particle injector from the particle data passed
// to it.  Particle injection from custom boundary handlers is
// processed in boundary_p() in boundary_p.cxx after particles are
// exchanged between processors across domain boundaries.
//
// Note that the way maxwellian_reflux_t is defined we have a hard
// maximum of only 32 species handled by the boundary condition.  This
// is not dynamically sized at runtime since it makes it harder to get
// the restart working right if we do.  If more are needed, one needs
// to adjust the size of the maxwellian_reflux_t data space by
// changing the array sizes in boundary_handler.h
//
// Procedure to generate new particle displacements:
//
// 1) From face information, particle species type, and parameters for
//   species defined in the boundary handler data, obtain new
//   Maxwellian- refluxed particle momenta.
//
// 2) From the old particle displacements (stored in the mover data),
//   compute the new displacements according to:
//
// dx_new = dx_old * (ux_new/ux_old) * sqrt((1+|u_old|**2)/(1+|u_new|**2))
//
// Written by:  Brian J. Albright, X-1, LANL   April, 2005
// Revamped by KJB, May 2008, Sep 2009
// Modified for new head vers. VPIC + add capability for Lin's CBET problem Mar 2018


#define BUFLEN (256)

#define IN_boundary
#include "boundary/boundary_private.h"

/* Private interface ********************************************************/


typedef struct maxwellian_reflux_tally {
  species_t * sp_list;
  rng_t     * rng;

  float     * ut_para;  // NSPEC 
  float     * ut_perp;  // NSPEC
  float     * kemax;    // NSPEC
  float     * dke;      // NSPEC
  double    * ke;       // NSPEC*NBINS - double precision to avoid roundoff
  char      * fbase;    // BUFLEN
  char      * fname;    // BUFLEN

  // -----------------------------
  char      * fbase_ke; // BUFLEN
  char      * fname_ke; // BUFLEN
  char      * fbase_ninj; // BUFLEN
  char      * fname_ninj; // BUFLEN
  double    * ke_net;   // NSPEC - double precision to avoid roundoff
  double    * ke_lost;  // NSPEC - double precision to avoid roundoff
  double    * ke_inj;   // NSPEC - double precision to avoid roundoff
  double    * ninj;     // NSPEC - double precision since I don't know if IDL can 
                        //         handle uint64_t; IEEE standard says mantissa 
                        //         of double can store integers up to 4.5e16 
  // -----------------------------

  int       write_interval; 
  int       next_write; 
  int       use_pinhole; 
  float     pinhole_angle_yx_min; 
  float     pinhole_angle_yx_max; 
  float     pinhole_angle_zx_min; 
  float     pinhole_angle_zx_max; 

} maxwellian_reflux_tally_t;

#ifndef M_SQRT2
#define M_SQRT2 (1.4142135623730950488)
#endif

/* FIXME: DON'T IGNORE MAX_PI */
int
interact_maxwellian_reflux_tally( maxwellian_reflux_tally_t * RESTRICT mr,
                                  species_t                 * RESTRICT sp,
                                  particle_t                * RESTRICT p,
                                  particle_mover_t          * RESTRICT pm,
                                  particle_injector_t       * RESTRICT pi,
                                  int                                  max_pi,
                                  int                                  face ) {
  const grid_t * RESTRICT g   = sp->g;
  /**/  rng_t  * RESTRICT rng = mr->rng;

  const int32_t sp_id   = sp->id;
  const float   ut_para = mr->ut_para[sp_id];
  const float   ut_perp = mr->ut_perp[sp_id];

  float u[3];                // u0 = para, u1 & u2 = perp
  float ux, uy, uz;          // x, y, z normalized momenta
  float dispx, dispy, dispz; // Particle displacement
  float ratio;

  /**/                      // axis x  y  z 
  static const int perm[6][3] = { { 0, 1, 2 },   // -x face
                                  { 2, 0, 1 },   // -y face
                                  { 1, 2, 0 },   // -z face 
                                  { 0, 1, 2 },   // +x face
                                  { 2, 0, 1 },   // +y face
                                  { 1, 2, 0 } }; // +z face
  static const float scale[6] = {  M_SQRT2,  M_SQRT2,  M_SQRT2,
                                  -M_SQRT2, -M_SQRT2, -M_SQRT2 };

  //----------------------------------------------------------------------
  // Write histogram if appropriate
  // 
  while ( g->step > mr->next_write ) {  // Write histogram data

    //MESSAGE(( "Starting to write histogram for custom boundary handler at"
    //          " step %d", mr->next_write ));

    FileIO       fileIO;
    FileIOStatus status;

    // MESSAGE(( "Writing pinhole diagnostic output: %s", mr->fbase ));
    status=fileIO.open( mr->fname,
                        (mr->next_write==0 ? io_write : io_read_write) );
    if ( status==fail ) ERROR(("Could not open file."));
    fileIO.seek( uint64_t(   num_species( mr->sp_list )*NBINS*sizeof(double)
                           * (mr->next_write/mr->write_interval) ),
                 SEEK_SET );
    fileIO.write( mr->ke, num_species( mr->sp_list )*NBINS );
    fileIO.close();

    //MESSAGE(( "Finished writing histogram data." ));


    //-------------------------
    status=fileIO.open( mr->fname_ke,
                        (mr->next_write==0 ? io_write : io_read_write) );
    if ( status==fail ) ERROR(("Could not open file."));
    fileIO.seek( uint64_t(   num_species( mr->sp_list )*(3*sizeof(double))
                           * (mr->next_write/mr->write_interval) ),
                 SEEK_SET );
    fileIO.write( mr->ke_net,  num_species( mr->sp_list ) );
    fileIO.write( mr->ke_lost, num_species( mr->sp_list ) );
    fileIO.write( mr->ke_inj,  num_species( mr->sp_list ) );
    fileIO.close();

    status=fileIO.open( mr->fname_ninj,
                        (mr->next_write==0 ? io_write : io_read_write) );
    if ( status==fail ) ERROR(("Could not open file."));
    fileIO.seek( uint64_t(   num_species( mr->sp_list )*sizeof(double)
                           * (mr->next_write/mr->write_interval) ),
                 SEEK_SET );
    fileIO.write( mr->ninj, num_species( mr->sp_list ) );
    fileIO.close();
    //-------------------------

    mr->next_write += mr->write_interval;

  } // while

  // FIXME:  need to modify this for simulation variables where g->cvac != 1
  ux = p->ux;
  uy = p->uy;
  uz = p->uz;

  // n.b.:   ke  = mc^2 * ( sqrt(1+u*u)-1 ) =  mc^2 *u*u/(sqrt(1+u*u)+1) 
  double u2  = ux*ux + uy*uy + uz*uz; 
  double ke  = u2 / ( sqrt(1.0 + u2) + 1.0 ); // gamma - 1
  ke        *= sp->m * g->cvac * g->cvac; 

  //------------------------
  double ke_old = ke;                  // for energy tally diagnostic 
  //------------------------

  if ( ke > mr->kemax[sp_id] ) {
    ke = mr->kemax[sp_id];
  } // if
  int ibin  = (int)(ke/mr->dke[sp_id]);

  //----------------------------------------------------------------------
  // Tally particle 
  //
  if ( mr->use_pinhole ) {

    // Simulate pinhole diagnostics.  
    // simulation is assumed to be in x-z plane with x the laser direction
    // 
    // Trident diagnostics are set up at small angle w.r.t. target normal; for speed,
    // use small angle approximation to arctangent (atan()). 
    // for |x| <1, atan(x) = x-x^3/3 + ...
    //  float theta_yx, theta_zx;
    //  theta_yx = (uy/ux) * (180./M_PI);  // Degrees
    //  theta_zx = (uz/ux) * (180./M_PI);  // Degrees 

    float theta_yx = atan( uy/ux );
    float theta_zx = atan( uz/ux );

    if (    theta_yx > mr->pinhole_angle_yx_min * M_PI / 180.
         && theta_yx < mr->pinhole_angle_yx_max * M_PI / 180.
         && theta_zx > mr->pinhole_angle_zx_min * M_PI / 180.
         && theta_zx < mr->pinhole_angle_zx_max * M_PI / 180. ) {

      (mr->ke)[KEINDEX(sp_id,ibin)] += p->w;

    } // if

  } else {  // no pinhole - tally all particles' energies

    (mr->ke)[KEINDEX(sp_id,ibin)] += p->w;

  } // if

  //----------------------------------------------------------------------
  // Now perform the Maxwellian refluxing. 
  // 
  // compute velocity of injected particle
  //
  // Suppose you have a Maxwellian at a boundary: p(u) ~ exp(-u^2/(2
  // ub^2)) where u is the || speed and ub is the thermal speed.  In a
  // time delta t, if the boundary has surface area delta A, there
  // will be
  //   
  //   p_inj(u) du ~ u exp(-u^2/(2 ub^2)) (delta t)(delta A) du
  //   
  // particles injected from the boundary between speeds u and
  // u+du. p_inj(u) is the distribution function we wish to sample.
  // It has a cumulative i distribution function
  //   
  //   cdf(u) = \int_0^u du p_inj(u) = 1 - exp(-u^2/(2 ub^2))
  //   
  // (I've adjusted the constants out front to give a proper cdf
  // ranging from 0 to 1, the range of h).
  //   
  // Let mu be a uniformly distributed random number from 0 to 1.
  // Setting cdf(u)=mu and solving for u gives the means for sampling
  // u:
  //   
  //   exp(-u^2/(2 ub^2)) = mu - 1 = mu
  //
  // (Note that 1-mu has same dist as mu.)  This implies that
  //
  //   u = sqrt(2) ub sqrt( -log(mu) ).
  //
  // Note that -log(mu) is an _exponentially_ distributed random
  // number.

  // Note: This assumes ut_para > 0

  u[0] = ut_para*scale[face]*sqrtf(frande(rng));
  u[1] = ut_perp*frandn(rng);
  u[2] = ut_perp*frandn(rng);
  ux   = u[perm[face][0]];
  uy   = u[perm[face][1]];
  uz   = u[perm[face][2]];

  // Compute the amount of aging to due of the refluxed particle.
  //
  // The displacement of the refluxed particle should be:
  //
  //   dr' = c dt u' (1-a) / gamma'
  //
  // where u' and gamma' refer to the refluxed 4-momentum and
  // a is when the particle's time step "age" when it hit the
  // boundary.
  //
  //   1-a = |remaining_dr| / ( c dt |u| / gamma )
  //
  // Thus, we have:
  //
  //   dr' = u' gamma |remaining_dr| / ( gamma' |u| )
  //
  // or:
  //
  //   dr' = u' sqrt(( (1+|u|^2) |remaining_dr|^2 ) / ( (1+|u'|^2) |u|^2 ))

  dispx = g->dx * pm->dispx;
  dispy = g->dy * pm->dispy;
  dispz = g->dz * pm->dispz;
  ratio = p->ux*p->ux + p->uy*p->uy + p->uz*p->uz;
  ratio = sqrtf( ( ( 1+ratio )*( dispx*dispx + dispy*dispy + dispz*dispz ) ) /
                 ( ( 1+(ux*ux+uy*uy+uz*uz) )*( FLT_MIN+ratio ) ) );
  dispx = ux * ratio * g->rdx;
  dispy = uy * ratio * g->rdy;
  dispz = uz * ratio * g->rdz;

  // If disp and u passed to this are consistent, ratio is sane in and
  // the displacment is non-zero in exact arithmetic.  However,
  // paranoid checking like the below can be done here if desired.
  //
  // if( ratio<=0 || ratio>=g->dt*g->cvac )
  //   WARNING(( "Bizarre behavior detected in maxwellian_reflux" ));

  pi->dx    = p->dx;
  pi->dy    = p->dy;
  pi->dz    = p->dz;
  pi->i     = p->i;
  pi->ux    = ux;
  pi->uy    = uy;
  pi->uz    = uz;
  pi->w     = p->w;
  pi->dispx = dispx;
  pi->dispy = dispy;
  pi->dispz = dispz;
  pi->sp_id = sp_id;

  //-----------------------------
  // Update ke_net to reflect difference in kinetic energy
  
  // n.b.:   ke/(mc^2) = sqrt(1+u*u)-1  =  u*u/(sqrt(1+u*u)+1) 
  double u2_new       = ux*ux + uy*uy + uz*uz;
  double ke_new       = sp->m * g->cvac * g->cvac * u2_new / ( 1.0 + sqrt(1.0 + u2_new) );
  mr->ke_net[sp_id]  += p->w * ( ke_new - ke_old ); 
  mr->ke_lost[sp_id] += p->w * ke_old;
  mr->ke_inj[sp_id]  += p->w * ke_new;
  mr->ninj[sp_id]    += 1.0;  
  //-----------------------------

  return 1;
}

void
checkpt_maxwellian_reflux_tally( const particle_bc_t * RESTRICT pbc ) {
  const maxwellian_reflux_tally_t * RESTRICT mr =
    (const maxwellian_reflux_tally_t *)pbc->params;
  CHECKPT( mr, 1 );
  CHECKPT_PTR( mr->sp_list );
  CHECKPT_PTR( mr->rng     );
  CHECKPT( mr->ut_para, num_species( mr->sp_list ) );
  CHECKPT( mr->ut_perp, num_species( mr->sp_list ) );
  CHECKPT( mr->kemax, num_species( mr->sp_list ) );
  CHECKPT( mr->dke,   num_species( mr->sp_list ) );
  CHECKPT( mr->ke,    num_species( mr->sp_list ) * (NBINS) );

  //-----------------------------------
  CHECKPT( mr->fbase_ke, BUFLEN ); 
  CHECKPT( mr->fname_ke, BUFLEN ); 
  CHECKPT( mr->ke_net, num_species( mr->sp_list ) ); 
  CHECKPT( mr->ke_lost, num_species( mr->sp_list ) ); 
  CHECKPT( mr->ke_inj, num_species( mr->sp_list ) ); 
  CHECKPT( mr->fbase_ninj, BUFLEN ); 
  CHECKPT( mr->fname_ninj, BUFLEN ); 
  CHECKPT( mr->ninj, num_species( mr->sp_list ) ); 
  //-----------------------------------

  CHECKPT( mr->fbase, BUFLEN ); 
  CHECKPT( mr->fname, BUFLEN ); 
  CHECKPT_VAL( int,   mr->write_interval       ); 
  CHECKPT_VAL( int,   mr->next_write           ); 
  CHECKPT_VAL( int,   mr->use_pinhole          ); 
  CHECKPT_VAL( float, mr->pinhole_angle_yx_min ); 
  CHECKPT_VAL( float, mr->pinhole_angle_yx_max ); 
  CHECKPT_VAL( float, mr->pinhole_angle_zx_min ); 
  CHECKPT_VAL( float, mr->pinhole_angle_zx_max ); 

  checkpt_particle_bc_internal( pbc );
}

particle_bc_t *
restore_maxwellian_reflux_tally( void ) {
  maxwellian_reflux_tally_t * mr;
  RESTORE( mr );
  RESTORE_PTR( mr->sp_list );
  RESTORE_PTR( mr->rng     );
  RESTORE( mr->ut_para );
  RESTORE( mr->ut_perp );
  RESTORE( mr->kemax );
  RESTORE( mr->dke );
  RESTORE( mr->ke );

  //-----------------------------------
  RESTORE( mr->fbase_ke ); 
  RESTORE( mr->fname_ke ); 
  RESTORE( mr->ke_net ); 
  RESTORE( mr->ke_lost ); 
  RESTORE( mr->ke_inj ); 
  RESTORE( mr->fbase_ninj ); 
  RESTORE( mr->fname_ninj ); 
  RESTORE( mr->ninj ); 
  //-----------------------------------

  RESTORE( mr->fbase ); 
  RESTORE( mr->fname ); 
  RESTORE_VAL( int, mr->write_interval       ); 
  RESTORE_VAL( int, mr->next_write           ); 
  RESTORE_VAL( int, mr->use_pinhole          ); 
  RESTORE_VAL( float, mr->pinhole_angle_yx_min ); 
  RESTORE_VAL( float, mr->pinhole_angle_yx_max ); 
  RESTORE_VAL( float, mr->pinhole_angle_zx_min ); 
  RESTORE_VAL( float, mr->pinhole_angle_zx_max ); 

  return restore_particle_bc_internal( mr );
}

void
delete_maxwellian_reflux_tally( particle_bc_t * RESTRICT pbc ) {
  FREE( pbc->params );
  delete_particle_bc_internal( pbc );
}


/* Public interface *********************************************************/
// FIXME: get rid of NBINS macro and pass NBINS as an argument

particle_bc_t *
maxwellian_reflux_tally( species_t  * RESTRICT sp_list,
                         rng_pool_t * RESTRICT rp ) {
  if( !sp_list || !rp ) ERROR(( "Bad args" ));
  maxwellian_reflux_tally_t * mr;
  MALLOC( mr, 1 );
  mr->sp_list = sp_list;
  mr->rng     = rp->rng[0];

  // DEBUG 
  if ( num_species( mr->sp_list ) <= 0 ) ERROR(("Bad num species.")); 

  MALLOC( mr->ut_para, num_species( mr->sp_list ) );
  CLEAR( mr->ut_para, num_species( mr->sp_list ) );

  MALLOC( mr->ut_perp, num_species( mr->sp_list ) );
  CLEAR( mr->ut_perp, num_species( mr->sp_list ) );

  MALLOC( mr->kemax, num_species( mr->sp_list ) );
  CLEAR( mr->kemax,  num_species( mr->sp_list ) );

  MALLOC( mr->dke,   num_species( mr->sp_list ) );
  CLEAR( mr->dke,    num_species( mr->sp_list ) );

  MALLOC( mr->ke,    num_species( mr->sp_list ) * (NBINS) );
  CLEAR( mr->ke,     num_species( mr->sp_list ) * (NBINS) );

  //---------------------------------------
  MALLOC( mr->fbase_ke, BUFLEN ); 
  CLEAR( mr->fbase_ke,  BUFLEN ); 

  MALLOC( mr->fname_ke, BUFLEN ); 
  CLEAR( mr->fname_ke,  BUFLEN ); 

  MALLOC( mr->ke_net, num_species( mr->sp_list ) ); 
  CLEAR( mr->ke_net, num_species( mr->sp_list ) ); 

  MALLOC( mr->ke_lost, num_species( mr->sp_list ) ); 
  CLEAR( mr->ke_lost, num_species( mr->sp_list ) ); 

  MALLOC( mr->ke_inj, num_species( mr->sp_list ) ); 
  CLEAR( mr->ke_inj, num_species( mr->sp_list ) ); 

  MALLOC( mr->fbase_ninj, BUFLEN ); 
  CLEAR( mr->fbase_ninj,  BUFLEN ); 

  MALLOC( mr->fname_ninj, BUFLEN ); 
  CLEAR( mr->fname_ninj,  BUFLEN ); 

  MALLOC( mr->ninj, num_species( mr->sp_list ) ); 
  CLEAR( mr->ninj, num_species( mr->sp_list ) ); 
  //---------------------------------------

  MALLOC( mr->fbase, BUFLEN ); 
  CLEAR( mr->fbase,  BUFLEN ); 

  MALLOC( mr->fname, BUFLEN ); 
  CLEAR( mr->fname,  BUFLEN ); 

  mr->next_write  = 0; 
  mr->use_pinhole = 0; // Default - FIXME: write method to override and set pinhole vals

  sprintf( mr->fbase, "boundary" );
  sprintf( mr->fname, "pinhole/%s.%i", mr->fbase, world_rank );

  //------------------------------
  sprintf( mr->fbase_ke, "ke" );
  sprintf( mr->fname_ke, "energy_flux/%s.%i", mr->fbase_ke, world_rank );
  sprintf( mr->fbase_ninj, "ninj" );
  sprintf( mr->fname_ninj, "energy_flux/%s.%i", mr->fbase_ninj, world_rank );
  //------------------------------

  return new_particle_bc_internal( mr,
                                   (particle_bc_func_t)interact_maxwellian_reflux_tally,
                                   delete_maxwellian_reflux_tally,
                                   (checkpt_func_t)checkpt_maxwellian_reflux_tally,
                                   (restore_func_t)restore_maxwellian_reflux_tally,
                                   NULL );
}

/* FIXME: NOMINALLY, THIS INTERFACE SHOULD TAKE kT */
void
set_reflux_tally_temp( /**/  particle_bc_t * RESTRICT pbc,
                       const species_t     * RESTRICT sp,
                       float ut_para,
                       float ut_perp ) {
  if( !pbc || !sp || ut_para<0 || ut_perp<0 ) ERROR(( "Bad args" ));
  maxwellian_reflux_tally_t * RESTRICT mr = (maxwellian_reflux_tally_t *)pbc->params;
  mr->ut_para[sp->id] = ut_para;
  mr->ut_perp[sp->id] = ut_perp;
}

// Set kemax value for species sp; then set dke for the species 
void 
set_reflux_kemax( /**/  particle_bc_t * RESTRICT pbc,
                  const species_t     * RESTRICT sp,
                  double                         kemax ) { 
  if( !pbc || !sp || kemax<0 ) ERROR(( "Bad args" ));
  maxwellian_reflux_tally_t * RESTRICT mr = (maxwellian_reflux_tally_t *)pbc->params;

  mr->kemax[sp->id] = kemax * 0.999999;    // Max ke value allowed; ensure no roundoff
                                           // can make value outside histogram (particle 
                                           // k.e. > kemax will fall into the last bin). 
  mr->dke[sp->id]   = kemax / (float)NBINS;
}

void 
set_reflux_write_interval( /**/  particle_bc_t * RESTRICT pbc,
                           int                            write_interval ) { 
  if( !pbc || write_interval<0 ) ERROR(( "Bad args" ));
  maxwellian_reflux_tally_t * RESTRICT mr = (maxwellian_reflux_tally_t *)pbc->params;

  mr->write_interval = write_interval;
}


// FIXME: finish this helper method
// FIXME: add ability to set pinhole flag in data
#if 0
// -----------------------------------------------------------------------------
// Set min, max yx, zx angles 
void
set_reflux_tally_angle( /**/  particle_bc_t * RESTRICT pbc,
                        float ut_para,
                        float ut_perp ) {
  if( !pbc || ut_para<0 || ut_perp<0 ) ERROR(( "Bad args" ));
  maxwellian_reflux_tally_t * RESTRICT mr = (maxwellian_reflux_tally_t *)pbc->params;
  mr->ut_para[sp->id] = ut_para;
  mr->ut_perp[sp->id] = ut_perp;
}
#endif 

// Undefine to avoid messing up namespace. 

#undef NSPEC
#undef NBINS
#undef KEINDEX

//---------------------------------------------------------------------------
//
// End histogram user boundary handler
//
//---------------------------------------------------------------------------


begin_globals {
  float emax;                   // E0 of the laser pump
  float omega_0;                // w0/wpe - frequency of pump
  float vthe; 			// vthe/c   <- these are needed to make movie files
  float vthi_He;                // vthi_He/c
  float vthi_H;                 // vthi_H/c
  int field_interval;           // how frequently to dump field built-in diagnostic
  int energies_interval;
  int restart_interval; 	// how frequently to write restart file. 
  int quota_check_interval;     // how often to check if runtime quota exceeded
  int poynting_interval;        // how frequently to dump poynting flux at boundaries. 
  int velocity_interval;        // how frequently to dump velocity space
  int fft_ex_interval;          // how frequently to save ex fft data
  int fft_ez_interval;          // how frequently to save ez fft data
  int fft_ey_interval;          // how frequently to save ey fft data
  int eparticle_interval;       // how frequently to dump particle data
  int Hparticle_interval;       //  
  int Heparticle_interval;      //
//  int ehydro_interval;
//  int Hhydro_interval;
//  int Hehydro_interval;
  int mobile_ions;		// flag: 0 if ions are not to be pushed
  int H_present;                // flag nonzero when H ions are present. 
  int He_present;               // flag nonzero when He ions are present.  
  int rtoggle;                  // Enables save of last two restart dumps for safety
  double quota_sec;             // Run quota in seconds

  int load_particles;           // Whether to load particles 
  double mime_H;                // proton to electron mass ratio
  double mime_He;               // alpha to electron mass ratio

  // Parameters for 2d and 3d Gaussian wave launch
  float waist;			// how wide the focused beam is
  float width;			
  float zcenter;		// center of beam at boundary in z
  float ycenter;		// center of beam at boundary in y
  float xfocus;			// how far from boundary to focus
  float mask;			// # gaussian widths from beam center where I nonzero

  double topology_x;            // domain topology needed to normalize Poynting diagnostic 
  double topology_y;
  double topology_z;

  double Lz;                    // Size of box in z

  // Used for fft diagnostic
  double Lx;                    // Size of box in x
  double xmin_domain;           // Location in x of left boundary 
  double ey_xloc;               // x location of ey(z) taken for FFT diagnostic 

  double lambda;                // Wavelength in terms of skin depths
  double wpe1ps;

  // Ponyting diagnostic flags - which output to turn on

  // write_backscatter_only flag:  when this flag is nonzero, it means to only compute 
  // poynting data for the lower-x surface.  This flag affects both the summed poynting 
  // data as well as the surface data. 

  int write_backscatter_only;  // nonzero means we only write backscatter diagnostic for fields

  // write_poynting_sum is nonzero if you wish to write a file containing the integrated
  // poynting data on one or more surfaces.  If write_backscatter_only is nonzero, then 
  // the output file will be a time series of double precision numbers containing the 
  // integrated poynting flux at that point in time.  If write_backscatter_only is zero,
  // then for each time step, six double precision numbers are written, containing the 
  // poynting flux through the lower x, upper x, lower y, upper y, lower z, and upper z
  // surfaces. 

  int write_poynting_sum;      // nonzero if we wish to write integrated Poynting data

  // write_poynting_faces is probably useless, but I put it here anyway in case it's not. 
  // When this flag is nonzero, it will print out the poynting flux at each of a 2D array 
  // of points on the surface.  When this flag is turned on and write_backscatter_only is 
  // nonzero, then only the 2D array of points on the lower-x boundary surface are written
  // for each time step.  When this flag is turned on and write_bacscatter_only is 
  // zero, then it will write 2D array data for the lower x, upper x, lower y, upper y, 
  // lower z, upper z surfaces for each time step. 

  int write_poynting_faces;    // nonzero if we wish to write Poynting data on sim boundaries 

  // write_eb_faces is nonzero when we wish to get the raw e and b data at the boundary
  // (e.g., to be used with a filter to distinguish between SRS and SBS backscatter).  
  // When this flag is on and write_backscatter_only is nonzero, then only the 2D array
  // of points on the lower-x boundary surface are written for each time step.  When 
  // this flag is turned on and write_backscatteR_only is zero, then it will write 2D
  // array data for the lower x, upper x, lower y, upper y, lower z, upper z surfaces for
  // each time step.  When turned on, four files are produced: e1, e2, cb1, cb2.  The 
  // values of the quantities printed depend on the face one is considering:  for the 
  // x faces, e1 = ey, e2 = ez, cb1 = cby, cb2 = cbz.  Similarly for y and z surfaces, 
  // but where 1 and 2 stand for (z,x) and (x,y) coordinates, respectively.  

  int write_eb_faces;          // nonzero if we wish to write E and B data on sim boundaries

  // write_side_scatter is nonzero when we want to write side scatter data.  For now
  // it doesn't turn off side scatter writes of poynting and field surface data, just 
  // the poynting sum data.  This flag can be zero while write_backscatter_only and 
  // write_poynting_sum flags are nonzero and we will write poynting sum data for left
  // and right most processors (two sum data files).  If this flag is on while the 
  // other two  are on, then we only write a single sum data and we use mpi collectives
  // to aggregate the data. 
  
  int write_side_scatter;      // nonzero if we wish to write side scatter sum data

  int launch_laser;            // whether to launch pump laser

  // Dump parameters for standard VPIC output formats
  DumpParameters fdParams;
  DumpParameters hedParams;
  DumpParameters hHdParams;
  DumpParameters hHedParams;
  std::vector<DumpParameters *> outputParams;

  // -------------------------------------------------------------------------------------------------
  // For the time-averaging field diagnostic:

  // The adjustable parameters of these, dis_interval and dis_nav, are now set by the variables
  // AVG_SPACING and AVG_TOTAL_STEPS in the beginning of the include file so they can be more
  // easily found and edited
  //
  // The data are saved in the interval
  //
  // j*dis_interval <= step < j*dis_interval + dis_nav
  //
  // where j is an integer. the output is assigned time index corresponding to the start of the interval
  //
  // The code is restart-aware. The associated global variables are
  //
  // global->restart_interval is assumed to be defined

  int dis_nav;                             // number of steps to average over
  int dis_interval;                        // number of steps between outputs
  int dis_iter;                            // iteration count. 0 means we are not averaging at the moment
  int dis_begin_int;                       // first time step of the interval

  // -------------------------------------------------------------------------------------------------

  // For integrade poynting flux tally diagnostic 
  double psum_integrated_poynting_flux_tally; 
  int    psum_integration_offset; 

};

begin_initialization {

  sim_log( "*** Begin initialization. ***" ); 
  mp_barrier(); // Barrier to ensure we started okay. 
  sim_log( "*** Begin initialization2. ***" ); 

  float elementary_charge  = 4.8032e-10;       // stat coulomb
  float elementary_charge2 = elementary_charge * elementary_charge; 
  float speed_of_light     = 2.99792458e10;    // cm/sec
  float m_e                = 9.1094e-28;       // g
  float k_boltz            = 1.6022e-12;       // ergs/eV
  float mec2 = m_e*speed_of_light*speed_of_light/k_boltz;
  float mpc2 = mec2*1836.0;
  float eps0 = 1;                       

  double cfl_req   = 0.98;      // How close to Courant should we try to run
  double damp      = 0;         // Level of radiation damping
  double iv_thick  = 2;         // Thickness (in cells) of imperm. vacuum region
//int    psum_integration_offset = int(iv_thick/2); 
  int    psum_integration_offset = 0; 

  double t_e               = 500;         // electron temperature, eV
  double t_i               = 150;         // ion temperature, eV
  double n_e_over_n_crit   = 0.05;         // n_e/n_crit
  double vacuum_wavelength = 527 * 1e-7;   // third micron light (cm)
  float  laser_intensity  = 2.0e15; // Units of W/cm^2

  float box_size_x = 158.0 * 1e-4;   // Microns
  float box_size_z = 14.5 *  1e-4;   // Microns (ignored if 1d or 2d in plane)

  int mobile_ions         = 1;        // Whether or not to push ions 
  int He_present=1, H_present=1;      // Parameters used later. Set to unity to initialize. 
  double f_He             = 0.5;        // Ratio of number density of He to total ion density
  double f_H              = 1-f_He;   // Ratio of number density of H  to total ion density 

  int load_particles = 1;         // Flag to turn off particle load for testing wave launch. 
//  double nppc        = 7168;
  double nppc        = 512;

// Here _He is C+6
  double A_H                = 1;
  double Z_H                = 1;
  double A_He               = 12;
  double Z_He               = 6;
  float mic2_H   = mpc2*A_H;
  float mic2_He  = mpc2*A_He;
  float mime_H   = mic2_H/mec2;
  float mime_He  = mic2_He/mec2;

  double uthe    = sqrt(t_e/mec2);    // vthe/c
  double uthi_H  = sqrt(t_i/mic2_H);  // vthi/c 
  double uthi_He = sqrt(t_i/mic2_He); // vthi/c

  double delta = (vacuum_wavelength/(2.0*M_PI))/sqrt(n_e_over_n_crit);
  double n_e = speed_of_light*speed_of_light*m_e/(4.0*M_PI*elementary_charge2*delta*delta);
  double debye = uthe*delta;
  double wpe1ps=1e-12* speed_of_light/delta;

  double nx                = 1200; //11232;
  double ny                = 1;    // 2D problem in x-z plane
  double nz                = 100; //756;  // was 549;
//  double nx                = 55; //11232;
//  double ny                = 55;    // 2D problem in x-z plane
//  double nz                = 55; //756;  // was 549;

#if 0
  // DEBUG - run on 64 proc. 
  nx = nx/16;
  box_size_x /= 16.;  
#endif

  double hx = box_size_x/(delta*nx);   // in c/wpe
  double hz = box_size_z/(delta*nz);
  double hy = hz;

  double cell_size_x  = delta*hx/debye;         // Cell size in Debye lengths
  double cell_size_z  = delta*hz/debye;         // Cell size in Debye lengths

  double Lx = nx*hx;          // in c/wpe
  double Ly = ny*hy;   
  double Lz = nz*hz;   

  double dt = cfl_req*courant_length(Lx, Ly, Lz, nx, ny, nz); 

  double topology_x = nproc(); //208; // 54 cells per MPI
  double topology_y = 1;
  double topology_z = 1;

// laser focusing parameters
  int  launch_laser = 1;        // Whether to launch pump laser

  double f_number   = 6.9;                      // f number of beam
  double lambda     = vacuum_wavelength/delta;  // Wavelength in terms of skin depths
  double waist      = f_number*lambda;          // in c/wpe, width of beam at focus
  double xfocus     = Lx/2;                     // in c/wpe, the x0 in NPIC
  double ycenter    = 0;                        // spot centered in y on lhs boundary
  double zcenter    = 0;                        // spot centered in z on lhs boundary
  double mask       = 1.9;                      // Set drive I>0 if r>mask*width at boundary.
  double width = waist*sqrt(1+(lambda*xfocus/(M_PI*waist*waist))*(lambda*xfocus/(M_PI*waist*waist)));

  double omega_0            = sqrt(1.0/n_e_over_n_crit); // w0/wpe
  double intensity_cgs = 1e7*laser_intensity;        // [ergs/(s*cm^2)]

  double emax = 
    sqrt(2.0*intensity_cgs/
	 (m_e*speed_of_light*speed_of_light*speed_of_light*n_e)); // at the waist, of NPIC
  emax = emax*sqrt(waist/width); // 2D, at entrance

  double ey_xloc = Lx * 0.5;              // x location for Ey(z) write in fft diagnostic 

// intervals 
  float t_stop = 100*wpe1ps;     // Runtime in 1/omega_pe
  int poynting_interval    = int(M_PI/(dt*(omega_0+1.5)));       // Num. steps between dumping poynting flux to resolve w/wpe=8
  int fft_ex_interval     = poynting_interval ;       // Num steps between writing Ex in fft_slice
  int fft_ey_interval     = poynting_interval ;       // Num steps between writing Ey in fft_slice
  int fft_ez_interval     = poynting_interval ;       // Num steps between writing Ez in fft_slice
//  int field_interval       = int(0.25*wpe1ps/dt);         // Num. steps between saving field, hydro data
  int field_interval       = 1000000;         // Num. steps between saving field, hydro data
//  int ehydro_interval = field_interval;
//  int Hhydro_interval = field_interval;
//  int Hehydro_interval = field_interval;
//  int eparticle_interval = 200000*field_interval;
//  int Hparticle_interval = 200000*field_interval;
//  int Heparticle_interval = 200000*field_interval;
  int energies_interval = field_interval;
// restart_interval has to be multiples of field_interval
  int restart_interval       = 8*field_interval;
//restart_interval     = 0;  //DEBUG      // Num. steps between restart dumps

  int quota_check_interval = 20;
  int velocity_interval   = field_interval;     //  Num steps between writing poynting flux; not used in NIC

  int ele_sort_freq       = 20*2; 
  int ion_sort_freq       = 5*ele_sort_freq; 
//  int ele_sort_freq       = -1; 
//  int ion_sort_freq       = -1; 

  double quota = 11.7;            // Run quota in hours.  
  double quota_sec = quota*3600;  // Run quota in seconds. 

  double Ne    = nppc*nx*ny*nz;             // Number of macro electrons in box
  Ne = trunc_granular(Ne, nproc());         // Make Ne divisible by number of processors       
  double Ni    = Ne;                        // Number of macro ions of each species in box
  double Npe   = Lx*Ly*Lz;                  // Number of physical electrons in box, wpe = 1
  double Npi   = Npe/(Z_H*f_H+Z_He*f_He);   // Number of physical ions in box
  double qe    = -Npe/Ne;                   // Charge per macro electron
  double qi_H  = f_H*Npi/Ni;          // Charge per H macro ion
  //double qi_He = Z_He*f_He*Npi/Ni;          // Charge per He macro ion
  double qi_He = f_He*Npi/Ni;                    // Charge per He macro ion

# if 0
  double Ne    = nppc*nx*ny*nz;             // Number of macro electrons in box
  Ne = trunc_granular(Ne, nproc());         // Make Ne divisible by number of processors       
  double Ni    = Ne;                        // Number of macro ions of each species in box
  double Npe   = Lx*Ly*Lz;                  // Number of physical electrons in box, wpe = 1
  double Npi   = Npe/(Z_H*f_H+Z_He*f_He);   // Number of physical ions in box
  double qe    = -Npe/Ne;                   // Charge per macro electron
  double qi_H  = Z_H *f_H *Npi/Ni;          // Charge per H macro ion
//double qi_He = Z_He*f_He*Npi/Ni;          // Charge per He macro ion
  double qi_He = Npi/Ni;                    // Charge per He macro ion
# endif 

  // Turn on integrated backscatter poynting diagnostic - right now there is a bug in this, so we 
  // only write the integrated backscatter time history on the left face. 
 
  int write_backscatter_only = 0;                 // Nonzero means only write lower x face
  int write_poynting_sum   = 1;                   // Whether to write integrated Poynting data 
  int write_side_scatter   = 1;                   // Turns on side scatter if nonzero
  int write_poynting_faces = 0;                   // Whether to write poynting data on sim boundary faces
  int write_eb_faces       = 0;                   // Whether to write e and b field data on sim boundary faces

  // PRINT SIMULATION PARAMETERS 
  sim_log("***** Simulation parameters *****");
  sim_log("* Processors:                    "<<nproc());
  sim_log("* Time step, max time, nsteps =  "<<dt<<" "<<t_stop<<" "<<int(t_stop/(dt))); 
  sim_log("* wpe1ps =                       "<<wpe1ps); 
  sim_log("* Debye length, delta =          "<<debye<<" "<<delta);
  sim_log("* cell size in x, z =            "<<cell_size_x<<" "<<cell_size_z);
  sim_log("* Lx, Ly, Lz =                   "<<Lx<<" "<<Ly<<" "<<Lz);
  sim_log("* nx, ny, nz =                   "<<nx<<" "<<ny<<" "<<nz);
  sim_log("* Charge/macro electron =        "<<qe);
  sim_log("* Charge/macro He =              "<<qi_He);
  sim_log("* Charge/macro H =               "<<qi_H);
  sim_log("* Average particles/processor:   "<<Ne/nproc());
  sim_log("* Average particles/cell:        "<<nppc);
  sim_log("* Do we have mobile ions?        "<<(mobile_ions ? "Yes" : "No"));
  sim_log("* Is there He present?           "<<(He_present ? "Yes" : "No")); 
  sim_log("* Is there H present ?           "<<(H_present ? "Yes" : "No")); 
  sim_log("* Omega_0:                       "<<omega_0);
  sim_log("* Omega_pe:                      "<<1);
  sim_log("* Plasma density, ne/nc:         "<<n_e<<" "<<n_e_over_n_crit);
  sim_log("* Vac wavelength,I_laser:        "<<vacuum_wavelength<<" "<<laser_intensity);
  sim_log("* T_e, T_i, m_e, m_i_H, m_i_He:  "<<t_e<<" "<<t_i<<" "<<1<<" "<<mime_H<<" "<<mime_He);
  sim_log("* Radiation damping:             "<<damp);
  sim_log("* Fraction of courant limit:     "<<cfl_req);
  sim_log("* vthe/c:                        "<<uthe);
  sim_log("* vthi_H/c, vth_He/c:            "<<uthi_H<<" "<<uthi_He);
  sim_log("* emax:                          "<<emax);
  sim_log("* restart interval:              "<<restart_interval); 
  sim_log("* energies_interval:             "<< energies_interval );
  sim_log("* quota_check_interval:           "<<quota_check_interval);
  sim_log("* velocity interval:             "<<velocity_interval); 
  sim_log("* poynting interval:             "<<poynting_interval); 
  sim_log("* fft ex save interval:          "<<fft_ex_interval); 
  sim_log("* fft ey save interval:          "<<fft_ey_interval); 
  sim_log("* fft ez save interval:          "<<fft_ez_interval); 
  sim_log("* f#, waist:                     "<<f_number<<" "<<waist);
  sim_log("* width, xfocus:                 "<<width<<" "<<xfocus);
  sim_log("* ycenter, zcenter, mask:        "<<ycenter<<" "<<zcenter<<mask);
  sim_log("* quota (hours):                 "<<quota);
  sim_log("* load_particles:                "<<(load_particles ? "Yes" : "No")); 
  sim_log("* mime_H:                        "<<mime_H); 
  sim_log("* mime_He:                       "<<mime_He); 
  sim_log("* ele_sort_freq:                 "<<ele_sort_freq); 
  sim_log("* ion_sort_freq:                 "<<ion_sort_freq); 
  sim_log("* launch_laser:                  "<<launch_laser);  
  sim_log("* ey_xloc (for Ey(z) FFT write): "<<ey_xloc); 
  sim_log("* psum_integration_offset:       "<<psum_integration_offset); 
  sim_log("*********************************");

  // SETUP HIGH-LEVEL SIMULATION PARMETERS
  sim_log("Setting up high-level simulation parameters. "); 
  num_step             = 200; //int(t_stop/(dt)); 
  status_interval      = 2000; 
//  status_interval      = -1; 
//  sync_shared_interval = status_interval/1;
//  clean_div_e_interval = status_interval/1;
//  clean_div_b_interval = status_interval/10;
//  status_interval      = 200; 
  sync_shared_interval = status_interval/1;
  clean_div_e_interval = status_interval/1;
  clean_div_b_interval = status_interval/10;
    kokkos_field_injection = true;
    field_injection_interval = 1;
    particle_injection_interval = -1;
    current_injection_interval = -1;
    field_copy_interval = -1;
    particle_copy_interval = -1;

  // For maxwellian reinjection, we need more than the default number of
  // passes (3) through the boundary handler
  // Note:  We have to adjust sort intervals for maximum performance on Cell.
  num_comm_round = 6;

  global->field_interval           = field_interval; 
  global->restart_interval         = restart_interval;
  global->energies_interval        = energies_interval;
  global->quota_check_interval     = quota_check_interval;
  global->poynting_interval        = poynting_interval; 
  global->velocity_interval        = velocity_interval; 
  global->fft_ex_interval          = fft_ex_interval; 
  global->fft_ey_interval          = fft_ey_interval; 
  global->fft_ez_interval          = fft_ez_interval; 
  global->vthe                     = uthe;     // c=1
  global->vthi_He                  = uthi_He;  // c=1
  global->vthi_H                   = uthi_H;   // c=1
  global->emax                     = emax; 
  global->omega_0                  = omega_0;
  global->mobile_ions              = mobile_ions; 
  global->H_present                = H_present; 
  global->He_present               = He_present; 
  global->lambda                   = lambda; 
  global->wpe1ps                   = wpe1ps; 
  global->waist                    = waist; 
  global->width                    = width; 
  global->xfocus                   = xfocus; 
  global->ycenter                  = ycenter; 
  global->zcenter                  = zcenter; 
  global->mask                     = mask; 
  global->quota_sec                = quota_sec;
  global->rtoggle                  = 0; 
  global->load_particles           = load_particles; 
  global->mime_H                   = mime_H; 
  global->mime_He                  = mime_He; 

  global->topology_x               = topology_x; 
  global->topology_y               = topology_y; 
  global->topology_z               = topology_z; 

  global->Lz                       = Lz;

  global->Lx                       = Lx; 
  global->xmin_domain              = 0;  
  global->ey_xloc                  = ey_xloc; 
  global->write_poynting_sum       = write_poynting_sum;
  global->write_poynting_faces     = write_poynting_faces;
  global->write_eb_faces           = write_eb_faces;
  global->write_backscatter_only   = write_backscatter_only;
  global->write_side_scatter       = write_side_scatter;    

  global->launch_laser             = launch_laser; 

  global->psum_integrated_poynting_flux_tally = 0; // initialization 
  global->psum_integration_offset             = psum_integration_offset; 

//  global->ehydro_interval = ehydro_interval;
//  global->Hhydro_interval = Hhydro_interval;
//  global->Hehydro_interval = Hehydro_interval;

  // SETUP THE GRID ===============================================================
  sim_log("Setting up computational grid."); 
  grid->dx = hx;
  grid->dy = hy;
  grid->dz = hz;
  grid->dt = dt;
  grid->cvac = 1;
  grid->eps0 = eps0;

  // FIXME:  Set up the mesh for load balancing with vacuum boundaries. 

  // Partition a periodic box among the processors sliced uniformly in x: 
  define_periodic_grid( 0,         -0.5*Ly,    -0.5*Lz,       // Low corner
                        Lx,         0.5*Ly,     0.5*Lz,       // High corner
                        nx,         ny,         nz,           // Resolution
                        topology_x, topology_y, topology_z ); // Topology

  int ix, iy, iz; 

  RANK_TO_INDEX( int(rank()), ix, iy, iz );  // Get position of domain in global topology

  // Override field boundary conditions 
  if ( ix == 0) {                                 // Leftmost proc.
    set_domain_field_bc( BOUNDARY(-1,0,0), absorb_fields );
  }
  if ( ix == topology_x - 1 ) {                   // Rightmost proc.
    set_domain_field_bc( BOUNDARY( 1,0,0), absorb_fields );
  }
  if ( iz == 0) {                                 // Topmost proc.
    set_domain_field_bc( BOUNDARY(0,0,-1), absorb_fields );
  }
  if ( iz == topology_z - 1 ) {                   // Bottommost proc.
    set_domain_field_bc( BOUNDARY(0,0, 1), absorb_fields );
  }

  // SETUP THE SPECIES ==============================================================================
  sim_log("Setting up species. ");
  

  double max_local_np              = 1.3*Ne/nproc();
  double max_local_nm              = max_local_np / 10.0;
  sim_log( "num electron, ion macroparticles: "<<max_local_np );
  sim_log("- Creating electron species.");
  species_t *electron = NULL; 
  species_t *ion_H    = NULL;
  species_t *ion_He   = NULL;
  if ( mobile_ions ) {
    if ( He_present ) {
      sim_log("- Creating He species.");
//      ion_He = define_species("He", Z_He, mime_He, qi_He, max_local_np, max_local_nm, ion_sort_freq, 0);
      ion_He = define_species("He", Z_He, mime_He, max_local_np, max_local_nm, ion_sort_freq, 1);
    }
    if ( H_present ) {
      sim_log("- Creating H species.");
//      ion_H  = define_species("H",  Z_H,  mime_H,  qi_H, max_local_np, max_local_nm, ion_sort_freq, 0);
      ion_H  = define_species("H",  Z_H,  mime_H,  max_local_np, max_local_nm, ion_sort_freq, 1);
    }
  }
//  electron = define_species("electron", -1, 1, fabs(qe), max_local_np, max_local_nm, ele_sort_freq, 0);
  electron = define_species("electron", -1, 1, max_local_np, max_local_nm, ele_sort_freq, 1);
  // Light error checking on define_species 
  if ( electron==NULL )                              sim_log_local(" ERROR: electron species not defined.");  
  if ( mobile_ions && H_present  && ion_H  == NULL ) sim_log_local(" ERROR: ion_H    species not defined.");  
  if ( mobile_ions && He_present && ion_He == NULL ) sim_log_local(" ERROR: ion_He   species not defined.");  

  sim_log("Done setting up species.");


  // Paint the simulation volume with materials and boundary conditions
# define iv_region (   x<      hx*iv_thick || x>Lx  -hx*iv_thick  \
                    || z<-Lz/2+hz*iv_thick || z>Lz/2-hz*iv_thick ) /* all boundaries are i.v. */

  //#ifdef USE_MAXWELLIAN
  // SETUP PARTICLE BOUNDARY HANDLER ==============================================

  sim_log("Setting up Maxwellian reflux tally boundary condition.");

  double kemax_mult = 100.;                    // max ke in tally in mc2 units is 
                                               // kemax_mult * temperature, if t_e = 3 keV then the kemax is 300 keV 
  particle_bc_t * maxwellian_reinjection_tally =
    define_particle_bc( maxwellian_reflux_tally( species_list, entropy ) );
  set_reflux_tally_temp( maxwellian_reinjection_tally, electron, uthe, uthe );
  set_reflux_kemax( maxwellian_reinjection_tally, electron, kemax_mult*1.0*uthe*uthe ); 
  if ( mobile_ions ) {
    if ( H_present  ) {
      set_reflux_tally_temp( maxwellian_reinjection_tally, ion_H,  uthi_H,  uthi_H  );
      set_reflux_kemax( maxwellian_reinjection_tally, ion_H, kemax_mult*mime_H*uthi_H*uthi_H ); 
    } 
    if ( He_present ) { 
      set_reflux_tally_temp( maxwellian_reinjection_tally, ion_He, uthi_He, uthi_He );
      set_reflux_kemax( maxwellian_reinjection_tally, ion_He, kemax_mult*mime_He*uthi_He*uthi_He ); 
    } 
  }

  // Set write interval to velocity interval for the time being
  set_reflux_write_interval( maxwellian_reinjection_tally, velocity_interval ); 

  // Create pinhole output directory 
  if ( rank()==0 ) dump_mkdir("pinhole");

  //--------------------------------
  // energy_flux directory also holds accumulated poynting flux 
  if ( rank()==0 ) dump_mkdir("energy_flux");
  //--------------------------------

  sim_log("Done setting up Maxwellian reflux tally boundary condition.");

  // SETUP THE MATERIALS ============================================================================
  sim_log("Setting up materials. "); 
  define_material( "vacuum", 1 );
  define_field_array( NULL, damp ); 

  // Paint the simulation volume with materials and boundary conditions
# define iv_region (   x<      hx*iv_thick || x>Lx  -hx*iv_thick  \
                    || z<-Lz/2+hz*iv_thick || z>Lz/2-hz*iv_thick ) /* all boundaries are i.v. */ 

  //set_region_bc( iv_region, maxwellian_reinjection_tally, maxwellian_reinjection_tally, maxwellian_reinjection_tally );
  set_region_bc( iv_region, reflect_particles,reflect_particles,reflect_particles ); 
  // LOAD THE PARTICLES =============================================================================
  // 
  // BJA - load a linear ramp from NE_NCR_MIN to NE_NCR_MAX

  // Mean ne/ncr for the simulation - found at center of box in x
# define NE_NCR_MEAN   (0.05)

  // ne/ncr variation as one goes from middle of box in x to either edge in x
# define NE_NCR_CHANGE (0.0)

  // Automatically defined from the above
# define NE_NCR_MAX (NE_NCR_MEAN + NE_NCR_CHANGE)
# define NE_NCR_MIN (NE_NCR_MEAN - NE_NCR_CHANGE)

  // Density macro - given x and z values, return ne/ncr
  //                 assumes that 0 <= X <= Lx and that -Lz/2 <= Z <= Lz/2
# define DENSITY( Z, X ) \
    ( NE_NCR_MIN * (1.0 - (X)/Lx) + NE_NCR_MAX * (X)/Lx )

  // Load particles
  if ( load_particles ) {
    sim_log("Loading particles.");
    // Fast load of particles--don't bother fixing artificial domain correlations
    double xmin=grid->x0, xmax=grid->x1;
    double ymin=grid->y0, ymax=grid->y1;
    double zmin=grid->z0, zmax=grid->z1;

    repeat( Ne * (NE_NCR_MAX / NE_NCR_MEAN) / (topology_x*topology_y*topology_z) ) {
      double x = uniform( rng(0), xmin, xmax );
      double y = uniform( rng(0), ymin, ymax );
      double z = uniform( rng(0), zmin, zmax );
      if ( iv_region ) continue;           // Particle fell in iv_region.  Don't load.

      // Rejection method - 2D density profile in (x,z) plane
      if ( uniform( rng(0), 0, NE_NCR_MAX ) > DENSITY( z, x ) ) continue;

      inject_particle( electron, x, y, z,
                       normal( rng(0), 0, uthe ), 
                       normal( rng(0), 0, uthe ), 
                       0.5,
//                       normal( rng(0), 0, uthe ), 
                       fabs(qe), 0, 0 );

      if ( mobile_ions ) {
        if ( H_present )  // Inject an H macroion on top of macroelectron
          inject_particle( ion_H, x, y, z,
                           normal( rng(0), 0, uthi_H ), 
                           normal( rng(0), 0, uthi_H ), 
                           0.25,
//                           normal( rng(0), 0, uthi_H ), 
                           fabs(qi_H), 0, 0 );
        if ( He_present ) // Inject an He macroion on top of macroelectron
          inject_particle( ion_He, x, y, z,
                           normal( rng(0), 0, uthi_He ), 
                           normal( rng(0), 0, uthi_He ), 
                           0.75, 
//                           normal( rng(0), 0, uthi_He ), 
                           fabs(qi_He), 0, 0 );
      }
      // DEBUG 
      if ( electron->np > 0.95*electron->max_np ) 
        sim_log_local( "Electron np, max_np = "<<electron->np<<" "<<electron->max_np ); 
      if ( ion_He  ->np > 0.95*ion_He  ->max_np ) 
        sim_log_local( "ion_He   np, max_np = "<<ion_He  ->np<<" "<<ion_He  ->max_np ); 
      if ( ion_H  ->np > 0.95*ion_H  ->max_np ) 
        sim_log_local( "ion_H   np, max_np = "<<ion_H  ->np<<" "<<ion_H  ->max_np ); 
    } // repeat 
    // DEBUG 
    // sim_log( "Electron np = "<<electron->np ); 
    // sim_log( "ion_He   np = "<<ion_He  ->np ); 
  } // if 

#ifdef PARAVIEW_DUMP
 /*--------------------------------------------------------------------------
  * New dump definition
  *------------------------------------------------------------------------*/

 /*--------------------------------------------------------------------------
  * Set data output format
  * 
  * This option allows the user to specify the data format for an output
  * dump.  Legal settings are 'band' and 'band_interleave'.  Band-interleave
  * format is the native storage format for data in VPIC.  For field data,
  * this looks something like:
  * 
  *   ex0 ey0 ez0 div_e_err0 cbx0 ... ex1 ey1 ez1 div_e_err1 cbx1 ...
  *   
  * Banded data format stores all data of a particular state variable as a
  * contiguous array, and is easier for ParaView to process efficiently. 
  * Banded data looks like:
  * 
  *   ex0 ex1 ex2 ... exN ey0 ey1 ey2 ...
  *   
  *------------------------------------------------------------------------*/
//  sim_log("Setting up hydro and field diagnostics.");

  global->fdParams.format = band;
  sim_log ( "Field output format          : band" );

  global->hedParams.format = band;
  sim_log ( "Electron hydro output format : band" );

  global->hHdParams.format = band;
  sim_log ( "Hydrogen hydro output format : band" );

  global->hHedParams.format = band;
  sim_log ( "Helium hydro output format   : band" );

 /*--------------------------------------------------------------------------
  * Set stride
  * 
  * This option allows data down-sampling at output.  Data are down-sampled
  * in each dimension by the stride specified for that dimension.  For
  * example, to down-sample the x-dimension of the field data by a factor
  * of 2, i.e., half as many data will be output, select:
  * 
  *   global->fdParams.stride_x = 2;
  *
  * The following 2-D example shows down-sampling of a 7x7 grid (nx = 7,
  * ny = 7.  With ghost-cell padding the actual extents of the grid are 9x9.
  * Setting the strides in x and y to equal 2 results in an output grid of
  * nx = 4, ny = 4, with actual extents 6x6.
  *
  * G G G G G G G G G
  * G X X X X X X X G
  * G X X X X X X X G         G G G G G G
  * G X X X X X X X G         G X X X X G
  * G X X X X X X X G   ==>   G X X X X G
  * G X X X X X X X G         G X X X X G
  * G X X X X X X X G         G X X X X G
  * G X X X X X X X G         G G G G G G
  * G G G G G G G G G
  *
  * Note that grid extents in each dimension must be evenly divisible by
  * the stride for that dimension:
  *
  *   nx = 150;
  *   global->fdParams.stride_x = 10; // legal -> 150/10 = 15
  *
  *   global->fdParams.stride_x = 8; // illegal!!! -> 150/8 = 18.75
  *------------------------------------------------------------------------*/

  // Strides for field and hydro arrays.  Note that here we have defined them 
  // the same for fields and all hydro species; if desired, we could use different
  // strides for each.   Also note that strides must divide evenly into the number 
  // of cells in a given domain. 

  // Define strides and test that they evenly divide into grid->nx, ny, nz
  int stride_x = 1, stride_y = 1, stride_z = 1; 
  if ( int(grid->nx)%stride_x ) ERROR(("Stride doesn't evenly divide grid->nx.")); 
  if ( int(grid->ny)%stride_y ) ERROR(("Stride doesn't evenly divide grid->ny.")); 
  if ( int(grid->nz)%stride_z ) ERROR(("Stride doesn't evenly divide grid->nz.")); 

  //----------------------------------------------------------------------
  // Fields

  // relative path to fields data from global header
  sprintf(global->fdParams.baseDir, "fields");

  // base file name for fields output
  sprintf(global->fdParams.baseFileName, "fields");

  // set field strides
  global->fdParams.stride_x = stride_x;
  global->fdParams.stride_y = stride_y;
  global->fdParams.stride_z = stride_z;
  sim_log ( "Fields x-stride " << global->fdParams.stride_x );
  sim_log ( "Fields y-stride " << global->fdParams.stride_y );
  sim_log ( "Fields z-stride " << global->fdParams.stride_z );

  // add field parameters to list
  global->outputParams.push_back(&global->fdParams);

  //----------------------------------------------------------------------
  // Electron hydro

  // relative path to electron species data from global header
  sprintf(global->hedParams.baseDir, "hydro");

  // base file name for fields output
  sprintf(global->hedParams.baseFileName, "e_hydro");

  // set electron hydro strides
  global->hedParams.stride_x = stride_x;
  global->hedParams.stride_y = stride_y;
  global->hedParams.stride_z = stride_z;
  sim_log ( "Electron species x-stride " << global->hedParams.stride_x );
  sim_log ( "Electron species y-stride " << global->hedParams.stride_y );
  sim_log ( "Electron species z-stride " << global->hedParams.stride_z );

  // add electron hydro parameters to list
  global->outputParams.push_back(&global->hedParams);

  //----------------------------------------------------------------------
  // Hydrogen hydro

  // relative path to electron species data from global header
  sprintf(global->hHdParams.baseDir, "hydro");

  // base file name for fields output
  sprintf(global->hHdParams.baseFileName, "H_hydro");

  // set hydrogen hydro strides
  global->hHdParams.stride_x = stride_x;
  global->hHdParams.stride_y = stride_y;
  global->hHdParams.stride_z = stride_z;
  sim_log ( "Ion species x-stride " << global->hHdParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hHdParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hHdParams.stride_z );

  // add hydrogen hydro parameters to list
  global->outputParams.push_back(&global->hHdParams);

  //----------------------------------------------------------------------
  // Helium hydro

  // relative path to electron species data from global header
  sprintf(global->hHedParams.baseDir, "hydro");

  // base file name for fields output
  sprintf(global->hHedParams.baseFileName, "He_hydro");

  // set helium hydro strides
  global->hHedParams.stride_x = stride_x;
  global->hHedParams.stride_y = stride_y;
  global->hHedParams.stride_z = stride_z;
  sim_log ( "Ion species x-stride " << global->hHedParams.stride_x );
  sim_log ( "Ion species y-stride " << global->hHedParams.stride_y );
  sim_log ( "Ion species z-stride " << global->hHedParams.stride_z );

  // add helium hydro parameters to list
  global->outputParams.push_back(&global->hHedParams);

 /*-----------------------------------------------------------------------
  * Set output fields
  *
  * It is now possible to select which state-variables are output on a
  * per-dump basis.  Variables are selected by passing an or-list of
  * state-variables by name.  For example, to only output the x-component
  * of the electric field and the y-component of the magnetic field, the
  * user would call output_variables like:
  *
  *   global->fdParams.output_variables( ex | cby );
  *
  * NOTE: OUTPUT VARIABLES ARE ONLY USED FOR THE BANDED FORMAT.  IF THE
  * FORMAT IS BAND-INTERLEAVE, ALL VARIABLES ARE OUTPUT AND CALLS TO
  * 'output_variables' WILL HAVE NO EFFECT.
  *
  * ALSO: DEFAULT OUTPUT IS NONE!  THIS IS DUE TO THE WAY THAT VPIC
  * HANDLES GLOBAL VARIABLES IN THE INPUT DECK AND IS UNAVOIDABLE.
  *
  * For convenience, the output variable 'all' is defined:
  *
  *   global->fdParams.output_variables( all );
  *------------------------------------------------------------------------*/
 /* CUT AND PASTE AS A STARTING POINT
  * REMEMBER TO ADD APPROPRIATE GLOBAL DUMPPARAMETERS VARIABLE

   output_variables( all );

   output_variables( electric | div_e_err | magnetic | div_b_err |
                     tca      | rhob      | current  | rhof |
                     emat     | nmat      | fmat     | cmat );

   output_variables( current_density  | charge_density |
                     momentum_density | ke_density     | stress_tensor );
  */

  //global->fdParams.output_variables( all );
  global->fdParams.output_variables( electric | magnetic );

  //global->hedParams.output_variables( all );
  //global->hedParams.output_variables( current_density | momentum_density );
  global->hedParams.output_variables(  current_density  | charge_density |
                                       momentum_density | ke_density |
                                       stress_tensor );
  global->hHdParams.output_variables(  current_density  | charge_density |
                                       momentum_density | ke_density |
                                       stress_tensor );
  global->hHedParams.output_variables( current_density  | charge_density |
                                       momentum_density | ke_density |
                                       stress_tensor );

 /*--------------------------------------------------------------------------
  * Convenience functions for simlog output
  *------------------------------------------------------------------------*/
  char varlist[BUFLEN];

  create_field_list(varlist, global->fdParams);
  sim_log ( "Fields variable list: " << varlist );

  create_hydro_list(varlist, global->hedParams);
  sim_log ( "Electron species variable list: " << varlist );

  create_hydro_list(varlist, global->hHdParams);
  sim_log ( "H species variable list: " << varlist );

//  create_hydro_list(varlist, global->hHedParams);
//  sim_log ( "He species variable list: " << varlist );

 /*------------------------------------------------------------------------*/

  sim_log("*** Finished with user-specified initialization ***"); 

  // Upon completion of the initialization, the following occurs:
  // - The synchronization error (tang E, norm B) is computed between domains
  //   and tang E / norm B are synchronized by averaging where discrepancies
  //   are encountered.
  // - The initial divergence error of the magnetic field is computed and
  //   one pass of cleaning is done (for good measure)
  // - The bound charge density necessary to give the simulation an initially
  //   clean divergence e is computed.
  // - The particle momentum is uncentered from u_0 to u_{-1/2}
  // - The user diagnostics are called on the initial state
  // - The physics loop is started
  //
  // The physics loop consists of:
  // - Advance particles from x_0,u_{-1/2} to x_1,u_{1/2}
  // - User particle injection at x_{1-age}, u_{1/2} (use inject_particles)
  // - User current injection (adjust field(x,y,z).jfx, jfy, jfz)
  // - Advance B from B_0 to B_{1/2}
  // - Advance E from E_0 to E_1
  // - User field injection to E_1 (adjust field(x,y,z).ex,ey,ez,cbx,cby,cbz)
  // - Advance B from B_{1/2} to B_1
  // - (periodically) Divergence clean electric field
  // - (periodically) Divergence clean magnetic field
  // - (periodically) Synchronize shared tang e and norm b
  // - Increment the time step
  // - Call user diagnostics
  // - (periodically) Print a status message
#endif
}

#define should_dump(x)							\
  (global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)
begin_diagnostics {
#if 0 //disable DIAGNOSTICS 
   int mobile_ions=global->mobile_ions,
        H_present=global->H_present,
        He_present=global->He_present;

  if ( step()%200==0 ) sim_log("Time step: "<<step());
//sim_log("Time step: "<<step); 

#if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier(); 
  sim_log( "Beginning diagnostics block.  Synchronized." ); 
#endif 

# define should_dump(x) \
  (global->x##_interval>0 && remainder(step(),global->x##_interval)==0)

  // Do a mkdir at time t=0 to ensure we have all the directories we need
  // Put a barrier here to avoid a race condition
  if ( step()==0 ) {
    mp_barrier(); 
    if ( rank()==0 ) {
      dump_mkdir("rundata"); 
      dump_mkdir("fft"); 
      dump_mkdir("field"); 
      dump_mkdir("ehydro"); 
      dump_mkdir("Hhydro"); 
      dump_mkdir("Hehydro"); 
      dump_mkdir("restart"); 
      dump_mkdir("particle"); 
      dump_mkdir("poynting"); 
      dump_mkdir("velocity"); 

      // Turn off rundata for now
      // dump_grid("rundata/grid");
      // dump_materials("rundata/materials");
      // dump_species("rundata/species");
      global_header("global", global->outputParams);
    } 
    mp_barrier(); 
  }

  // Energy dumps store all the energies in various directions of E and B
  // and the total kinetic (not including rest mass) energies of each species
  // species in a simple text format. By default, the energies are appended to
  // the file. However, if a "0" is added to the dump_energies call, a new
  // energies dump file will be created. The energies are in the units of the
  // problem and are all time centered appropriately. Note: When restarting a
  // simulation from a restart dump made at a prior time step to the last
  // energies dump, the energies file will have a "hiccup" of intervening
  // time levels. This "hiccup" will not occur if the simulation is aborted
  // immediately following a restart dump. Energies dumps are in a text
  // format and the layout is documented at the top of the file. Only rank 0
  // makes makes an energies dump.
  if( should_dump(energies) ) dump_energies( "rundata/energies", step()==0 ? 0 : 1 );

  // Field and hydro data writes contained in time_average_v3_He.cxx

//#include "time_average_v3_He.cxx"

#if 1
  // Field and hydro data

  if ( should_dump(field) ) {
    field_dump( global->fdParams );
//  dump_fields( "field/fields", (int)step );

    if ( global->load_particles ) {
      hydro_dump( "electron", global->hedParams );
      if ( global->mobile_ions ) {
        if ( global->H_present  ) hydro_dump( "H",  global->hHdParams );
        if ( global->He_present ) hydro_dump( "He", global->hHedParams );
      }
    }

  }
#endif


  // Particle dump data
#if 0  
  if ( should_dump(particle) && global->load_particles ) {
    dump_particles( "electron", "particle/eparticle" );
    if ( global->mobile_ions ) {
      if ( global->H_present  ) dump_particles( "H",  "particle/Hparticle" );
      if ( global->He_present ) dump_particles( "He", "particle/Heparticle" );
    }
  } 
#endif

  
#if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier(); 
  sim_log( "setup done; going to standard VPIC output." ); 
#endif 

#if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier();
  sim_log( "standard VPIC output done; going to FFT." );
#endif

#if 1
// kx fft diag.
  // ------------------------------------------------------------------------
  // Custom diagnostic where we write out Ex for each cell in grid.
  // These data are stored for every fft_ex_interval time step.
  //
  // Refactored for 2D LPI problem.
# define FFT_HEADER_SIZE (sizeof(int))

# define WRITE_FFT(SUFF,INTERVAL)                                              \
  BEGIN_PRIMITIVE {                                                            \
    status = fileIO_##SUFF.open( fname_##SUFF, io_read_write);                 \
    if ( status==fail ) ERROR(("Could not open file."));                       \
    fileIO_##SUFF.seek( uint64_t( FFT_HEADER_SIZE                              \
                                  + uint64_t((step()/INTERVAL)*stride*sizeof(float)) ),  \
                        SEEK_SET );                                            \
    fileIO_##SUFF.write( SUFF, stride );                                       \
    fileIO_##SUFF.close();                                                     \
  } END_PRIMITIVE

# define WRITE_FFT_HEADER(SUFF)                                                \
  BEGIN_PRIMITIVE {                                                            \
    status = fileIO_##SUFF.open( fname_##SUFF, io_write);                      \
    if ( status==fail ) ERROR(("Could not open file."));                       \
    fileIO_##SUFF.write( &grid->nx, 1 );                                       \
    fileIO_##SUFF.close();                                                     \
  } END_PRIMITIVE

  static int initted=0;
  FileIOStatus status;

  static float *ex;
  static char fname_ex[256];
  FileIO       fileIO_ex;
  int stride=grid->nx;

  static float *ey; 
  static char  fname_ey[256]; 
  FileIO       fileIO_ey; 

  if ( !initted ) {
    // Allocate space for data arrays
    long array_length=grid->nx;
    ALLOCATE(ex,  array_length, float);
    ALLOCATE(ey,  array_length, float);
    // Define filenames
    sprintf( fname_ex,  "fft/fft_ex.%i",  (int)rank() );
    sprintf( fname_ey,  "fft/fft_ey.%i",  (int)rank() );
    // On first timestep prepend a header with number of x meshpoints to each file
    if ( step()==0 ) {
      WRITE_FFT_HEADER(ex);
      WRITE_FFT_HEADER(ey);
    }
    initted=1;
  }

  // Note: Ex and Ey on the cut at Lz/2 are both normalized to global->emax

  // *** Ex ***
  if ( !(step()%global->fft_ex_interval) ) {
    // Store data into array ex
    for ( int i=0; i<grid->nx; ++i ) {
      int k=INDEX_FORTRAN_3(i+1,1,grid->nz/2+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
      ex[i]=field(k).ex / global->emax;
    } // for 
    // Write array to file
    sim_log("Writing FFT data for Ex fields.");
    DIAG_LOG("Starting to dump FFT Ex data.");
    WRITE_FFT(ex, global->fft_ex_interval);
    DIAG_LOG("Finished dumping FFT Ex data.");
  } // if 

  // *** Ey ***
  if ( !(step()%global->fft_ey_interval) ) { 
    // Store data into array ey
    for ( int i=0; i<grid->nx; ++i ) { 
      int k=INDEX_FORTRAN_3(i+1,1,grid->nz/2+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
      ey[i]=field(k).ey / global->emax;  
    } // for 
    // Write array to file 
    sim_log("Writing FFT data for Ey fields."); 
    DIAG_LOG("Starting to dump FFT Ey data."); 
    WRITE_FFT(ey, global->fft_ey_interval); 
    DIAG_LOG("Finished dumping FFT Ey data."); 
  } // if 

#endif


#if 1
  // FFT DIAGNOSTIC ========================================================================

  // kx and kz fft diag.
  // use emax as norm factor
  // ------------------------------------------------------------------------
  // Custom diagnostic where we write out Ex, Ey, and cBz for each cell in grid.
  // These data are stored for each time step.
  //
  // Refactored for 2D LPI problem.
# define FFT_HEADER_SIZE (sizeof(int))

# define WRITE_FFT(SUFF,INTERVAL)                                              \
  BEGIN_PRIMITIVE {                                                            \
    status = fileIO_##SUFF.open( fname_##SUFF, io_read_write);                 \
    if ( status==fail ) ERROR(("Could not open file."));                       \
    fileIO_##SUFF.seek( uint64_t( FFT_HEADER_SIZE                              \
                                  + uint64_t((step()/INTERVAL)*stride*sizeof(float)) ),  \
                        SEEK_SET );                                            \
    fileIO_##SUFF.write( SUFF, stride );                                       \
    fileIO_##SUFF.close();                                                     \
  } END_PRIMITIVE

# define WRITE_FFT_HEADER(SUFF)                                                \
  BEGIN_PRIMITIVE {                                                            \
    status = fileIO_##SUFF.open( fname_##SUFF, io_write);                      \
    if ( status==fail ) ERROR(("Could not open file."));                       \
    fileIO_##SUFF.write( &grid->nx, 1 );                                       \
    fileIO_##SUFF.close();                                                     \
  } END_PRIMITIVE

  BEGIN_PRIMITIVE { 
    int ix, iy, iz; 
    RANK_TO_INDEX( int(rank()), ix, iy, iz );  // Get position of domain in global topology

    // Write Ey(z) along laser entrance plane
 
    BEGIN_PRIMITIVE { 
      if ( ix == 0 ) {
        static int initted=0;
        static float *ey;
        static char fname_ey[BUFLEN];
        FileIO fileIO_ey;
        FileIOStatus status;
        int stride=grid->nz;
  
        if ( !initted ) {
          // Allocate space for data arrays
          long array_length=grid->nz;
          ALLOCATE( ey, array_length, float );
          // Define filenames
          sprintf( fname_ey,  "fft/fft_ey_0.%i", iz );
          // On first timestep prepend a header with number of x meshpoints to each file
          if ( step()==0 ) {
            WRITE_FFT_HEADER(ey);
            status = fileIO_ey.open( fname_ey, io_write);
            if ( status==fail ) ERROR(("Could not open file."));
            fileIO_ey.write( &grid->nz, 1 );
            fileIO_ey.close();
          } // if 
          initted=1;
        } // if 
  
        // *** Ey ***
        if ( !(step()%global->fft_ey_interval) ) {
          // Store data into array ey
          for ( int i=0; i<grid->nz; ++i ) {
            int k=INDEX_FORTRAN_3(1,1,i+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            ey[i]  = field(k).ey / global->emax;
          } // for 
          // Write array to file
          sim_log( "Writing FFT data for Ey fields: x=0" );
          WRITE_FFT(ey,  global->fft_ey_interval);
  
          status = fileIO_ey.open( fname_ey, io_read_write);
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO_ey.seek( uint64_t( FFT_HEADER_SIZE
                                        + uint64_t((step()/global->fft_ey_interval)*stride*sizeof(float)) ),
                              SEEK_SET );
          fileIO_ey.write( ey, stride );
          fileIO_ey.close();
  
        } // if 
      } // if 
    } END_PRIMITIVE; 

    // Write Ey(z) along laser exit plane
 
    BEGIN_PRIMITIVE { 
      if ( ix == global->topology_x-1 ) {
        static int initted=0;
        static float *ey;
        static char fname_ey[BUFLEN];
        FileIO fileIO_ey;
        FileIOStatus status;
        int stride=grid->nz;
  
        if ( !initted ) {
          // Allocate space for data arrays
          long array_length=grid->nz;
          ALLOCATE( ey, array_length, float );
          // Define filenames
          sprintf( fname_ey,  "fft/fft_ey_Lx.%i", iz );
          // On first timestep prepend a header with number of x meshpoints to each file
          if ( step()==0 ) {
            WRITE_FFT_HEADER(ey);
            status = fileIO_ey.open( fname_ey, io_write);
            if ( status==fail ) ERROR(("Could not open file."));
            fileIO_ey.write( &grid->nz, 1 );
            fileIO_ey.close();
          } // if 
          initted=1;
        } // if 
  
        // *** Ey ***
        if ( !(step()%global->fft_ey_interval) ) {
          // Store data into array ey
          for ( int i=0; i<grid->nz; ++i ) {
            int k=INDEX_FORTRAN_3(grid->nx,1,i+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            ey[i]  = field(k).ey / global->emax;
          } // for 
          // Write array to file
          sim_log( "Writing FFT data for Ey fields: x=" << global->Lx );
          WRITE_FFT(ey,  global->fft_ey_interval);
  
          status = fileIO_ey.open( fname_ey, io_read_write);
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO_ey.seek( uint64_t( FFT_HEADER_SIZE
                                        + uint64_t((step()/global->fft_ey_interval)*stride*sizeof(float)) ),
                              SEEK_SET );
          fileIO_ey.write( ey, stride );
          fileIO_ey.close();
  
        } // if 
      } // if 
    } END_PRIMITIVE; 


    // Write Ey(z) along intermediate plane given by point global->ey_xloc

    BEGIN_PRIMITIVE { 
      int ix_plane; 
      int ix_index;    
      double dx_domain = ( global->Lx - global->xmin_domain ) / global->topology_x; 
      
      // ix index for MPI domains that write the ey(z) output
      ix_plane = ( global->ey_xloc - global->xmin_domain ) / dx_domain; 
 
      // x index value of field to write for this domain (used only if this rank writes output) 
      ix_index = ( global->ey_xloc - grid->x0 ) / grid->dx;  
  
      if ( ix == ix_plane ) {
        static int initted=0;
        static float *ey;
        static char fname_ey[BUFLEN];
        FileIO fileIO_ey;
        FileIOStatus status;
        int stride=grid->nz;
  
        if ( !initted ) {
          // Allocate space for data arrays
          long array_length=grid->nz;
          ALLOCATE( ey, array_length, float );
          // Define filenames
          sprintf( fname_ey,  "fft/fft_ey_xloc.%i", iz );
          // On first timestep prepend a header with number of x meshpoints to each file
          if ( step()==0 ) {
            WRITE_FFT_HEADER(ey);
            status = fileIO_ey.open( fname_ey, io_write);
            if ( status==fail ) ERROR(("Could not open file."));
            fileIO_ey.write( &grid->nz, 1 );
            fileIO_ey.close();
          } // if 
          initted=1;
        } // if 
  
        // *** Ey ***
        if ( !(step()%global->fft_ey_interval) ) {
          // Store data into array ey
          for ( int i=0; i<grid->nz; ++i ) {
            int k=INDEX_FORTRAN_3(ix_index+1,1,i+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            ey[i]  = field(k).ey / global->emax;
          } // for 
          // Write array to file
          sim_log( "Writing FFT data for Ey fields: x=" << global->ey_xloc );
          WRITE_FFT(ey,  global->fft_ey_interval);
  
          status = fileIO_ey.open( fname_ey, io_read_write);
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO_ey.seek( uint64_t( FFT_HEADER_SIZE
                                        + uint64_t((step()/global->fft_ey_interval)*stride*sizeof(float)) ),
                              SEEK_SET );
          fileIO_ey.write( ey, stride );
          fileIO_ey.close();
  
        } // if 
      } // if 
    } END_PRIMITIVE; 

    // Write Ez(z) at x = 0.5*Lx 

    BEGIN_PRIMITIVE { 
      int ix_plane; 
      int ix_index;    
      double dx_domain = ( global->Lx - global->xmin_domain ) / global->topology_x; 
      
      // ix index for MPI domains that write the ey(z) output
      ix_plane = ( global->Lx*0.5 - global->xmin_domain ) / dx_domain; 
 
      // x index value of field to write for this domain (used only if this rank writes output) 
      ix_index = ( global->Lx*0.5 - grid->x0 ) / grid->dx;  
  
      if ( ix == ix_plane ) {
        static int initted=0;
        static float *ez;
        static char fname_ez[BUFLEN];
        FileIO fileIO_ez;
        FileIOStatus status;
        int stride=grid->nz;
  
        if ( !initted ) {
          // Allocate space for data arrays
          long array_length=grid->nz;
          ALLOCATE( ez, array_length, float );
          // Define filenames
          sprintf( fname_ez,  "fft/fft_ez_0.5_Lx.%i", iz );
          // On first timestep prepend a header with number of x meshpoints to each file
          if ( step()==0 ) {
            WRITE_FFT_HEADER(ez);
            status = fileIO_ez.open( fname_ez, io_write);
            if ( status==fail ) ERROR(("Could not open file."));
            fileIO_ez.write( &grid->nz, 1 );
            fileIO_ez.close();
          } // if 
          initted=1;
        } // if 
  
        // *** Ez ***
        if ( !(step()%global->fft_ez_interval) ) {
          // Store data into array ez
          for ( int i=0; i<grid->nz; ++i ) {
            int k=INDEX_FORTRAN_3(ix_index+1,1,i+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            ez[i]  = field(k).ez / global->emax;
          } // for 
          // Write array to file
          sim_log( "Writing FFT data for Ez fields: x=" << global->Lx * 0.5 );
          WRITE_FFT(ez,  global->fft_ez_interval);
  
          status = fileIO_ez.open( fname_ez, io_read_write);
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO_ez.seek( uint64_t( FFT_HEADER_SIZE
                                        + uint64_t((step()/global->fft_ez_interval)*stride*sizeof(float)) ),
                              SEEK_SET );
          fileIO_ez.write( ez, stride );
          fileIO_ez.close();
  
        } // if 
      } // if 
    } END_PRIMITIVE; 

  } END_PRIMITIVE;
#endif



  //------------------------------------
  //
  // Sum Poynting flux on boundaries; write summed Poynting flux to file every global->velocity_interval steps 
  //
  // Poynting flux in a given direction is defined as the projection
  // of E x B along the unit vector pointing into the simulation domain normal to box face


  BEGIN_PRIMITIVE { 
    int ix, iy, iz; 

    // MKS 
    // n.b. 1/mu0 = c^2 * eps0 and poynting should be dA * dt * ExB * (1/mu0)
    float norm_xface = grid->dt * grid->dy * grid->dz * grid->eps0 * grid->cvac * grid->cvac;
    float norm_zface = grid->dt * grid->dx * grid->dy * grid->eps0 * grid->cvac * grid->cvac;
  
    RANK_TO_INDEX( rank(), ix, iy, iz );
  
    // 3D decomposition
    // Test whether processor is on an MPI domain edge
    // if (    ( ix==0 || ix==global->topology_x-1 )  
    //      || ( iy==0 || iy==global->topology_y-1 ) 
    //      || ( iz==0 || iz==global->topology_z-1 ) ) {

    // 2D decomposition in x and z
    if (    ( ix==0 || ix==global->topology_x-1 )  
         || ( iz==0 || iz==global->topology_z-1 ) ) {
      int i, j, k; 
      int offset = global->psum_integration_offset; 
 
      //--------------------------------------------------------------- 
      // if on lower x boundary, sum poynting flux through lower x face 
      if ( ix==0 ) { 

        // Ey * cBz on lower x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,    j+0.5,k (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbz @ i+0.5,j+0.5,k (all 1:nx,  1:ny,1:nz+1; int 1:nx,1:ny,2:nz)
        //
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ey, cbz;
            ey    = 0.25*(  field(1+offset,j,k).ey  + field(1+offset,j,k+1).ey
                          + field(2+offset,j,k).ey  + field(2+offset,j,k+1).ey );
            cbz   = 0.50*(  field(1+offset,j,k).cbz + field(1+offset,j,k+1).cbz ); 
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( ey*cbz )*norm_xface;
          } // for 
        } // for 

        // Ez * cBy on lower x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ez  @ i,j,k+0.5     (all 1:nx+1,1:ny+1,1:nz; int 2:nx,2:ny,1:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,  1:ny+1,1:nz; int 1:nx,2:ny,1:nz)
        // 
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ez, cby;
            ez    = 0.25*(  field(1+offset,j,k).ez  + field(1+offset,j+1,k).ez    
                          + field(2+offset,j,k).ez  + field(2+offset,j+1,k).ez );
            cby   = 0.50*(  field(1+offset,j,k).cby + field(1+offset,j+1,k).cby );    
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( -ez*cby )*norm_xface;
          } // for 
        } // for 

      } // if 
  
      //--------------------------------------------------------------- 
      // if on upper x boundary, sum poynting flux through upper x face 
      if ( ix==global->topology_x-1 ) {

        // Ey * cBz on upper x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,    j+0.5,k (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbz @ i+0.5,j+0.5,k (all 1:nx,  1:ny,1:nz+1; int 1:nx,1:ny,2:nz)
        //
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ey, cbz;
            ey    = 0.25*(  field(grid->nx+0-offset,j,k).ey  + field(grid->nx+0-offset,j,k+1).ey
                          + field(grid->nx+1-offset,j,k).ey  + field(grid->nx+1-offset,j,k+1).ey );
            cbz   = 0.50*(  field(grid->nx+0-offset,j,k).cbz + field(grid->nx+0-offset,j,k+1).cbz ); 
            // unit normal to box pointing inward is -ix so -= below 
            global->psum_integrated_poynting_flux_tally -= ( ey*cbz )*norm_xface;
          } // for 
        } // for 

        // Ez * cBy on upper x, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ez  @ i,j,k+0.5     (all 1:nx+1,1:ny+1,1:nz; int 2:nx,2:ny,1:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,  1:ny+1,1:nz; int 1:nx,2:ny,1:nz)
        //
        for ( j=1; j<=grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            double ez, cby;
            ez    = 0.25*(  field(grid->nx+0-offset,j,k).ez  + field(grid->nx+0-offset,j+1,k).ez    
                          + field(grid->nx+1-offset,j,k).ez  + field(grid->nx+1-offset,j+1,k).ez );
            cby   = 0.50*(  field(grid->nx+0-offset,j,k).cby + field(grid->nx+0-offset,j+1,k).cby );    
            // unit normal to box pointing inward is -ix so -= below 
            global->psum_integrated_poynting_flux_tally -= ( -ez*cby )*norm_xface;
          } // for 
        } // for 

      } // if 
 
      //--------------------------------------------------------------- 
      // if on lower z boundary, sum poynting flux through lower z face 
      if ( iz==0 ) {

        // Ex * cBy on lower z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ex  @ i+0.5,j,k     (all 1:nx,1:ny+1,1:nz+1; int 1:nx,2:ny,2:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,1:ny+1,1:nz;   int 1:nx,2:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ex, cby;
            ex    = 0.25*(  field(i,j  ,1+offset).ex + field(i,j  ,2+offset).ex 
                          + field(i,j+1,1+offset).ex + field(i,j+1,2+offset).ex );
            cby   = 0.50*(  field(i,j  ,1+offset).cby+ field(i,j+1,1+offset).cby ); 
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( ex*cby )*norm_zface;
          } // for 
        } // for 

        // Ey * cBx on lower z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,j+0.5,k     (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbx @ i,j+0.5,k+0.5 (all 1:nx+1,1:ny,1:nz;   int 2:nx,1:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ey, cbx;
            ey    = 0.25*(  field(i  ,j,1+offset).ey  + field(i  ,j,2+offset).ey       
                          + field(i+1,j,1+offset).ey  + field(i+1,j,2+offset).ey );
            cbx   = 0.50*(  field(i  ,j,1+offset).cbx + field(i+1,j,1+offset).cbx );    
            // unit normal to box pointing inward is +ix so += below 
            global->psum_integrated_poynting_flux_tally += ( -ey*cbx )*norm_zface;
          } // for 
        } // for 

      } // if

      //--------------------------------------------------------------- 
      // if on upper z boundary, sum poynting flux through upper z face 
      if ( iz==global->topology_z-1 ) {

        // Ex * cBy on upper z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ex  @ i+0.5,j,k     (all 1:nx,1:ny+1,1:nz+1; int 1:nx,2:ny,2:nz)
        // f(i,j,k).cby @ i+0.5,j,k+0.5 (all 1:nx,1:ny+1,1:nz;   int 1:nx,2:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ex, cby;
            ex    = 0.25*(  field(i,j  ,grid->nz+0-offset).ex  + field(i,j  ,grid->nz+1-offset).ex 
                          + field(i,j+1,grid->nz+0-offset).ex  + field(i,j+1,grid->nz+1-offset).ex );
            cby   = 0.50*(  field(i,j  ,grid->nz+0-offset).cby + field(i,j+1,grid->nz+0-offset).cby ); 
            // unit normal to box pointing inward is +iz so -= below 
            global->psum_integrated_poynting_flux_tally -= ( ex*cby )*norm_zface;
          } // for 
        } // for 

        // Ey * cBx on upper z, spatial average at cell center (E and B are already time centered)
        //
        // f(i,j,k).ey  @ i,j+0.5,k     (all 1:nx+1,1:ny,1:nz+1; int 2:nx,1:ny,2:nz)
        // f(i,j,k).cbx @ i,j+0.5,k+0.5 (all 1:nx+1,1:ny,1:nz;   int 2:nx,1:ny,1:nz)
        //
        for ( i=1; i<=grid->nx; ++i ) {
          for ( j=1; j<=grid->ny; ++j ) {
            double ey, cbx;
            ey    = 0.25*(  field(i  ,j,grid->nz+0-offset).ey +  field(i  ,j,grid->nz+1-offset).ey       
                          + field(i+1,j,grid->nz+0-offset).ey +  field(i+1,j,grid->nz+1-offset).ey );
            cbx   = 0.50*(  field(i  ,j,grid->nz+0-offset).cbx + field(i+1,j,grid->nz+0-offset).cbx );    
            // unit normal to box pointing inward is +iz so -= below 
            global->psum_integrated_poynting_flux_tally -= ( -ey*cbx )*norm_zface;
          } // for 
        } // for 

      } // if                        

      // global->velocity_interval is the same interval as used in the pbc dumps
      if ( step()>0 && should_dump(velocity) ) {
        FileIO        fileIO;
        FileIOStatus  status;
        char          fname_poynting_tally[BUFLEN];

        sprintf( fname_poynting_tally, "energy_flux/poynting.%d", int(rank()) ); 
        status=fileIO.open( fname_poynting_tally,
                            (step()==global->velocity_interval ? io_write : io_read_write) );
        if ( status==fail ) ERROR(("Could not open file."));
        fileIO.seek( uint64_t( sizeof(double) * ( step()/global->velocity_interval-1 ) ), 
                     SEEK_SET );
        fileIO.write( &(global->psum_integrated_poynting_flux_tally), 1 );
        fileIO.close();

      } // if  
    } // if  


  } END_PRIMITIVE;
  //------------------------------------


#if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier(); 
  sim_log( "FFT done; going to Poynting." ); 
#endif 

  // POYNTING DIAGNOSTIC ========================================================================
# if 1
  //----------------------------------------------------------------------------
  // Poynting diagnostic.  Lin needs the raw E and B fields at the boundary
  // in order to perform digital filtering to extract the SRS component from the 
  // SBS.  We use random access binary writes with a stride of length:
  // 
  // stride =   2* grid->nz * global->topology_z  // lower, upper x faces
  //          + 2* grid->nx * global->topology_x; // lower, upper z faces
  // 
  // On the x faces, e1 = ey, e2 = ez, cb1 = cby, cb2 = cbz
  // On the z faces, e1 = ex, e2 = ey, cb1 = cbx, cb2 = cby
  //
  // We also write 4-element arrays of integrated poynting flux on each face:
  // 
  // vals = {lower x, upper x, lower z, upper z}
  // 
  // Note:  This diagnostic assumes uniform domains.
  // 
  // Also note:  Poynting flux in a given direction is defined as the projection
  // of E x B along the unit vector in that direction.  
  //---------------------------------------------------------------------------- 

# define ALLOCATE(A,LEN,TYPE)                                             \
    if ( !((A)=(TYPE *)malloc((size_t)(LEN)*sizeof(TYPE))) ) ERROR(("Cannot allocate.")); 

  // From grid/partition.c: used to determine which domains are on edge

  BEGIN_PRIMITIVE {
    static double *pvec =NULL, *e1vec =NULL, *e2vec =NULL, *cb1vec =NULL, *cb2vec =NULL;
    static double *gpvec=NULL, *ge1vec=NULL, *ge2vec=NULL, *gcb1vec=NULL, *gcb2vec=NULL;
    static double *psum, *gpsum, norm;
    FileIO       fileIO;
    FileIOStatus status;
    static char fname_poynting_sum_l[]="poynting/poynting_sum_l",
                fname_poynting_sum_r[]="poynting/poynting_sum_r",
                fname_poynting_sum[]  ="poynting/poynting_sum",
                fname_poynting[]      ="poynting/poynting"    ,
                fname_e1[]            ="poynting/e1"          ,
                fname_e2[]            ="poynting/e2"          ,
                fname_cb1[]           ="poynting/cb1"         ,
                fname_cb2[]           ="poynting/cb2"         ;
    static uint64_t stride;  // Force tmp variable in seek() to be uint64_t and not int!
    static int sum_stride, initted=0;
    int ncells_x = int(grid->nx*global->topology_x);
    int ncells_z = int(grid->nz*global->topology_z);

    if ( !initted ) {
      if ( global->write_backscatter_only ) {
        stride     = uint64_t(ncells_z); // x faces
        sum_stride = 1;
      } else {
        stride     = uint64_t( 2*( ncells_x + ncells_z ) );
        sum_stride = 4;
      }
      ALLOCATE( psum,   sum_stride, double ); ALLOCATE( gpsum,   sum_stride, double );
      ALLOCATE( pvec,   stride,     double ); ALLOCATE( gpvec,   stride,     double );
      ALLOCATE( e1vec,  stride,     double ); ALLOCATE( ge1vec,  stride,     double );
      ALLOCATE( e2vec,  stride,     double ); ALLOCATE( ge2vec,  stride,     double );
      ALLOCATE( cb1vec, stride,     double ); ALLOCATE( gcb1vec, stride,     double );
      ALLOCATE( cb2vec, stride,     double ); ALLOCATE( gcb2vec, stride,     double );
      norm = 1.0 / (grid->cvac*grid->cvac*global->emax*global->emax);
      initted=1;
    }

    // FIXME:  Rewrite using permutation-symmetric macros
    // FIXME:  Don't we have to do something special for mp on Roadrunner? 

    // Note:  Will dump core if we dump poynting by mistake on time t=0  

    if ( step()>0 && should_dump(poynting) ) {
      uint64_t ii;  // To shut the compiler up.
      int i, j, k, k1, k2, ix, iy, iz, skip, index;

      // Initialize arrays to zero
      for ( ii=0; ii<stride; ++ii ) {
        pvec[ii]    = 0;
        e1vec[ii]   = 0;
        e2vec[ii]   = 0;
        cb1vec[ii]  = 0;
        cb2vec[ii]  = 0;
        gpvec[ii]   = 0;
        ge1vec[ii]  = 0;
        ge2vec[ii]  = 0;
        gcb1vec[ii] = 0;
        gcb2vec[ii] = 0;
      }
      RANK_TO_INDEX( int(rank()), ix, iy, iz );  // Get position of domain in global topology

      skip=0;

      // Lower x face
      if ( ix==0 ) {
        for ( j=0; j< grid->ny; ++j ) {
          for ( k=1; k<=grid->nz; ++k ) {
            float e1, e2, cb1, cb2;
            // In output, the 2D surface arrays A[j,k] are FORTRAN indexed: 
            // The j quantity varyies fastest, k, slowest. 
//          index = int(  ((iy*grid->ny) + j-1)  // FIXED 
            index = int(  ((iy*grid->ny) + j-0)
                        + ((iz*grid->nz) + k-1) * (grid->ny*global->topology_y)
                        + skip);
            k1  = INDEX_FORTRAN_3(1,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            k2  = INDEX_FORTRAN_3(2,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
            e1  = field(k2).ey;
            e2  = field(k2).ez;
            cb1 = 0.5*(field(k1).cby+field(k2).cby);
            cb2 = 0.5*(field(k1).cbz+field(k2).cbz);
            pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
            e1vec[index]  = e1;
            e2vec[index]  = e2;
            cb1vec[index] = cb1;
            cb2vec[index] = cb2;
          }
        }
      }

      if ( global->write_backscatter_only==0 ) {

        skip+=ncells_z;

        // Upper x face
        if ( ix==global->topology_x-1 ) {
          for ( j=0; j< grid->ny; ++j ) {
            for ( k=1; k<=grid->nz; ++k ) {
              float e1, e2, cb1, cb2;
              index = int(  ((iy*grid->ny) + j-0)
                          + ((iz*grid->nz) + k-1) * (grid->ny*global->topology_y)
                          + skip);
              k1  = INDEX_FORTRAN_3(grid->nx-1,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
              k2  = INDEX_FORTRAN_3(grid->nx  ,j+1,k+1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
              e1  = field(k2).ey;
              e2  = field(k2).ez;
              cb1 = 0.5*(field(k1).cby+field(k2).cby);
              cb2 = 0.5*(field(k1).cbz+field(k2).cbz);
              pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
              e1vec[index]  = e1;
              e2vec[index]  = e2;
              cb1vec[index] = cb1;
              cb2vec[index] = cb2;
            }
          }
        }
        skip+=ncells_z;


        if ( global->write_side_scatter ) {
          // Lower z face
          if ( iz==0 ) {
            for ( j=0; j< grid->nx; ++j ) {
              for ( k=1; k<=grid->ny; ++k ) {
                float e1, e2, cb1, cb2;
                index = int(  ((ix*grid->nx) + j-0)  
                            + ((iy*grid->ny) + k-1) * (grid->nx*global->topology_x)
                            + skip);
                k1  = INDEX_FORTRAN_3(j+1,k+1,1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                k2  = INDEX_FORTRAN_3(j+1,k+1,2,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                e1  = field(k2).ex;
                e2  = field(k2).ey;
                cb1 = 0.5*(field(k1).cbx+field(k2).cbx);
                cb2 = 0.5*(field(k1).cby+field(k2).cby);
                pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
                e1vec[index]  = e1;
                e2vec[index]  = e2;
                cb1vec[index] = cb1;
                cb2vec[index] = cb2;
              }
            }
          }
          skip+=ncells_x;

          // Upper z face
          if ( iz==global->topology_z-1 ) {
            for ( j=0; j< grid->nx; ++j ) {
              for ( k=1; k<=grid->ny; ++k ) {
                float e1, e2, cb1, cb2;
                index = int(  ((ix*grid->nx) + j-0)
                            + ((iy*grid->ny) + k-1) * (grid->nx*global->topology_x)
                            + skip);
                k1  = INDEX_FORTRAN_3(j+1,k+1,grid->nz-1,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                k2  = INDEX_FORTRAN_3(j+1,k+1,grid->nz  ,0,grid->nx+1,0,grid->ny+1,0,grid->nz+1);
                e1  = field(k2).ex;
                e2  = field(k2).ey;
                cb1 = 0.5*(field(k1).cbx+field(k2).cbx);
                cb2 = 0.5*(field(k1).cby+field(k2).cby);
                pvec[index]   = ( e1*cb2-e2*cb1 )*norm;
                e1vec[index]  = e1;
                e2vec[index]  = e2;
                cb1vec[index] = cb1;
                cb2vec[index] = cb2;
              } // for
            } // for
          } // if                        
        } // if
      } // if

      if ( global->write_poynting_sum ) {
        // Sum poynting flux on surface
        skip=0;

        // Lower x face
        for ( i=0, psum[0]=0; i<ncells_z; ++i ) psum[0]+=pvec[i+skip];
        if ( global->write_backscatter_only==0 ) {
          // Upper x face
          skip+=ncells_z;
          for ( i=0, psum[1]=0; i<ncells_z; ++i ) psum[1]+=pvec[i+skip];
          if ( global->write_side_scatter ) {
            // Lower z face
            skip+=ncells_z;
            for ( i=0, psum[2]=0; i<ncells_x; ++i ) psum[2]+=pvec[i+skip];
            // Upper z face
            skip+=ncells_x;
            for ( i=0, psum[3]=0; i<ncells_x; ++i ) psum[3]+=pvec[i+skip];
          }
        }

        // Only do the mpi when we are summing with side scatter
        if ( global->write_backscatter_only==0 && global->write_side_scatter ) {
          // Sum over all surfaces
          mp_allsum_d(psum, gpsum, sum_stride);

          // Divide by number of mesh points summed over
          gpsum[0] /= ncells_z;
          if ( global->write_backscatter_only==0 ) {
            gpsum[1] /= ncells_z;
            gpsum[2] /= ncells_x;
            gpsum[3] /= ncells_x;
          } // if

          // Write summed Poynting data
          if ( rank()==0 ) {
            status=fileIO.open( fname_poynting_sum,
                                (step()==global->poynting_interval ? io_write : io_read_write) );
            if ( status==fail ) ERROR(("Could not open file."));
            fileIO.seek( uint64_t(sum_stride*(step()/global->poynting_interval-1)*sizeof(double)),
                         SEEK_SET );
            fileIO.write( gpsum, sum_stride );
            fileIO.close();
          } // if

        } else {   // no MPI collectives if either write_backscatter_only is 1 
                   // or write_side_scatter is 0

          // use local sums instead of global mpi-gathered sums
          gpsum[0] = psum[0]/ncells_z;
          gpsum[1] = psum[1]/ncells_z;
          gpsum[2] = psum[2]/ncells_x;
          gpsum[3] = psum[3]/ncells_x;

          // Write summed Poynting data for leftmost processor
          if ( rank()==0 ) {
            status=fileIO.open( fname_poynting_sum_l,
                                (step()==global->poynting_interval ? io_write : io_read_write) );
            if ( status==fail ) ERROR(("Could not open file."));
            fileIO.seek( uint64_t(sum_stride*(step()/global->poynting_interval-1)*sizeof(double)),
                         SEEK_SET );
            fileIO.write( gpsum, sum_stride );
            fileIO.close();
          } // if

          // Write summed Poynting data for rightmost processor
          if ( global->write_backscatter_only == 0 ) {
            if ( rank()==nproc()-1 ) {
              status=fileIO.open( fname_poynting_sum_r,
                                  (step()==global->poynting_interval ? io_write : io_read_write) );
              if ( status==fail ) ERROR(("Could not open file."));
              fileIO.seek( uint64_t(sum_stride*(step()/global->poynting_interval-1)*sizeof(double)),
                           SEEK_SET );
              fileIO.write( gpsum, sum_stride );
              fileIO.close();
            } // if
          } // if

        } // if/else

      } // if

      // Write sum data to screen
      if ( rank()==0 )
        sim_log_local("** step = "<<step()<<" Lower x Poynting = "<<gpsum[0]);  // Dump data to stdout

      if ( global->write_backscatter_only==0 ) {

        if ( rank()==nproc()-1 )
          sim_log_local("**        "<<step()<<" Upper x Poynting = "<<gpsum[1]);  // Dump data to stdout

        if ( global->write_side_scatter ) {
          if ( rank()==0 ) {
            sim_log_local("**        "<<step()<<" Lower z Poynting = "<<gpsum[2]);  // Dump data to stdout
            sim_log_local("**        "<<step()<<" Upper z Poynting = "<<gpsum[3]);  // Dump data to stdout
          } // if
        } // if

      } // if


      // FIXME: Change field and poynting writes on surfaces to have the
      // option of turning off side scatter.


      // FIXME:  Is the paranoia of explicit casts inside fileIO.seek() necessary?  


      if ( global->write_poynting_faces ) {
        // Sum across all processors to get quantities on each surface, then write from proc 0 
        mp_allsum_d(pvec,   gpvec,   stride);
        if ( rank()==0 ) {
          status=fileIO.open( fname_poynting,
                              (step()==global->poynting_interval ? io_write : io_read_write) );
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO.seek( stride * uint64_t(step()/global->poynting_interval-1)
                              * uint64_t(sizeof(double)),
                       SEEK_SET );
          fileIO.write( gpvec, stride );
          fileIO.close();
        }
      }

      if ( global->write_eb_faces ) {
        // Sum across all processors to get quantities on each surface, then write from proc 0 
        mp_allsum_d(e1vec,  ge1vec,  stride);
        mp_allsum_d(e2vec,  ge2vec,  stride);
        mp_allsum_d(cb1vec, gcb1vec, stride);
        mp_allsum_d(cb2vec, gcb2vec, stride);

        if ( ix==0 ) {
          // Write e1 face data
          status=fileIO.open( fname_e1,
                              (step()==global->poynting_interval ? io_write : io_read_write) );
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO.seek( stride * uint64_t(step()/global->poynting_interval-1)
                              * uint64_t(sizeof(double)),
                       SEEK_SET );
          fileIO.write( ge1vec, stride );
          fileIO.close();

          // Write e2 data
          status=fileIO.open( fname_e2,
                              (step()==global->poynting_interval ? io_write : io_read_write) );
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO.seek( stride * uint64_t(step()/global->poynting_interval-1)
                              * uint64_t(sizeof(double)),
                       SEEK_SET );
          fileIO.write( ge2vec, stride );
          fileIO.close();

          // Write cb1 data
          status=fileIO.open( fname_cb1,
                              (step()==global->poynting_interval ? io_write : io_read_write) );
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO.seek( stride * uint64_t(step()/global->poynting_interval-1)
                              * uint64_t(sizeof(double)),
                       SEEK_SET );
          fileIO.write( gcb1vec, stride );
          fileIO.close();

          // Write cb2 Poynting data
          status=fileIO.open( fname_cb2,
                              (step()==global->poynting_interval ? io_write : io_read_write) );
          if ( status==fail ) ERROR(("Could not open file."));
          fileIO.seek( stride * uint64_t(step()/global->poynting_interval-1)
                              * uint64_t(sizeof(double)),
                       SEEK_SET );
          fileIO.write( gcb2vec, stride );
          fileIO.close();
        }
      }
    } // if
  } END_PRIMITIVE;
# endif // switch for poynting diagnostic 


  // NEW VELOCITY SPACE DIAGNOSTIC ========================================================================

#if 1
  // -------------------------------------------------------------------------
  // Diagnostic to write a 2d vx, vz binned velocity distribution 
  // Note that we have converted, using the relativistic gammas, from momentum
  // (which the code uses) to velocity prior to binning. 
  //
#define NVX 100
#define NVZ 100
/* Set PZMASK to an appropriate value */
// PZMAX is the distance in simulation units on each side from line z = fixed value in sim units
#define PZMAX 0.5*global->Lz
// distance in sim units
#define PZCENTER (0)  // at 0 micron
#define PMASK ((pz-PZCENTER)*(pz-PZCENTER)<PZMAX*PZMAX)
#define PREPARE_VELOCITY_SPACE_DATA(SUFF, NAME)                           \
    {                                                                     \
      species_t *sp;                                                      \
      for (int i=0; i<NVX; ++i)                                           \
        for (int j=0; j<NVZ; ++j)                                         \
	  f_##SUFF[i*NVZ+j]=0;                                            \
      sp = find_species_name(NAME, species_list);                         \
      for (int ip=0; ip<sp->np; ip++) {                                   \
	particle_t *p=&sp->p[ip];                                         \
        /* Lots of un-used stuff commented because PMASK only has pz */           \
	int nxp2=grid->nx+2;                                              \
	int nyp2=grid->ny+2;                                              \
	/* int nzp2=grid->nz+2;    */                                     \
	/* Turn index i into separate ix, iy, iz indices */               \
	int iz = p->i/(nxp2*nyp2);                                        \
	/* int iy = (p->i-iz*nxp2*nyp2)/nxp2;  */                         \
	/* int ix = p->i-nxp2*(iy+nyp2*iz); */                            \
	/* Compute real particle position from relative coords and grid data */ \
	/* double px = grid->x0+((ix-1)+(p->dx+1)*0.5)*grid->dx; */       \
	/* double py = grid->y0+((iy-1)+(p->dy+1)*0.5)*grid->dy; */       \
	double pz = grid->z0+((iz-1)+(p->dz+1)*0.5)*grid->dz;             \
	float invgamma=1/sqrt(1+p->ux*p->ux+p->uy*p->uy+p->uz*p->uz);     \
	float vx=p->ux*grid->cvac*invgamma;                               \
	float vz=p->uz*grid->cvac*invgamma;                               \
	long ivx=long(vx/dvx_##SUFF+(NVX/2));                             \
	long ivz=long(vz/dvz_##SUFF+(NVZ/2));                             \
	if ( abs(ivx)<NVX && abs(ivz)<NVZ && PMASK ) f_##SUFF[ivx*NVZ+ivz]+=p->w;  \
      }                                                                   \
    }

#define INCLUDE_VELOCITY_HEADER 0
#if INCLUDE_VELOCITY_HEADER 
#  define VELOCITY_HEADER_SIZE (2*sizeof(int)+2*sizeof(float))
#  define WRITE_VELOCITY_HEADER(SUFF)                               \
    {                                                               \
      int nvpts[2] = { NVX, NVZ };                                  \
      float dv[2];                                                  \
      dv[0] = dvx_##SUFF; dv[1] = dvz_##SUFF;                       \
      status = fileIO_##SUFF.open( fname_##SUFF, io_write );        \
      if ( status==fail ) ERROR(("Could not open file."));          \
      fileIO_##SUFF.write( &nvpts, 2 );                             \
      fileIO_##SUFF.write( &dv,    2 );                             \
      fileIO_##SUFF.close();                                        \
    }
#else
#  define VELOCITY_HEADER_SIZE 0
#  define WRITE_VELOCITY_HEADER(SUFF) 
#endif

// NOTE:  WE DO NOT WRITE ON TIME STEP 0! (see -1 in seek())
#define DUMP_VELOCITY_DATA(SUFF,LEN,HEADER_SIZE)                    \
    {                                                               \
      status = fileIO_##SUFF.open( fname_##SUFF,                    \
                                   ( ( INCLUDE_VELOCITY_HEADER==0 && \
                                       step()==global->velocity_interval ) \
                                     ? io_write                     \
                                     : io_read_write ) );           \
      if ( status==fail ) ERROR(("Could not open file."));          \
      fileIO_##SUFF.seek( uint64_t( HEADER_SIZE +                   \
                                    (step()/global->velocity_interval-1) \
                                    * LEN * sizeof(float)) ,        \
                          SEEK_SET );                               \
      fileIO_##SUFF.write( f_##SUFF, LEN );                         \
      fileIO_##SUFF.close();                                        \
    }

//  if ( 1 ) {
  if ( 0 ) {
    float        f_e[NVX*NVZ], f_He[NVX*NVZ], f_H[NVX*NVZ]; 
    float        vmax_e  = 10*global->vthe,    dvx_e,  dvz_e;     
    float        vmax_He = 10*global->vthi_He, dvx_He, dvz_He; 
    float        vmax_H  = 10*global->vthi_H,  dvx_H,  dvz_H ; 
    FileIO       fileIO_e, fileIO_He, fileIO_H; 
    FileIOStatus status; 
    static char  fname_e[BUFLEN], fname_He[BUFLEN], fname_H[BUFLEN]; 
    dvx_e  = dvz_e  = 2*vmax_e /NVX;
    dvx_He = dvz_He = 2*vmax_He/NVX;
    dvx_H  = dvz_H  = 2*vmax_H /NVX;
    sprintf( fname_e,  "velocity/velocity_e.%i",  (int)rank() );   
    sprintf( fname_He, "velocity/velocity_He.%i", (int)rank() );   
    sprintf( fname_H,  "velocity/velocity_H.%i",  (int)rank() );   
    if ( !step() ) {
      WRITE_VELOCITY_HEADER(e); 
      WRITE_VELOCITY_HEADER(He); 
      WRITE_VELOCITY_HEADER(H); 
    }

    // NOTE: We don't write on time step 0, as per comment above. 
    if ( step()!=0 && (step()%global->velocity_interval)==0 ) {
      PREPARE_VELOCITY_SPACE_DATA(e,  "electron"); 
      PREPARE_VELOCITY_SPACE_DATA(He, "He");
      PREPARE_VELOCITY_SPACE_DATA(H,  "H");
      DUMP_VELOCITY_DATA(e,  NVX*NVZ, VELOCITY_HEADER_SIZE); 
      DUMP_VELOCITY_DATA(He, NVX*NVZ, VELOCITY_HEADER_SIZE); 
      DUMP_VELOCITY_DATA(H,  NVX*NVZ, VELOCITY_HEADER_SIZE); 
    }

  }

#endif

#if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier(); 
  sim_log( "Velocity-space diagnostic done; going to restart." ); 
#endif 



# if DEBUG_SYNCHRONIZE_IO
  // DEBUG 
  mp_barrier(); 
  sim_log( "Velocity-space diagnostic done; going to restart." ); 
# endif 

  //--------------------------------------------------------------------------- 

  // Restart dump filenames are by default tagged with the current timestep.
  // If you do not want to do this add "0" to the dump_restart arguments. One
  // reason to not tag the dumps is if you want only one restart dump (the most
  // recent one) to be stored at a time to conserve on file space. Note: A
  // restart dump should be the _very_ _last_ dump made in a diagnostics
  // routine. If not, the diagnostics performed after the restart dump but
  // before the next timestep will be missed on a restart. Restart dumps are
  // in a binary format. Each rank makes a restart dump.

  // Restarting from restart files is accomplished by running the executable 
  // with "restart restart" as additional command line arguments.  The executable
  // must be identical to that used to generate the restart dumps or else 
  // the behavior may be unpredictable. 

  // Note:  restart is not currently working with custom boundary conditions
  // (such as the reflux condition) and has not been tested with emission 
  // turned on.  
  
  if ( step()>0 && should_dump(restart) ) {
    static const char * restart_fbase[2] = { "restart/restart0", "restart/restart1" };
    double dumpstart = uptime();

    // Employ turnstiles to partially serialize the writes
    // NUM_TURNSTILES is define above
    BEGIN_TURNSTILE(NUM_TURNSTILES) {
      checkpt( restart_fbase[global->rtoggle], 0 );
    } END_TURNSTILE;

    double dumpelapsed = uptime() - dumpstart;

    sim_log("Restart duration "<<dumpelapsed);

    global->rtoggle^=1;
  }

  // Shut down simulation when wall clock time exceeds global->quota_sec.
  // Note that the mp_elapsed() is guaranteed to return the same value for all
  // processors (i.e., elapsed time on proc #0), and therefore the abort will
  // be synchronized across processors.

  if ( step()>0 && global->quota_check_interval && (step()%global->quota_check_interval)==0 ) {
    if ( uptime() > global->quota_sec ) {

      // Employ turnstiles to partially serialize the writes
      // NUM_TURNSTILES is define above
      BEGIN_TURNSTILE(NUM_TURNSTILES) {
        checkpt( "restart/restart", 0 );
      } END_TURNSTILE;

      sim_log( "Restart dump restart completed." );
      sim_log( "Allowed runtime exceeded for this job.  Terminating." );
      mp_barrier(); // Just to be safe
      halt_mp();
      exit(0);
    }
  }

#if DEBUG_SYNCHRONIZE_IO
  // DEBUG
  mp_barrier();
  sim_log( "All diagnostics done in begin_diagnostics" );
#endif
#endif 
  
//#ifdef PARAVIEW_DUMP  
//  /*--------------------------------------------------------------------------
//   * NOTE: YOU CANNOT DIRECTLY USE C FILE DESCRIPTORS OR SYSTEM CALLS ANYMORE
//   *
//   * To create a new directory, use:
//   *
//   *   dump_mkdir("full-path-to-directory/directoryname")
//   *
//   * To open a file, use: FileIO class
//   *
//   * Example for file creation and use:
//   *
//   *   // declare file and open for writing
//   *   // possible modes are: io_write, io_read, io_append,
//   *   // io_read_write, io_write_read, io_append_read
//   *   FileIO fileIO;
//   *   FileIOStatus status;
//   *   status= fileIO.open("full-path-to-file/filename", io_write);
//   *
//   *   // formatted ASCII  output
//   *   fileIO.print("format string", varg1, varg2, ...);
//   *
//   *   // binary output
//   *   // Write n elements from array data to file.
//   *   // T is the type, e.g., if T=double
//   *   // fileIO.write(double * data, size_t n);
//   *   // All basic types are supported.
//   *   fileIO.write(T * data, size_t n);
//   *
//   *   // close file
//   *   fileIO.close();
//   *------------------------------------------------------------------------*/
//  
//  /*--------------------------------------------------------------------------
//   * Data output directories
//   * WARNING: The directory list passed to "global_header" must be
//   * consistent with the actual directories where fields and species are
//   * output using "field_dump" and "hydro_dump".
//   *
//   * DIRECTORY PATHES SHOULD BE RELATIVE TO
//   * THE LOCATION OF THE GLOBAL HEADER!!!
//   *------------------------------------------------------------------------*/
//  
//  
//  /*--------------------------------------------------------------------------
//   * Normal rundata dump
//   *------------------------------------------------------------------------*/
//  if(step()==0) {
//    dump_mkdir("fields");
//    dump_mkdir("hydro");
//    dump_mkdir("rundata");
//    dump_mkdir("restart1");  // 1st backup
//    dump_mkdir("restart2");  // 2nd backup
//    dump_mkdir("particle");
//    
//    dump_grid("rundata/grid");
//    dump_materials("rundata/materials");
//    dump_species("rundata/species");
//    global_header("global", global->outputParams);
//  } // if
//  
//  /*--------------------------------------------------------------------------
//   * Normal rundata energies dump
//   *------------------------------------------------------------------------*/
//  if(should_dump(energies)) {
//    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
//  } // if
//  
//  /*--------------------------------------------------------------------------
//   * Field data output
//   *------------------------------------------------------------------------*/
//  
//  if(step() == 1 || should_dump(field)) field_dump(global->fdParams);
//  
//  /*--------------------------------------------------------------------------
//   * Electron species output
//   *------------------------------------------------------------------------*/
//  
//  if(should_dump(ehydro)) hydro_dump("electron", global->hedParams);
//  
//  /*--------------------------------------------------------------------------
//   * Ion species output
//   *------------------------------------------------------------------------*/
//  
//  if(should_dump(Hhydro)) hydro_dump("H", global->hHdParams);
//
//  if(should_dump(Hehydro)) hydro_dump("He", global->hHedParams);
//  
//  /*--------------------------------------------------------------------------
//   * Energy Spectrum Output
//   *------------------------------------------------------------------------*/
//  
//  //#include "energy.cxx"   // Subroutine to compute energy spectrum diagnostic
//  
//  //Vadim: 
//  //#include "dissipation.cxx"
//  //#include "Ohms_exp_all_v2.cxx"
//  
//  /*--------------------------------------------------------------------------
//   * Restart dump
//   *------------------------------------------------------------------------*/
///*
//  // jgw: cannot write a restart every step with this logic, which is really
//  //      only useful for debugging.
//  // Vadim:
//  if (step() && !(step()%global->restart_interval))
//    global->write_restart = 1; // set restart flag. the actual restart files
//                               // are written during the next step
//  else
//    if (global->write_restart) {
//      
//      global->write_restart = 0; // reset restart flag
//      
//      double dumpstart = uptime();
//      
//      if(!global->rtoggle) {
//	global->rtoggle = 1;
//	checkpt("restart1/restart", 0);
//      }
//      else {
//	global->rtoggle = 0;
//	checkpt("restart2/restart", 0);
//      } // if
//      
//      mp_barrier(  ); // Just to be safe
//      
//      double dumpelapsed = uptime() - dumpstart;
//      sim_log("Restart duration "<<dumpelapsed);
//      
//      //Vadim
//      if (rank()==0) {
//	
//        FileIO fp_restart_info;
//        if ( ! (fp_restart_info.open("latest_restart", io_write)==ok) )
//	  ERROR(("Cannot open file."));
//        if(!global->rtoggle) {
//          fp_restart_info.print("restart restart2/restart\n");
//        } else
//          fp_restart_info.print("restart restart1/restart\n");
//	
//        fp_restart_info.close();
//      }
//      
//    } // if
//*/ 
//  
//  // Dump particle data
//  char subdir[36];
//  if ( should_dump(eparticle) && step() !=0 &&
//       step() > 0*(global->field_interval)  ) {
//    sprintf(subdir,"particle/T.%d",step()); 
//    dump_mkdir(subdir);
//    sprintf(subdir,"particle/T.%d/eparticle",step()); 
//    dump_particles("electron",subdir);
//  }
//  
//  // Shut down simulation when wall clock time exceeds global->quota_sec. 
//  // Note that the mp_elapsed() is guaranteed to return the same value for all
//  // processors (i.e., elapsed time on proc #0), and therefore the abort will 
//  // be synchronized across processors. Note that this is only checked every
//  // few timesteps to eliminate the expensive mp_elapsed call from every
//  // timestep. mp_elapsed has an ALL_REDUCE in it!
///* 
//  //Vadim:
//  if  (( step()>0 && global->quota_check_interval>0
//	 && (step()&global->quota_check_interval)==0)
//       || (global->write_end_restart) ) {
//    if ( (global->write_end_restart) ) {
//		   
//      global->write_end_restart = 0; // reset restart flag
//      
//      sim_log( "Allowed runtime exceeded for this job.  Terminating....\n");
//      double dumpstart = uptime();
//      
//      checkpt("restart0/restart",0);
//      
//      mp_barrier(  ); // Just to be safe
//      sim_log( "Restart dump restart completed." );
//      double dumpelapsed = uptime() - dumpstart;
//      sim_log("Restart duration "<<dumpelapsed);
//      
//      //Vadim:
//      if (rank()==0) {
//	FileIO fp_restart_info;
//	if ( ! (fp_restart_info.open("latest_restart", io_write)==ok) )
//          ERROR(("Cannot open file."));
//	fp_restart_info.print("restart restart0/restart");
//	fp_restart_info.close();
//      }
//      
//      exit(0); // Exit or abort?
//    }
//    if( uptime( ) > global->quota_sec )   global->write_end_restart = 1;
//  }
//*/  
//#endif
}// end diagnostics


begin_particle_injection {
  // No particle injection for this simulation
}


begin_current_injection {
  // No current injection for this simulation
}

begin_particle_collisions {
  // No particle collisions for this simulation
}

begin_field_injection { 
  // Inject a light wave from lhs boundary with E aligned along y
  // Use scalar diffraction theory for the Gaussian beam source.  (This is approximate). 

  // For quiet startup (i.e., so that we don't propagate a delta-function noise
  // pulse at time t=0) we multiply by a constant phase term exp(i phi) where: 
  //   phi = k*global->xfocus+atan(h)    (3d) 

  // Inject from the left a field of the form ey = e0 sin( omega t )

  if ( grid->x0==0 ) {               // Node is on left boundary
    double alpha      = grid->cvac*grid->dt/grid->dx;
    double emax_coeff = (4/(1+alpha))*global->omega_0*grid->dt*global->emax;
    double prefactor  = emax_coeff*sqrt(2/M_PI); 
    double t          = grid->dt*step(); 

    // Compute Rayleigh length in c/wpe
    double rl         = M_PI*global->waist*global->waist/global->lambda; 

    double pulse_shape_factor = 1;
    float pulse_length        = 70;  // units of 1/wpe
    float sin_t_tau           = sin(0.5*t*M_PI/pulse_length);
    pulse_shape_factor        = ( t<pulse_length ? sin_t_tau : 1 );
    double h                  = -global->xfocus/rl;   // Distance / Rayleigh length

// Original injection
//# define DY    ( grid->y0 + (iy-0.5)*grid->dy - global->ycenter )
//# define DZ    ( grid->z0 + (iz-1  )*grid->dz - global->zcenter )
//# define R2    ( DY*DY + DZ*DZ )                                   
//# define R2Z    ( DZ*DZ )                                   
//# define PHASE ( -global->omega_0*t + h*R2Z/(global->width*global->width) )
//# define MASK  ( R2Z<=pow(global->mask*global->width,2) ? 1 : 0 )

    // Loop over all Ey values on left edge of this node
//    for ( int iz=1; iz<=grid->nz+1; ++iz ) 
//      for ( int iy=1; iy<=grid->ny; ++iy )  
//        field(1,iy,iz).ey += prefactor 
//                             * cos(PHASE) 
////                           * exp(-R2/(global->width*global->width))  // 3D
//                             * exp(-R2Z/(global->width*global->width))
//                             * MASK * pulse_shape_factor;

// Kokkos Port
    int ny = grid->ny;
    int nz = grid->nz;
    float dy = grid->dy;
    float dz = grid->dz;
    float y0 = grid->y0;
    float z0 = grid->z0;
    float ycenter = global->ycenter;
    float zcenter = global->zcenter;
    float width = global->width;
    float mask = global->mask;
    float omega_0 = global->omega_0;
    float dy_offset = y0 - ycenter;
    float dz_offset = z0 - zcenter;
    int sy = grid->sy;
    int sz = grid->sz;

    k_field_t& kfield = field_array->k_f_d;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> left_edge({1, 1}, {nz+2, ny+1});
    Kokkos::parallel_for("Field injection", left_edge, KOKKOS_LAMBDA(const int iz, const int iy) {
        auto DY   =( (iy-0.5)*dy + dy_offset );
        auto DZ   =( (iz-1  )*dz + dz_offset );
        auto R2   =( DY*DY + DZ*DZ );
        auto R2Z  = ( DZ*DZ );                                   
        auto PHASE=( -omega_0*t + h*R2Z/(width*width) );
        auto MASK =( R2Z<=pow(mask*width,2) ? 1 : 0 );
        kfield(voxel(1, iy, iz, sy, sz), field_var::ey) += (prefactor * cos(PHASE) * exp(-R2Z/(width*width)) * MASK * pulse_shape_factor);
    });

  }
}

