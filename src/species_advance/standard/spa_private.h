#ifndef _spa_private_h_
#define _spa_private_h_

#ifndef IN_spa
#error "Do not include spa_private.h; include species_advance.h"
#endif

#include "../species_advance.h"

///////////////////////////////////////////////////////////////////////////////
// advance_p_pipeline interface

typedef struct particle_mover_seg {

  MEM_PTR( particle_mover_t, 16 ) pm; // First mover in segment
  int max_nm;                         // Maximum number of movers
  int nm;                             // Number of movers used
  int n_ignored;                      // Number of movers ignored

  PAD_STRUCT( SIZEOF_MEM_PTR+3*sizeof(int) )

} particle_mover_seg_t;

typedef struct advance_p_pipeline_args {

  MEM_PTR( particle_t,           128 ) p0;       // Particle array
  MEM_PTR( particle_mover_t,     128 ) pm;       // Particle mover array
  MEM_PTR( accumulator_t,        128 ) a0;       // Accumulator arrays
  MEM_PTR( const interpolator_t, 128 ) f0;       // Interpolator array
  MEM_PTR( particle_mover_seg_t, 128 ) seg;      // Dest for return values
  MEM_PTR( const grid_t,         1   ) g;        // Local domain grid params

  float                                qdt_2mc;  // Particle/field coupling
  float                                cdt_dx;   // x-space/time coupling
  float                                cdt_dy;   // y-space/time coupling
  float                                cdt_dz;   // z-space/time coupling
  float                                qsp;      // Species particle charge

  int                                  np;       // Number of particles
  int                                  max_nm;   // Number of movers
  int                                  nx;       // x-mesh resolution
  int                                  ny;       // y-mesh resolution
  int                                  nz;       // z-mesh resolution
 
  PAD_STRUCT( 6*SIZEOF_MEM_PTR + 5*sizeof(float) + 5*sizeof(int) )

} advance_p_pipeline_args_t;

PROTOTYPE_PIPELINE( advance_p, advance_p_pipeline_args_t );

///////////////////////////////////////////////////////////////////////////////
// center_p_pipeline and uncenter_p_pipeline interface

typedef struct center_p_pipeline_args {

  MEM_PTR( particle_t,           128 ) p0;      // Particle array
  MEM_PTR( const interpolator_t, 128 ) f0;      // Interpolator array
  float                                qdt_2mc; // Particle/field coupling
  int                                  np;      // Number of particles

  PAD_STRUCT( 2*SIZEOF_MEM_PTR + sizeof(float) + sizeof(int) )

} center_p_pipeline_args_t;

#ifdef USE_LEGACY_PARTICLE_ARRAY
PROTOTYPE_PIPELINE( center_p,   center_p_pipeline_args_t );
#endif
PROTOTYPE_PIPELINE( center_p_dump,   center_p_pipeline_args_t );
PROTOTYPE_PIPELINE( uncenter_p, center_p_pipeline_args_t );

///////////////////////////////////////////////////////////////////////////////
// energy_p_pipeline interface

#ifdef USE_LEGACY_PARTICLE_ARRAY
typedef struct energy_p_pipeline_args {

  MEM_PTR( const particle_t,     128 ) p;       // Particle array
  MEM_PTR( const interpolator_t, 128 ) f;       // Interpolator array
  MEM_PTR( double,               128 ) en;      // Return values
  float                                qdt_2mc; // Particle/field coupling
  float                                msp;     // Species particle rest mass
  int                                  np;      // Number of particles

  PAD_STRUCT( 3*SIZEOF_MEM_PTR + 2*sizeof(float) + sizeof(int) )

} energy_p_pipeline_args_t;

PROTOTYPE_PIPELINE( energy_p, energy_p_pipeline_args_t );
#endif // USE_LEGACY_PARTICLE_ARRAY

#endif // _spa_private_h_
