// Benchmark advance_p
//
// Written by:
//   Kevin J. Bowers, Ph.D.
//   Plasma Physics Group (X-1)
//   Applied Physics Division
//   Los Alamos National Lab
// March/April 2004 - Adapted into input deck format and heavily revised from
//                    earlier V4PIC versions

begin_globals {
};

begin_initialization {

  sim_log( "Injecting" );

  double dt       = 1e-17; //dt=10 //1e-16; //dt=100 //2e-18; //dt=1
  //double wpdt     = 0.2;
  double debye    = 1;
  double wp       = 1; //wpdt / dt;
  double vt       = 1e-5; //debye*wp;
  double v0_x     = 0.5e-5;
  double vt_fl    = 0.428485705712571e-5;
  double N        = 1; 
  double L        = debye;
  double dx       = L/N;

  double nppc     = 10000;
  double np       = nppc*N*N*N;
  double q        = 1;
  double m        = 1;
  double m_fl     = 0.01;

  double n0       = 1; //w*nppc/(dx*dx*dx);
  double w        = n0/np;
  double kT0      = vt*vt*m;
  double kT0_fl   = vt_fl*vt_fl*m_fl;
  
  // The probability a physical particle has a collision in a timestep dt
  // is ~pi bmax^2 |vr| dt n.  For self collisions of a Maxwellian species,
  // the relative velocities are Gaussian with twice the variance as the
  // particle velocities.  This gives the expectiation of |vr| as:
  //   |vr| = sqrt(8/pi) sqrt(2) vt = (4/sqrt(pi)) wp debye
  // Substituting into the probability above gives:
  //   p_coll = 4 sqrt(pi) bmax^2 wp dt debye nppc w / V
  // where nppc is the number of computational particles per voxel, w is
  // their physical weight and V is the volume of voxel.  Inverting gives
  // the below.

  double interval = 1;
  double p_coll   = 0.1;
  double bmax     = sqrt( p_coll / ( 4.*sqrt(M_PI)*wp*debye*dt*interval*n0 ) );
  double sample   = 1;

  int n_step = 10;

  define_units( 1, 1 );
  define_timestep( dt );
  define_periodic_grid( 0, 0, 0,   // Grid low corner
                        L, L, L,   // Grid high corner
                        N, N, N,   // Grid resolution
                        1, 1, 1 ); // Processor topology

  define_material( "vacuum", 1.0, 1.0, 0.0 );
  define_field_array();

  fluid_species_t * sp_fl = define_fluid_species( "test_fluid_species", q, m_fl );
  set_region_fluid( everywhere, "test_fluid_species", n0, kT0_fl, n0*kT0_fl ); //density, tmperature, pressure
  //sp_fl->init_k_fl(grid->nv);
  sp_fl->copy_to_device();
  species_t * sp = define_species( "test_species", q, m, np, 1, 0, 0 );
  repeat( np ) inject_particle( sp,
                                uniform( rng(0), grid->x0, grid->x1 ),
                                uniform( rng(0), grid->y0, grid->y1 ),
                                uniform( rng(0), grid->z0, grid->z1 ),
                                normal(  rng(0), v0_x, vt ),
                                normal(  rng(0), 0,    vt ),
                                normal(  rng(0), 0,    vt ),
				w, 0, 0 );
  sp->copy_to_device();
  interpolator_array->copy_to_device();
  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;

  Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos_nomove_ngp(
      particles,
      particles_i,
      hydro_array->k_h_d,
      interpolators_k,
      sp
  );
  hydro_array->copy_to_host(true); //print==true
  auto M_ln_Lamda = 10;
  auto mu= m*m/(m+m);
  auto dV=dx*dx*dx;
  double cvar0 = q*q*q*q*M_ln_Lamda/(8.0*M_PI);       //in SI: (e^4*n0*Lambda)/(8*pi*eps0^2*m_e^2*c^3)
  // --> internal operator multiplies by dt!
  int sort_interval = 1;  
  int ncoll = (int) sort_interval;  // How frequently to do collisions

#define FAK field_array->kernel
  FAK->clear_rhof( field_array );
  accumulate_rho_p( field_array, sp );
  set_region_te( everywhere, kT0_fl ); // Set the electron temperature
  FAK->hyb_init(field_array,-1);
  field_array->copy_to_device();
  auto &k_field = field_array->k_f_d;
  
  define_collision_op(lemons("lemons_coll", sp, sp_fl, cvar0, ncoll,field_array) );
  
/*
  define_collision_op( langevin( kT0, 1./dt, sp, entropy, 1*(int)interval ) );

  define_collision_op( large_angle_coulomb_fluid( "lac_fluid",
                                                  n0, 0,0,0, kT0, q, m,
                                                  sp, bmax,
                                                  entropy, 0*(int)interval ) );

  define_collision_op( large_angle_coulomb( "lac", sp, sp, bmax, entropy,
                                             sample, 0*(int)interval ) );

  // FIXME: Both seem somewhat unstable ... probably a bug in the momentum
  // transfer computation
  define_collision_op( hard_sphere_fluid( "hs_fluid",
                                          n0, 0,0,0, kT0, m, 0.5*bmax,
                                          sp, 0.5*bmax,
                                          entropy, 0*(int)interval ) );
  define_collision_op( hard_sphere( "hs", sp, 0.5*bmax, sp, 0.5*bmax, entropy,
                                    sample, 0*(int)interval ) );
*/
  //sim_log( "Colliding" );

  // Hack input VPIC internals

  // Don't pollute the benchmark with the sort time
  // sort_p( sp );
  sp->last_indexed = -1;
  // Warm up the caches
  /*
  repeat( 3 ) {
      apply_collision_op_list( collision_op_list, *kokkos_rng);
      Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
      accumulate_hydro_p_kokkos(
				particles,
				particles_i,
				hydro_array->k_h_d,
				interpolators_k,
				sp
				);
      hydro_array->copy_to_host();      
  }
  */
  // Do the benchmark
  double elapsed = wallclock();
  printf("#start collisions\n");
  repeat( n_step ) {
      printf("#step=%d\n",n_step-_remain);
      apply_collision_op_list( collision_op_list, *kokkos_rng );

      Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
      accumulate_hydro_p_kokkos_nomove_ngp(
      				particles,
      				particles_i,
      				hydro_array->k_h_d,
      				interpolators_k,
      				sp
      				);
      hydro_array->copy_to_host(true); //print_to_screen == true
      transfer_mom_en_src(k_field, sp_fl);
  }
  elapsed = wallclock() - elapsed;

  sim_log( (double)np*(double)n_step/elapsed/1e6 );

  exit(0);
}

begin_diagnostics {
}

begin_particle_injection {
}

begin_current_injection {
}

begin_field_injection {
}

begin_particle_collisions {
}


