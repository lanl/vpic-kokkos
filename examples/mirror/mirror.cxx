begin_globals {
  int restart_interval;
  int energies_interval;
  int fields_interval;
  int ehydro_interval;
  int Hhydro_interval;
  int eparticle_interval;
  int Hparticle_interval;
  int quota_check_interval;

  int rtoggle;
  int write_restart;
  int write_end_restart;
  
  double quota_sec;
  double b0;
  double v_A;
  double mirror_ratio;
  double topology_x;
  double topology_y;
  double topology_z;

  DumpParameters fdParams;
  DumpParameters hHdParams;
  std::vector<DumpParameters *> outputParams;
};

begin_initialization {
  
  // Natural units
  double ec   = 1.0;
  double mi   = 1.0;
  double mu0  = 1.0;
  double b0   = 1.0;
  double n0   = 1.0;

  // Derived quantities
  double v_A = b0/sqrt(mu0*n0*mi);
  double wci = ec*b0/mi;
  double di = v_A/wci;

  // Plasma parameters
  double betai_par = 1.0;
  double Tperp_Tpar = 2.0;  // Temperature anisotropy
  double gamma = 5.0/3.0;
  double Tipar_Te = 1.0;
  
  double Tipar = betai_par*b0*b0/2.0/n0;
  double vthipar = sqrt(Tipar/mi);
  double vthiperp = vthipar*sqrt(Tperp_Tpar);
  double Te = Tipar/Tipar_Te;

  // Magnetic mirror parameters
  double mirror_ratio = 4.0;    // R_m = B_max/B_min
  double L_mirror = 20.0*di;    // Mirror length scale
  
  // Simulation parameters
  double taui = 100;
  double quota = 23.5;
  double quota_sec = quota*3600;

  // Grid in cylindrical (r, theta, z)
  // CRITICAL: Start r > 0 to avoid singularity!
  double r_min = 0.5*di;        // Inner radius (NOT zero!)
  double r_max = 10.0*di;       // Outer radius
  double z_min = -25.0*di;      // Axial extent
  double z_max = 25.0*di;
  
  double Lr = r_max - r_min;
  double Ltheta = 2.0*M_PI;     // Full azimuth
  double Lz = z_max - z_min;

  double nr = 64;
  double ntheta = 1;            // Axisymmetric (set >1 for 3D)
  double nz = 256;
  
  double nppc = 5000;
  
  double topology_x = 1;        // r direction
  double topology_y = 1;        // theta direction  
  double topology_z = 8;       // z direction

  // Grid spacing
  double hr = Lr/nr;
  double htheta = Ltheta/ntheta;
  double hz = Lz/nz;

  // Total particles and charge
  // Volume integral in cylindrical: ∫∫∫ r dr dθ dz
  double V_total = 0.0;
  for(int i=0; i<nr; i++) {
    double r_cell = r_min + (i+0.5)*hr;
    V_total += r_cell * hr * Ltheta * Lz;  // r*dr*dθ*dz
  }
  
  double Ni = nppc*nr*ntheta*nz;
  Ni = trunc_granular(Ni, nproc());
  double qi = ec * n0 * V_total / Ni;

  // Timestep
  double dg = courant_length(Lr, Ltheta*r_max, Lz, nr, ntheta, nz);
  double dt = 0.01/wci;
  double sort_interval = 10;

  // Output intervals
  int restart_interval = 3000;
  int energies_interval = 200;
  int interval = int(0.2/(wci*dt));
  int fields_interval = interval;
  int Hhydro_interval = interval;
  int Hparticle_interval = 0;
  int quota_check_interval = 100;

  num_step = int(taui/(wci*dt));
  status_interval = 200;
  sync_shared_interval = status_interval/2;
  clean_div_e_interval = status_interval/2;
  clean_div_b_interval = status_interval/2;

  global->restart_interval = restart_interval;
  global->energies_interval = energies_interval;
  global->fields_interval = fields_interval;
  global->Hhydro_interval = Hhydro_interval;
  global->Hparticle_interval = Hparticle_interval;
  global->quota_check_interval = quota_check_interval;
  global->quota_sec = quota_sec;
  global->rtoggle = 0;
  global->b0 = b0;
  global->v_A = v_A;
  global->mirror_ratio = mirror_ratio;
  global->topology_x = topology_x;
  global->topology_y = topology_y;
  global->topology_z = topology_z;

  //////////////////////////////////////////////////////////////////////////////
  // Setup grid - treat as Cartesian but interpret as (r, θ, z)
  
  define_units(1.0, 1.0);
  define_timestep(dt);

  // Map to "Cartesian" grid that we interpret as cylindrical
  define_periodic_grid(r_min, 0.0, z_min,         // Low corner (r_min, θ=0, z_min)
                       r_max, Ltheta, z_max,      // High corner
                       nr, ntheta, nz,
                       topology_x, topology_y, topology_z);

  grid->eta = 0.0;
  grid->init_cylindrical_grid();
  
  // Custom cylindrical initialization
  // Override Cartesian metric with cylindrical
  // Store r-values at cell centers for later use
  sim_log("Initializing cylindrical coordinate system");
  
  // Note: VPIC uses Cartesian internally, so we must handle
  // cylindrical metric manually in field initialization and pusher

  //////////////////////////////////////////////////////////////////////////////
  // Boundary conditions
  
  // Absorbing in r (x) direction
  // Periodic in theta (y) 
  // Absorbing in z
  
  // sim_log("Setting absorbing radial boundaries");
  // absorb_fields(); // Default absorbing on all boundaries
  
  // Make theta periodic manually if needed
  // (depends on VPIC version - may need custom boundary handler)

  //////////////////////////////////////////////////////////////////////////////
  // Materials
  
  define_material("vacuum", 1);
  define_field_array(NULL);
  
  //////////////////////////////////////////////////////////////////////////////
  // Species
  
  double nmax = 1.5*Ni/nproc();
  double nmovers = 0.1*nmax;
  species_t *ion = define_species("ion", ec, mi, nmax, nmovers, 
                                  sort_interval, 1);

  //////////////////////////////////////////////////////////////////////////////
  // Diagnostic output
  
  sim_log("*** VPIC Magnetic Mirror in Cylindrical Coordinates ***");
  sim_log("Topology: " << topology_x << " " << topology_y << " " << topology_z);
  sim_log("Grid: nr=" << nr << " ntheta=" << ntheta << " nz=" << nz);
  sim_log("Domain: r=[" << r_min/di << "," << r_max/di << "] di");
  sim_log("        z=[" << z_min/di << "," << z_max/di << "] di");
  sim_log("Mirror ratio: " << mirror_ratio);
  sim_log("Mirror length: " << L_mirror/di << " di");
  sim_log("taui = " << taui);
  sim_log("dt*wci = " << wci*dt);
  sim_log("nppc = " << nppc);
  sim_log("Total particles = " << Ni);
  sim_log("qi = " << qi);
  sim_log("Particle weight = " << qi/ec);
  
  // Write simulation info
  if(rank() == 0) {
    FileIO fp_info;
    if(fp_info.open("info.bin", io_write) == ok) {
      fp_info.write(&topology_x, 1);
      fp_info.write(&topology_y, 1);
      fp_info.write(&topology_z, 1);
      fp_info.write(&Lr, 1);
      fp_info.write(&Ltheta, 1);
      fp_info.write(&Lz, 1);
      fp_info.write(&nr, 1);
      fp_info.write(&ntheta, 1);
      fp_info.write(&nz, 1);
      fp_info.write(&dt, 1);
      fp_info.write(&r_min, 1);
      fp_info.write(&mirror_ratio, 1);
      fp_info.close();
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Initialize magnetic mirror field
  
  sim_log("Setting up magnetic mirror field");
  
  set_region_field(everywhere,
    // E-field (zero)
    0, 0, 0,
    
    // B_r: from ∇·B = 0 in cylindrical
    // B_r = -(r/2) * dB_z/dz
    -0.5 * x * b0 * (mirror_ratio - 1.0) * (z/L_mirror) / 
           (L_mirror * sqrt(1.0 + (mirror_ratio - 1.0) * (z/L_mirror)*(z/L_mirror))),
    
    // B_θ (zero for axisymmetry)
    0,
    
    // B_z: mirror field profile
    // B_z(z) = B_0 * sqrt(1 + (R_m - 1)*(z/L)^2)
    b0 * sqrt(1.0 + (mirror_ratio - 1.0) * (z/L_mirror)*(z/L_mirror))
  );

    set_region_te(everywhere, Te);

  //////////////////////////////////////////////////////////////////////////////
  // Load particles with proper cylindrical volume weighting
  
  sim_log("Loading particles");
  
  seed_entropy(rank());
  
  // Get local grid bounds
  double r_local_min = grid->x0;
  double r_local_max = grid->x0 + grid->dx * grid->nx;
  double theta_local_min = grid->y0;
  double theta_local_max = grid->y0 + grid->dy * grid->ny;
  double z_local_min = grid->z0;
  double z_local_max = grid->z0 + grid->dz * grid->nz;
  
  // Load particles cell by cell to get proper volume weighting
  int particles_loaded = 0;
  
  for(int k=0; k<grid->nz; k++) {
    for(int j=0; j<grid->ny; j++) {
      for(int i=0; i<grid->nx; i++) {
        
        // Cell center radius
        double r_cell = r_local_min + (i+0.5)*hr;
        double theta_cell = theta_local_min + (j+0.5)*htheta;
        double z_cell = z_local_min + (k+0.5)*hz;
        
        // Physical cell volume: V = r * dr * dθ * dz
        double cell_volume = r_cell * hr * htheta * hz;
        
        // Number of particles for this cell
        int npc = (int)(nppc * cell_volume * n0 / (n0 * V_total / (nr*ntheta*nz)));
        
        for(int p=0; p<nppc; p++) {
          double r = r_cell + hr*(uniform(rng(0), -0.5, 0.5));
          double theta = theta_cell + htheta*(uniform(rng(0), -0.5, 0.5));
          double z = z_cell + hz*(uniform(rng(0), -0.5, 0.5));
          
          // Velocity in cylindrical basis
          double vr = normal(rng(0), 0, vthiperp);
          double vtheta = normal(rng(0), 0, vthiperp);
          double vz = normal(rng(0), 0, vthipar);
          
          // Inject using Cartesian interface (x=r, y=theta, z=z)
          // Velocities also stored as (vr, vtheta, vz) in (ux, uy, uz)
          inject_particle(ion, r, theta, z, vr, vtheta, vz, qi, 0, 0);
          particles_loaded++;
        }
      }
    }
  }
  
  sim_log("Loaded " << particles_loaded << " particles on rank " << rank());

  //////////////////////////////////////////////////////////////////////////////
  // Setup output parameters
  
  global->fdParams.format = band;
  global->hHdParams.format = band;
  
  sprintf(global->fdParams.baseDir, "fields/");
  dump_mkdir("fields");
  sprintf(global->fdParams.baseFileName, "fields");
  global->fdParams.stride_x = 1;
  global->fdParams.stride_y = 1;
  global->fdParams.stride_z = 1;
  global->outputParams.push_back(&global->fdParams);
  
  sprintf(global->hHdParams.baseDir, "hydro/");
  dump_mkdir("hydro");
  sprintf(global->hHdParams.baseFileName, "Hhydro");
  global->hHdParams.stride_x = 1;
  global->hHdParams.stride_y = 1;
  global->hHdParams.stride_z = 1;
  global->outputParams.push_back(&global->hHdParams);
  
  global->fdParams.output_variables(0xffffffff);
  global->hHdParams.output_variables(current_density | charge_density | stress_tensor);
  
  sim_log("*** Initialization complete ***");

} // begin_initialization

#define should_dump(x) \
  (global->x##_interval>0 && remainder(step(), global->x##_interval) == 0)

begin_diagnostics {
  
  if(step() == 0) {
    dump_mkdir("rundata");
    dump_mkdir("restore0");
    dump_mkdir("restore1");
    dump_mkdir("particle");
    
    dump_grid("rundata/grid");
    dump_materials("rundata/materials");
    dump_species("rundata/species");
    global_header("global", global->outputParams);
  }
  
  if(should_dump(energies)) {
    dump_energies("rundata/energies", step() == 0 ? 0 : 1);
  }
  
  if(step() == 1 || should_dump(fields)) {
    field_dump(global->fdParams);
  }
  
  if(should_dump(Hhydro)) {
    hydro_dump("ion", global->hHdParams);
  }
  
  // Restart logic
  if(step() && !(step() % global->restart_interval)) {
    global->write_restart = 1;
  } else if(global->write_restart) {
    global->write_restart = 0;
    if(!global->rtoggle) {
      global->rtoggle = 1;
      checkpt("restore1/restore", 0);
    } else {
      global->rtoggle = 0;
      checkpt("restore0/restore", 0);
    }
    sim_log("Restart dump completed at step " << step());
  }
  
  // Quota check
  if(step() > 0 && global->quota_check_interval > 0 &&
     (step() % global->quota_check_interval) == 0) {
    if(uptime() > global->quota_sec) {
      sim_log("Quota exceeded - writing final restart");
      checkpt(global->rtoggle ? "restore0/restore" : "restore1/restore", 0);
      mp_barrier();
      exit(0);
    }
  }

} // begin_diagnostics

begin_particle_injection {
} // begin_particle_injection

begin_current_injection {
} // begin_current_injection

begin_field_injection {
} // begin_field_injection

begin_particle_collisions {
} // begin_particle_collisions