// =============================================================================
// hdf5/write_correctness.cxx
// =============================================================================
// HDF5-Only Verification Deck (Single Rank)
// Tests: M2M (many-to-many) and M2O (many-to-one) modes
// Grid: 4x4x4 cells, 1 rank (matches binary test for easy comparison)
//
// This deck ONLY tests HDF5 writers - binary writers are tested separately
// in binary/write_correctness.cxx
// =============================================================================

begin_globals {
  // EMPTY! Avoid global struct serialization issues
};

begin_initialization {
  // ---------------------------------------------------------------------------
  // 1. SYSTEM SETUP
  // ---------------------------------------------------------------------------
  double L = 1.0;
  double nx = 4, ny = 4, nz = 4;  // Small grid for easy verification

  define_units(1.0, 1.0);
  define_timestep(0.1);
  
  // Single rank periodic grid
  define_periodic_grid(0, 0, 0, L, L, L, nx, ny, nz, 1, 1, 1);
  
  define_material("vacuum", 1.0);
  define_field_array(NULL, 0.0);

  // Species with enough capacity for 64 particles (1 per cell)
  species_t * electron = define_species("electron", -1.0, 1.0, 128, -1, 10, 1);

  // ---------------------------------------------------------------------------
  // 2. DETERMINISTIC PARTICLE INJECTION (Golden Pattern)
  // ---------------------------------------------------------------------------
  sim_log(" [HDF5-Single] Injecting Deterministic Particles...");
  
  for (int iz = 1; iz <= grid->nz; ++iz) {
    for (int iy = 1; iy <= grid->ny; ++iy) {
      for (int ix = 1; ix <= grid->nx; ++ix) {
        // Cell-centered position
        double px = grid->x0 + ((double)ix - 0.5) * grid->dx;
        double py = grid->y0 + ((double)iy - 0.5) * grid->dy;
        double pz = grid->z0 + ((double)iz - 0.5) * grid->dz;

        // Global index pattern (same as binary test)
        long global_id = (long)iz * grid->ny * grid->nx + 
                        (long)iy * grid->nx + ix;
        double unique_val = (double)global_id * 1.0e-9;
        
        // Inject with anisotropic velocity (ux=1x, uy=2x, uz=3x)
        inject_particle(electron, px, py, pz, 
                       unique_val, unique_val * 2.0, unique_val * 3.0, 
                       1.0, 0, 0);
      }
    }
  }
}

begin_diagnostics {
  if (step() == 0) {
    sim_log(" [HDF5-Single] Starting HDF5 Verification Dump...");
    
    // -------------------------------------------------------------------------
    // SETUP: Create Output Directories
    // -------------------------------------------------------------------------
    dump_mkdir("field");
    dump_mkdir("ehydro");
    dump_mkdir("particle");

    // =========================================================================
    // 1. PARTICLES - Both Modes
    // =========================================================================
    sim_log(" [HDF5-Single] Writing Particles...");

    // M2M Mode (one file per rank - here just rank 0)
    write_particles_hdf5("particle/test_particles_m2m", "electron", {ParticleCoordinates::Logical, OutputLayout::PerRank});

    // M2O Mode (single shared file - same output for 1 rank but tests the code path)
    write_particles_hdf5("particle/test_particles_m2o", "electron", {ParticleCoordinates::Logical, OutputLayout::Shared});

    // Physical output is stored once in directly XDMF-readable numeric arrays.
    write_particles_hdf5("particle/physical_m2m", "electron", {ParticleCoordinates::Physical, OutputLayout::PerRank});
    write_particles_hdf5("particle/physical_m2o", "electron", {ParticleCoordinates::Physical, OutputLayout::Shared});
    write_xdmf_timeseries("particle_series.xmf", {0}, "particle", "physical_m2m.electron");

    // =========================================================================
    // 2. FIELDS - Both Modes
    // =========================================================================
    sim_log(" [HDF5-Single] Setting Deterministic Fields...");
    
    // Set golden pattern: 1000 + CellIndex * 1e-8
    for (int iz = 0; iz <= grid->nz + 1; ++iz) {
      for (int iy = 0; iy <= grid->ny + 1; ++iy) {
        for (int ix = 0; ix <= grid->nx + 1; ++ix) {
          double cell_idx = (double)(iz * (grid->ny+2) * (grid->nx+2) + 
                                     iy * (grid->nx+2) + ix);
          double base_val = 1000.0 + (cell_idx * 1.0e-8);

          field(ix, iy, iz).ex  = base_val + 0.1;
          field(ix, iy, iz).ey  = base_val + 0.2;
          field(ix, iy, iz).ez  = base_val + 0.3;
          field(ix, iy, iz).cbx = base_val + 0.4;
          field(ix, iy, iz).cby = base_val + 0.5;
          field(ix, iy, iz).cbz = base_val + 0.6;
        }
      }
    }
    
    // Mark as fresh so HDF5 writers see the host data
    field_array->last_copied = step();

    // Configure field dump parameters
    DumpParameters fdParams;
    fdParams.format = band;
    fdParams.stride_x = 1;
    fdParams.stride_y = 1;
    fdParams.stride_z = 1;
    sprintf(fdParams.baseDir, "field");
    fdParams.output_vars = BitField(electric | magnetic);

    // M2M Mode
    sprintf(fdParams.baseFileName, "test_fields_m2m");
    write_fields_hdf5(fdParams, field_array, false);

    // M2O Mode
    sprintf(fdParams.baseFileName, "test_fields_m2o");
    write_fields_hdf5(fdParams, field_array, true);

    // =========================================================================
    // 3. HYDRO - Both Modes
    // =========================================================================
    sim_log(" [HDF5-Single] Zeroing fields for Hydro safety...");
    
    // Zero fields to avoid EÃ—B effects in hydro accumulation
    for (int iz = 0; iz <= grid->nz + 1; ++iz) {
      for (int iy = 0; iy <= grid->ny + 1; ++iy) {
        for (int ix = 0; ix <= grid->nx + 1; ++ix) {
          field(ix, iy, iz).ex = 0;
          field(ix, iy, iz).ey = 0;
          field(ix, iy, iz).ez = 0;
          field(ix, iy, iz).cbx = 0;
          field(ix, iy, iz).cby = 0;
          field(ix, iy, iz).cbz = 0;
        }
      }
    }
    field_array->last_copied = step();

    // Configure hydro dump parameters
    DumpParameters hedParams;
    hedParams.format = band;
    hedParams.stride_x = 1;
    hedParams.stride_y = 1;
    hedParams.stride_z = 1;
    sprintf(hedParams.baseDir, "ehydro");
    hedParams.output_vars = BitField(current_density | charge_density);

    // M2M Mode (physics accumulation happens inside write_hydro_hdf5)
    sprintf(hedParams.baseFileName, "test_hydro_m2m");
    write_hydro_hdf5(hedParams, hydro_array, "electron", false);

    // M2O Mode (re-accumulates physics)
    sprintf(hedParams.baseFileName, "test_hydro_m2o");
    write_hydro_hdf5(hedParams, hydro_array, "electron", true);

    // =========================================================================
    // CLEANUP & EXIT
    // =========================================================================
    sim_log(" [HDF5-Single] Verification dump complete. Halting.");
    mp_barrier();
    halt_mp();
    exit(0);
  }
}

// Stubs (no active physics injection)
begin_particle_injection {}
begin_current_injection {}
begin_field_injection {}
begin_particle_collisions {}