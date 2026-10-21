// Binary-only deck (No HDF5 calls)

begin_globals {};

begin_initialization {
  double L = 1.0;
  define_units(1.0, 1.0);
  define_timestep(0.1);
  define_periodic_grid(0, 0, 0, L, L, L, 4, 4, 4, 1, 1, 1);
  define_material("vacuum", 1.0);
  define_field_array(NULL, 0.0);
  species_t * electron = define_species("electron", -1.0, 1.0, 128, -1, 10, 1);

  // Inject particles with deterministic pattern
  sim_log(" [Binary] Injecting test particles...");
  for (int iz = 1; iz <= grid->nz; ++iz) {
    for (int iy = 1; iy <= grid->ny; ++iy) {
      for (int ix = 1; ix <= grid->nx; ++ix) {
        double px = grid->x0 + ((double)ix - 0.5) * grid->dx;
        double py = grid->y0 + ((double)iy - 0.5) * grid->dy;
        double pz = grid->z0 + ((double)iz - 0.5) * grid->dz;
        
        long gid = iz * grid->ny * grid->nx + iy * grid->nx + ix;
        double u = (double)gid * 1.0e-9;
        
        inject_particle(electron, px, py, pz, u, u, u, 1.0, 0, 0);
      }
    }
  }
}

begin_diagnostics {
  if (step() == 0) {
    sim_log(" [Binary] Writing output...");
    
    dump_mkdir("field");
    dump_mkdir("ehydro");
    dump_mkdir("particle");

    // =====================================================================
    // BINARY WRITERS ONLY
    // =====================================================================

    // Set deterministic field pattern
    for (int iz = 0; iz <= grid->nz + 1; ++iz) {
      for (int iy = 0; iy <= grid->ny + 1; ++iy) {
        for (int ix = 0; ix <= grid->nx + 1; ++ix) {
          int cell_idx = iz * (grid->ny+2) * (grid->nx+2) + 
                         iy * (grid->nx+2) + ix;
          float base = 1000.0f + (cell_idx * 1.0e-8f);
          
          field(ix, iy, iz).ex = base + 0.1f;
          field(ix, iy, iz).ey = base + 0.2f;
          field(ix, iy, iz).ez = base + 0.3f;
          field(ix, iy, iz).cbx = base + 0.4f;
          field(ix, iy, iz).cby = base + 0.5f;
          field(ix, iy, iz).cbz = base + 0.6f;
        }
      }
    }
    field_array->last_copied = step();

    // Fields
    DumpParameters fdParams;
    fdParams.format = band;
    fdParams.stride_x = 1; fdParams.stride_y = 1; fdParams.stride_z = 1;
    sprintf(fdParams.baseDir, "field");
    sprintf(fdParams.baseFileName, "fields_bin");
    fdParams.output_vars = BitField(electric | magnetic);
    write_fields_binary(fdParams, field_array);

    // Zero fields for hydro safety
    for(int iz=0; iz<=grid->nz+1; ++iz) {
      for(int iy=0; iy<=grid->ny+1; ++iy) {
        for(int ix=0; ix<=grid->nx+1; ++ix) {
          field(ix,iy,iz).ex = 0; field(ix,iy,iz).ey = 0; field(ix,iy,iz).ez = 0;
          field(ix,iy,iz).cbx = 0; field(ix,iy,iz).cby = 0; field(ix,iy,iz).cbz = 0;
        }
      }
    }
    field_array->last_copied = step();

    // Hydro
    DumpParameters hedParams;
    hedParams.format = band;
    hedParams.stride_x = 1; hedParams.stride_y = 1; hedParams.stride_z = 1;
    sprintf(hedParams.baseDir, "ehydro");
    sprintf(hedParams.baseFileName, "hydro_bin");
    hedParams.output_vars = BitField(current_density | charge_density);
    write_hydro_binary(hedParams, hydro_array, "electron");

    // Particles (both modes)
    write_particles_binary("particle/particles_logical", "electron", {ParticleCoordinates::Logical, OutputLayout::PerRank});
    write_particles_binary("particle/particles_physical", "electron", {ParticleCoordinates::Physical, OutputLayout::PerRank});

    sim_log(" [Binary] Complete.");
    mp_barrier();
    halt_mp();
    exit(0);
  }
}

begin_particle_injection {}
begin_current_injection {}
begin_field_injection {}
begin_particle_collisions {}