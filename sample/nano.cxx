// nano.cxx - Manual Verification Deck
// 2x2x1 Grid, Deterministic Pattern for Hex/Log comparison

begin_globals {
  // Required by dump.cc
  DumpParameters fdParams;
  DumpParameters hedParams;
  std::vector<DumpParameters *> outputParams;
};

begin_initialization {
  // ------------------------------------------------------------------
  // 1. SYSTEM SETUP
  // ------------------------------------------------------------------
  // Use a tiny 2x2x1 grid for manual hex verification
  double L = 1.0;
  double nx = 4, ny = 4, nz = 4;
  
  define_units(1.0, 1.0); // c=1, eps0=1
  define_timestep(0.1);   // Arbitrary dt

  // Define Grid (Periodic)
  define_periodic_grid(0, 0, 0,    // Low Corner
                       L, L, L,    // High Corner
                       nx, ny, nz, // Resolution
                       1, 1, 1);   // Topology (1 Rank)

  // Define Material (Required for dumps)
  define_material("vacuum", 1.0);

  // Define Fields (Allocates field_array->f)
  define_field_array(NULL, 0.0);

  // Define Species (Allocates hydro_array)
  // 4 particles max (1 per cell)
  species_t * electron = define_species("electron", -1.0, 1.0, 128, -1, 10, 1);

  // ------------------------------------------------------------------
  // 2. DUMP CONFIGURATION (Matching dump.cc requirements)
  // ------------------------------------------------------------------
  
  // Field Dump: Banded, Stride 1
  global->fdParams.format = band; 
  global->fdParams.stride_x = 1;
  global->fdParams.stride_y = 1;
  global->fdParams.stride_z = 1;
  sprintf(global->fdParams.baseDir, "field");
  sprintf(global->fdParams.baseFileName, "fields");
  // Output Ex, Ey, Ez, Bx, By, Bz (Bitmask)
  global->fdParams.output_variables( electric | magnetic ); 
  global->outputParams.push_back(&global->fdParams);

  // Hydro Dump: Banded, Stride 1
  global->hedParams.format = band;
  global->hedParams.stride_x = 1;
  global->hedParams.stride_y = 1;
  global->hedParams.stride_z = 1;
  sprintf(global->hedParams.baseDir, "ehydro");
  sprintf(global->hedParams.baseFileName, "e_hydro");
  // Output J (Current) and Rho (Charge)
  global->hedParams.output_variables( current_density | charge_density );
  global->outputParams.push_back(&global->hedParams);

  // ------------------------------------------------------------------
  // 3. DETERMINISTIC INJECTION (The Golden Pattern)
  // ------------------------------------------------------------------
  
  sim_log(" [Nano-Verif] Injecting Deterministic Particles...");
  
  // Loop Physical Cells (1..nx)
  for (int iz = 1; iz <= grid->nz; ++iz) {
    for (int iy = 1; iy <= grid->ny; ++iy) {
        for (int ix = 1; ix <= grid->nx; ++ix) {
            
            // Center of cell
            double px = grid->x0 + ((double)ix - 0.5) * grid->dx;
            double py = grid->y0 + ((double)iy - 0.5) * grid->dy;
            double pz = grid->z0 + ((double)iz - 0.5) * grid->dz;

            // Pattern: Rank + GlobalIndex * 1e-9
            long global_id = (long)iz * grid->ny * grid->nx + (long)iy * grid->nx + ix;
            double unique_val = (double)rank() + (double)global_id * 1.0e-9;
            
            // LOG for Manual Verification
            sim_log("INJECT: Cell(" << ix << "," << iy << ") ID=" << global_id 
                    << " ux=" << unique_val);

            // Inject (ux=uy=uz for Symmetry Check)
            inject_particle(electron, px, py, pz, 
                            unique_val, unique_val, unique_val, 
                            1.0, 0, 0);
        }
    }
  }
  /*
  sim_log(" [Nano-Verif] Overwriting Fields...");
  
  // Loop Physical + Ghost (0..nx+1)
  for (int iz = 0; iz <= grid->nz + 1; ++iz) {
    for (int iy = 0; iy <= grid->ny + 1; ++iy) {
        for (int ix = 0; ix <= grid->nx + 1; ++ix) {
            
            // Pattern: 1000 + Rank + Index*1e-8 + ComponentOffset
            double cell_idx = (double)(iz * (grid->ny+2) * (grid->nx+2) + 
                                       iy * (grid->nx+2) + ix);
            double base_val = 1000.0 + (double)rank() + (cell_idx * 1.0e-8);

            field(ix, iy, iz).ex  = base_val + 0.1;
            field(ix, iy, iz).ey  = base_val + 0.2;
            field(ix, iy, iz).ez  = base_val + 0.3;
            field(ix, iy, iz).cbx = base_val + 0.4;
            field(ix, iy, iz).cby = base_val + 0.5;
            field(ix, iy, iz).cbz = base_val + 0.6;
            
            // LOG First Physical slice for Manual Verification
            if (iz==1 && ix>0 && ix<=nx && iy>0 && iy<=ny) {
               sim_log("FIELD: Cell(" << ix << "," << iy << ") Ex=" << base_val + 0.1);
            }
        }
    }
  }
  */
}

begin_diagnostics {
  if (step() == 0) {
      sim_log(" [Nano-Verif] Starting Dump Sequence...");
      
      dump_mkdir("field");
      dump_mkdir("ehydro");
      dump_mkdir("particle");
      dump_mkdir("rundata");

      global_header("global", global->outputParams);
      dump_grid("rundata/grid");
      dump_materials("rundata/materials");

      // 1. Dump Particles FIRST (Fields are 0.0 -> No Kick -> Golden Pattern Preserved)
      dump_particles("electron", "particle/eparticle");
      write_particles_binary("particle/clean_particles", "electron"); // NEW WRITE

      // 2. Set Golden Pattern Fields for Field Dump
      sim_log(" [Nano-Verif] Setting Fields for Field Dump...");
      for (int iz = 0; iz <= grid->nz + 1; ++iz) {
        for (int iy = 0; iy <= grid->ny + 1; ++iy) {
            for (int ix = 0; ix <= grid->nx + 1; ++ix) {
                double cell_idx = (double)(iz * (grid->ny+2) * (grid->nx+2) + 
                                           iy * (grid->nx+2) + ix);
                double base_val = 1000.0 + (double)rank() + (cell_idx * 1.0e-8);

                field(ix, iy, iz).ex  = base_val + 0.1;
                field(ix, iy, iz).ey  = base_val + 0.2;
                field(ix, iy, iz).ez  = base_val + 0.3;
                field(ix, iy, iz).cbx = base_val + 0.4;
                field(ix, iy, iz).cby = base_val + 0.5;
                field(ix, iy, iz).cbz = base_val + 0.6;
            }
        }
      }

      // CRITICAL FIX: Tell VPIC that Host data is up-to-date.
      // This prevents field_dump from downloading 0.0s from the GPU/Device.
      field_array->last_copied = step();

      // 3. Dump Fields
      field_dump(global->fdParams);
      write_fields_binary("field/clean_fields", field_array); // NEW WRITE

      // 4. Zero Fields for Hydro Dump safety
      sim_log(" [Nano-Verif] Zeroing fields for Hydro safety...");
      for(int iz=0; iz<=grid->nz+1; ++iz) {
        for(int iy=0; iy<=grid->ny+1; ++iy) {
           for(int ix=0; ix<=grid->nx+1; ++ix) {
               field(ix,iy,iz).ex = 0; field(ix,iy,iz).ey = 0; field(ix,iy,iz).ez = 0;
               field(ix,iy,iz).cbx = 0; field(ix,iy,iz).cby = 0; field(ix,iy,iz).cbz = 0;
           }
        }
      }
      // Mark as fresh again so next dump doesn't pull old data (good practice)
      field_array->last_copied = step(); 
      
      // 5. Dump Hydro
      hydro_dump("electron", global->hedParams);
      write_hydro_binary("ehydro/clean_hydro", hydro_array, "electron"); // NEW WRITE

      sim_log(" [Nano-Verif] Complete. Halt.");
      mp_barrier();
      halt_mp();
      exit(0);
  }
}

// Stubs
begin_particle_injection {}
begin_current_injection {}
begin_field_injection {}
begin_particle_collisions {}