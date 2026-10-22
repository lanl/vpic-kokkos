// giga.cxx - Scalability Verification Deck
// 1 Node, 32 Ranks
// Global: 16x16x8, Topology: 4x4x2 -> Local: 4x4x4 (Matches Nano density)

//#include "Write.h" // Ensure Write.cc functionality is available

begin_globals {
  DumpParameters fdParams;
  DumpParameters hedParams;
  std::vector<DumpParameters *> outputParams;
};

begin_initialization {
  // ------------------------------------------------------------------
  // 1. SYSTEM SETUP
  // ------------------------------------------------------------------
  double L = 1.0;
  
  // GLOBAL GRID DIMENSIONS
  // Targeted for 32 Ranks to yield 4x4x4 local grids
  double nx = 16, ny = 16, nz = 8; 
  
  // TOPOLOGY (4 * 4 * 2 = 32 Ranks)
  double px = 4, py = 4, pz = 2;

  define_units(1.0, 1.0); 
  define_timestep(0.1);   

  // Define Grid (Periodic)
  define_periodic_grid(0, 0, 0,    
                       L, L, L,    
                       nx, ny, nz, 
                       px, py, pz); // 32 Ranks

  // Define Material
  define_material("vacuum", 1.0);

  // Define Fields (Starts at 0.0)
  define_field_array(NULL, 0.0);

  // Define Species
  // Local grid is 4x4x4 = 64 cells. 
  // max_np = 128 is sufficient for 1 particle/cell injection.
  species_t * electron = define_species("electron", -1.0, 1.0, 128, -1, 10, 1);

  // ------------------------------------------------------------------
  // 2. DUMP CONFIGURATION
  // ------------------------------------------------------------------
  
  // Field Dump
  global->fdParams.format = band; 
  global->fdParams.stride_x = 1; global->fdParams.stride_y = 1; global->fdParams.stride_z = 1;
  sprintf(global->fdParams.baseDir, "field");
  sprintf(global->fdParams.baseFileName, "fields");
  global->fdParams.output_variables( electric | magnetic ); 
  global->outputParams.push_back(&global->fdParams);

  // Hydro Dump
  global->hedParams.format = band;
  global->hedParams.stride_x = 1; global->hedParams.stride_y = 1; global->hedParams.stride_z = 1;
  sprintf(global->hedParams.baseDir, "ehydro");
  sprintf(global->hedParams.baseFileName, "e_hydro");
  global->hedParams.output_variables( current_density | charge_density );
  global->outputParams.push_back(&global->hedParams);

  // ------------------------------------------------------------------
  // 3. DETERMINISTIC INJECTION
  // ------------------------------------------------------------------
  
  sim_log(" [Giga-Verif] Injecting Deterministic Particles...");
  
  // Loops over LOCAL physical cells (1..4)
  // Logic mirrors Nano.cxx exactly
  for (int iz = 1; iz <= grid->nz; ++iz) {
    for (int iy = 1; iy <= grid->ny; ++iy) {
        for (int ix = 1; ix <= grid->nx; ++ix) {
            
            // Physical Coordinates
            double px = grid->x0 + ((double)ix - 0.5) * grid->dx;
            double py = grid->y0 + ((double)iy - 0.5) * grid->dy;
            double pz = grid->z0 + ((double)iz - 0.5) * grid->dz;

            // Pattern: Rank + LocalIndex * 1e-9
            // This ensures every particle across the cluster has a unique-ish ID
            // derived from its specific rank and local voxel.
            long local_id = (long)iz * grid->ny * grid->nx + (long)iy * grid->nx + ix;
            double unique_val = (double)rank() + (double)local_id * 1.0e-9;
            
            // Inject
            inject_particle(electron, px, py, pz, 
                            unique_val, unique_val, unique_val, 
                            1.0, 0, 0);
        }
    }
  }
}

begin_diagnostics {
  if (step() == 0) {
      sim_log(" [Giga-Verif] Starting Dump Sequence on Rank " << rank());
      
      // Note: In MPI, only Rank 0 usually creates dirs, but dump_mkdir handles checking.
      dump_mkdir("field");
      dump_mkdir("ehydro");
      dump_mkdir("particle");
      dump_mkdir("rundata");

      global_header("global", global->outputParams);
      dump_grid("rundata/grid");
      dump_materials("rundata/materials");

      // ----------------------------------------------------------
      // 1. PARTICLES (Dump First to preserve Injection Pattern)
      // ----------------------------------------------------------
      // Legacy Dump
      dump_particles("electron", "particle/eparticle");
      // New Write (No Ghosts)
      write_particles_binary("particle/clean_particles", "electron");

      // ----------------------------------------------------------
      // 2. FIELDS (Set Pattern -> Dump -> Write)
      // ----------------------------------------------------------
      sim_log(" [Giga-Verif] Setting Fields...");
      for (int iz = 0; iz <= grid->nz + 1; ++iz) {
        for (int iy = 0; iy <= grid->ny + 1; ++iy) {
            for (int ix = 0; ix <= grid->nx + 1; ++ix) {
                // Pattern: 1000 + Rank + LocalIndex*1e-8
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
      
      // Mark as fresh so writes don't pull zeros from device
      field_array->last_copied = step();

      field_dump(global->fdParams); // Legacy
      write_fields_binary("field/clean_fields", field_array); // New Write

      // ----------------------------------------------------------
      // 3. HYDRO (Legacy -> New Write)
      // ----------------------------------------------------------
      // Legacy dump wipes hydro, accumulates, syncs, and writes.
      // We zero fields first for safety, though legacy dump handles it.
      
      sim_log(" [Giga-Verif] Zeroing fields for Hydro safety...");
      for(int iz=0; iz<=grid->nz+1; ++iz) {
        for(int iy=0; iy<=grid->ny+1; ++iy) {
           for(int ix=0; ix<=grid->nx+1; ++ix) {
               field(ix,iy,iz).ex = 0; field(ix,iy,iz).ey = 0; field(ix,iy,iz).ez = 0;
               field(ix,iy,iz).cbx = 0; field(ix,iy,iz).cby = 0; field(ix,iy,iz).cbz = 0;
           }
        }
      }
      field_array->last_copied = step(); // Keep sync state happy

      hydro_dump("electron", global->hedParams); // Legacy
      
      // New Write (Idempotent: Clears, Re-Accumulates, Syncs, Writes)
      write_hydro_binary("ehydro/clean_hydro", hydro_array, "electron"); 

      sim_log(" [Giga-Verif] Complete. Halt.");
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