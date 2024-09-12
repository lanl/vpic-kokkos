Running with field ionization
******************************

To enable field ionization in VPIC, you need to set the
`option(FIELD_IONIZATION "Enable field ionization" ON)` in
`CMakeLists.txt`. This is enabled by default in branch
`91-add-field-ionization`, but it is worthwhile to verify.

To enable field ionization in your input deck the following additions
and/or changes are required. Pro tip, these changes can be wrapped in
`#if defined(FIELD_IONIZATION)` to maintain a usable deck with or
without field ionization.


1. define_species(): The user needs to pass the number of ionization energies and quantum
   numbers (principle `qn`, magnetic `qm`, and angular `ql`) to define_species().

   .. code-block:: c++

		   int n_energy_I2 = 6; // Carbon for instance 
		   int qn = 1; // cannot be 0
		   int qm = 0;
		   int ql = 0;
                   ion_I2 = define_species("I2",
                                           n_energy_I2, //Number of ionization energies
                                           qn,qm,ql,  // quantum numbers
                                           m_I2_c,max_local_np_i2, max_local_nm_i2, 10, 0);

  `REQUIRED`: It is required that the electron species has the exact
  name `electron` and that it is the LAST defined species in the deck.


2. Define ionization energies

   The user should use the species parameter sp->ionization_energy to define the ionizable species.
   Ionizable species will have non-zero values in eV. This parameter needs to be set in
   the input deck after the species is defined (see above).

   Example of ionizable species:

   .. highlight:: c++
   .. code-block:: c++

		     double ionization_energy_I2_values[] = {11.26030, 24.38332, 47.8878, 64.4939, 392.087, 489.99334}; // in eV 
                     for (int i = 0; i < num_elements_I2; ++i) {
                         ion_I2->ionization_energy[i] = ionization_energy_I2_values[i];
                     }

   Example of unionizable species (set the ionization energy to 0):

   .. highlight:: c++
   .. code-block:: c++

                     electron->ionization_energy[0] = 0;



3. inject_particle: need to give charge

   When field ionization is enabled, charge is in the `particle data`
   not the `species data`.

   .. code-block:: c++

                  inject_particle( ion_I2, x, y, z,
                  		   normal( rng(0), 0, px_I2_norm ),
                                   normal( rng(0), 0, px_I2_norm ),
                                   normal( rng(0), 0, px_I2_norm ),
                                   fabs(qi_I2), // weight                                                                                                       
                                   Z_I2*e_c, // charge                                                                                                          
                                   0, 0 ); // age, update_rhob 


4. Grid parameters

   - When setting up the grid, users need to pass the laser wavelength in SI units via
     `grid->lambda` 
   - It is also required to define the conversions from code units to
     SI units for time, length, charge, and mass via: `grid->t_to_SI`, `grid->l_to_SI`, `grid->q_to_SI`, `grid->m_to_SI`


**Optional diagnostics**

- New variables are available for hydro output: number_density, maximum_charge, average_charge
- New time history output for the ionization states distribution can
  be added to `begin_diagnostics`. This outputs a text file with the
  number of particles in each charge state a given timestep. It is
  added to the deck similar to `dump_energies` via

  .. code-block:: c++

		  if( should_dump(ionization_states) ) {
                    dump_ionization_states( "rundata/ionization_states", step() ==0 ? 0 : 1 );
                  }

  This diagnostic also outputs a file with the number of macro-electrons in
  the reqested timesteps. This is specifically done for the species
  specifically named `electron`. This diagnostic doesn't create files
  for species that do not ionize ( i.e., ionization_energy set to 0)
