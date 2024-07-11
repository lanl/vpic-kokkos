Running VPIC
============

Running on multiple GPUs
************************

To run on multiple GPU's, you can pass the flag: `--kokkos-num-devices=N` (which replaced `--kokkos-ndevices`), where
`N` specifies the number of GPUs (per node). This works by VPIC passing through
options it doesn't understand to Kokkos, and thus VPIC will generate a warning
as it thinks you may have tried to tell it something it doesn't understand...

Running with field ionization
******************************
Currently, field ionization is only available in the git branch `91-add-field-ionization`

To enable field ionization in VPIC, you need to set the
`option(FIELD_IONIZATION "Enable field ionization" ON)` in
`CMakeLists.txt`. This is enabled by default in branch
`91-add-field-ionization`, but it is worthwhile to verify.

To enable field ionization in your input deck the following additions
and/or changes are required. Pro tip, these changes can be wrapped in
`#if defined(FIELD_IONIZATION)` to maintain a usable deck with or
without field ionization.

1. Define ionization energies

   The user should use this parameter to define the ionizable species.
   Ionizable species will have non-zero values in eV. This parameter needs
   to be defined as a kokkos::View, which is passed to
   `define_species`, see below. Note, copying the ionization energy to
   the host and device allows the deck to be run on CPUs and GPUs.
   
   Example of ionizable species:

   .. highlight:: c++
   .. code-block:: c++
		   
		     const int num_elements_I2 = 6;
		     double ionization_energy_I2_values[] = {11.26030, 24.38332, 47.8878, 64.4939, 392.087, 489.99334}; // in eV 
                     Kokkos::View<double*,Kokkos::HostSpace> ionization_energy_I2("my_kokkos_view", num_elements_I2);                                                                              
                     for (int i = 0; i < num_elements_I2; ++i) {
                         ionization_energy_I2(i) = ionization_energy_I2_values[i];
                     }
                     Kokkos::View<double*> ionization_energy_I2_d("ionization_energy_I2_d",num_elements_I2);
                     Kokkos::deep_copy(ionization_energy_I2_d, ionization_energy_I2); // copy to device 

   Example of unionizable species (set the ionization energy to 0):

   .. highlight:: c++
   .. code-block:: c++

                     Kokkos::View<double*,Kokkos::HostSpace> ionization_energy_electron("my_kokkos_view", 1);
                     ionization_energy_electron(0) = 0; // in eV
                     Kokkos::View<double*>
		     ionization_energy_electron_d("ionization_energy_electron_d", 1);
                     Kokkos::deep_copy(ionization_energy_electron_d, ionization_energy_electron); // copy to device 
		   		     
     
2. define_species: need to give ionization energies and quantum
   numbers (principle `qn`, magnetic `qm`, and angular `ql`)
   
   .. code-block:: c++
		   
		   ion_I2 = define_species("I2",
		                           ionization_energy_I2_d, 
		                           qn,qm,ql,  // quantum numbers
				           m_I2_c,max_local_np_i2, max_local_nm_i2, 10, 0);

  `REQUIRED`: It is required that the electron species has the exact
  name `electron` and that it is the last defined species in the deck.					   
					   
					   
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
  for species that do not ionize ( i.e., ionization_energy set to 0).
