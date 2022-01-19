Particle Boundary Diagnostic
============================

The particle boundary diagnostic is a tool to record particle data when it interacts with a boundary.  Currently, it works only with absorbing boundaries, uses an ad hoc binary output, and interfaces with c postprocessors and python plotters that rely on the precise layout of the data.  This page should get you started making some plots.

Enabling the diagnostic in your deck
************************************

The `species_t` class now has a `pb_diagnostic` struct as a member which controls the particle boundary diagnostic for that species.  You must, in your deck, tell the diagnostic what values to write and finalize the diagnostic to enable it.

.. highlight:: c++
.. code-block:: c++

   species_t * electron = define_species("electron", -1.*e_c, m_e_c, max_local_np_e, max_local_nm_e, 20, 0);
   electron->pb_diag->write_ux = 1;
   electron->pb_diag->write_uy = 1;
   electron->pb_diag->write_uz = 1;
   electron->pb_diag->write_weight = 1;
   electron->pb_diag->write_posx = 1;
   electron->pb_diag->write_posy = 1;
   electron->pb_diag->write_posz = 1;
   finalize_pb_diagnostic(electron);

This will save 7 floats every time an electron hits an absorbing boundary.  The data are first saved to a buffer in main memory, and will write to disk when the buffer is full.  The buffers won't all fill up at the same time, so you should manually call the writer at a regular interval in your deck in `user_diagnostics`.  Doing this when you write the fields is usually a good choice.

.. code-block:: c++

   for(species_t *sp=species_list; sp; sp=sp->next){
       if(sp->pb_diag) pbd_buff_to_disk(sp->pb_diag);
   }

You must also call the writer on the last time step to get all of the particles actually written to disk.  You may also want to put any remaining particles into the diagnostic to analyze them with those that left the simulation.

.. code-block:: c++

   if(step()==num_step){
       for(species_t *sp=species_list; sp; sp=sp->next){
           if(sp->pb_diag){
               sp->copy_to_host();
               for(int p_index=0; p_index<sp->np; p_index++){
                   pbd_write_to_buffer(sp, sp->k_p_h, sp->k_p_i_h, p_index);
               }
               // Flush the buffers
               pbd_buff_to_disk(sp->pb_diag);
           }
       }
   }

Processing the output
*********************

There are two `*.c` example files in `vpic-kokkos/post`.  At present, these are extreemely ad hoc, and make assumptions about what values the particle boundary diagnostic wrote and what species are present in your simulation.  They work well with the short_pulse.cxx sample deck.  See the files for compilation and usage instructions.

Plotting the results
********************
There are two python plotters in `vpic-kokkos/post` that will make plots from the data from the c postprocessors.  If you ran the postprocessors correctly, these should be very easy to use.
