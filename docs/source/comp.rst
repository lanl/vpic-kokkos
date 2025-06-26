Compiling VPIC
==============

Compiling VPIC consists of two steps: compiling or finding a Kokkos install, and building VPIC against that Kokkos install.

Quickstart
**********

Compile Kokkos and VPIC
-----------------------

1. Do a *recursive* clone of this repo, this will pull down a copy of Kokkos for you.  ``git clone --recursive git@github.com:lanl/vpic-kokkos.git``  If you switch branches, you might need to update the Kokkos submodule.  ``git submodule update --init``
2. Load modules for CMake, your compiler, MPI, and any platform specific packages like CUDA.  GNU compilers are the most consistent.  Other compilers might perform slightly better and/or break the build system and/or break physics and/or cause any maner of problems.
3. Find a file in `arch/` that is close to your intended system and modify as necessary.  "Cray" means "Cray."  As in, it's for Cray systems.  People seem to not understand that.  Pay particular attention to lines like::
    
    -DENABLE_KOKKOS_CUDA=ON
    -DKokkos_ARCH_VOLTA70=ON
    -DKokkos_ARCH_POWER9=ON

4. Make a build directory and run the arch file, keeping in mind that the arch files expect the source directory to be its parent.
5. Type ``make``.

This should give you a simple working of the code, and if you selected the correct backend (CUDA/HIP) and architecture optimization targets, there's a good chance VPIC will select the best performance optimizations too.

Build a deck
------------

Compiling VPIC creates a script `bin/vpic` that compiles decks.  From the folder your deck is in, type ``$BUILD_PATH/bin/vpic MyDeck.cxx`` to produce an executable.

Run the executable
------------------

The executable can be run with a simple MPI command: ``mpirun -np $NUM_PROCS MyDeck.Linux``.  Consider saving the stdout and stderr to a text file by appending ``2>&1 | tee out.txt``.

Examine `sample/short_pulse.slurm` for an example submission script.  We recommend using one MPI rank per CPU core or one rank per GPU as a baseline, but that may not work best for your simulation on your hardware.

Manual Kokkos Install (more powerful, more effort)
**************************************************

It is possible to have a version of Kokkos tuned for a specific machine that may outperform VPIC's internal Kokkos build.  This does not seem to be very popular at present, and VPIC's builds are generally very good with the right architectures and backends set, but you can link to an external Kokkos build.  Make sure the `BUILD_INTERNAL_KOKKOS` option is off.

Further Reading
***************

One can cherry pick the Kokkos specific details from
[here](https://github.com/ECP-copa/Cabana/wiki/Build-Instructions) to get
detailed build instructions for Kokkos (ignore the things about Cabana).

The advanced user should review `CMakeLists.txt` for the Kokkos specific
options that are available. These include:

1. `ENABLE_KOKKOS_OPENMP`
2. `ENABLE_KOKKOS_CUDA`
3. `BUILD_INTERNAL_KOKKOS`
4. `VPIC_KOKKOS_DEBUG`
5. `KOKKOS_ARCH`

Optimization Options
********************

VPIC has compilation flags for enabling/disabling various optimizations. VPIC will automatically select optimizations settings based on hardware targets according to developer experience. Users can supply their own settings for potentially better performance. The optimization options are as follows:

1. `VPIC_ENABLE_AUTO_TUNING=ON`

  Control whether to use the automatically determined optimization settings or user supplied compile time flags.

2. `VPIC_ENABLE_HIERARCHICAL=OFF` 

  Allow finer control over how work is distributed among threads. Automatically enabled by certain optimizations (Team reduction, Vectorization) that require explicit control over threads and vector lanes. Performance is highly dependent on how work is distributed. See kokkos_tuning.hpp for setting the number of leagues (thread teams) and team size (threads per team).

3. `VPIC_ENABLE_TEAM_REDUCTION=OFF` 

  Reduce number of atomic writes in the particle push. Checks if all the particles being processed by active threads / vector lanes belong to the same cell. If so, use fast register based methods to reduce current so that only 1 thread/lane needs to update the fields.

4. `VPIC_ENABLE_VECTORIZATION=OFF` 

  Enables vectorization with OpenMP SIMD for greater performance on the CPU

5. `VPIC_ENABLE_ACCUMULATORS=OFF`

  Use an explicit accumulator for collecting current in advance_p. The accumulator results in better memory access patterns when writing current. This is useful on CPUs but not necessary on GPUs which have better random access characteristics.

