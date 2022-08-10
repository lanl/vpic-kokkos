Automated VPIC building and benchmarking
========================================

`VPIC <https://github.com/lanl/vpic-kokkos>`_ includes a collection of \
shell scripts that should simplify building, testing  and benchmarking the \
code, especially when evaluating performance in distinct programming \
environments when the code is built with distinct combinations of Kokkos and \
multithreading options. Supported shell scripts, currently found in \
*vpic-kokkos/TestScripts/*:

* ``vpic2.0-build.sh`` -- script that builds the code in different *Cray* programming environments (currently on the *Chicoma* system at LANL) with different multithreading (*PThreads*, *OpenMP*) and Kokkos options

* ``vpic2.0-test.sh`` -- script that performs scaling tests (weak scaling at the moment) of the code built previously with ``vpic2.0-build.sh``

* ``vpic2.0_weak_scaling.sh`` -- script called by ``vpic2.0-test.sh`` in order to conduct a weak scaling study of the requested input deck and plot obtained results

* ``create_plot.pl`` -- *perl* script that plots run-time data

* ``extract_output_data.pl`` -- *perl* script, called by *create_plot.pl*, that extracts data from the *slurm* output file generated at run-time

Below we discuss usage of these scripts in more detail. 

Automated building
***********************

``vpic2.0-build.sh`` is a batch shell script that can build multiple versions of \
VPIC and targeted input decks that may require distinct programming environments \
and options, such as the Kokkos and multithreading options. ``vpic2.0-build.sh`` \
also makes it easy to quickly rebuild input desks using already pre-compiled code. \
This script can be easily edited / fully customized by the users.    

Note that on the *Cray*-based systems, such as the *Chicoma* system at LANL, \
the following programming environments are available: 

* PrgEnv-cray (default Cray environment)
* PrgEnv-aocc (*aocc* compiler based) 
* PrgEnv-gnu (*gnu* compiler based) 
* PrgEnv-intel (*Intel* compiler based) 
* PrgEnv-nvidia (NVIDIA environment) 

``vpic2.0-build.sh`` offers two functions, ``vpic_cmake()`` and ``build_deck()`` that make it easy to rebuild VPIC based on specific multithreading and Kokkos requirements. 

``vpic_cmake()`` is a *CMake* configuration function, that is currently configured \
with *Chicoma* architecture in mind, i.e. we indicate that it is a ZEN2 CPU and \ 
NVIDIA AMPERE GPU based system:
::
      function vpic_cmake(){
      cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_INTEGRATED_TESTS=ON \
        -DENABLE_UNIT_TESTS=ON \
        -DBUILD_INTERNAL_KOKKOS=ON \
        -DENABLE_KOKKOS_CUDA=ON \
        -DVPIC_ENABLE_TEAM_REDUCTION=$6 \
        -DVPIC_ENABLE_HIERARCHICAL=$5\
        -DKokkos_ARCH_ZEN2=ON \
        -DKokkos_ARCH_AMPERE80=ON \
        -DCMAKE_CXX_COMPILER="$(readlink -f $1/kokkos/bin/nvcc_wrapper)" \
        -DCMAKE_CXX_FLAGS=$4 \
        -DKokkos_ENABLE_OPENMP=$3 \
        -DKokkos_ENABLE_PTHREAD=$2 \
        $1
      }

At the moment ``vpic_cmake()`` takes the following positional options:

.. py:function:: vpic_cmake()  

   Configure VPIC *CMake* using provided positional input parameters

   :arg 1: a path to the source directory (e.g. $openmp/team_hierarch)   
   :arg 2: an "ON" or "OFF" string, indicating whether to compile with *pthreads*    
   :arg 3: an "ON" or "OFF" string, indicating whether to compile with *openmp*
   :arg 4: a string with C++ flags, e.g. "-fopenmp -rdynamic -dynamic" 
   :arg 5: an "ON" or "OFF" string, indicating whether Kokkos hierarchical parallelism should be enabled 
   :arg 6: an "ON" or "OFF" string, indicating whether Kokkos team reduction should be enabled

``build_deck()`` function builds input deck files based on the directory path to the build directory and directory path to the input deck:

.. py:function:: build_deck()

    Build input decks (at the moment hardwired in the script) using provided positional input parameters:

   :arg 1: directory path to the VPIC build
   :arg 2: directory	path to the input deck (\*.cxx file)

Below we provide an example that illustrates ``vpic_cmake()`` and ``build_deck()`` usage:
::
          build_dir=$openmp/team_hierarch
          mkdir -p $build_dir
          cd $build_dir
          cmake_openmp="ON"
          cmake_pthreads="OFF"
          cmake_team_reduction="OFF"
          cmake_hierarchical="OFF"
          cmake_cxx_flags="-fopenmp -rdynamic -dynamic"
          vpic_cmake $src_dir $cmake_pthreads $cmake_openmp $cmake_cxx_flags $cmake_hierarchical $cmake_team_reduction
          make
          make test
          build_deck $build_dir $src_dir
      
Automated benchmarking
**********************

``vpic2.0-test.sh`` is a shell script that uses function ``vpic_kokkos_benchmark()`` to construct VPIC \
test cases that run in batch using VPIC weak scaling executables created by the ``vpic_cmake()`` and \
``build_deck()`` scripts:

.. py:function:: vpic_kokkos_benchmark()

   Calls VPIC weak scaling benchmark, previously built by the ``vpic_cmake()`` and ``build_deck()`` scripts. 

   :arg 1: programming environment that takes the following values: "gnu" (and, in the nearest future: "aocc", "cray", "intel", "nvidia") 
   :arg 2: directory path to the input deck in the build directory
   :arg 3: multithreading type {"openmp" or "pthreads"}

Below we provide an example that illustrates ``vpic_kokkos_benchmark()`` usage:
::
      prog_env="gnu"
      thread="pthreads"
      bench_path=pthreads/team_hierarch
      vpic_kokkos_benchmark $prog_env $bench_path $thread


``vpic2.0_weak_scaling.sh`` is a shell script that launches a weak scaling function \
of the requested input deck and plots obtained results; it is called by ``vpic2.0-test.sh``, 
but can also used on its own. ``vpic2.0_weak_scaling.sh`` is located in the 
*$HOME/VPIC/vpic-kokkos/TestScripts/* directory and takes the following positional arguments: 
::
	$prg_env -- programming environment {"PrgEnv-cray", "PrgEnv-aocc", "PrgEnv-gnu", "PrgEnv-nvidia",  "PrgEnv-intel"}
	$thread -- multithreading support {"openmp", "pthreads"}
	$deck4 $deck3 $deck2 $deck1 -- (cpu scaling: decks that run on 64, 32, 16, 8 ranks per node) 
	$vpic_scripts -- directory path *vpic-kokkos/TestScripts*
	$slurm_output -- the name of the slurm output file

Here is an example illustrating how this script can be called from within the ``vpic2.0-test.sh``:
::
      sbatch -o $slurm_output -p gpu -N 1 -t 6:00:00 $vpic_scripts/vpic2.0_weak_scaling.sh $prg_env $thread $deck4 $deck3 $deck2 $deck1 $vpic_scripts $slurm_output


``vpic2.0_weak_scaling.sh`` launches ``weak_run()`` function that runs a weak scaling study for the provided input decks:

.. py:function:: weak_run()

   Runs VPIC weak scaling benchmark with the following parameters: 

   :arg 1: $np -- the number of processors requested  
   :arg 2: $num_threads -- the number of hardware threads requested
   :arg 3: $code -- the name of the input deck

After the ``weak_run()`` is done ``vpic2.0_weak_scaling.sh`` calls perl script ``create_plot.pl`` in order to visualize generated performance data. ``create_plot.pl`` calls ``extract_output_data.pl`` -- a *perl* script that  extracts data from the *slurm* output file $slurm_output and uses `GnuPlot <http://gnuplot.sourceforge.net>`_ to visualize collected data. Usage example:
::
      $plot_script=./vpic-kokkos/TestScripts/create_plot.pl      
      perl $plot_script $slurm_output

Extracted data will be written into the file `vpic_plot_data.dat` and performance plots will be placed in the file `vpic_perf_plot.png`:
