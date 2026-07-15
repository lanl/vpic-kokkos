#!/usr/bin/env python3
"""
Program compiles, runs, and post-process 2D Hybrid-VPIC simulation
of a plasma mirror including neutral beam injection and fusion reactions.

Author:  Mike Lavell, XCP-6, Los Alamos National Laboratory
Created: April 2026

Command line arguments:
  --compile: create cxx and slurm files, and 'make clean and make'
  --submit: submit simulation
  --postproc: run post-processing

References:
Ref 1: Bosch, H. S., & Hale, G. M. (1992). Improved formulas for fusion 
       cross-sections and thermal reactivities. Nuclear fusion, 32(4), 611-631.
Ref 3: Higginson, D. P., Link, A., & Schmidt, A. (2019). A pairwise nuclear fusion 
       algorithm for weighted particle-in-cell plasma simulations. Journal of 
       Computational Physics, 388, 439-453.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as con
import argparse, sys, os, h5py, subprocess, math
from jinja2 import Environment, FileSystemLoader

import utils # from local utils.py

if (subprocess.check_output(["whoami"])[:-1] == b'mlavell'):
    ws_dir = os.getenv('WS_DIR')
    plt.style.use(f'{ws_dir}/extras/mjl.mplstyle')
good_colors = ["#db6d00","#006ddb","#920000","#52a736","#edd455"]

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--compile', action='store_true',
                    help='Flag to enable compiling')
parser.add_argument('--execute', action='store_true',
                    help='Flag to execute simulation on current node')
parser.add_argument('--submit', action='store_true',
                    help='Flag to enable submitting slurm')
parser.add_argument('--postproc', action='store_true',
                    help='Flag to enable post-processing output')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    # Get user args
    args = parser.parse_args()

    # Template file base names
    deck = 'mirror_nbi_2d'
    slurm_file = 'slurm-runVPIC'


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define simulation parameters

    # run time parameters that populate tnf.cxx using template tnf.tmpl
    plasma_params = {
        'Ti' : 1,          # ion temperature in (keV)
        'Te' : 1,           # electron temperature (keV)
        'ni' : 1e14,        # seed density (cm^-3)    
        'nppc' : 300,       # number of particles per cell
        'tritium_fuel' : 0, # flag to include tritium in seed plasma (precompiler option)
        'DT_ratio' : 1,     # ratio of deuteirum ions to tritium ions in seed plasma (nD/nT)
        'b0' : 3.0e4,       # field strength at throat (G)
        'n0_profile' : 2    # initial density profile (1: square, 2: oval)
    }

    beam_params = {
        'energy' : 25.0, # beam energy (keV)
        'power'  : 0.4,   # beam power (MW)
        'T_para' : 1.0,   # temperature parallel to beam direction (keV)
        'T_perp' : 1.0,   # temperature perpendicular to beam direction (keV)
        'interval' : 20,  # injection interval
        'nppc' : 20,      # particles per cell
        'radius' : 5.0,   # beam radius (di)
    }

    fusion_params = {
        'enable_fusion' : 0,        # flag to turn on/off fusion module (precompiler option)
        'pmult' : 1e5,              # fusion production multiplier (for tuning product macroparticle count)
        'anisotropic_emission' : 1, # flag turn on/off anisotropic emission model (precompiler option)
    }

    runtime_params = {
        'dt' : 0.01,   # time step  (1/wci)
        'Lt' : 2000.0, # total sim time (1/wci)
        'Lx' : 36,     # size in x-direction (di)
        'Ly' : 4,      # size in y-direction (di)
        'Lz' : 360,    # size in z-direction (di)
        'nx' : 256,    # number of cells in x-direction
        'nz' : 1024,   # number of cells in z-direction
        'topo_x' : 4,  # number of mpi domains in x-direction
        'topo_z' : 32, # number of mpi domains in z-direction
    }

    interval_params = {
        'fields' : 100,     # interval to save fields and hydro
        'particles' : 5000, # interval to save particles
        'metrics' : 100,    # interval to save energy metrics
        'restart' : 10000,    # interval to save restarts
        'sort' : 20,        # particle sort interval
        'collision' : 1,    # collision interval (units of sort_interval)
        'fusion' : 1,       # fusion interval (units of sort_interval)
    }

    job_params = {
        'deck_name' : deck,      # name off cxx file
        'sim_name' : 'mirror2d', # job name
        'nodes' : 4,             # number of cluster nodes
        'n_ranks' : runtime_params['topo_x'] * runtime_params['topo_z']
    }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run safety checks

    if (runtime_params['nx'] % runtime_params['topo_x'] != 0):
        sys.exit("ERROR: nx % topo_x != 0")

    if (runtime_params['nz'] % runtime_params['topo_z'] != 0):
        sys.exit("ERROR: nz % topo_z != 0")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.compile:

        print('\n -- Compiling program.')

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Make slurm script

        rundir = os.getcwd()
        environment = Environment(loader=FileSystemLoader(rundir))

        slurm_template = environment.get_template(f'{slurm_file}.tmpl')
        slurm_content = slurm_template.render(job_params = job_params)
        with open(f'{slurm_file}', mode='w', encoding='utf-8') as jfile:
            jfile.write(slurm_content)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write and compile deck

        # generate c++ file from template
        template = environment.get_template(f'{deck}.tmpl')
        content = template.render(plasma_params = plasma_params,
                                  beam_params = beam_params,
                                  fusion_params = fusion_params,
                                  runtime_params = runtime_params,
                                  interval_params = interval_params)

        with open(f'{deck}.cxx', mode='w', encoding='utf-8') as jfile:
            jfile.write(content)

        subprocess.call('make clean',shell=True)
        subprocess.call('make',shell=True)

        if not os.path.exists(f'{deck}.Linux'):
            sys.exit('  --- Compilation failed.')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.execute:
        
        print('\n -- Executing program.')

        if not os.path.exists(f'{deck}.Linux'):
            sys.exit('  --- No binary.')

        cmd = f'srun -n {job_params["n_ranks"]} ./{deck}.Linux' # >& outfile &
        print(f"\n  --- Executing with: {cmd}")
        
        subprocess.run(cmd, shell=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.submit:
        
        print('\n -- Submitting program.')

        if not os.path.exists(f'{deck}.Linux'):
            sys.exit('  --- No binary.')

        subprocess.run(f'sbatch {slurm_file}', shell=True)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if args.postproc:
    #     print(f'\n -- Post-processing program')
