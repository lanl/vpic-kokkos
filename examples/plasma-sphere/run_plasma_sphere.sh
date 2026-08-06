#!/bin/bash
cd ../../build
rm -rf hydro fields data rundata restore0 restore1 particle
module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx
export OMP_PROC_BIND=true
./bin/vpic ../examples/plasma-sphere/plasma_sphere.cxx
srun -N1 -n1 ./plasma_sphere.Linux
cd ../examples/plasma-sphere
module load miniconda3
python3 plots_plasma_sphere.py
