#!/bin/bash
# Build, run, translate, and plot the plasma-sphere verification deck.
# Mirrors examples/iaw/run_iaw.sh and examples/mirror/run_*.sh.

cd ../../build
rm -rf hydro fields data rundata restore0 restore1 particle

module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx
export OMP_PROC_BIND=true

# Build the deck
./bin/vpic ../examples/plasma-sphere/plasma_sphere.cxx

# Run (single rank; matches topology 1x1x1 in the deck)
srun -N1 -n1 ./plasma_sphere.Linux

# Translate the raw dumps into (r,theta,phi) .gda files under build/data
cp ../examples/plasma-sphere/conf.dat ./conf.dat
mpif90 -o translate_faster ../examples/plasma-sphere/translate_faster.f90
mkdir -p data
mpirun -np 1 ./translate_faster

# Plot
cd ../examples/plasma-sphere
module load miniconda3
python3 plots_plasma_sphere.py
