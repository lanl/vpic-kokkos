
cd ../../build
rm -rf hydro
rm -rf fields
rm -rf data
rm -rf rundata
rm -rf restore0
rm -rf restore1
rm -rf particle
module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
export NVCC_WAPPER_DEFAULT_COMPILER=mpicxx
./bin/vpic ../examples/whistler/whistler_stretch.cxx
export OMP_PROC_BIND=true
srun -N1 -n1 ./whistler_stretch.Linux
mpif90 -o translateWhistler ../examples/whistler/translateWhistler.f90
mkdir data
mpirun -np 1 ./translateWhistler
cd ../examples/whistler
module load miniconda3
python plotsWhistler_stretch.py
