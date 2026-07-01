cd ../../../
# touch src/species_advance/standard/advance_p.cc
# # # touch src/species_advance/species_advance.h
# # touch src/grid/grid.h
# . run_rebuild.sh

cd build
module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
export NVCC_WAPPER_DEFAULT_COMPILER=mpicxx
./bin/vpic ../test/integrated/curv/cyclo-curv.deck
export OMP_PROC_BIND=true
mpirun -np 1 ./cyclo-curv.deck.Linux
cd ../

cd test/integrated/curv
module load miniconda3
python plot_trajectory.py