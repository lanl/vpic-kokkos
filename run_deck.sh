# cd build
# ./bin/vpic ../examples/iaw/iaw.cxx
# module load openmpi gcc/11.2.0
# export OMP_PROC_BIND=true
# mpirun -np 1 iaw.Linux
# cd ../

cd build
module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
export NVCC_WAPPER_DEFAULT_COMPILER=mpicxx
./bin/vpic ../test/integrated/curv/iaw.deck
export OMP_PROC_BIND=true
mpirun -np 1 ./iaw.deck.Linux
cd ../

# cd build
# module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
# export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx
# ./bin/vpic ../test/integrated/to_completion/pcai.deck
# export OMP_PROC_BIND=true
# mpirun -np 1 ./pcai.deck.Linux
# cd ../
