module load openmpi gcc/11.2.0
export OMP_PROC_BIND=true
cd ../../build
./bin/vpic ../examples/pcai/pcai.cxx
srun -N4 -n4 ./pcai.Linux
mpif90 -o translate_pcai ../examples/pcai/translate_pcai.f90
mkdir data
mpirun -np 1 ./translate_pcai
python3 ../examples/pcai/plotsPCAI.py
