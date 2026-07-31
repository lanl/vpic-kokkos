#module load openmpi gcc/12.2.0
export OMP_PROC_BIND=true
cd ../../build
rm -rf hydro
rm -rf fields
rm -rf data
rm -rf rundata
rm -rf restore0
rm -rf restore1
rm -rf particle
./bin/vpic ../examples/pcai/pcai.cxx
srun -N1 -n1 ./pcai.Linux
mpif90 -o translate_pcai ../examples/pcai/translate_pcai.f90
mkdir data
mpirun -np 1 ./translate_pcai
cd ../examples/pcai
python3 plotsPCAI.py
