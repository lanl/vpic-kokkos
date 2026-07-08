module load openmpi gcc/11.2.0
export OMP_PROC_BIND=true
cd ../../build
rm -rf hydro
rm -rf fields
rm -rf data
rm -rf rundata
rm -rf restore0
rm -rf restore1
rm -rf particle
./bin/vpic ../examples/mirror/mirror.cxx
srun -N4 -n8 ./mirror.Linux
cd ../examples/mirror
python3 plotsMirror.py