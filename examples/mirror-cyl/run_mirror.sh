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
./bin/vpic ../examples/mirror-cyl/mirror_cyl_RZ.cxx
srun -N4 -n4 ./mirror_cyl_RZ.Linux
module load openmpi/4.1.4-intel_2022.1.0
cp ../examples/mirror-cyl/conf.dat ./conf.dat
mpif90 -o translate_faster ../examples/mirror-cyl/translate_faster.f90
mkdir data
mpirun -np 1 ./translate_faster
cd ../examples/mirror-cyl
python3 plotsMirror.py