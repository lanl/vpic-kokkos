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
./bin/vpic ../examples/mirror/mirror_cyl.cxx
srun -N1 -n1 ./mirror_cyl.Linux
module load openmpi/4.1.4-intel_2022.1.0
cp ../examples/mirror/conf.dat ./conf.dat
mpif90 -o translate_mirror ../examples/mirror/translate_mirror.f90
mkdir data
mpirun -np 1 ./translate_mirror
mpif90 -o ay_gda_integrate ../examples/mirror/ay_gda_integrate.f90
mpirun -np 1 ./ay_gda_integrate
cd ../examples/mirror
python3 plotsMirror.py