cd ../../build
module load openmpi/4.1.4-intel_2022.1.0
cp ../examples/mirror/conf.dat ./conf.dat
mpif90 -o translate_mirror ../examples/mirror/translate_mirror.f90
mkdir data
mpirun -np 1 ./translate_mirror
mpif90 -o ay_gda_integrate ../examples/mirror/ay_gda_integrate.f90
mpirun -np 1 ./ay_gda_integrate
cd ../examples/mirror
python3 plotsMirror_cyl.py