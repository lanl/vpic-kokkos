cd ../../build
module load openmpi/4.1.4-intel_2022.1.0
cp ../examples/mirror-cyl/conf.dat ./conf.dat
mpif90 -o translate_faster ../examples/mirror-cyl/translate_faster.f90
mkdir data
mpirun -np 1 ./translate_faster
mpif90 -o ay_gda_integrate ../examples/mirror-cyl/ay_gda_integrate.f90
mpirun -np 1 ./ay_gda_integrate
cd ../examples/mirror-cyl
python3 plotsMirror_cyl.py