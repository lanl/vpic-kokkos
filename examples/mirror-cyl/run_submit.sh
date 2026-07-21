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
./bin/vpic ../examples/mirror-cyl/mirror_RZ_cyl.cxx
sbatch ../examples/mirror-cyl/submit
tail -f slurm.err