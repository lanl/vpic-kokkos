module load openmpi gcc/11.2.0
scancel -n mir
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
sbatch ../examples/mirror/submit
rm slurm.err
touch slurm.err
tail -f slurm.err