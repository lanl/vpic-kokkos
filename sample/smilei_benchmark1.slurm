#!/bin/bash
#SBATCH -J smilei_benchmark1
#SBATCH -o smilei_benchmark1%j.out # name of the stdout
#SBATCH -e smilei_benchmark1%j.err # name of the stderr
#SBATCH -N 1       # number of cluster nodes
#SBATCH -n 1      # number of MPI tasks
#SBATCH -t 00:15:00 # walltime,
#SBATCH -A xd_g     # slurm account 
#SBATCH --qos=debug
#SBATCH -p gpu
#SBATCH --reservation gpu_debug

# load modules
module load PrgEnv-gnu
module load cmake cuda cpe-cuda
module list


pwd
date
ls -lh
echo '*** Starting Parallel Job ***'

# run the program
srun -n $SLURM_NTASKS ./smilei_benchmark1.Linux

date
echo '*** All Done ***'
