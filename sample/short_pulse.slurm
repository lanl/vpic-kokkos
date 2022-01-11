#!/bin/bash
#SBATCH -J short_pulse
#SBATCH -o short_pulse%j.out # name of the stdout
#SBATCH -e short_pulse%j.err # name of the stderr
#SBATCH -N 1       # number of cluster nodes
#SBATCH -n 4      # number of MPI tasks
#SBATCH -t 16:00:00 # walltime,
#SBATCH -A my_account     # slurm account 
#SBATCH --qos=standard

# load modules
module purge
module load PrgEnv-cray
module load cmake

module list


pwd
date
ls -lh
echo '*** Starting Parallel Job ***'

# run the program
srun -n $SLURM_NTASKS ./short_pulse.Linux

date
echo '*** All Done ***'
