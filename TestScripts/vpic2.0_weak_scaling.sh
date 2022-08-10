#!/bin/bash

prog_env=$1            # programming environment
multi_threading=$2     # multithreading support ("openmp" or "pthreads")
nn=1                   # the number of nodes
num_threads=2          # the number of cpu threads
p_scale=2              # scaling constant
log=./log-weak         # log file
let np=64\*$nn         # number of processes / ranks

if [ $prog_env = "aocc" ] 
then    
    module swap PrgEnv-cray PrgEnv-aocc
elif [ $prog_env = "gnu" ] 
then
    module swap PrgEnv-cray PrgEnv-gnu
elif [ $prog_env = "intel" ]
then
    module swap PrgEnv-cray PrgEnv-intel
fi

if[ $multi_threading = "openmp" ]
then
    export OMP_PROC_BIND=spread
    export OMP_PLACES=threads
    export OMP_NUM_THREADS=$num_threads
fi

module load cmake cuda gnuplot
module list
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
echo "NVCC_WRAPPER_DEFAULT_COMPILER:" $NVCC_WRAPPER_DEFAULT_COMPILER
CC --version
lscpu
nvidia-smi
#env 

function weak_run(){

    srun -n $1 -c $2 --cpu-bind=cores $3 --tpp $2 >> $log
}

code=$3                      
weak_run $np $num_threads $code

code=$4                      
let np=$np/$p_scale          
weak_run $np $num_threads $code

code=$5                      
let np=$np/$p_scale

weak_run $np $num_threads $code

code=$6                      
let np=$np/$p_scale          
weak_run $np $num_threads $code

##### create performance plot #####

plot_script=$7/create_plot.pl
slurm_output=$8

perl $plot_script $slurm_output

##################################
