simd-analysis is a helpful utility for evaluating different platforms using 
different vectorization strategies (**auto, guided, manual**). The utility 
consists of several computational kernels implemented with the different 
strategies.

# Building the microbenchmark
## Dependencies
- CMake
- Kokkos

## Build
The **build** directory contains a build script (**build.sh**) with an example 
of the CMake commands for setting the code up as well as a script (**clean.sh**) 
for cleaning the build directory of CMake generated files.

cd build
. build.sh
make -j

## Clean
make clean
. clean.sh

# Running the microbenchmark
Arguments
- bench-mode     : What vectorization strategy to use (all|auto|guided|manual)
- -l, --limit    : Memory usage limit
- -n, --num-runs : How many times to run each test
- --log          : Output csv file name

The included **runscript.sh** will run each benchmark 50 times and automatically
concatenate the resulting csv files together. Remember to adjust the architecture
string and the number of CPU threads to fit the target platform.

# Visualizing runtime performance
Plotting results is done with the **plot\_simd\_microbench.py** script. The 
script accepts the concatenated log file as an argument. Runtimes are
normalized to the **auto** vectorization strategy.

python plot\_simd\_microbench.py simd\_microbench\_results.csv
