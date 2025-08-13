gather-scatter-analysis is a microbenchmark for testing gather scatter patterns
that frequently pop up in applications like hash tables and stencil codes. 
Different sorting functions can be applied to test how the platform reacts to 
more coalesced memory access patterns. The microbenchmark also includes a simple
Kokkos version of stream to act a a baseline for achievable bandwidth. 

# Building the microbenchmark
## Dependencies
- CMake
- Kokkos

## Build
The **build** directory contains a build script (**build.sh**) with an example 
of the CMake commands for setting the code up as well as a script (**clean.sh**) 
for cleaning the build directory of CMake generated files.

```
cd build
. build.sh
make -j
```

## Clean
```
make clean
. clean.sh
```

# Running the microbenchmark
## CLI
Usage: s-g-bench [--help] [--version] [--sort VAR] [--pattern VAR] [--data-length VAR] [--sample-size VAR] [--unique-keys VAR] [--block-size VAR] [--tile-size VAR] [--stencil-radius VAR] [--num-iter VAR] [--grid-dims VAR...] [--stride VAR] [--log VAR] bench-mode

Benchmark for different scatter/gather/stencil patterns and sorting orders

Positional arguments:
  bench-mode         Benchmark mode: [scatter, gather, gather-scatter, scatter-stencil, gather-stencil, gather-scatter-stencil] [nargs=0..1] [default: "scatter"]

Optional arguments:
  -h, --help         shows help message and exits
  -v, --version      prints version information and exits
  --sort             Sort function: [default, random, standard, strided, tiled, tiled-strided] [nargs=0..1] [default: "default"]
  -p, --pattern      Pattern for keys: [random, unique, repeat, contig, stride] [nargs=0..1] [default: "random"]
  -d, --data-length  Length of data to scatter/gather from [nargs=0..1] [default: 268435456]
  -s, --sample-size  Number of samples to gather/scatter [nargs=0..1] [default: 1000000]
  -u, --unique-keys  Number of unique keys to generate. [nargs=0..1] [default: 1000000]
  -b, --block-size   Number of elements to read per sample [nargs=0..1] [default: 1]
  -t, --tile-size    Number of elements in each tile for tiled and tiled-strided sort [nargs=0..1] [default: 1]
  --stencil-radius   Radius of stencil [nargs=0..1] [default: 1]
  -n, --num-iter     Number of iterations to run [nargs=0..1] [default: 5]
  -g, --grid-dims    Dimensions of grid for stencil benchmark [nargs=0..3] [default: {100 100 100}]
  --stride           Stride for key generation [nargs=0..1] [default: 1]
  --log              Filename for logging results [nargs=0..1] [default: "result_log"]

## Scripts
The included scripts are used to run experiments testing different patterns
and variations of keys for each sorting algorithm. **The user must edit the 
platform specifc parameters in the script to match the hardware.**
- gather\_contig\_jobscript.sh
- gather\_repeat\_jobscript.sh
- scatter\_contig\_jobscript.sh
- scatter\_repeat\_jobscript.sh

# Visualizing runtime performance
Plotting results is done with the plot\_p3hpc\_sort.sh script. The script accepts
 the log directory containing the performance results across different platforms.

```
python plot\_simd\_microbench.py simd\_microbench\_results.csv
```
