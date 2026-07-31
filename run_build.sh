module load cmake gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cuda/12.9.1
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx
mkdir build
cd build
../arch/CUDA-Release
make -j$(nproc)

#build/bin/vpic:
# #! /usr/bin/env bash

# deck=`echo $1 | sed 's,\.cxx,,g;s,\.cc,,g;s,\.cpp,,g;s,.*\/,,g'`

# clean_kokkos_path()
# {
#     # $1 = path
#     # $2 = swap ; for $2
#     if grep -q ";" <<< "$1"; then
#       echo $(echo $1 | sed "s/;/ $2/g")
#     else
#       echo ${2}${1}
#     fi
# }


# KOKKOS_CORE_INCLUDES=$( cat /vast/home/cgraham/vpic-kokkos/build/kokkos_core_includes )
# KOKKOS_CONTAINER_INCLUDES=$( cat /vast/home/cgraham/vpic-kokkos/build/kokkos_container_includes )
# KOKKOS_COMPILE_OPTIONS=$( cat /vast/home/cgraham/vpic-kokkos/build/kokkos_compile_options )

# KOKKOS_CORE_LIBS=$KOKKOS_CORE_INCLUDES
# KOKKOS_CONTAINER_LIBS=$KOKKOS_CONTAINER_INCLUDES

# echo $KOKKOS_CORE_LIBS

# # Add hack to include ../lib and ../lib64, only required for "external" builds
# # If only a single path was passed append the paths
# if ! grep -q ";" <<< "$KOKKOS_CORE_LIBS"; then
#     # We pre append ";" to make the above subsition work when cleaning the path
#     KOKKOS_CORE_LIBS=";${KOKKOS_CORE_LIBS};${KOKKOS_CORE_LIBS}/../lib/;${KOKKOS_CORE_LIBS}/../lib64/"
# fi

# # Add hack to include kokkos_random
# KOKKOS_CONTAINER_INCLUDES=";${KOKKOS_CONTAINER_INCLUDES};${KOKKOS_CONTAINER_INCLUDES}/../../algorithms/src/"
# echo $KOKKOS_CONTAINER_INCLUDES

# KOKKOS_CORE_INCLUDES=$(clean_kokkos_path $KOKKOS_CORE_INCLUDES -I)
# KOKKOS_CONTAINER_INCLUDES=$(clean_kokkos_path $KOKKOS_CONTAINER_INCLUDES -I)

# KOKKOS_CORE_LIBS=$(clean_kokkos_path $KOKKOS_CORE_LIBS -L)
# KOKKOS_CONTAINER_LIBS=$(clean_kokkos_path $KOKKOS_CONTAINER_LIBS -L)

# #KOKKOS_LIBS="-l:libkokkoscore.a -l:libkokkoscontainers.a"
# KOKKOS_LIBS="-lkokkoscore -lkokkoscontainers"

# echo /vast/home/cgraham/vpic-kokkos/kokkos/bin/nvcc_wrapper  -I/projects/darwin-nv/rhel8/x86_64/packages/cuda/12.4.1/targets/x86_64-linux/include -I/projects/opt/centos8/x86_64/openmpi/4.1.2-gcc_11.2.0/include -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include/c++/11.2.0 -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include/c++/11.2.0/x86_64-pc-linux-gnu -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include/c++/11.2.0/backward -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include-fixed -I/usr/local/include -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include -I/usr/include -g -rdynamic -fopenmp -I. -I/vast/home/cgraham/vpic-kokkos/src -fopenmp -std=c++17  -O2 -g -DNDEBUG -DSHAPE_NGP -DINPUT_DECK='"'$1'"' /vast/home/cgraham/vpic-kokkos/deck/main.cc /vast/home/cgraham/vpic-kokkos/deck/wrapper.cc -o $deck.Linux -Wl,-rpath,/vast/home/cgraham/vpic-kokkos/build -L/vast/home/cgraham/vpic-kokkos/build -lvpic     -lpthread -ldl $KOKKOS_CORE_LIBS $KOKKOS_CONTAINER_LIBS $KOKKOS_CORE_INCLUDES $KOKKOS_CONTAINER_INCLUDES $KOKKOS_LIBS $KOKKOS_COMPILE_OPTIONS

# echo "$1"
# /vast/home/cgraham/vpic-kokkos/kokkos/bin/nvcc_wrapper  -I/projects/darwin-nv/rhel8/x86_64/packages/cuda/12.4.1/targets/x86_64-linux/include -I/projects/opt/centos8/x86_64/openmpi/4.1.2-gcc_11.2.0/include -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include/c++/11.2.0 -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include/c++/11.2.0/x86_64-pc-linux-gnu -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include/c++/11.2.0/backward -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include-fixed -I/usr/local/include -I/vast/projects/opt/centos8/x86_64/gcc/11.2.0/include -I/usr/include -g -rdynamic -fopenmp -I. -I/vast/home/cgraham/vpic-kokkos/src -fopenmp -std=c++17  -O2 -g -DNDEBUG -DSHAPE_NGP -DINPUT_DECK='"'$1'"' /vast/home/cgraham/vpic-kokkos/deck/main.cc /vast/home/cgraham/vpic-kokkos/deck/wrapper.cc -o $deck.Linux -Wl,-rpath,/vast/home/cgraham/vpic-kokkos/build -L/vast/home/cgraham/vpic-kokkos/build -lvpic   -lcuda  -lpthread -ldl $KOKKOS_CORE_LIBS $KOKKOS_CONTAINER_LIBS $KOKKOS_CORE_INCLUDES $KOKKOS_CONTAINER_INCLUDES $KOKKOS_LIBS $KOKKOS_COMPILE_OPTIONS
