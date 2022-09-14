Known Issues
============

1. CMake has a bug where MPI detection was changed, and is faulty. So far we've
seen:

  - Does *NOT* work: 3.12.1
  - Does work: 3.12.4, 3.11.1, 3.7.1

2. During a GPU compile, nvcc warns about multiply defined c-standard flags.
   This warning can be safely ignored
3. If you see `../src/util/pipelines/pipelines_thread.cc:239: undefined
   reference to omp_get_num_threads'` when building a deck, it's because
   `-fopenmp` (or equivelent) didn't get added to `./bin/vpic`. This happens
   because of a cmake bug where it fails to detect OpenMP. To fix it, you can
   manually add `-fopenmp` to the build flags in the file.
