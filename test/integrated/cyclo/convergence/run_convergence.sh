#!/bin/bash
module load gcc/12.2.0 openmpi/4.1.5-gcc_12.2.0 cmake/3.29.2 cuda/12.9.1
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx

cd /vast/home/cgraham/vpic-kokkos/build

dts=(0.001 0.01 0.1 1.0 10.0 100.0)

output_file="convergence_results.txt"
> $output_file

for dt in "${dts[@]}"; do
  echo "Running with dt=$dt..."

  temp_deck="cyclo_temp_${dt}.deck"
  sed "s/dt_test_value/$dt/g" ../test/integrated/cyclo/convergence/cyclo-convergence.deck > "$temp_deck"
  ./bin/vpic "$temp_deck" > cyclo_compile_${dt}.log 2>&1

  if [ -f cyclo_temp_${dt}.deck.Linux ]; then
    ./cyclo_temp_${dt}.deck.Linux 2>&1 | tee cyclo_output_${dt}.txt | grep -oP '(?<=: )[0-9].* [0-9].*' >> $output_file
    rm -f cyclo_temp_${dt}.deck.Linux "$temp_deck"
  else
    echo "Compilation failed for dt=$dt"
    echo "Check cyclo_compile_${dt}.log for details"
  fi
done

echo "Results written to $output_file"
cat $output_file

cd ../test/integrated/cyclo/convergence/
module load miniconda3
python plot_convergence.py