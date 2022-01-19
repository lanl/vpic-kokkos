Running VPIC
============

Running on multiple GPUs
************************

To run on multiple GPU's, you can pass the flag: `--kokkos-num-devices=N` (which replaced `--kokkos-ndevices`), where
`N` specifies the number of GPUs (per node). This works by VPIC passing through
options it doesn't understand to Kokkos, and thus VPIC will generate a warning
as it thinks you may have tried to tell it something it doesn't understand...

