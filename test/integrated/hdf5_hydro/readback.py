#!/usr/bin/env python

import h5py
import numpy as np
import os.path
import sys

if len(sys.argv) != 2:
    sys.stderr.write("Usage: "+str(sys.argv[0])+" rundir\n")
    sys.exit(1)

nx = 64
ny = 64
nz = 1
nxg = nx + 2
nyg = ny + 2
nzg = nz + 2

rundir = sys.argv[1]

step_names = ["0",  "1"]

for step_name in step_names:
    filename = rundir + "/hydro_hdf5/T." + step_name + "/hydro_ion_" + step_name + ".h5"

    if not os.path.isfile(filename):
        print("FAIL: " + filename + " is missing")
        sys.exit(1)

    infile = h5py.File(filename, 'r')
    datagroup = infile["Timestep_" + step_name]

    # Binary data
    bin_filename = "Hhydro." + step_name + ".0"
    with open(bin_filename, 'r') as fh:
        hydro_data_bi_all = np.fromfile(bin_filename, dtype=np.float32, offset=123)

    hydro_names = ["jx", "jy", "jz", "rho",
                   "px", "py", "pz", "ke",
                   "txx", "tyy", "tzz",
                   "tyz", "tzx", "txy"]

    for ihydro, hydro_name in enumerate(hydro_names):
        hydro_data_h5 = np.array(datagroup[hydro_name]).flatten()
        fdata_tmp = hydro_data_bi_all[ihydro::16].reshape([nzg, nyg, nxg])
        hydro_data_bi = np.ascontiguousarray((np.transpose(fdata_tmp[1:-1, 1:-1, 1:-1], axes=[2, 1, 0]))).flatten()
        
        if np.allclose(hydro_data_bi, hydro_data_h5, atol=0.0000000001) == False:
            print(hydro_name, 'in ', 'hydro_data_h5', " does not contain same value as ", bin_filename)
            print_max_element = 0
            print("     Binary Output", ",  ", "HDF5 Output",  ",  ", "Difference")
            for i in range(0, hydro_data_h5.shape[0]):
                if hydro_data_bi[i] - hydro_data_h5[i] > 0.0000000001:
                    print(i, hydro_data_bi[i], ",  ", hydro_data_h5[i], ",  ", hydro_data_bi[i] - hydro_data_h5[i])
                    print_max_element = print_max_element + 1
                if print_max_element == 10:
                    break
            sys.exit(1)

    infile.close()
