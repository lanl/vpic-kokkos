#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys

if len(sys.argv) > 3:
    sys.stderr.write("Usage: "+str(sys.argv[0])+" rundir\n")
    sys.exit(1)

nx = 64
ny = 64
nz = 1
nxg = nx + 2
nyg = ny + 2
nzg = nz + 2
num_var = 16

rundir = sys.argv[1]

step_names = ["0",  "1"]

hydro_names = ["jx", "jy", "jz", "rho",
               "px", "py", "pz", "rho_m",
               "txx", "tyy", "tzz",
               "tyz", "tzx", "txy"]

if len(sys.argv) == 3 and sys.argv[2] == '--variable-charge':
    hydro_names.append("qmin")
    hydro_names.append("qmax")
    hydro_names.append("n_q0")
    hydro_names.append("n_q1")
    hydro_names.append("n_q2")
    hydro_names.append("n_q3")
    hydro_names.append("n_q4")
    hydro_names.append("n_q5")
    num_var = 24

print(hydro_names)

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
        hydro_data_bi_all = np.fromfile(bin_filename, dtype=np.float64, offset=123) # Changes to hydro made all hydro variables doubles
        print(hydro_data_bi_all.shape)

    for ihydro, hydro_name in enumerate(hydro_names):
        print(hydro_name)
        hydro_data_h5 = np.array(datagroup[hydro_name]).flatten()
        fdata_tmp = hydro_data_bi_all[ihydro::num_var].reshape([nzg, nyg, nxg])
        hydro_data_bi = np.ascontiguousarray((np.transpose(fdata_tmp[1:-1, 1:-1, 1:-1], axes=[2, 1, 0]))).flatten()
        
        #binary_image = np.transpose(fdata_tmp[1:-1, 1:-1, 1:-1], axes=[2,1,0])
        #hdf5_image = hydro_data_h5.reshape([nx, ny, nz])

        #min_val = np.min([np.min(binary_image), np.min(hdf5_image)])
        #max_val = np.max([np.max(binary_image), np.max(hdf5_image)])
        #print("Binary Min val = " + str(np.min(binary_image)))
        #print("Binary Max val = " + str(np.max(binary_image)))
        #print("HDF5 Min val = "   + str(np.min(hdf5_image)))
        #print("HDF5 Max val = "   + str(np.max(hdf5_image)))

        #fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4), sharex=True, sharey=True)
        #bi_pcm = ax[0].imshow(binary_image, vmin=min_val, vmax=max_val, aspect='auto')
        #h5_pcm = ax[1].imshow(hdf5_image,   vmin=min_val, vmax=max_val, aspect='auto')
        ##fig.subplots_adjust(right=0.85)
        ##cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        ##cbar = fig.colorbar(bi_pcm, cax=cbar_ax)
        #plt.colorbar(bi_pcm, ax=ax[0])
        #plt.colorbar(h5_pcm, ax=ax[1])

        #ax[0].set_title("Binary: " + hydro_name)
        #ax[1].set_title("HDF5: " + hydro_name)
        #plt.show(block=True)
        
        if np.allclose(hydro_data_bi, hydro_data_h5, rtol=1e-5, atol=1e-10, equal_nan=True) == False:
            print(hydro_name, 'in ', 'hydro_data_h5', " does not contain same value as ", bin_filename)
            print_max_element = 0
            print("     Binary Output", ",  ", "HDF5 Output",  ",  ", "Difference")
            for i in range(0, hydro_data_h5.shape[0]):
                if np.isclose(hydro_data_bi[i], hydro_data_h5[i], rtol=1e-5, atol=1e-10, equal_nan=True) == False:
                    print(i, hydro_data_bi[i], ",  ", hydro_data_h5[i], ",  ", hydro_data_bi[i] - hydro_data_h5[i])
                    print_max_element = print_max_element + 1
                if print_max_element == 10:
                    break
            sys.exit(1)

    infile.close()
