#!/usr/bin/env python

import h5py
import numpy as np
import os.path
import sys

if len(sys.argv) > 3:
    sys.stderr.write("Usage: "+str(sys.argv[0])+" rundir\n")
    sys.exit(1)

nx = 10
ny = 10
nz = 1
nxg = nx + 2
nyg = ny + 2
nzg = nz + 2

rundir = sys.argv[1]

step_names = ["0",  "1"]

particle_names = ["dX", "dY", "dZ", "i",
               "Ux", "Uy", "Uz", "q"]
abs_tol = 0.0000000001
rel_tol = 1e-05
num_vars = 8
if len(sys.argv) == 3 and sys.argv[2] == '--variable-charge':
    particle_names.append("qp")
    num_vars = 9

error = 0
for step_name in step_names:
    print('step ' + step_name)
    filename = rundir + "/particle_hdf5/T." + step_name + "/ion_" + step_name + ".h5"

    if not os.path.isfile(filename):
        print("FAIL: " + filename + " is missing")
        sys.exit(1)

    infile = h5py.File(filename, 'r')
    datagroup = infile["Timestep_" + step_name]

    # Binary data
    bin_filename = "Hion." + step_name + ".0"
    with open(bin_filename, 'r') as fh:
        particle_data_bi_all = np.fromfile(bin_filename, dtype=np.float32, offset=115)
    numpart = int(len(particle_data_bi_all)/num_vars)
    #print(numpart)

    #print(particle_data_bi_all[:num_vars])
    particle_view_bi_all = particle_data_bi_all.reshape((numpart, num_vars))
    #print(particle_view_bi_all.shape)

    #print(particle_data_bi_all.shape)
    #print(type(datagroup['dX']))
    #print(len(datagroup['dX']))


    for iparticle, particle_name in enumerate(particle_names):
        #particle_data_h5 = np.array(datagroup[particle_name]).flatten()
        #fdata_tmp = particle_data_bi_all[iparticle::16].reshape([nzg, nyg, nxg])
        #particle_data_bi = np.ascontiguousarray((np.transpose(fdata_tmp[1:-1, 1:-1, 1:-1], axes=[2, 1, 0]))).flatten()

        #print(type(particle_data_bi_all[iparticle*numpart:(iparticle+1)*numpart]))
        #print(len(particle_data_bi_all[iparticle*numpart:(iparticle+1)*numpart]))
        particle_data_h5 = datagroup[particle_name][:]
        particle_data_bi = particle_view_bi_all[:,iparticle]
        #print(type(particle_data_h5))
        #print(type(particle_data_bi))
        #print(particle_data_h5.shape)
        #print(particle_data_bi.shape)
        #print(particle_data_h5.shape[0])

        if particle_name == "i":
          particle_data_bi = particle_data_bi.view(np.int32)
          continue
        
        particle_data_bi.sort()
        particle_data_h5.sort()
        if np.allclose(particle_data_bi, particle_data_h5, atol=abs_tol, rtol=rel_tol) == False:
            print(particle_name, 'in ', 'particle_data_h5', " does not contain same value as ", bin_filename)
            print_max_element = 0
            print("     Binary Output", ",  ", "HDF5 Output",  ",  ", "Difference")
            for i in range(0, numpart):
#                if abs(particle_data_bi[i] - particle_data_h5[i]) > abs_tol:
                if np.isclose(particle_data_bi[i], particle_data_h5[i]) == False:
                    print(i, '{:.11E}'.format(particle_data_bi[i]), ",  ", '{:.11E}'.format(particle_data_h5[i]), ",  ", '{:.11E}'.format(particle_data_bi[i] - particle_data_h5[i]))
                    print_max_element = print_max_element + 1
                if print_max_element == 10:
                    break
#            sys.exit(1)
            error = 1

    infile.close()
sys.exit(error)
