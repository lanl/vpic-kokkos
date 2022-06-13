#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.constants as const

'''  This plotter, mcnpTallySpec.py, makes a spectrum plot from an MCNP tally
file.  I am not sure how specific this is to the way I set up MCNP, but it is
probably not hard to adjust this script as necessary.  You need to copy the
parameter file created by anglehist.c to the directory you run this from.

Written by Scott V. Luedtke, XCP-6, January 18, 2022'''


if __name__=="__main__":
    if len(sys.argv)<1:
        print("Usage: python mcnpTallySpec.py [tally file]")

    tfile = open(sys.argv[1], 'r')

    line = tfile.readline()
    test = 'et'
    while line.split()[0] != test:
        line = tfile.readline()

    # Read in hist-specific parameters from a file
    particle = "electron"
    params = open(particle + "anglehistparams.txt", 'r')
    params.readline()
    Amin = float(params.readline().split()[0])
    Amax = float(params.readline().split()[0])
    Emin = float(params.readline().split()[0])
    Emax = float(params.readline().split()[0])
    numbinsA = int(params.readline().split()[0])
    numbinsE = int(params.readline().split()[0])
    params.readline()
    params.readline()
    params.readline()
    params.readline()
    numphys = float(params.readline().split()[0])
    params.close()

    bins = np.zeros(1)
    test = 't'
    line = tfile.readline()
    #print(test)
    #print(line.split()[0])
    while line.split()[0] != test:
        for num in line.split():
            #print("doing an append")
            bins = np.append(bins, float(num))
        line = tfile.readline()
    #print(bins)

    spec = np.zeros(0)
    line = tfile.readline()
    line = tfile.readline()
    test = 'tfc'
    count = 0
    while line.split()[0] != test:
        for num in line.split():
            if not(count%2):
                spec = np.append(spec, float(num))
            count+=1
        line = tfile.readline()
    spec = np.delete(spec, -1)
    #print(spec)
    spec = spec*numphys

    
    #print(hist)
    plt.figure()
    plt.xlabel('Energy (MeV)', fontsize=20)
    plt.ylabel('Photons (#/MeV)', fontsize=20)
    plt.yscale('log', nonposy='clip')
    plt.xlim(bins[0],bins[-1])
    plt.plot(bins[:-1] + .5*(bins[1]-bins[0]), spec[0:100])#, label="0.1mm")
    #plt.legend(loc='upper right')
    plt.show()
