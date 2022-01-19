#!/usr/bin/env python3
import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.constants as const
from matplotlib.colors import LogNorm

'''  This plotter, 2Dbinhistplot.py, makes plots from a binary dump of bin
values, presumably like that from anglehist.c.  It requires a very specifically
formated parameters file from anglehist.c to function, though you can probably
figure out how to modify it and input parameters manually.

Written by Scott V. Luedtke, XCP-6, September 18, 2019'''


# Load a histogram file from disk
def loadHist(filename, nbinsx=100, nbinsy=100, type=np.float64, typesize=8):
    hist = np.zeros([nbinsx, nbinsy], dtype=type)
    structstring = "=" + nbinsy*"d"
    data = open(filename, 'rb')
    #chunck = histnum*nbins*typesize
    #data.seek(-chunck, 2) # Set read pointer to chunck from the end
    for i in range(nbinsx):
        hist[i] = np.asarray(struct.unpack(structstring, data.read(nbinsy*typesize)))
    data.close()
    return hist

if __name__=="__main__":
    if len(sys.argv)<1:
        print("Usage: python 2Dbinhistplot.py [particle to plot]")

    useelec = False
    usecarb = False
    # See if the user specified to use electrons
    if len(sys.argv)>1:
        useelec = (sys.argv[1]) in ['E', 'e', 'Electron', 'electron', 'elec']
    # See if the user specified to use carbons
        usecarb = (sys.argv[1]) in ['C', 'c', 'Carbon', 'carbon', 'carb', 'I2']

    # Determine which particle to use
    if useelec:
        print('using electron')
        particle = "electron"
    elif usecarb:
        print('using carbon')
        particle = "I2"
    # Nothing specified, use default
    else:
        #particle = "I2"
        particle = "electron"

    # Read in hist-specific parameters from a file
    params = open(particle + "anglehistparams.txt", 'r')
    params.readline()
    xmin = float(params.readline().split()[0])
    xmax = float(params.readline().split()[0])
    ymin = float(params.readline().split()[0])
    ymax = float(params.readline().split()[0])
    numbinsx = int(params.readline().split()[0])
    numbinsy = int(params.readline().split()[0])
    params.close()

    # Get density normalization
    # You should check this, and definitely don't blame me if it's wrong.
    #omega = 2.*np.pi*const.c/lamb
    #ncr = const.epsilon_0 * const.m_e * omega**2 / const.e**2
    #delta = lamb/(np.sqrt(ne_over_nc)*(2.*np.pi))
    #num_physical_per_macro = delta**3 * ne_over_nc*ncr
    # I'm pretty sure that this should just be unity now that all my units make
    # sense.  You may want to effectively adjust the thickness in unused
    # dimensions by changing this, though.
    num_physical_per_macro = 1.

    # Where are the data?
    filepath = particle + "lostspec.bin"

    hist = np.zeros([numbinsx, numbinsy], dtype=float)

    hist = loadHist(filepath, numbinsx, numbinsy)
    #print(hist)
    #print(hist.sum())
    hist *= num_physical_per_macro

    xmin = xmin*180./np.pi
    xmax = xmax*180./np.pi
    #print(hist)
    hist = hist.transpose()
    #print(numbinsy,numbinsx)
    #print(ymin,ymax)
    aspect = (xmax-xmin)/(ymax-ymin)
    plt.imshow(hist, origin='lower', interpolation='none', aspect=aspect, norm=LogNorm(), extent=(xmin,xmax,ymin,ymax))
    plt.colorbar().set_label('Flux (#/(sr MeV)', size=18)
    plt.xlabel('Angle from laser axis (degrees)', fontsize=18)
    plt.ylabel('Energy (MeV)', fontsize=18)
    plt.show()
    #plt.savefig(particle + "_anghist.pdf")
    #np.savetxt("AngleSpec.txt", hist, delimiter=",")
