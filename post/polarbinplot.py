#!/usr/bin/env python3
import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.constants as const
from matplotlib.colors import LogNorm

'''  This plotter, polarbinplot.py, makes plots from a binary dump of bin
values, presumably like that from edep.c.  It requires a very specifically
formated parameters file from edep.c to function, though you can probably
figure out how to modify it and input parameters manually.

Written by Scott V. Luedtke, XCP-6, September 23, 2019'''


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

# Make bins in the azimuthal angle such that the area on a sphere subtending a bin is constant
def makeThetabins(N):
    bins = np.zeros(N+1)
    for i in range(N+1):
        bins[i] = np.arccos(1. - (2.*i)/(N))
    return bins

if __name__=="__main__":
    if len(sys.argv)<1:
        print("Usage: python polarbinplot.py [particle to plot]")

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
    params = open(particle + "edepparams.txt", 'r')
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

    ### ********************************************************** ###
    ### This block contains parameters that are commonly changed.  ###
    ### ********************************************************** ###
    # Where are the data?
    filepath = particle + "edep.bin"

    hist = np.zeros([numbinsx, numbinsy], dtype=float)

    hist = loadHist(filepath, numbinsx, numbinsy)
    #print(hist)
    #print(hist.sum())
    #hist *= num_physical_per_macro
    #print(ncr, delta, ne_over_nc)
    #print(num_physical_per_macro)
    #print(hist[3])

    print("Total energy in particles is ", hist.sum(), " MeV, or ", hist.sum()*1e6*const.e, "J")

    thetbins = makeThetabins(numbinsx)
    phibins = np.linspace(ymin, ymax, numbinsy+1)
    binsx = phibins
    binsy = thetbins*180./np.pi
    
    from pylab import figure, subplot, pcolormesh, axis, colorbar, tight_layout, show, savefig
    figure(figsize=(.6*9,.6*6))
    ax = subplot(111, polar=True)
    pcolormesh(binsx, binsy, hist, norm=LogNorm(), cmap='viridis')#vmin=17266172.332, vmax=10513534524.8))
    axis([0., 2.*np.pi, 0., 180.])
    colorbar().set_label('Energy / Solid Angle (MeV/sr)',size=16)
    ax.set_yticks(np.arange(45,181,45))
    ax.set_rlabel_position(60)
    tight_layout()
    show()
    #savefig('TPW_elec_edep.pdf')
