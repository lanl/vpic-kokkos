#!/usr/bin/env python3
import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import struct
import scipy.constants as const
from matplotlib.colors import LogNorm

'''  This script, VPICtoMCNP.py, takes electron data from anglehist.c and
creates an MCNP deck that will calculate the photons produced by said electrons
in a tungsten converter.  This script requires the data and parameter files
created by anglehist.c.  If you want to make adjustments, you can do so in this
file or in the resulting MCNP deck.  MCNP will throw a fit if a line in the
input deck is too long, so you may need to reduce the number of angle bins when
you run anglehist.c.

Written by Scott V. Luedtke, XCP-6, April 14, 2022'''


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
        print("Usage: python specplot.py [particle to plot] [make the histograms from scratch?]")
        print("All command line options are optional")
        print("There is a block in the source with commonly changed parameters, which you probably want to look at.")

    particle = "electron"
    deck = open("elecWBrem.mcnp","w")

    # Read in hist-specific parameters from a file
    params = open(particle + "anglehistparams.txt", 'r')
    params.readline()
    Amin = float(params.readline().split()[0])
    Amax = float(params.readline().split()[0])
    Emin = float(params.readline().split()[0])
    Emax = float(params.readline().split()[0])
    numbinsA = int(params.readline().split()[0])
    numbinsE = int(params.readline().split()[0])
    params.close()

    if (numbinsA % 2 != 0):
        raise ValueError("You need to have an even number of angle bins so that one of the bin boundaries is on 90 degrees from the axis")
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

    hist = np.zeros([numbinsA, numbinsE], dtype=float)

    hist = loadHist(filepath, numbinsA, numbinsE)
    # We need the number of particles per bin, not the number per bin per unit
    # solid angle, so undo the normalization anglehist.c did.
    da = (Amax-Amin)/numbinsA
    de = (Emax-Emin)/numbinsE
    for i in range(numbinsA):
        norm = (np.cos(da*i)-np.cos(da*(i+1)))*de/2.
        for j in range(numbinsE):
            hist[i][j] *= norm
    #print(hist)
    #print(hist.sum())
    hist *= num_physical_per_macro

    deck.write("""c This deck is auto-generated from makeMCNPdeck.py
c ********************* BLOCK 1: CELLS *********************************
c 0.5mm slab of tungsten
1 1 -19.2 1 -2 3 -4 5 -6 imp:p,e=1
c Void around tungsten
12 0       -100   (-1:2:-3:4:-5:6) imp:p,e=1
13 0        100                    imp:p,e=0

c ********************* BLOCK 2: Surface cards *************************
1 px -2
2 px 2
3 py -2
4 py 2
5 pz 0
6 pz 0.1
100  so 10

c ********************* BLOCK 3: Data cards ****************************
c Materials---just tungsten
M1 74184 1.0
c
c Tally photon flux and bin in energy
F1:P 6 
E1    0.5 98I 50
c
c Ignore particles below a cutoff
CUT:e j 0.5
CUT:p j 0.1
MODE P E
c I am very unsure what physics is necessary for these runs
c These are the same as the default
PHYS:P 100 0 0 0 0 $ 100 MeV, brems, coh scat, no photonuc, Doppler
c  emax, e prod by p ON, p prod by e ON, istrag (default), bnum (default)
PHYS:E 100 0  0  0  1
NPS 1e7
c
c
c Electron source, energy from distribution, direction from distribution,
c radial distance D4 (what is D4?), just in front of the tungsten
sdef  par=e vec 0 0 1 erg=Fdir d10  dir=d1 rad=D4 pos=0 0 -0.002  
c
SI4 0 0.001 $ disk source with radius on 10um=0.001cm
SP4 -21 1    $r^1 for disk source  
c
c
c Angular distribution --- (angle as cos(theta), probability can be based on total # electrons)
#       SI1             SP1  $ histogram for cosine bin limits (each bin is the upper boundary for the probability)
        H               D    $  cos(90)=0, cos(0deg)=1 <-- all electrons are within this range, i.e. directed forward
	0       	0\n""")

    #Need to calculate total in each angular bin
    Atotes = hist.sum(axis=1)
    #print(Atotes)
    cosAngles = np.cos(np.linspace(0.,np.pi,numbinsA+1))
    #print(cosAngles)
    for i in range(numbinsA//2):
        deck.write("    {:.8e}   {:.8e}\n".format(cosAngles[numbinsA//2 - 1 - i],Atotes[numbinsA//2 -i]))

    deck.write("c These specify the energy distribution for the bins\n")
    deck.write("c First determine the angle, then the energy spectrum for that angle\n")
    deck.write("DS10 S")
    for i in range(numbinsA//2):
        deck.write(" {:d}".format(i+101))
    deck.write("\n")

    Ebins = np.linspace(Emin,Emax,numbinsE+1)
    for i in range(numbinsA//2):
        deck.write("c Electron spectrum {:d}\nc\n".format(i+101))
        deck.write("#        SI{:d}            SP{:d}\n".format(i+101,i+101))
        deck.write("        H           D\n")
        deck.write("    {:.8e}        0\n".format(Ebins[0]))
        for j in range(numbinsE):
            deck.write("    {:.8e}  {:.8e}\n".format(Ebins[j+1],hist[i][j]))

    deck.write("print\n")
    deck.write("prdmp 2j 1  $  Creat MCTAL file")
    deck.close()

