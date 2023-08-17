#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import h5py

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )

parser = argparse.ArgumentParser(description='Visualize trajectories')
parser.add_argument('filenames', metavar='N', type=str, nargs='+', help='Input files')
parser.add_argument('--fig-name', type=str, nargs=1, help='Name of output figure')
parser.add_argument('--origin', type=tuple, nargs=2, default=(-282395872, -423593792), help='Origin point (2D coordinates)')
parser.add_argument('--corner', type=tuple, nargs=2, default=(564791744, 423593824), help='Upper right corner (2D coordinates')
parser.add_argument('--overlay-var', type=str, nargs=1, help='Name of dataset to use for determining the color of the scatter plot')

args = parser.parse_args()

xmin = args.origin[0]
ymin = args.origin[1]
xmax = args.corner[0]
ymax = args.corner[1]

fig = plt.figure(figsize=(12.8,9.6), dpi=200)

min_energy = []
max_energy = []
for filename in args.filenames:
  traj = h5py.File(filename, 'r')
  print(list(traj.keys()))
  tracer_id = traj.attrs['TracerID']
  variables = list(traj.keys())
  print(variables)
  
  xdata = np.array(traj['posx'][:])
  zdata = np.array(traj['posz'][:])
  if args.overlay_var != None:
    print(args.overlay_var)
    sc = plt.scatter(xdata, zdata, s=10, c=traj[args.overlay_var[0]][:], cmap='viridis')
  else:
    plt.plot(xdata, zdata, label=tracer_id)
  
cb = plt.colorbar(label=args.overlay_var[0])
cb.set_label(label=args.overlay_var[0], size=18)
cb.ax.tick_params(labelsize=18)

plt.legend()
plt.xlabel('Global X Pos', fontsize=18)
plt.ylabel('Global Z Pos', fontsize=18)
axes=fig.axes
for ax in axes:
  ax.tick_params(axis='both', labelsize=18)
#plt.ylim(bottom=ymin, top=ymax)
#plt.xlim(left=xmin, right=xmax)
plt.tight_layout()
plt.savefig(args.fig_name[0])



