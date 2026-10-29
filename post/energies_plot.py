#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

''' This plotter, energies_plot.py, plots the data from VPIC's internal
energies diagnostic.  Run 'python energies_plot.py -h' to see options.

Written by Scott V. Luedtke, XCP-6, April 24, 2026'''

# ============================================================================
# USER CONFIGURATION - Edit these defaults for your typical use case.  Command
# line arguments will override these.
# ============================================================================

# File and plotting options
DEFAULT_FILENAME = './rundata/energies'
DEFAULT_PLOT_FIELDS = True      # Set to False to skip E and B field plots
DEFAULT_PLOT_FIELD_TOTAL = False    # Plot total field energy (E+B)
DEFAULT_PLOT_PARTICLES = True       # Plot individual particle species energies
DEFAULT_PLOT_PARTICLE_TOTAL = False # Plot total particle energy (all species)
DEFAULT_PLOT_TOTAL = True       # Plot grand total energy
DEFAULT_LOGY = False            # Set to True for logarithmic y-axis
DEFAULT_OUTPUT = None           # Set to filename (e.g., 'plot.png') to auto-save
DEFAULT_FIGSIZE = [10, 6]       # Figure size in inches [width, height]

# Unit conversions
DEFAULT_ENERGY_SCALE = 1.0      # Factor to multiply energy values
DEFAULT_ENERGY_UNITS = 'code units'  # Label for energy axis
DEFAULT_TIME_SCALE = 1.0        # Factor to multiply time values
DEFAULT_TIME_UNITS = 'time step'     # Label for time axis

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


# Parse command-line arguments (these override the defaults above)
parser = argparse.ArgumentParser(
    prog='energies_plot.py',
    description='''
Plot VPIC energy diagnostics. All arguments are optional.  Edit the DEFAULT_*
variables at the top of the script to customize default behavior.  Command
line options override default values.
    '''.strip(),
    epilog='''
examples:
  %(prog)s                           # plot with defaults
  %(prog)s --no-plot-fields          # skip E/B field plots  
  %(prog)s --logy                    # use log scale
  %(prog)s mydata/energies --logy -o result.png
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('filename', nargs='?', default=DEFAULT_FILENAME, 
                    help=f'Path to energies file (default: {DEFAULT_FILENAME})')
parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT,
                    help='Save figure to file instead of displaying')
parser.add_argument('--plot-fields', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_PLOT_FIELDS,
                    help=f'Plot field energies (default: {DEFAULT_PLOT_FIELDS})')
parser.add_argument('--plot-field-total', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_PLOT_FIELD_TOTAL,
                    help=f'Plot total field energy (E+B) (default: {DEFAULT_PLOT_FIELD_TOTAL})')
parser.add_argument('--plot-particles', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_PLOT_PARTICLES,
                    help=f'Plot individual particle species energies (default: {DEFAULT_PLOT_PARTICLES})')
parser.add_argument('--plot-particle-total', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_PLOT_PARTICLE_TOTAL,
                    help=f'Plot total particle energy (default: {DEFAULT_PLOT_PARTICLE_TOTAL})')
parser.add_argument('--plot-total', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_PLOT_TOTAL,
                    help=f'Plot grand total energy (default: {DEFAULT_PLOT_TOTAL})')
parser.add_argument('--logy', action=argparse.BooleanOptionalAction,
                    default=DEFAULT_LOGY,
                    help=f'Use logarithmic y-axis (default: {DEFAULT_LOGY})')
parser.add_argument('--figsize', nargs=2, type=float, default=DEFAULT_FIGSIZE,
                    metavar=('WIDTH', 'HEIGHT'),
                    help=f'Figure size in inches (default: {DEFAULT_FIGSIZE})')
parser.add_argument('--energy-scale', type=float, default=DEFAULT_ENERGY_SCALE,
                    metavar='FACTOR',
                    help=f'Conversion factor for energy (default: {DEFAULT_ENERGY_SCALE})')
parser.add_argument('--energy-units', type=str, default=DEFAULT_ENERGY_UNITS,
                    metavar='UNITS',
                    help=f'Units for energy (default: "{DEFAULT_ENERGY_UNITS}")')
parser.add_argument('--time-scale', type=float, default=DEFAULT_TIME_SCALE,
                    metavar='FACTOR',
                    help=f'Conversion factor for time (default: {DEFAULT_TIME_SCALE})')
parser.add_argument('--time-units', type=str, default=DEFAULT_TIME_UNITS,
                    metavar='UNITS',
                    help=f'Units for time (default: "{DEFAULT_TIME_UNITS}")')

args = parser.parse_args()

filename = args.filename

# Read the header to get column labels and timestep
labels = []
timestep = None
try:
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('% step'):
                parts = line.strip().split()[1:]
                labels = [part.strip('"') for part in parts]
            elif 'timestep' in line.lower():
                timestep = float(line.split('=')[1].strip())
except FileNotFoundError:
    print(f"Error: File '{filename}' not found")
    sys.exit(1)
except Exception as e:
    print(f"Error reading file header: {e}")
    sys.exit(1)

if not labels:
    print("Warning: Could not find column labels in file header")

# Load the data
try:
    d = np.loadtxt(filename, comments="%")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Apply conversion factors
d[:,0] *= args.time_scale
d[:,1:] *= args.energy_scale

time = d[:,0]

E = d[:,1] + d[:,2] + d[:,3]
B = d[:,4] + d[:,5] + d[:,6]
field_total = E + B
particle_total = np.sum(d[:,7:], axis=1)  # Sum all species energies
total = field_total + particle_total

# Create figure
fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=100)

if args.plot_fields:
    plt.plot(time, E, label="Electric Field")
    plt.plot(time, B, label="Magnetic Field")
if args.plot_field_total:
    plt.plot(time, field_total, label="Total Fields", linewidth=2, linestyle='--')
if args.plot_particle_total:
    plt.plot(time, particle_total, label="Total Particles", linewidth=2, linestyle='--')
if args.plot_total:
    plt.plot(time, total, 'k-', linewidth=2.5, label="Grand Total")

if args.plot_particles:
    # Plot species data (columns 7 onwards) using labels from header
    for i in range(7, d.shape[1]):
        ax.plot(time, d[:,i], label=labels[i])

# Formatting
ax.set_ylabel(f"Energy ({args.energy_units})", fontsize=20)
ax.set_xlabel(f"Time ({args.time_units})", fontsize=20)
ax.legend(loc="best", framealpha=0.9, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
plt.xlim(left=0)

if args.logy:
    ax.set_yscale('log')
else:
    plt.ylim(bottom=0)

plt.tight_layout()

# Save or show
if args.output:
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {args.output}")
else:
    plt.show()
