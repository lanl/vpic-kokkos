import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import argparse
import pandas as pd
import seaborn as sns
import math
from math import log10,floor
import sys

def plot_roofline(df, fig, ax):
  peak_fp32 = df['Peak FP32 FLOPS'].max()
  l1_bw   = df['L1 Bandwidth'].max()
  l2_bw   = df['L2 Bandwidth'].max()
  dram_bw = df['DRAM Bandwidth'].max()
  l1_x   = [0.01, peak_fp32/l1_bw, 100]
  l2_x   = [0.01, peak_fp32/l2_bw, 100]
  dram_x = [0.01, peak_fp32/dram_bw, 100]
  l1_y   = [l1_x[0] * l1_bw / (10**9), peak_fp32/(10**9), peak_fp32/(10**9)]
  l2_y   = [l1_x[0] * l2_bw / (10**9), peak_fp32/(10**9), peak_fp32/(10**9)]
  dram_y = [l1_x[0] * dram_bw / (10**9), peak_fp32/(10**9), peak_fp32/(10**9)]
  print((l1_x, l1_y))
  print((l2_x, l2_y))
  print((dram_x, dram_y))

  l1_dx = (np.log2(l1_y[1])-np.log2(l1_y[0]))
  l1_dy = (np.log2(l1_x[1])-np.log2(l1_x[0]))
  angle = np.rad2deg(np.arctan2(l1_dy, l1_dx))

  # Plot L1 roofline
  plt.loglog(l1_x, l1_y, c='k', label='_nolegend_')
  ax.annotate("L1", xy=(l1_x[0], l1_y[0]*1.1), xytext=(6,8), textcoords='offset points', rotation_mode='anchor', rotation=angle, color='black', fontweight='bold')
  
  # Plot L2 roofline
  plt.loglog(l2_x, l2_y, c='b', label='_nolegend_')
  ax.annotate("L2", xy=(l2_x[0], l2_y[0]*0.6), xytext=(6,8), textcoords='offset points', rotation_mode='anchor', rotation=angle, color='blue', fontweight='bold')
  
  # Plot DRAM roofline
  plt.loglog(dram_x, dram_y, c='r', label='_nolegend_')
  ax.annotate("DRAM", xy=(dram_x[0], dram_y[0]*1.1), xytext=(6,8), textcoords='offset points', rotation_mode='anchor', rotation=angle, color='red', fontweight='bold')

def plot_kernels(df, fig, ax):
  df['Kernel GFLOP/s'] = df['Kernel FLOPS'] / (10**9)
  sns.scatterplot(data=df, x='L1 AI',   y='Kernel GFLOP/s', c='k', style='Sort Mode', s=64, ax=ax, legend=False)
  sns.scatterplot(data=df, x='L2 AI',   y='Kernel GFLOP/s', c='b', style='Sort Mode', s=64, ax=ax, legend=False)
  sns.scatterplot(data=df, x='DRAM AI', y='Kernel GFLOP/s', c='r', style='Sort Mode', s=64, ax=ax)
  ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)", fontsize=14)
  ax.set_ylabel("Performance (GFLOPs)", fontsize=14)
  for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
  ax.set_yticklabels(ax.get_yticks(), size=12)
  ax.set_xticklabels(ax.get_xticks(), size=12)

parser = argparse.ArgumentParser(description="Roofline visualizer")
parser.add_argument('roofline_data', metavar='N', type=str, nargs=1, help='Roofline data (csv)')

args = parser.parse_args()
df = pd.read_csv(args.roofline_data[0])
print(df)

a100_df = df[df['Chip'] == 'A100']
h100_df = df[df['Chip'] == 'H100']
mi250_df = df[df['Chip'] == 'MI250']
mi300_df = df[df['Chip'] == 'MI300A']

fig,ax = plt.subplots()

plot_roofline(mi300_df, fig, ax)
plot_kernels(mi300_df, fig, ax)

plt.tight_layout()
plt.show()
