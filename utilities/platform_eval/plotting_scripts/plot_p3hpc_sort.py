import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import argparse
import pandas as pd
import math
from math import log10,floor
import sys
import copy
import re
import os

plt.style.use('tableau-colorblind10')

def plot_contig(df, title=""):
  bar_order = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU', 'A64FX', 'Zen3', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A\nCPU']
  if df['Chip'].str.contains('V100').any():
    bar_order = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU']
  else:
    bar_order = ['A64FX', 'Zen3', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A\nCPU']
  contig_df = df[(df['Pattern'] == 'contig') & ( (df['Kernel'] == 'gather') | (df['Kernel'] == 'scatter') )]
  fig,ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,6))
  sns.barplot(ax=ax[0], data=contig_df[contig_df['Kernel'] == 'gather'],  x='Chip', y='Bandwidth GB/s', hue='Sort mode', errorbar='sd', order=bar_order)
  sns.barplot(ax=ax[1], data=contig_df[contig_df['Kernel'] == 'scatter'], x='Chip', y='Bandwidth GB/s', hue='Sort mode', errorbar='sd', order=bar_order, legend=False)
  ax[0].set_title("Gather", fontsize=14)
  ax[1].set_title("Scatter", fontsize=14)
  ax[0].set_xlabel("")
  ax[1].set_xlabel("")
  ax[0].set_yticklabels(ax[0].get_yticklabels(), size=12)
  ax[0].set_xticklabels(ax[0].get_xticklabels(), size=12)
  ax[1].set_xticklabels(ax[1].get_xticklabels(), size=12)
  fig.supxlabel(None)
  ax[0].set_ylabel("Bandwidth (GB/s)", fontsize=14)
  ax[0].legend(fontsize=14)
  fig.suptitle(title, fontsize=14)
  plt.tight_layout()
  fig.savefig(title.replace(': ', '_').replace(' ', '_') + ".pdf", format='pdf')

def plot_repeat(df, title=""):
  bar_order = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU', 'A64FX', 'Zen3', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A\nCPU']
  if df['Chip'].str.contains('V100').any():
    bar_order = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU']
  else:
    bar_order = ['A64FX', 'Zen3', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A\nCPU']
  repeat_df = df[(df['Pattern'] == 'repeat') & ( (df['Kernel'] == 'gather') | (df['Kernel'] == 'scatter') )]
  scatter = repeat_df[repeat_df['Sort mode'] == 'Standard'][['Kernel', 'Chip', 'Bandwidth GB/s', 'Sort mode', 'Kernel time']]
  fig,ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,6))
  sns.barplot(ax=ax[0], data=repeat_df[repeat_df['Kernel'] == 'gather'],  x='Chip', y='Bandwidth GB/s', hue='Sort mode', errorbar='sd', order=bar_order)
  sns.barplot(ax=ax[1], data=repeat_df[repeat_df['Kernel'] == 'scatter'], x='Chip', y='Bandwidth GB/s', hue='Sort mode', errorbar='sd', order=bar_order, legend=False)
  ax[0].set_title("Gather", fontsize=14)
  ax[1].set_title("Scatter (Atomic)", fontsize=14)
  ax[0].set_xlabel("")
  ax[1].set_xlabel("")
  ax[0].set_yticklabels(ax[0].get_yticklabels(), size=12)
  ax[0].set_xticklabels(ax[0].get_xticklabels(), size=12)
  ax[1].set_xticklabels(ax[1].get_xticklabels(), size=12)
  ax[0].set_ylabel("Bandwidth (GB/s)", fontsize=14)
  ax[0].legend(fontsize=14)
  fig.supxlabel(None)
  fig.suptitle(title, fontsize=14)
  #plt.yscale('log')
  plt.tight_layout()
  fig.savefig(title.replace(': ', '_').replace(' ', '_') + ".pdf", format='pdf')

def plot_stencil(df, title=""):
  bar_order = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU', 'A64FX', 'Zen3', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A\nCPU']
  if df['Chip'].str.contains('V100').any():
    bar_order = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU']
  else:
    bar_order = ['A64FX', 'Zen3', 'SPR\nDDR', 'SPR\nHBM', 'Grace', 'MI300A\nCPU']
  repeat_df = df[(df['Pattern'] == 'repeat') & ( (df['Kernel'] == 'gather-stencil') | (df['Kernel'] == 'scatter-stencil') )]
  scatter = repeat_df[repeat_df['Sort mode'] == 'Standard'][['Kernel', 'Chip', 'Bandwidth GB/s', 'Sort mode', 'Kernel time']]
  fig,ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,6))
  sns.barplot(ax=ax[0], data=repeat_df[repeat_df['Kernel'] == 'gather-stencil'],  x='Chip', y='Bandwidth GB/s', hue='Sort mode', errorbar='sd', order=bar_order)
  sns.barplot(ax=ax[1], data=repeat_df[repeat_df['Kernel'] == 'scatter-stencil'], x='Chip', y='Bandwidth GB/s', hue='Sort mode', errorbar='sd', order=bar_order, legend=False)
  ax[0].set_title("Gather", fontsize=14)
  ax[1].set_title("Scatter (Atomic)", fontsize=14)
  ax[0].set_xlabel("")
  ax[1].set_xlabel("")
  ax[0].set_yticklabels(ax[0].get_yticklabels(), size=12)
  ax[0].set_xticklabels(ax[0].get_xticklabels(), size=12)
  ax[1].set_xticklabels(ax[1].get_xticklabels(), size=12)
  ax[0].set_ylabel("Bandwidth (GB/s)", fontsize=14)
  ax[0].legend(fontsize=14)
  fig.supxlabel(None)
  fig.suptitle(title, fontsize=14)
  #plt.yscale('log')
  plt.tight_layout()
  fig.savefig(title.replace(': ', '_').replace(' ', '_') + ".pdf", format='pdf')

parser = argparse.ArgumentParser(description="Plot gather-scatter data")
parser.add_argument('logs', metavar='N', type=str, nargs='+', help='CSV logs')

args = parser.parse_args()

file_list = []
for fname in args.logs:
  if os.path.isfile(fname):
    file_list.append(fname)
  else:
    dir_contents = os.listdir(fname)
    dir_files = [fname+'/'+f for f in dir_contents if os.path.isfile(fname+'/'+f)]
    file_list.append(dir_files)
file_list = [x for item in file_list for x in item]
print(file_list)

df_list = []
for fname in file_list:
  frame = pd.read_csv(fname)
  df_list.append(frame)
df = pd.concat(df_list)
chip_id = {"V100": int(0), "A100": int(1), "H100": int(2), 'MI100': int(3), 'MI250': int(4), 'MI300A': int(5), 'A64FX': int(6), 'Zen3': int(7),  "SPRDDR": int(8), "SPRHBM":int(9), 'Grace': int(10), 'MI300A': int(11)}
name_map = {"V100": "V100", "A100": "A100", "H100": "H100", 'MI100': 'MI100', 'MI250': 'MI250\n1 GCD', 'MI300A GPU': 'MI300A\nGPU', 'A64FX': 'A64FX', 'Grace': 'Grace', 'Zen3': 'Zen3',  "SPRDDR": "SPR\nDDR", "SPRHBM":"SPR\nHBM", 'MI300A CPU': 'MI300A\nCPU'}
df['Chip ID'] = df['Chip'].map(chip_id)
df['Chip'] = df['Chip'].map(name_map)
sort_map = {'random': 'Random', 'standard': 'Standard', 'strided': 'Strided', 'tiled-strided': 'Tiled Strided'}
df['Sort mode'] = df['Sort mode'].map(sort_map)
print(df)

df["Bandwidth GB/s"] = df["Bytes touched"] / (df["Kernel time"] * 1e9)

gpus = ['V100', 'A100', 'H100', 'MI100', 'MI250\n1 GCD', 'MI300A\nGPU']
gpu_df = df[df['Chip'].isin(gpus)]
cpu_df = df[~df['Chip'].isin(gpus)]

plot_contig(cpu_df, "CPU: Contiguous keys")
plot_repeat(cpu_df, "CPU: Repeating keys")
plot_stencil(cpu_df, "CPU: Stencil")

plot_contig(gpu_df, "GPU: Contiguous keys")
plot_repeat(gpu_df, "GPU: Repeating keys")
plot_stencil(gpu_df, "GPU: Stencil")

plt.tight_layout()
plt.show()
