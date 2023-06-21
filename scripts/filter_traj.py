#!/usr/bin/python3

import argparse
import h5py
import os
import re

# Basic example for filtering out tracers
# Extract any necessary values for filtering into a dictionary 
# then use the dictionary to select which tracers to keep

# Store energy of particle for each trajectory file in a dictionary with the tracer ID as the key
def extract_quantity(tracer_data):
  # Get last timestep
  timestep_list = list(tracer_data.keys())
  timestep_list.sort(key=natural_sort_key)
  last_timestep = tracer_data[timestep_list[-1]]
  
  # Retrieve Tracer ID and energy datasets
  ids = last_timestep['TracerID']
  energy = last_timestep['ke']
  
  # Create dictionary
  energy_dict = dict(zip(ids, energy))
  return energy_dict

# Sort tracers by energy at the last timestep and select the N highest energy particles
def select_trajectories(data_dict, N):
  sorted_list = list(dict(sorted(data_dict.items(), key=lambda item: item[1])))
  return sorted_list[-N:]

# Natural sort
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
  return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

# Read in args
parser = argparse.ArgumentParser(description='Filter trajectories')
parser.add_argument('filename', metavar='filename', type=str, nargs=1, help='Input file')
args = parser.parse_args()

filename = args.filename[0]

# Read in tracer data
tracer_data = h5py.File(filename, 'r')

# Retrieve any data necessary for filtering tracers
data_dict = extract_quantity(tracer_data)

# Filter tracers
selected_entries = select_trajectories(data_dict, 10)

# Print out selected tracers
for entry in selected_entries[-10:]:
  print(entry, data_dict[entry])
