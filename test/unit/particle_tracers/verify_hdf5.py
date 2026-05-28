#!/usr/bin/env python
import h5py
import numpy as np
import pandas as pd
import os.path
import sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
if len(sys.argv) != 2:
    sys.stderr.write("Usage: "+str(sys.argv[0])+" rundir\n")
    sys.exit(1)

rundir = sys.argv[1]
print(rundir)

for test in ['tracers/ion_tracers_percentage', 'tracers/ion_tracers_predicate']:
  for buffered in [False, True]:
    print("Testing: " + test)
    print("HDF5 buffering: " + str(buffered))

    h5_filename = rundir + '/' + test
    if buffered:
      h5_filename += '_buffered'
    
    f = h5py.File(h5_filename, 'r')
    print("Opened up file " + h5_filename)
    
    for tframe in range(0,110,10):
      csv_filename = rundir + '/' + test + '.' + str(tframe) + '.0.csv'
    
      csv_df = pd.read_csv(csv_filename)
      csv_df.rename(columns={'cell_id': 'i'}, inplace=True)
      csv_df = csv_df[csv_df['Timestep'] == tframe]
      filtered_cols = list(csv_df.columns)
      filtered_cols.remove('Timestep')
      filtered_cols.remove('rank')
      filtered_cols.remove('tracer_id')
      filtered_csv_df = csv_df[filtered_cols]
      filtered_csv_df = filtered_csv_df.sort_values(by='TracerID', ignore_index=True)
      
      step = f['Timestep_' + str(tframe)]
      h5_dict = {}
      for k,v in step.items():
        h5_dict[k] = np.array(v)
      
      h5_df = pd.DataFrame(h5_dict)
      reordered_h5_df = h5_df[filtered_cols]
      reordered_h5_df = reordered_h5_df.sort_values(by='TracerID', ignore_index=True)
      
      pd.set_eng_float_format(accuracy=6)
      frames_same = np.allclose(filtered_csv_df, reordered_h5_df)
      print("Step " + str(tframe) + ": CSV and HDF5 match? " + str(frames_same))
      if not frames_same:
        print("CSV")
        print(filtered_csv_df)
        print('====')
        print("HDF5")
        print(reordered_h5_df)
        print('====')
        print(np.isclose(filtered_csv_df, reordered_h5_df))
        print(np.logical_not(np.isclose(filtered_csv_df, reordered_h5_df)))
        print('====')
        print(filtered_csv_df.compare(reordered_h5_df))
        print('====')
        not_close = pd.DataFrame(np.logical_not(np.isclose(filtered_csv_df, reordered_h5_df)))
        not_close.columns = filtered_csv_df.columns
        #print(not_close.to_string())
        print('====')
        print(filtered_csv_df[not_close].to_string())
        print('====')
        print(reordered_h5_df[not_close].to_string())
        sys.exit(1)
