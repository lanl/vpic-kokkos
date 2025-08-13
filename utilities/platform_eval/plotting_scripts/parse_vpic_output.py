import pandas as pd
import argparse
import copy
import re
import os

def parse_filename(filename):
  info = {}
  split_filename = re.split("_|/|\\.", filename)
  #split_filename = filename.split('_')
  print(split_filename)
  if 'a64fx' in split_filename:
    info['Chip'] = 'a64fx'
  elif 'ampere1' in split_filename:
    info['Chip'] = 'ampere1'
  elif 'epyc7742' in split_filename:
    info['Chip'] = 'epyc7742'
  elif 'epyc7763' in split_filename:
    info['Chip'] = 'epyc7763'
  elif 'icelake' in split_filename:
    info['Chip'] = 'icelake'
  elif 'sprddr' in split_filename:
    info['Chip'] = 'sprddr'
  elif 'sprhbm' in split_filename:
    info['Chip'] = 'sprhbm'
  elif 'thunderx2' in split_filename:
    info['Chip'] = 'thunderx2'
  elif 'a100' in split_filename:
    info['Chip'] = 'a100'
  elif 'h100' in split_filename:
    info['Chip'] = 'h100'
  elif 'mi100' in split_filename:
    info['Chip'] = 'mi100'
  elif 'mi250' in split_filename:
    info['Chip'] = 'mi250'
  elif 'mi300a' in split_filename:
    info['Chip'] = 'mi300a'
  elif 'grace' in split_filename:
    info['Chip'] = 'grace'
  
  info['Vectorization'] = 'auto'
  if 'neon' in split_filename:
    info['Vectorization'] = 'neon'
  elif 'auto-simd-avx2' in split_filename:
    info['Vectorization'] = 'auto_simd_avx2'
  elif 'auto-simd-avx512' in split_filename:
    info['Vectorization'] = 'auto_simd_avx512'
  elif 'simd-avx2' in split_filename:
    info['Vectorization'] = 'simd_avx2'
  elif 'simd-avx512' in split_filename:
    info['Vectorization'] = 'simd_avx512'
  elif 'avx2' in split_filename:
    info['Vectorization'] = 'avx2'
  elif 'guided' in split_filename:
    info['Vectorization'] = 'guided'
  elif 'manual' in split_filename:
    info['Vectorization'] = 'manual'
  elif 'adhoc' in split_filename:
    info['Vectorization'] = 'ad-hoc'

  if 'standard' in split_filename:
    info['Sort Order'] = 'standard'
  elif 'tiled' in split_filename:
    info['Sort Order'] = 'tiled-strided'
  elif 'strided' in split_filename:
    info['Sort Order'] = 'strided'
  else:
    info['Sort Order'] = 'random'

#  for substr in split_filename:
#    if 'vpic' in substr:
#      info['Version'] = re.findall(r"\d+\.\d+", substr)[0]
#    if 'rank' in substr:
#      info['Ranks'] = re.findall(r"\d+", substr)[0]
#    if 'thread' in substr:
#      info['Threads'] = re.findall(r"\d+", substr)[0]
  
  print(info)
  return info
      
#def read_profiles_old(file_list, timer_labels):
#  df = pd.DataFrame()
#  for fname in file_list:
##    row_dict = parse_filename(fname)
#  
#    lines = []
#    with open(fname) as file:
#      lines = [line.rstrip() for line in file]
#    
#    starts = []
#    ends = []
#    for i in range(len(lines)):
#      line = lines[i]
#      if '*** Initializing' in line:
#        starts.append(i)
#      if 'normal exit' in line:
#        ends.append(i)
#  
#    row_dicts = []
#    for i in range(len(starts)):
#      temp_dict = row_dict
#      for j in range(starts[i], ends[i]):
#        line = lines[j]
#        if '*** Done' in line:
#          numbers = re.findall(r"\d+\.\d+", line)
#          temp_dict['Total Time'] = float(numbers[0])
#        for label in timer_labels:
#          if label in line:
#            split_line = line.split()
#            temp_dict[label] = float(split_line[8])
#      row_dicts.append(temp_dict)
#      temp_df = pd.DataFrame(data=temp_dict, columns=column_labels, index=[0])
#      df = pd.concat([df, temp_df], ignore_index=True)
#  
#  return df

def read_profiles(file_list, details_labels):
  frames = [];
  for fname in file_list:
    df = pd.DataFrame()
    fname_dict = parse_filename(fname)
    #print(fname)
    #print(fname_dict)
    lines = []
    with open(fname) as file:
      lines = [line for line in file]

    starts = []
    ends = []
    start_found = False

    details_dict = {}
    for i in range(len(lines)):
      line = lines[i]
      if 'Initialization complete' in line or 'Completed step' in line or 'Cleaning up' in line:
        starts.append(i+5)
        start_found = True
        #print("Start (" + str(i+5) + "): " + line)
      if (len(starts) == len(ends)+1) and (i>starts[-1]) and (line in ['\n', '\r\n']):
        ends.append(i)
        start_found = False
        #print("End (" + str(i) + "): " + line)
      for label,search_str in details_labels.items():
        if search_str in line:
          details_dict[label] = line.split()[-1]
        if '# px' in line:
          details_dict['Topology X'] = line.split()[2]
          details_dict['Topology Y'] = line.split()[4]
          details_dict['Topology Z'] = line.split()[6]
        if '# gnx' in line:
          details_dict['Global NX'] = line.split()[2]
          details_dict['Global NY'] = line.split()[4]
          details_dict['Global NZ'] = line.split()[6]
        if '# dx' in line:
          details_dict['Global DX'] = line.split()[2]
          details_dict['Global DY'] = line.split()[4]
          details_dict['Global DZ'] = line.split()[6]
        if '# dt' in line:
          details_dict['Dt'] = line.split()[2]
          details_dict['Cvac'] = line.split()[4]
          details_dict['Eps0'] = line.split()[6]
        if '# nx' in line:
          details_dict['Local NX'] = line.split()[2]
          details_dict['Local NY'] = line.split()[4]
          details_dict['Local NZ'] = line.split()[6]
        if 'Completed step' in line:
          details_dict['Num Time Steps'] = line.split()[5]
    ends.append(len(lines))

    rows = []
    for i in range(len(starts)):
      step = 0
      done_time = 0
      if 'Initialization complete' in lines[starts[i]-5]:
        step = 0
        done_time = 0.0
      elif 'Completed step' in lines[starts[i]-5]:
        step = int(lines[starts[i]-5].split()[3])
        done_time = 0.0 
      elif 'Cleaning up' in lines[starts[i]-5]:
        step = int(details_dict['Num Time Steps'])+1
        done_time = float(lines[starts[i]-6].split()[2][1:-2])

      for j in range(starts[i], ends[i]):
        temp_dict = {'Step': step, 'Done Time': done_time}
        line = lines[j]
        splitline = line.split()
        #print(splitline)
        if len(splitline) == 11:
          temp_dict['Timer']                       = splitline[0]
          temp_dict['% Since Last Update']         = splitline[2]
          temp_dict['Time Since Last Update']      = splitline[3]
          temp_dict['Count Since Last Update']     = splitline[4]
          temp_dict['Per Step Since Last Update']  = splitline[5]
          temp_dict['% Since Last Restore']        = splitline[7]
          temp_dict['Time Since Last Restore']     = splitline[8]
          temp_dict['Count Since Last Restore']    = splitline[9]
          temp_dict['Per Step Since Last Restore'] = splitline[10]
          #temp_dict['Min (All Ranks)']          = splitline[7]
          #temp_dict['Max (All Ranks)']          = splitline[8]
          #temp_dict['Min/Max (All Ranks)']      = splitline[9]
          #temp_dict['Me/Max (All Ranks)']       = splitline[10]
          #temp_dict['Max Rank']                 = splitline[11]
          #temp_dict['% Since Last Restore']     = splitline[13]
          #temp_dict['Time Since Last Restore']  = splitline[14]
          #temp_dict['Count Since Last Restore'] = splitline[15]
        elif len(splitline) == 7:
          temp_dict['Timer']                       = splitline[0]
          temp_dict['% Since Last Update']         = 0 
          temp_dict['Time Since Last Update']      = 0 
          temp_dict['Count Since Last Update']     = 0 
          temp_dict['Per Step Since Last Update']  = 0 
          temp_dict['% Since Last Restore']        = splitline[3]
          temp_dict['Time Since Last Restore']     = splitline[4]
          temp_dict['Count Since Last Restore']    = splitline[5]
          temp_dict['Per Step Since Last Restore'] = splitline[6]
          #temp_dict['Min (All Ranks)']          = 0 
          #temp_dict['Max (All Ranks)']          = 0 
          #temp_dict['Min/Max (All Ranks)']      = 0 
          #temp_dict['Me/Max (All Ranks)']       = 0 
          #temp_dict['Max Rank']                 = 0 
          #temp_dict['% Since Last Restore']     = splitline[4]
          #temp_dict['Time Since Last Restore']  = splitline[5]
          #temp_dict['Count Since Last Restore'] = splitline[6]

        #print(temp_dict)
        rows.append(temp_dict)
      #print(rows)
    #print("Rows")
    #print(rows)
    df = pd.DataFrame.from_dict(rows, orient='columns')
    for k,v in fname_dict.items():
      df[k] = v
    for k,v in details_dict.items():
      df[k] = v
    #print(df.columns)
    frames.append(df)
  return pd.concat(frames)
 

parser = argparse.ArgumentParser(description="VPIC runtime output parser")
parser.add_argument('profiles', metavar='N', type=str, nargs='+', help='VPIC output')
parser.add_argument('--output-fname', type=str, nargs=1, default='vpic_profiles.csv', help='Filename for writing profile output')
args = parser.parse_args()

files = args.profiles
file_list = []
for fname in files:
  if os.path.isfile(fname):
    file_list.append(fname)
  else:
    dir_contents = os.listdir(fname)
    dir_files = [fname+f for f in dir_contents if os.path.isfile(fname+'/'+f)]
    file_list.append(dir_files)

# columns for simd tests
#column_labels = [ 'Chip', 'Version', 'Sort Order', 'Vectorization', 'Ranks', 'Threads',
#                  'Total Time']
details_labels = {######## Build Details ##########
                 'VPIC Git Hash': 'VPIC Git Hash:',
                 'Deck Name': '.cxx', 
                 'Built Date': 'Built on:',
                 # CMake options ##
                 'VARIABLE_CHARGE': 'VARIABLE_CHARGE:',
                 'ENABLE_INTEGRATED_TESTS': 'ENABLE_INTEGRATED_TESTS:', 
                 'ENABLE_UNIT_TESTS':'ENABLE_UNIT_TESTS:', 
                 'USE_V4_ALTIVEC':'USE_V4_ALTIVEC:', 
                 'USE_V4_PORTABLE':'USE_V4_PORTABLE:', 
                 'USE_V4_SSE':'USE_V4_SSE:', 
                 'ENABLE_OPENSSL':'ENABLE_OPENSSL:', 
                 'ENABLE_KOKKOS_OPENMP':'ENABLE_KOKKOS_OPENMP:', 
                 'ENABLE_KOKKOS_CUDA':'ENABLE_KOKKOS_CUDA:', 
                 'BUILD_INTERNAL_KOKKOS':'BUILD_INTERNAL_KOKKOS:', 
                 'VPIC_DUMP_ENERGIES':'VPIC_DUMP_ENERGIES:', 
                 'VPIC_ENABLE_AUTO_TUNING':'VPIC_ENABLE_AUTO_TUNING:', 
                 'VPIC_ENABLE_HIERARCHICAL':'VPIC_ENABLE_HIERARCHICAL:', 
                 'VPIC_ENABLE_TEAM_REDUCTION':'VPIC_ENABLE_TEAM_REDUCTION:', 
                 'VPIC_ENABLE_VECTORIZATION':'VPIC_ENABLE_VECTORIZATION:', 
                 'VPIC_ENABLE_ACCUMULATOR':'VPIC_ENABLE_ACCUMULATOR:', 
                 'Sort Method':'Using sort method:',
                 ######## End Build Details ########
                 ######## Begin Run Details ########
                 'Num Processes': 'MPI Ranks:',
                 'Num Threads per Process': 'Threads:',
                 'Num Time Steps': 'Num Step',
                 'Topology X': 'px',
                 'Topology Y': 'py',
                 'Topology Z': 'pz',
                 'Global NX': 'gnx',
                 'Global NY': 'gny',
                 'Global NZ': 'gnz',
                 'Global DX': 'dx',
                 'Global DY': 'dx',
                 'Global DZ': 'dx',
                 'Dt': 'dt',
                 'Cvac': 'cvac',
                 'Eps0': 'eps0',
                 'Local NX': 'nx',
                 'Local NY': 'ny',
                 'Local NZ': 'nz',
                 'Num Particle Species': 'Local Particle Species:',
}
update_labels = ['Step',
                 'Timer',
                 '% Since Last Update',
                 'Time Since Last Update',
                 'Count Since Last Update',
                 'Min (All Ranks)',
                 'Max (All Ranks)',
                 'Min/Max (All Ranks)',
                 'Me/Max (All Ranks)',
                 'Max Rank',
                 '% Since Last Restore',
                 'Time Since Last Restore',
                 'Count Since Last Restore',
                 'Done Time'
]
timer_labels  = ['clear_accumulators',
                 'sort_p',
                 'sorting_particles',
                 'advance_p',
                 'reduce_accumulators',
                 'boundary_p',
                 'clear_jf',
                 'unload_accumulator',
                 'synchronize_jf',
                 'advance_b',
                 'advance_e',
                 'clear_rhof',
                 'accumulate_rho_p',
                 'synchronize_rho',
                 'compute_div_e_err',
                 'compute_rms_div_e_err',
                 'clean_div_e',
                 'compute_div_b_err',
                 'compute_rms_div_b_err',
                 'clean_div_b',
                 'synchronize_tang_e_norm_b',
                 'load_interpolator',
                 'compute_curl_b',
                 'compute_rhob',
                 'uncenter_p',
                 'user_initialization',
                 'user_particle_collisions',
                 'user_particle_injection',
                 'user_current_injection',
                 'user_field_injection',
                 'user_diagnostics'
]


#df = read_profiles_old(file_list, timer_labels)
df = read_profiles(file_list[0], details_labels)
print(df.columns)
print(df[['Step', 'Timer', 'Time Since Last Update']])
df.to_csv(args.output_fname[0])
