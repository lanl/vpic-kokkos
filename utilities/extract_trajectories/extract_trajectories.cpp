#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include "hdf5.h"
#include "mpi.h"

//#define DEBUG

#ifdef DEBUG
 #define DEBUG_PRINT(...) do { fprintf(stderr, __VA_ARGS__); } while(0)
#else
 #define DEBUG_PRINT(...) do { } while (0)
#endif

// Retrieve all group names
herr_t get_group_names(hid_t group, const char * name, const H5L_info2_t* info, void* op_data) {
  auto timesteps = reinterpret_cast<std::vector<std::string>*>(op_data);
  timesteps->push_back(std::string(name));
  return 0;
}

// Compare function for sorting groups by Timestep_N
bool timestep_sort(std::string i, std::string j) {
  std::string i_int_str;
  int i_int = atoi(i.c_str()+9);
  int j_int = atoi(j.c_str()+9);
  return (i_int < j_int);
}

template <typename Out>
void split(const std::string &s, char delim, Out result) {
  std::istringstream iss(s);
  std::string item;
  while(std::getline(iss, item, delim)) {
    if(!item.empty())
      *result++ = item;
  }
}

std::vector<std::string> split(const std::string &s, char delip) {
  std::vector<std::string> elems;
  split(s, delip, std::back_inserter(elems));
  return elems;
}

int main(int argc, char**argv) {
  MPI_Init(&argc, &argv);
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int arg_offset = 0;
  bool select = false;
  std::set<int> selected_tracers;
  auto select_arg = std::find(argv, argv+argc, std::string("--select-tracers"));
  if(select_arg != argv+argc) {
    select = true;
    arg_offset += 2;
    std::string tracers(*(select_arg+1));
    std::vector<std::string> selected_ids = split(tracers, ',');
    for(size_t i=0; i<selected_ids.size(); i++) {
      selected_tracers.insert(std::stoi(selected_ids[i].c_str(), NULL, 10));
    }
  }

  if(comm_rank == 0 && selected_tracers.size() > 0) {
    std::cout << "Selected tracers: ";
    for(auto it=selected_tracers.begin(); it!=selected_tracers.end(); it++) {
      std::cout << *it << ", ";
    }
    std::cout << std::endl;
  }

  const char* fname = argv[1+arg_offset];

  hid_t file_id, access_id;

  // Open HDF5 file
  access_id = H5Pcreate(H5P_FILE_ACCESS);
  herr_t ret = H5Pset_fapl_mpio(access_id, MPI_COMM_WORLD, MPI_INFO_NULL);
  if(ret == H5I_INVALID_HID)
    printf("Failed to set MPIO file access permission list\n");

  file_id = H5Fopen(fname, H5F_ACC_RDWR, access_id);

  std::vector<std::string> timesteps;
 
  // Get list of timestep names
  ret = H5Literate(file_id, H5_INDEX_NAME, H5_ITER_INC, NULL, get_group_names, &timesteps);
  hsize_t num_timesteps = timesteps.size();
  std::sort(timesteps.begin(), timesteps.end(), timestep_sort);

  std::vector<std::string> tracer_vars;
  std::vector<hid_t> tracer_var_types;
  // Maps for each main data type <TracerID, <VarName, Data> >
  std::map<int, std::map<std::string, std::vector<int> > >     traj_int;
  std::map<int, std::map<std::string, std::vector<int64_t> > > traj_int64_t;
  std::map<int, std::map<std::string, std::vector<float> > >   traj_float;
  std::map<int, std::map<std::string, std::vector<double> > >  traj_double;

  //***************************************************************************
  // Extract tracer info
  //***************************************************************************

  // Divide timesteps amoung ranks
  int steps_per_proc = timesteps.size()/comm_size;
  int leftover = 0;
  if(static_cast<uint32_t>(steps_per_proc*comm_size) < timesteps.size())
    leftover = timesteps.size()-steps_per_proc*comm_size;
  int beg_step = steps_per_proc*comm_rank;
  int end_step = steps_per_proc*(comm_rank+1);
  if(comm_rank > comm_size-leftover) {
    beg_step = (comm_size-leftover)*steps_per_proc + (comm_rank-(comm_size-leftover))*(steps_per_proc+1);
    end_step = beg_step + steps_per_proc+1;
  }
  if(static_cast<uint32_t>(end_step) > timesteps.size())
    end_step = timesteps.size();
  steps_per_proc = end_step - beg_step;
  DEBUG_PRINT("Rank %d [%d,%d]: %d out of %llu\n", comm_rank, beg_step, end_step, steps_per_proc, num_timesteps);

  // Extract trajectory data
  for(int i=beg_step; i<end_step; i++) {
    char stepstr[20];
    timesteps[i].copy(stepstr, 20, 9);
    int step = std::stoi(stepstr);

    // Open timestep
    hid_t group_id = H5Gopen(file_id, timesteps[i].c_str(), H5P_DEFAULT);
    hid_t dataset_id = H5Dopen(group_id, "TracerID", H5P_DEFAULT);
    hid_t dataspace_id = H5Dget_space(dataset_id);

    // Get number of tracers
    hsize_t ntracers;
    H5Sget_simple_extent_dims(dataspace_id, &ntracers, NULL);
    std::vector<int> tracer_ids(ntracers, -1);

    // Load tracer IDs
    ret = H5Dread(dataset_id, H5T_STD_I32LE, dataspace_id, H5S_ALL, H5P_DEFAULT, tracer_ids.data());
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);

    // Add timestep entry
    for(uint32_t j=0; j<tracer_ids.size(); j++) {
      if(i == beg_step) 
        traj_int[tracer_ids[j]]["Timestep"] = std::vector<int>();
      traj_int[tracer_ids[j]]["Timestep"].push_back(step);
    }

    // Get list of variables
    tracer_vars.clear();
    tracer_var_types.clear();
    ret = H5Literate(group_id, H5_INDEX_NAME, H5_ITER_INC, NULL, get_group_names, &tracer_vars);

    // Load each variable and store in trajectories
    for(uint32_t j=0; j<tracer_vars.size(); j++) {
      dataset_id = H5Dopen(group_id, tracer_vars[j].c_str(), H5P_DEFAULT);
      dataspace_id = H5Dget_space(dataset_id);
      tracer_var_types.push_back(H5Dget_type(dataset_id));
      auto data_type = H5Dget_type(dataset_id);
      if(H5Tequal(data_type, H5T_IEEE_F32LE)) {
        std::vector<float> scratch(ntracers, 0);
        ret = H5Dread(dataset_id, data_type, dataspace_id, H5S_ALL, H5P_DEFAULT, scratch.data());
        for(uint32_t k=0; k<tracer_ids.size(); k++) {
          if(i == beg_step) 
            traj_float[tracer_ids[k]][tracer_vars[j]] = std::vector<float>();
          traj_float[tracer_ids[k]][tracer_vars[j]].push_back(scratch[k]);
        }
      } else if(H5Tequal(data_type, H5T_IEEE_F64LE)) {
        std::vector<double> scratch(ntracers, 0);
        ret = H5Dread(dataset_id, data_type, dataspace_id, H5S_ALL, H5P_DEFAULT, scratch.data());
        for(uint32_t k=0; k<tracer_ids.size(); k++) {
          if(i == beg_step) 
            traj_double[tracer_ids[k]][tracer_vars[j]] = std::vector<double>();
          traj_double[tracer_ids[k]][tracer_vars[j]].push_back(scratch[k]);
        }
      } else if(H5Tequal(data_type,H5T_STD_I32LE)) {
        std::vector<int> scratch(ntracers, 0);
        ret = H5Dread(dataset_id, data_type, dataspace_id, H5S_ALL, H5P_DEFAULT, scratch.data());
        for(uint32_t k=0; k<tracer_ids.size(); k++) {
          if(i == beg_step) 
            traj_int[tracer_ids[k]][tracer_vars[j]] = std::vector<int>();
          traj_int[tracer_ids[k]][tracer_vars[j]].push_back(scratch[k]);
        }
      } else if(H5Tequal(data_type, H5T_STD_I64LE)) {
        std::vector<int64_t> scratch(ntracers, 0);
        ret = H5Dread(dataset_id, data_type, dataspace_id, H5S_ALL, H5P_DEFAULT, scratch.data());
        for(uint32_t k=0; k<tracer_ids.size(); k++) {
          if(i == beg_step) 
            traj_int64_t[tracer_ids[k]][tracer_vars[j]] = std::vector<int64_t>();
          traj_int64_t[tracer_ids[k]][tracer_vars[j]].push_back(scratch[k]);
        }
      }
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
    }
    H5Gclose(group_id);
  }
  ret = H5Pclose(access_id);
  ret = H5Fclose(file_id);

  MPI_Barrier(MPI_COMM_WORLD);
  if(comm_rank == 0)
    std::cout << "Done reading tracer data into vectors\n";
  
  //***************************************************************************
  // Write trajectory files
  //***************************************************************************

  // Identify all TracerIDs for each data type
  std::set<int> tracer_id_set;
  for(auto it=traj_int.begin(); it!=traj_int.end(); it++) {
    tracer_id_set.insert(it->first);
  }
  for(auto it=traj_int64_t.begin(); it!=traj_int64_t.end(); it++) {
    tracer_id_set.insert(it->first);
  }
  for(auto it=traj_float.begin(); it!=traj_float.end(); it++) {
    tracer_id_set.insert(it->first);
  }
  for(auto it=traj_double.begin(); it!=traj_double.end(); it++) {
    tracer_id_set.insert(it->first);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<int> tracer_id_vec;
  for(auto it=tracer_id_set.begin(); it!=tracer_id_set.end(); it++) {
    tracer_id_vec.push_back(*it);
  }
  if(comm_rank == 0)
    printf("Done loading tracers into a vector for communication\n");

  std::map<int,int> ntimesteps;
  for(auto it=tracer_id_set.begin(); it!=tracer_id_set.end(); it++) {
    if(traj_int.find(*it) != traj_int.end()) {
      ntimesteps[*it] = (((traj_int[*it]).begin())->second).size();
    }
    if(traj_int64_t.find(*it) != traj_int64_t.end()) {
      ntimesteps[*it] = (((traj_int64_t[*it]).begin())->second).size();
    }
    if(traj_float.find(*it) != traj_float.end()) {
      ntimesteps[*it] = (((traj_float[*it]).begin())->second).size();
    }
    if(traj_double.find(*it) != traj_double.end()) {
      ntimesteps[*it] = (((traj_double[*it]).begin())->second).size();
    }
  }

  for(auto it=ntimesteps.begin(); it!=ntimesteps.end(); it++) {
    DEBUG_PRINT("Rank: %d: Tracer %d has %d timesteps\n", comm_rank, it->first, it->second);
  }

  std::vector<int> tracers_per_proc(comm_size, 0);
  tracers_per_proc[comm_rank] = tracer_id_vec.size();
  DEBUG_PRINT("Rank %d should have %zu tracers\n", comm_rank, tracer_id_vec.size());
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tracers_per_proc.data(), 1, MPI_INT, MPI_COMM_WORLD); 
  if(comm_rank == 0) {
    for(uint32_t i=0; i<tracers_per_proc.size(); i++) {
      DEBUG_PRINT("Rank %d has %d tracers\n", i, tracers_per_proc[i]);
    }
  }
  int max_possible_tracers = 0;
  for(uint32_t i=0; i<tracers_per_proc.size(); i++) {
    max_possible_tracers += tracers_per_proc[i];
  }
  std::vector<int> all_possible_tracer_ids(max_possible_tracers, -1);
  std::vector<int> displ(comm_size, 0);
  for(int i=1; i<comm_size; i++) {
    displ[i] += tracers_per_proc[i-1] + displ[i-1];
  }
  if(comm_rank == 0) {
    for(uint32_t i=0; i<displ.size(); i++) {
      DEBUG_PRINT("Rank %d displ: %d\n", i, displ[i]);
    }
  }
  for(uint32_t i=0; i<tracer_id_vec.size(); i++) {
    all_possible_tracer_ids[displ[comm_rank]+i] = tracer_id_vec[i];
  }
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_possible_tracer_ids.data(), tracers_per_proc.data(), displ.data(), MPI_INT32_T, MPI_COMM_WORLD);
  if(comm_rank == 0) {
    for(uint32_t i=0; i<all_possible_tracer_ids.size(); i++) {
      DEBUG_PRINT("TracerID: %d\n", all_possible_tracer_ids[i]);
    }
  }

  std::set<int> all_tracers_set;
  for(uint32_t i=0; i<all_possible_tracer_ids.size(); i++) {
    all_tracers_set.insert(all_possible_tracer_ids[i]);
  }
  for(auto it=all_tracers_set.begin(); it!=all_tracers_set.end(); it++) {
    if(ntimesteps.find(*it) == ntimesteps.end()) {
      ntimesteps[*it] = 0;
    }
  }

  auto tracer_set = all_tracers_set;
  if(select)
    tracer_set = selected_tracers;

  if(comm_rank == 0) 
    std::cout << "Done setting up tracer data\n";

  // Iterate through all tracers
  for(auto it=tracer_set.begin(); it!=tracer_set.end(); it++) {
    int id = *it;
    // Create trajectory file name
    std::string particle_fname = std::string(fname);
    particle_fname.erase(particle_fname.end()-3, particle_fname.end());
    particle_fname = particle_fname + "." + std::to_string(id) + ".traj.h5";
    // Split communicator
    MPI_Comm stepcomm;
    MPI_Comm_split(MPI_COMM_WORLD, ntimesteps[id]>0, comm_rank, &stepcomm);
    int step_proc_rank, step_proc_size;
    MPI_Comm_rank(stepcomm, &step_proc_rank);
    MPI_Comm_size(stepcomm, &step_proc_size);
    // Only ranks with time steps that matter to the tracer write data
    if(ntimesteps[id]>0) {
      // Start offset and number of elements for this process
      hsize_t offset = beg_step;
      hsize_t count = ntimesteps[id];
      // Create file access property list with parallel I/O access
      hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
      H5Pset_fapl_mpio(plist_id, stepcomm, MPI_INFO_NULL);
      // Create trajectory file collectively
      hid_t tracer_fid = H5Fcreate(particle_fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
      H5Pclose(plist_id);
      // Create attribute for tracer ID
      hid_t acpl_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
      hid_t attr_fspace_id = H5Screate(H5S_SCALAR);
      hid_t attr_id = H5Acreate(tracer_fid, "TracerID", H5T_STD_I32LE, attr_fspace_id, acpl_id, H5P_DEFAULT);
      H5Awrite(attr_id, H5T_STD_I32LE, &id);
      H5Aclose(attr_id);
      H5Sclose(attr_fspace_id);
      H5Pclose(acpl_id);
      // Create memspace for this processes chunk of data
      hsize_t memspace_len = 0;
      if(tracer_id_set.find(id) != tracer_id_set.end()) 
        memspace_len = static_cast<hsize_t>(ntimesteps[id]);
      hid_t memspace_id = H5Screate_simple(1, &memspace_len, NULL);
      // Create property list for collective operations
      hid_t plist_id2 = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist_id2, H5FD_MPIO_COLLECTIVE);
      // Create dataspace for all timesteps
      num_timesteps = memspace_len;
      hsize_t total_timesteps;
      DEBUG_PRINT("Rank %d has %llu timesteps for tracer %d\n", comm_rank, num_timesteps, id);
      MPI_Allreduce(&num_timesteps, &total_timesteps, 1, MPI_LONG_LONG, MPI_SUM, stepcomm);
      MPI_Scan(&num_timesteps, &offset, 1, MPI_LONG_LONG, MPI_SUM, stepcomm);
      offset -= num_timesteps;
      DEBUG_PRINT("Rank %d Tracer: %d, offset: %llu, count: %llu\n", step_proc_rank, id, offset, count);
      if(comm_rank == 0) 
        DEBUG_PRINT("%llu timesteps for tracer %d\n", num_timesteps, id);
      hid_t dataspace_id = H5Screate_simple(1, (hsize_t*)(&total_timesteps), NULL);

      // Write slab of int data for all int variables
      for(auto var_it=traj_int[id].begin(); var_it!=traj_int[id].end(); var_it++) {
        if(strcmp("TracerID", var_it->first.c_str()) != 0) {
          hid_t var_id = H5Dcreate(tracer_fid, var_it->first.c_str(), H5T_STD_I32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
          hid_t filespace_id = H5Dget_space(var_id);
          H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL);
          ret = H5Dwrite(var_id, H5T_STD_I32LE, memspace_id, filespace_id, H5P_DEFAULT, var_it->second.data());
          H5Sclose(filespace_id);
          H5Dclose(var_id);
        }
      }

      // Write slab of int64_t data for all int64_t variables
      for(auto var_it=traj_int64_t[id].begin(); var_it!=traj_int64_t[id].end(); var_it++) {
        hid_t var_id = H5Dcreate(tracer_fid, var_it->first.c_str(), H5T_STD_I64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t filespace_id = H5Dget_space(var_id);
        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL);
        ret = H5Dwrite(var_id, H5T_STD_I64LE, memspace_id, filespace_id, H5P_DEFAULT, var_it->second.data());
        H5Sclose(filespace_id);
        H5Dclose(var_id);
      }

      // Write slab of float data for all float variables
      for(auto var_it=traj_float[id].begin(); var_it!=traj_float[id].end(); var_it++) {
        hid_t var_id = H5Dcreate(tracer_fid, var_it->first.c_str(), H5T_IEEE_F32LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t filespace_id = H5Dget_space(var_id);
        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL);
        ret = H5Dwrite(var_id, H5T_IEEE_F32LE, memspace_id, filespace_id, H5P_DEFAULT, var_it->second.data());
        H5Sclose(filespace_id);
        H5Dclose(var_id);
      }

      for(auto var_it=traj_double[id].begin(); var_it!=traj_double[id].end(); var_it++) {
        hid_t var_id = H5Dcreate(tracer_fid, var_it->first.c_str(), H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t filespace_id = H5Dget_space(var_id);
        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, &offset, NULL, &count, NULL);
        ret = H5Dwrite(var_id, H5T_IEEE_F64LE, memspace_id, filespace_id, H5P_DEFAULT, var_it->second.data());
        H5Sclose(filespace_id);
        H5Dclose(var_id);
      }

      // Close HDF5 objects
      H5Sclose(dataspace_id);
      H5Pclose(plist_id2);
      H5Sclose(memspace_id);
      H5Fclose(tracer_fid);
      if(comm_rank == 0)
        printf("Wrote trajectory file %s\n", particle_fname.c_str());
    }
    MPI_Comm_free(&stepcomm);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
};
