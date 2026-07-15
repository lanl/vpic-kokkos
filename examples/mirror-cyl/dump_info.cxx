// ============================================================================
// Code to dump info.hdf5 containing important simulation metadata,
// --Aaron Tran (atran@physics.wisc.edu), started 2023 Dec 14
// ============================================================================

#include "hdf5.h"
#include <vector>
using std::vector;

void dump_info(const char* fname,
               vector<int>& ivalues, vector<const char*>& inames,
               vector<double>& dvalues, vector<const char*>& dnames) {

  hid_t file_id, dataset_id, dataspace_id;
  herr_t status;

  const int datarank = 1;
  const hsize_t dims[1] = { 1 };

  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write integer parameters (key,value) to file
  for (size_t ii{0}; ii < ivalues.size(); ++ii) {

    const char* key = inames[ii];
    int value[1] = { ivalues[ii] };

    dataspace_id = H5Screate_simple(datarank, dims, NULL);
    dataset_id = H5Dcreate2(file_id, key, H5T_NATIVE_INT, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, value);
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
  }

  // Write double/float parameters (key,value) to file
  for (size_t ii{0}; ii < dvalues.size(); ++ii) {

    const char* key = dnames[ii];
    double value[1] = { dvalues[ii] };

    dataspace_id = H5Screate_simple(datarank, dims, NULL);
    dataset_id = H5Dcreate2(file_id, key, H5T_NATIVE_DOUBLE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, value);
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
  }

  status = H5Fclose(file_id);
  return;
}
