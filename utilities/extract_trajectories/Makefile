
H5Root=${HDF5_DIR}

extract_trajectories:
	mpicxx -O3 -Wall -o extract_trajectories -L ${H5Root}/lib -lhdf5 -I ${H5Root}/include extract_trajectories.cpp

clean:
	rm extract_trajectories 
