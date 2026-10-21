#!/usr/bin/env python3
"""
Shared Utilities for VPIC I/O Test Verification
Provides common readers, comparison functions, and logging
"""

import numpy as np
import struct
import os
import sys

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_pass(msg):
    """Print green PASS message"""
    print(f"[\033[92mPASS\033[0m] {msg}")

def log_fail(msg):
    """Print red FAIL message and exit"""
    print(f"[\033[91mFAIL\033[0m] {msg}")
    sys.exit(1)

def log_warn(msg):
    """Print yellow WARN message"""
    print(f"[\033[93mWARN\033[0m] {msg}")

def log_info(msg):
    """Print blue INFO message"""
    print(f"[\033[94mINFO\033[0m] {msg}")

# =============================================================================
# BINARY FORMAT READERS
# =============================================================================

def read_binary_header(fname):
    """
    Read VPIC binary header (64 bytes)
    
    Returns dict with keys:
        magic, version, step, nx, ny, nz, dt, dx, dy, dz,
        x0, y0, z0, q_m, num_vars, var_mask
    """
    if not os.path.exists(fname):
        return None
        
    with open(fname, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        step = struct.unpack('I', f.read(4))[0]
        nx, ny, nz = struct.unpack('III', f.read(12))
        dt = struct.unpack('f', f.read(4))[0]
        dx, dy, dz = struct.unpack('fff', f.read(12))
        x0, y0, z0 = struct.unpack('fff', f.read(12))
        q_m = struct.unpack('f', f.read(4))[0]
        num_vars = struct.unpack('I', f.read(4))[0]
        var_mask = struct.unpack('I', f.read(4))[0]
    
    return {
        'magic': magic,
        'version': version,
        'step': step,
        'nx': nx, 'ny': ny, 'nz': nz,
        'dt': dt,
        'dx': dx, 'dy': dy, 'dz': dz,
        'x0': x0, 'y0': y0, 'z0': z0,
        'q_m': q_m,
        'num_vars': num_vars,
        'var_mask': var_mask
    }

def read_binary_fields(fname, expected_vars=6):
    """
    Read field data from VPIC binary file
    
    Args:
        fname: Binary file path
        expected_vars: Number of variables to read
        
    Returns:
        numpy array of shape (num_vars, nz, ny, nx)
    """
    hdr = read_binary_header(fname)
    if hdr is None:
        return None
        
    vol = hdr['nx'] * hdr['ny'] * hdr['nz']
    
    with open(fname, 'rb') as f:
        f.seek(64)  # Skip header
        data = np.fromfile(f, dtype=np.float32)
    
    # Handle variable count mismatch gracefully
    num_vars = min(hdr['num_vars'], expected_vars)
    if data.size < num_vars * vol:
        log_warn(f"File {fname} truncated: expected {num_vars*vol}, got {data.size}")
        return None
        
    return data[:num_vars * vol].reshape((num_vars, hdr['nz'], hdr['ny'], hdr['nx']))

def read_binary_hydro(fname, expected_vars=4):
    """
    Read hydro data from VPIC binary file
    
    Args:
        fname: Binary file path
        expected_vars: Number of hydro variables (default 4: jx,jy,jz,rho)
        
    Returns:
        numpy array of shape (num_vars, nz, ny, nx)
    """
    return read_binary_fields(fname, expected_vars)

def read_binary_particles(fname):
    """Mode-aware binary particle reader."""
    data = {"ux": [], "uy": [], "uz": []}
    
    if not os.path.exists(fname):
        return data
    
    file_size = os.path.getsize(fname)
    header_size = 64
    payload_size = file_size - header_size
    
    # Read header to get num_vars (offset 48, int32)
    with open(fname, 'rb') as f:
        f.seek(56)
        num_vars = struct.unpack("I", f.read(4))[0]
        
        # Determine particle size from num_vars
        # Physical mode: num_vars=7 (x,y,z,ux,uy,uz,w)
        # Logical mode: num_vars=8 (dx,dy,dz,i,ux,uy,uz,w)
        if num_vars == 7:
            p_size = 28  # Physical: 7 floats
            format_str = "fffffff"  # 7 floats
        else:
            p_size = 32  # Logical: 3f + 1i + 4f
            format_str = "fffIffff"
            
        count = payload_size // p_size
        
        f.seek(header_size)
        for k in range(count):
            chunk = f.read(p_size)
            vals = struct.unpack(format_str, chunk)
            
            # Velocity indices differ by mode
            if num_vars == 7:
                data["ux"].append(vals[3])  # Physical: x,y,z,[ux],uy,uz,w
                data["uy"].append(vals[4])
                data["uz"].append(vals[5])
            else:
                data["ux"].append(vals[4])  # Logical: dx,dy,dz,i,[ux],uy,uz,w
                data["uy"].append(vals[5])
                data["uz"].append(vals[6])
    
    for k in data:
        data[k] = np.array(data[k])
    
    return data

# =============================================================================
# HDF5 HELPERS
# =============================================================================

def verify_hdf5_structure(fname, expected_datasets, dataset_type='field'):
    """
    Verify HDF5 file structure
    
    Args:
        fname: HDF5 file path
        expected_datasets: List of dataset names
        dataset_type: 'field', 'hydro', or 'particle'
        
    Returns:
        True if structure valid, False otherwise
    """
    try:
        import h5py
    except ImportError:
        log_warn("h5py not available, skipping HDF5 structure check")
        return False
        
    if not os.path.exists(fname):
        log_fail(f"HDF5 file not found: {fname}")
        return False
        
    with h5py.File(fname, 'r') as f:
        for ds in expected_datasets:
            if ds not in f:
                log_fail(f"Missing dataset: {ds}")
                return False
        if dataset_type == 'particle':
            if f.attrs.get('schema_version') != 2 or 'momentum' not in f or 'weight' not in f:
                log_fail("Missing versioned numeric particle schema")
                return False
                    
    return True

# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def arrays_close(arr1, arr2, rtol=1e-5, atol=1e-5, name="arrays"):
    """
    Compare two arrays with detailed error reporting
    
    Returns:
        True if arrays match within tolerance
    """
    if arr1.shape != arr2.shape:
        log_fail(f"{name}: Shape mismatch {arr1.shape} vs {arr2.shape}")
        return False
        
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    
    if max_diff > atol:
        rel_diff = diff / (np.abs(arr1) + 1e-15)
        max_rel = np.max(rel_diff)
        log_warn(f"{name}: Max abs diff {max_diff:.2e}, max rel diff {max_rel:.2e}")
        return False
        
    return True

def compare_binary_hdf5_fields(bin_fname, h5_fname, field_names, tol=1e-5):
    """
    Compare binary and HDF5 field outputs
    
    Returns:
        True if all fields match
    """
    try:
        import h5py
    except ImportError:
        log_warn("h5py not available, skipping HDF5 comparison")
        return False
        
    bin_data = read_binary_fields(bin_fname, len(field_names))
    if bin_data is None:
        return False
        
    with h5py.File(h5_fname, 'r') as f:
        for i, name in enumerate(field_names):
            if name not in f:
                log_fail(f"HDF5 missing field: {name}")
                return False
                
            h5_data = f[name][:]
            if not arrays_close(bin_data[i], h5_data, atol=tol, name=name):
                return False
                
    return True

# =============================================================================
# GRID RECONSTRUCTION
# =============================================================================

def stitch_m2m_to_global(file_pattern, num_ranks, topology, local_shape,
                         var_index=0, file_type='binary'):
    """
    Stitch M2M output files into global array
    
    Args:
        file_pattern: e.g., "field/T.0/fields.0.{rank}"
        num_ranks: Total number of ranks
        topology: (px, py, pz) decomposition
        local_shape: (nz, ny, nx) per rank
        var_index: Which variable to extract (for binary)
        file_type: 'binary' or 'hdf5'
        
    Returns:
        Global array
    """
    px, py, pz = topology
    nz_l, ny_l, nx_l = local_shape
    nz_g = nz_l * pz
    ny_g = ny_l * py
    nx_g = nx_l * px
    
    global_arr = np.zeros((nz_g, ny_g, nx_g), dtype=np.float32)
    
    for rank in range(num_ranks):
        # Compute rank offset
        rx = rank % px
        ry = (rank // px) % py
        rz = rank // (px * py)
        
        oz = rz * nz_l
        oy = ry * ny_l
        ox = rx * nx_l
        
        fname = file_pattern.format(rank=rank)
        
        if file_type == 'binary':
            data = read_binary_fields(fname)
            if data is None:
                log_warn(f"Failed to read {fname}")
                continue
            local_block = data[var_index]
        else:  # hdf5
            try:
                import h5py
                with h5py.File(fname, 'r') as f:
                    # Assume single dataset or user specifies
                    keys = list(f.keys())
                    local_block = f[keys[0]][:]
            except:
                log_warn(f"Failed to read HDF5 {fname}")
                continue
                
        global_arr[oz:oz+nz_l, oy:oy+ny_l, ox:ox+nx_l] = local_block
        
    return global_arr

# =============================================================================
# TEST FRAMEWORK HELPERS
# =============================================================================

class TestSuite:
    """Simple test suite manager"""
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        
    def run_test(self, test_name, test_func):
        """Run a test function and track result"""
        try:
            print(f"\n  [TEST] {test_name}...")
            test_func()
            log_pass(test_name)
            self.passed += 1
            return True
        except Exception as e:
            log_fail(f"{test_name}: {str(e)}")
            self.failed += 1
            return False
            
    def summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Test Suite: {self.name}")
        print(f"  Passed: {self.passed}/{total}")
        print(f"  Failed: {self.failed}/{total}")
        
        if self.failed == 0:
            print(f"\n\033[92m✓ ALL {self.passed} TESTS PASSED\033[0m")
            return 0
        else:
            print(f"\n\033[91m✗ {self.failed} TEST(S) FAILED\033[0m")
            return 1