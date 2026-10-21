#!/usr/bin/env python3
"""
Binary I/O Verification with Automatic Cleanup
"""

import sys
import os
import argparse
import shutil

# Add common utilities to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
from verify_utils import *

# =============================================================================
# CONFIGURATION
# =============================================================================
NX, NY, NZ = 4, 4, 4

# =============================================================================
# CLEANUP
# =============================================================================
def cleanup_test_output():
    """Remove test output directories"""
    dirs_to_remove = ["field", "ehydro", "particle", "rundata"]
    for dirname in dirs_to_remove:
        if os.path.exists(dirname):
            try:
                shutil.rmtree(dirname)
                print(f"  [CLEANUP] Removed {dirname}/")
            except Exception as e:
                print(f"  [WARN] Could not remove {dirname}: {e}")

# =============================================================================
# SINGLE-RANK TESTS
# =============================================================================
def verify_single():
    """Verify single-rank binary output"""
    suite = TestSuite("Binary Single-Rank")
    
    # Test 1: Field Headers
    def test_field_headers():
        hdr = read_binary_header("field/T.0/fields_bin.0.0")
        if hdr is None:
            raise Exception("Field file not found")
        
        assert hdr['magic'] == 0xBEEF0002, "Invalid magic number"
        assert hdr['nx'] == NX and hdr['ny'] == NY and hdr['nz'] == NZ, "Wrong grid size"
        assert hdr['num_vars'] == 6, "Expected 6 field variables"
    
    suite.run_test("Field Headers", test_field_headers)
    
    # Test 2: Field Data Pattern
    def test_field_data():
        fields = read_binary_fields("field/T.0/fields_bin.0.0", 6)
        if fields is None:
            raise Exception("Failed to read field data")
        
        # Verify first interior cell (1,1,1) has index 43
        base = 1000.0 + (43 * 1.0e-8)
        expected_offsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        for i, offset in enumerate(expected_offsets):
            expected = base + offset
            actual = fields[i, 0, 0, 0]
            diff = abs(actual - expected)
            if diff > 1e-4:
                raise Exception(f"Field component {i} mismatch: expected {expected}, got {actual}")
    
    suite.run_test("Field Data Pattern", test_field_data)
    
    # Test 3: Particle Data (Logical Mode)
    def test_particle_data_logical():
        particles = read_binary_particles("particle/particles_logical.electron.0.0")
        if particles is None:
            raise Exception("Failed to read logical particle data")
        
        num_particles = len(particles['ux'])
        expected_count = NX * NY * NZ
        
        if num_particles != expected_count:
            raise Exception(f"Particle count mismatch: expected {expected_count}, got {num_particles}")
        
        # Verify velocity pattern
        ux_sorted = np.sort(particles['ux'])
        expected = []
        for iz in range(1, NZ+1):
            for iy in range(1, NY+1):
                for ix in range(1, NX+1):
                    gid = iz * NY * NX + iy * NX + ix
                    expected.append(gid * 1.0e-9)
        expected = np.sort(np.array(expected))
        
        diff = np.max(np.abs(ux_sorted - expected))
        if diff > 1e-4:
            raise Exception(f"Logical particle velocity pattern mismatch: max diff {diff}")

    suite.run_test("Particle Data (Logical Mode)", test_particle_data_logical)

    # Test 4: Particle Data (Physical Mode)
    def test_particle_data_physical():
        particles = read_binary_particles("particle/particles_physical.electron.0.0")
        if particles is None:
            raise Exception("Failed to read physical particle data")
        
        num_particles = len(particles['ux'])
        expected_count = NX * NY * NZ
        
        if num_particles != expected_count:
            raise Exception(f"Particle count mismatch: expected {expected_count}, got {num_particles}")
        
        # Physical mode: velocities should still match pattern (coordinates are different)
        ux_sorted = np.sort(particles['ux'])
        expected = []
        for iz in range(1, NZ+1):
            for iy in range(1, NY+1):
                for ix in range(1, NX+1):
                    gid = iz * NY * NX + iy * NX + ix
                    expected.append(gid * 1.0e-9)
        expected = np.sort(np.array(expected))
        
        diff = np.max(np.abs(ux_sorted - expected))
        if diff > 1e-4:
            raise Exception(f"Physical particle velocity pattern mismatch: max diff {diff}")
        
        # BONUS: Verify physical coordinates are in simulation domain
        if 'x' in particles and 'y' in particles and 'z' in particles:
            x_min, x_max = np.min(particles['x']), np.max(particles['x'])
            y_min, y_max = np.min(particles['y']), np.max(particles['y'])
            z_min, z_max = np.min(particles['z']), np.max(particles['z'])
            
            # Should be within [0, L] where L=1.0 for your deck
            if not (0.0 <= x_min and x_max <= 1.0):
                raise Exception(f"Physical X coords out of bounds: [{x_min}, {x_max}]")
            if not (0.0 <= y_min and y_max <= 1.0):
                raise Exception(f"Physical Y coords out of bounds: [{y_min}, {y_max}]")
            if not (0.0 <= z_min and z_max <= 1.0):
                raise Exception(f"Physical Z coords out of bounds: [{z_min}, {z_max}]")

    suite.run_test("Particle Data (Physical Mode)", test_particle_data_physical)
    
    # Test 4: Hydro Headers
    def test_hydro_headers():
        hdr = read_binary_header("ehydro/T.0/hydro_bin.0.0")
        if hdr is None:
            raise Exception("Hydro file not found")
        # DIAGNOSTIC: Print actual vs expected
        print(f"  [DEBUG] Hydro num_vars: {hdr['num_vars']} (expected 4)")
        print(f"  [DEBUG] Hydro var_mask: {hdr['var_mask']}")
        print(f"  [DEBUG] Magic: {hex(hdr['magic'])} (expected 0xBEEF0002)")
        assert hdr['magic'] == 0xBEEF0002, "Invalid magic number"
        assert hdr['num_vars'] == 4, "Expected 4 hydro variables"
    
    suite.run_test("Hydro Headers", test_hydro_headers)
    
    # Test 5: Hydro Physics Recovery
    def test_hydro_physics():
        hydro = read_binary_hydro("ehydro/T.0/hydro_bin.0.0", 4)
        if hydro is None:
            raise Exception("Failed to read hydro data")
        
        jx = hydro[0]
        rho = hydro[3]
        
        valid = np.abs(rho) > 1e-15
        if not np.any(valid):
            raise Exception("No valid charge density found")
        
        ux = np.zeros_like(jx)
        ux[valid] = jx[valid] / rho[valid]
        
        # Verify first interior cell (gid=43)
        expected_ux = 43 * 1.0e-9
        actual_ux = ux.flatten()[0]
        
        if abs(actual_ux - expected_ux) > 1e-6:
            raise Exception(f"Hydro physics mismatch: expected {expected_ux}, got {actual_ux}")
    
    suite.run_test("Hydro Physics Recovery", test_hydro_physics)
    
    return suite.summary()

# =============================================================================
# 4-RANK TESTS
# =============================================================================
def verify_4rank():
    """Verify 4-rank binary output"""
    suite = TestSuite("Binary 4-Rank")
    
    NX_L, NY_L, NZ_L = 4, 4, 4
    NX_G, NY_G, NZ_G = 8, 8, 4
    NUM_RANKS = 4
    
    # Test 1: All Files Exist
    def test_files_exist():
        for rank in range(NUM_RANKS):
            files = [
                f"field/T.0/fields_bin.0.{rank}",
                f"ehydro/T.0/hydro_bin.0.{rank}",
                f"particle/particles_logical.electron.0.{rank}",
                f"particle/particles_physical.electron.0.{rank}"
            ]
            for fname in files:
                if not os.path.exists(fname):
                    raise Exception(f"Missing file: {fname}")
    
    suite.run_test("All Rank Files Present", test_files_exist)
    
    # Test 2: Global Field Stitching
    def test_field_stitching():
        stitched = stitch_m2m_to_global(
            "field/T.0/fields_bin.0.{rank}",
            NUM_RANKS,
            (2, 2, 1),
            (NZ_L, NY_L, NX_L),
            var_index=0,
            file_type='binary'
        )
        
        if stitched is None:
            raise Exception("Failed to stitch field data")
        
        # Verify known global cell (4,4,2) -> gid=164
        expected = 164.0 + 0.1
        actual = stitched[1, 3, 3]
        
        if abs(actual - expected) > 1e-5:
            raise Exception(f"Field stitching failed: expected {expected}, got {actual}")
    
    suite.run_test("Global Field Stitching", test_field_stitching)
    
    # Test 3: Particle Distribution (Test BOTH modes)
    def test_particle_distribution():
        # Test Logical Mode
        total_logical = 0
        all_ux_logical = []
        
        for rank in range(NUM_RANKS):
            fname = f"particle/particles_logical.electron.0.{rank}"
            particles = read_binary_particles(fname)
            if particles:
                total_logical += len(particles['ux'])
                all_ux_logical.extend(particles['ux'])
        
        # Test Physical Mode
        total_physical = 0
        all_ux_physical = []
        
        for rank in range(NUM_RANKS):
            fname = f"particle/particles_physical.electron.0.{rank}"
            particles = read_binary_particles(fname)
            if particles:
                total_physical += len(particles['ux'])
                all_ux_physical.extend(particles['ux'])
        
        # Both modes should have same particle count
        expected_total = NX_G * NY_G * NZ_G
        if total_logical != expected_total:
            raise Exception(f"Logical count mismatch: expected {expected_total}, got {total_logical}")
        if total_physical != expected_total:
            raise Exception(f"Physical count mismatch: expected {expected_total}, got {total_physical}")
        
        # Verify velocity pattern (should be identical in both modes)
        all_ux_logical_sorted = np.sort(np.array(all_ux_logical))
        all_ux_physical_sorted = np.sort(np.array(all_ux_physical))
        
        expected = []
        for iz in range(1, NZ_G+1):
            for iy in range(1, NY_G+1):
                for ix in range(1, NX_G+1):
                    gid = iz * NY_G * NX_G + iy * NX_G + ix
                    expected.append(gid * 1.0e-9)
        expected = np.sort(np.array(expected))
        
        # Check both modes match expected
        diff_logical = np.max(np.abs(all_ux_logical_sorted - expected))
        diff_physical = np.max(np.abs(all_ux_physical_sorted - expected))
        
        if diff_logical > 1e-5:
            raise Exception(f"Logical mode velocity mismatch: max diff {diff_logical}")
        if diff_physical > 1e-5:
            raise Exception(f"Physical mode velocity mismatch: max diff {diff_physical}")

    suite.run_test("Particle Distribution", test_particle_distribution)
    
    return suite.summary()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify VPIC Binary I/O")
    parser.add_argument("--mode", choices=["single", "4rank"], required=True)
    parser.add_argument("--preserve-output", action="store_true",
                       help="Keep output files after successful verification")
    args = parser.parse_args()
    
    exit_code = 0
    try:
        if args.mode == "single":
            exit_code = verify_single()
        elif args.mode == "4rank":
            exit_code = verify_4rank()
    
    except Exception as e:
        print(f"\n\033[91m✗ VERIFICATION ERROR: {e}\033[0m")
        exit_code = 1
    
    finally:
        if exit_code == 0 and not args.preserve_output:
            print("\nCleaning up test output...")
            cleanup_test_output()
        elif exit_code != 0:
            print("\n[PRESERVED] Output files kept for debugging")
        elif args.preserve_output:
            print("\n[PRESERVED] Output kept (--preserve-output flag)")
    
    sys.exit(exit_code)