import numpy as np
import struct
import glob
import matplotlib.pyplot as plt

def read_vpic_hydro(filepath):
    """Read VPIC hydro file - stream format"""
    
    print(f"\nReading: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read boilerplate (23 bytes total)
        sizearr = struct.unpack('5b', f.read(5))     # 5 bytes
        cafe = struct.unpack('H', f.read(2))[0]      # 2 bytes
        deadbeef = struct.unpack('I', f.read(4))[0]  # 4 bytes
        realone = struct.unpack('f', f.read(4))[0]   # 4 bytes
        doubleone = struct.unpack('d', f.read(8))[0] # 8 bytes
        
        # Read v0 header (80 bytes)
        v0_bytes = f.read(80)
        v0_data = struct.unpack('10i10f', v0_bytes)
        
        # Read itype, ndim, nc
        itype = struct.unpack('i', f.read(4))[0]
        ndim = struct.unpack('i', f.read(4))[0]
        nc = struct.unpack('3i', f.read(12))
        
        print(f"  Grid with ghost: {nc[0]}×{nc[1]}×{nc[2]}")
        
        # Read 10 variable arrays
        n_cells = nc[0] * nc[1] * nc[2]
        n_bytes = n_cells * 4
        
        variables = []
        var_names = ['jx', 'jy', 'jz', 'rho', 'px', 'py', 'pz', 'ke', 'txx', 'tyy']
        
        for var_name in var_names:
            data = f.read(n_bytes)
            arr = np.array(struct.unpack(f'{n_cells}f', data))
            arr_3d = arr.reshape(nc, order='F')
            interior = arr_3d[1:-1, 1:-1, 1:-1]
            variables.append(interior)
            
            if np.abs(interior).max() > 1e-10:
                print(f"  {var_name:3s}: [{interior.min():+.6e}, {interior.max():+.6e}]")
        
        return variables, nc

def analyze_all_timesteps():
    """Analyze all available timesteps"""
    
    files = sorted(glob.glob('../../build/hydro/T.*/Hhydro.*.0'))
    print(f"\nFound {len(files)} hydro files")
    
    if len(files) == 0:
        print("No files found!")
        return
    
    # Try several timesteps
    test_indices = [0, 1, 10, 50] if len(files) > 50 else range(min(10, len(files)))
    
    for idx in test_indices:
        if idx >= len(files):
            break
        result = read_vpic_hydro(files[idx])
        if result[0] is not None and np.abs(result[0][3]).max() > 1e-10:  # Check rho
            print(f"\n✅ Found non-zero data at index {idx}")
            return files, idx
    
    print("\n❌ All tested files have zero data")
    return files, 0

def plot_timestep(files, idx):
    """Plot specific timestep"""
    
    variables, nc = read_vpic_hydro(files[idx])
    if variables is None:
        return
    
    jx, jy, jz, rho, px, py, pz, ke, txx, tyy = variables
    
    # Extract 1D profile (it's a 1D simulation: 48×1×1)
    nx = nc[0] - 2  # Interior cells
    x = np.linspace(-8, 8, nx)  # Lx = 16 from input deck
    
    # Extract data (middle of y,z dimensions which are size 1)
    rho_1d = rho[:, 0, 0]
    jx_1d = jx[:, 0, 0]
    px_1d = px[:, 0, 0]
    ke_1d = ke[:, 0, 0]
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    timestep = int(files[idx].split('.')[-2].split('/')[-1])
    dt = 0.02  # From input deck
    time = timestep * dt
    
    # Charge density (should show sine wave perturbation)
    axes[0].plot(x, rho_1d, 'b.-', linewidth=2, markersize=4)
    axes[0].set_ylabel('Charge Density (ρ)', fontsize=12)
    axes[0].set_title(f'Ion Acoustic Wave - t={time:.2f}', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Current density  
    axes[1].plot(x, jx_1d, 'r.-', linewidth=2, markersize=4)
    axes[1].set_ylabel('Current Density (jx)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Momentum
    axes[2].plot(x, px_1d, 'g.-', linewidth=2, markersize=4)
    axes[2].set_ylabel('Momentum Density (px)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # Kinetic energy
    axes[3].plot(x, ke_1d, 'm.-', linewidth=2, markersize=4)
    axes[3].set_xlabel('x', fontsize=12)
    axes[3].set_ylabel('Kinetic Energy', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ion_acoustic_wave.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: ion_acoustic_wave.png")
    plt.show()

def main():
    files, idx = analyze_all_timesteps()
    if files:
        plot_timestep(files, max(1, idx))  # Use at least timestep 1

if __name__ == "__main__":
    main()