import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import os

class VPICFieldReader:
    """Read VPIC field dumps"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.info = None
        self.timesteps = None
        self.files = None
        
    def read_info(self, info_path='../../build/info.bin'):
        """Read info.bin for grid parameters"""
        with open(info_path, 'rb') as f:
            topology = struct.unpack('3d', f.read(24))
            Lx, Ly, Lz = struct.unpack('3d', f.read(24))
            nx, ny, nz = struct.unpack('3d', f.read(24))
            dt = struct.unpack('d', f.read(8))[0]
            
        self.info = {
            'topology': topology,
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'nx': int(nx), 'ny': int(ny), 'nz': int(nz),
            'dt': dt
        }
        return self.info
    
    def find_dumps(self):
        """Find all field dump files"""
        pattern = os.path.join(self.base_dir, 'T.*/fields.*.0')
        files = sorted(glob.glob(pattern))
        
        # Extract timesteps
        timesteps = []
        for f in files:
            parts = os.path.basename(f).split('.')
            timesteps.append(int(parts[1]))
        
        self.timesteps = timesteps
        self.files = files
        return timesteps, files
    
    def print_available_times(self):
        """Print information about available dumps"""
        if self.info is None:
            self.read_info()
        if self.timesteps is None:
            self.find_dumps()
            
        times = np.array(self.timesteps) * self.info['dt']
        
        print("\n" + "="*60)
        print("AVAILABLE FIELD DUMPS")
        print("="*60)
        print(f"Total dumps found: {len(self.timesteps)}")
        print(f"Timestep range:    {self.timesteps[0]} to {self.timesteps[-1]}")
        print(f"Time range:        {times[0]:.3f} to {times[-1]:.3f}")
        print(f"dt:                {self.info['dt']:.6f}")
        print(f"Grid size:         {self.info['nx']} x {self.info['ny']} x {self.info['nz']}")
        print("="*60 + "\n")
        
        # Print first 10 and last 10 available times
        print("First 10 dumps:")
        for i in range(min(10, len(self.timesteps))):
            print(f"  Index {i:3d}: step={self.timesteps[i]:6d}, time={times[i]:8.3f}")
        
        if len(self.timesteps) > 20:
            print("  ...")
            print("Last 10 dumps:")
            for i in range(max(10, len(self.timesteps)-10), len(self.timesteps)):
                print(f"  Index {i:3d}: step={self.timesteps[i]:6d}, time={times[i]:8.3f}")
        print()
    
    def read_field_dump(self, filename):
        """Read a single field dump file"""
        if self.info is None:
            self.read_info()
            
        nx, ny, nz = self.info['nx'], self.info['ny'], self.info['nz']
        
        # VPIC dumps have ghost cells (1 on each side)
        nx_ghost = nx + 2
        ny_ghost = ny + 2
        nz_ghost = nz + 2
        
        data = {}
        
        with open(filename, 'rb') as f:
            header_size = 0
            f.seek(header_size)
            
            try:
                raw = np.fromfile(f, dtype=np.float32)
                n_per_var = nx_ghost * ny_ghost * nz_ghost
                
                if len(raw) >= 7 * n_per_var:
                    # Extract fields (skip ghost cells)
                    data['ex'] = raw[0*n_per_var:1*n_per_var].reshape(nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    data['ey'] = raw[1*n_per_var:2*n_per_var].reshape(nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    data['ez'] = raw[2*n_per_var:3*n_per_var].reshape(nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    data['cbx'] = raw[4*n_per_var:5*n_per_var].reshape(nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    data['cby'] = raw[5*n_per_var:6*n_per_var].reshape(nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    data['cbz'] = raw[6*n_per_var:7*n_per_var].reshape(nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                return None
        
        return data

def plot_efield_at_time(time_index=None, time_value=None, 
                        slice_axis='z', slice_index=0,
                        components=['ex', 'ey', 'ez'],
                        save_prefix='efield'):
    """
    Plot E-field components at a specific time
    
    Parameters:
    -----------
    time_index : int, optional
        Index into the list of available dumps (0 to N-1)
    time_value : float, optional
        Actual simulation time (will find closest dump)
    slice_axis : str
        Which axis to slice through ('x', 'y', or 'z')
    slice_index : int
        Index along slice_axis to plot (default: middle)
    components : list
        Which field components to plot (e.g., ['ex', 'ey', 'ez'])
    save_prefix : str
        Prefix for saved figure filename
    """
    
    reader = VPICFieldReader('../../build/fields')
    reader.read_info()
    reader.find_dumps()
    reader.print_available_times()
    
    # Determine which dump to load
    if time_index is not None:
        idx = time_index
    elif time_value is not None:
        times = np.array(reader.timesteps) * reader.info['dt']
        idx = np.argmin(np.abs(times - time_value))
        print(f"Requested time {time_value:.3f}, closest dump at time {times[idx]:.3f}")
    else:
        # Default to middle time
        idx = len(reader.timesteps) // 2
        print(f"No time specified, using middle dump (index {idx})")
    
    # Load the data
    print(f"\nLoading dump index {idx}...")
    data = reader.read_field_dump(reader.files[idx])
    
    if data is None:
        print("Failed to load data!")
        return
    
    time = reader.timesteps[idx] * reader.info['dt']
    step = reader.timesteps[idx]
    
    # Get grid coordinates
    info = reader.info
    x = np.linspace(-info['Lx']/2, info['Lx']/2, info['nx'])
    y = np.linspace(-info['Ly']/2, info['Ly']/2, info['ny'])
    z = np.linspace(-info['Lz']/2, info['Lz']/2, info['nz'])
    
    # Determine if this is effectively 1D, 2D, or 3D
    is_1d = (info['ny'] == 1 and info['nz'] == 1)
    is_2d = (info['ny'] == 1 or info['nz'] == 1) and not is_1d
    
    print(f"Grid dimensions: {info['nx']} x {info['ny']} x {info['nz']}")
    print(f"Detected: {'1D' if is_1d else '2D' if is_2d else '3D'} simulation")
    
    # Create figure
    n_components = len(components)
    
    if is_1d:
        # 1D plot
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
        if n_components == 1:
            axes = [axes]
        
        for i, comp in enumerate(components):
            field_1d = data[comp].flatten()
            axes[i].plot(x, field_1d, 'b-', linewidth=2)
            axes[i].set_xlabel('x', fontsize=12)
            axes[i].set_ylabel(comp.upper(), fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(0, color='k', linestyle='--', alpha=0.3)
            
            # Show wavelength
            wavelength = info['Lx']
            axes[i].axvline(-wavelength/2, color='r', linestyle=':', alpha=0.5, label='Domain edge')
            axes[i].axvline(wavelength/2, color='r', linestyle=':', alpha=0.5)
            axes[i].legend()
        
        axes[0].set_title(f'Electric Field at t={time:.3f} (step {step})', 
                         fontsize=14, fontweight='bold', pad=15)
        
    else:
        # 2D/3D: show heatmap slices
        fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 5))
        if n_components == 1:
            axes = [axes]
        
        # Extract 2D slice
        if slice_axis == 'z' or info['nz'] == 1:
            slice_idx = slice_index if slice_index >= 0 else info['nz']//2
            X, Y = np.meshgrid(x, y)
            for i, comp in enumerate(components):
                field_2d = data[comp][slice_idx, :, :].T
                
                im = axes[i].pcolormesh(X, Y, field_2d, shading='auto', cmap='RdBu_r')
                axes[i].set_xlabel('x', fontsize=12)
                axes[i].set_ylabel('y', fontsize=12)
                axes[i].set_title(f'{comp.upper()}', fontsize=12)
                axes[i].set_aspect('equal')
                plt.colorbar(im, ax=axes[i])
        
        elif slice_axis == 'y' or info['ny'] == 1:
            slice_idx = slice_index if slice_index >= 0 else info['ny']//2
            X, Z = np.meshgrid(x, z)
            for i, comp in enumerate(components):
                field_2d = data[comp][:, slice_idx, :].T
                
                im = axes[i].pcolormesh(X, Z, field_2d, shading='auto', cmap='RdBu_r')
                axes[i].set_xlabel('x', fontsize=12)
                axes[i].set_ylabel('z', fontsize=12)
                axes[i].set_title(f'{comp.upper()}', fontsize=12)
                axes[i].set_aspect('equal')
                plt.colorbar(im, ax=axes[i])
        
        fig.suptitle(f'Electric Field at t={time:.3f} (step {step})', 
                    fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    filename = f'{save_prefix}_t{time:.3f}_step{step}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")
    plt.show()


def plot_efield_evolution_comparison(time_indices=None, n_times=6):
    """
    Plot E-field evolution showing multiple time snapshots side-by-side
    
    Parameters:
    -----------
    time_indices : list of int, optional
        Specific indices to plot. If None, evenly spaced times will be used
    n_times : int
        Number of time snapshots to show (if time_indices not specified)
    """
    
    reader = VPICFieldReader('../../build/fields')
    reader.read_info()
    reader.find_dumps()
    
    if time_indices is None:
        time_indices = np.linspace(0, len(reader.timesteps)-1, n_times, dtype=int)
    
    n_plots = len(time_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    x = np.linspace(-reader.info['Lx']/2, reader.info['Lx']/2, reader.info['nx'])
    
    for i, idx in enumerate(time_indices):
        data = reader.read_field_dump(reader.files[idx])
        
        if data is not None:
            ex_1d = data['ex'].flatten()
            time = reader.timesteps[idx] * reader.info['dt']
            
            axes[i].plot(x, ex_1d, 'b-', linewidth=2)
            axes[i].set_ylabel('Ex', fontsize=11)
            axes[i].set_title(f't = {time:.2f}', fontsize=11, loc='right')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(0, color='k', linestyle='--', alpha=0.3)
            
            # Mark domain edges
            axes[i].axvline(-reader.info['Lx']/2, color='r', linestyle=':', alpha=0.3)
            axes[i].axvline(reader.info['Lx']/2, color='r', linestyle=':', alpha=0.3)
    
    axes[-1].set_xlabel('x', fontsize=12)
    axes[0].set_title('Ex Field Evolution', fontsize=14, fontweight='bold', loc='left', pad=15)
    
    plt.tight_layout()
    plt.savefig('ex_evolution_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: ex_evolution_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("VPIC FIELD VISUALIZATION")
    print("="*60)
    
    # Example 1: Plot E-field at a specific time index
    print("\n>>> Plotting E-field at time index 50...")
    plot_efield_at_time(time_index=50, components=['ex'])
    
    # Example 2: Plot E-field at a specific simulation time
    print("\n>>> Plotting E-field at time t=25...")
    plot_efield_at_time(time_value=25.0, components=['ex'])
    
    # Example 3: Plot evolution comparison
    print("\n>>> Plotting evolution comparison...")
    plot_efield_evolution_comparison(n_times=6)
    
    # Example 4: Plot all three components at a specific time
    print("\n>>> Plotting all E-field components...")
    plot_efield_at_time(time_index=-1, components=['ex', 'ey', 'ez'])