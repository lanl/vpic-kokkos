import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import os

class VPICHydroReader:
    """Read VPIC hydro dumps for IAW analysis"""
    
    def __init__(self, base_dir='../../build/hydro'):
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
            'topology': tuple(int(t) for t in topology),
            'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
            'nx': int(nx), 'ny': int(ny), 'nz': int(nz),
            'dt': dt
        }
        return self.info
    
    def find_dumps(self):
        """Find all hydro dump files"""
        pattern = os.path.join(self.base_dir, 'T.*/Hhydro.*.0')
        files = sorted(glob.glob(pattern))
        
        timesteps = []
        for f in files:
            parts = os.path.basename(f).split('.')
            timesteps.append(int(parts[1]))
        
        self.timesteps = timesteps
        self.files = files
        return timesteps, files
    
    def read_hydro_dump(self, filename):
        """Read a single hydro dump file"""
        if self.info is None:
            self.read_info()
            
        nx, ny, nz = self.info['nx'], self.info['ny'], self.info['nz']
        nx_ghost = nx + 2
        ny_ghost = ny + 2
        nz_ghost = nz + 2
        
        data = {}
        
        with open(filename, 'rb') as f:
            try:
                raw = np.fromfile(f, dtype=np.float32)
                n_per_var = nx_ghost * ny_ghost * nz_ghost
                
                # Hydro variables: jx, jy, jz, rho, px, py, pz, ke, txx, tyy, tzz, txy, tyz, tzx
                if len(raw) >= 14 * n_per_var:
                    # Extract density (4th variable, index 3)
                    data['rho'] = raw[3*n_per_var:4*n_per_var].reshape(
                        nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    
                    # Extract current density
                    data['jx'] = raw[0*n_per_var:1*n_per_var].reshape(
                        nz_ghost, ny_ghost, nx_ghost)[1:-1, 1:-1, 1:-1]
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                return None
        
        return data
    
    def print_available_times(self, c_s=1.0, L=16.0):
        """Print available dumps in normalized units"""
        if self.info is None:
            self.read_info()
        if self.timesteps is None:
            self.find_dumps()
            
        times_physical = np.array(self.timesteps) * self.info['dt']
        times_normalized = times_physical * c_s / L
        
        print("\n" + "="*70)
        print("AVAILABLE HYDRO DUMPS - ION ACOUSTIC WAVE")
        print("="*70)
        print(f"Total dumps:       {len(self.timesteps)}")
        print(f"Timestep range:    {self.timesteps[0]} to {self.timesteps[-1]}")
        print(f"Physical time:     {times_physical[0]:.3f} to {times_physical[-1]:.3f}")
        print(f"Normalized time:   {times_normalized[0]:.3f} to {times_normalized[-1]:.3f} (t*C_s/L)")
        print(f"Grid: {self.info['nx']} x {self.info['ny']} x {self.info['nz']}")
        print(f"Domain: L = {self.info['Lx']:.1f}")
        print("="*70 + "\n")


def plot_density_perturbation_vs_time(time_indices=None, n_times=6, 
                                       save_name='iaw_density_evolution.png'):
    """
    Plot |dn| vs x/L at multiple times
    
    Parameters:
    -----------
    time_indices : list of int, optional
        Specific dump indices to plot
    n_times : int
        Number of snapshots to show (if time_indices not specified)
    save_name : str
        Output filename
    """
    
    # Physical parameters from your VPIC deck
    c_s = 1.0
    L = 16.0
    n0 = 1.0  # Background density
    pert = 0.02  # Initial perturbation amplitude
    
    reader = VPICHydroReader()
    reader.read_info()
    reader.find_dumps()
    reader.print_available_times(c_s, L)
    
    if time_indices is None:
        time_indices = np.linspace(0, len(reader.timesteps)-1, n_times, dtype=int)
    
    # Create x-coordinate normalized by L
    x = np.linspace(-reader.info['Lx']/2, reader.info['Lx']/2, reader.info['nx'])
    x_normalized = x / L
    
    # Create figure
    n_plots = len(time_indices)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    print(f"\nPlotting {n_plots} time snapshots...")
    
    for i, idx in enumerate(time_indices):
        print(f"  Loading dump {idx}...")
        data = reader.read_hydro_dump(reader.files[idx])
        
        if data is not None:
            # Get 1D density profile (average over y, z)
            rho_1d = np.mean(data['rho'], axis=(0, 1))
            
            # Compute density perturbation
            dn = rho_1d - n0
            dn_abs = np.abs(dn)
            
            # Normalize time
            time_physical = reader.timesteps[idx] * reader.info['dt']
            time_normalized = time_physical * c_s / L
            
            # Plot
            axes[i].plot(x_normalized, dn_abs, 'b-', linewidth=2, label='|δn|')
            axes[i].plot(x_normalized, dn, 'r--', linewidth=1, alpha=0.6, label='δn')
            
            # Formatting
            axes[i].set_ylabel('|δn|', fontsize=12)
            axes[i].set_title(f't·Cs/L = {time_normalized:.3f}', 
                            fontsize=11, loc='right')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[i].set_xlim(x_normalized[0], x_normalized[-1])
            
            # Mark wavelength
            axes[i].axvline(-0.5, color='gray', linestyle=':', alpha=0.3)
            axes[i].axvline(0.5, color='gray', linestyle=':', alpha=0.3)
            
            if i == 0:
                axes[i].legend(loc='upper right', fontsize=10)
    
    axes[-1].set_xlabel('x/L', fontsize=13)
    axes[0].set_title('Ion Acoustic Wave: Density Perturbation', 
                     fontsize=14, fontweight='bold', loc='left', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_name}")
    plt.show()


def plot_density_damping():
    """
    Plot density perturbation amplitude vs time to measure damping rate
    """
    
    # Physical parameters
    c_s = 1.0
    L = 16.0
    n0 = 1.0
    kx = 2.0*np.pi/L
    
    reader = VPICHydroReader()
    reader.read_info()
    reader.find_dumps()
    
    x = np.linspace(-reader.info['Lx']/2, reader.info['Lx']/2, reader.info['nx'])
    
    times_physical = []
    times_normalized = []
    amplitudes = []
    
    print("\nComputing density amplitude for each dump...")
    for i, (step, file) in enumerate(zip(reader.timesteps, reader.files)):
        data = reader.read_hydro_dump(file)
        
        if data is not None:
            # Get 1D density profile
            rho_1d = np.mean(data['rho'], axis=(0, 1))
            dn = rho_1d - n0
            
            # Compute Fourier amplitude at fundamental mode
            cos_sum = np.sum(dn * np.cos(kx * x))
            sin_sum = np.sum(dn * np.sin(kx * x))
            amplitude = np.sqrt(cos_sum**2 + sin_sum**2) / len(x)
            
            time_phys = step * reader.info['dt']
            times_physical.append(time_phys)
            times_normalized.append(time_phys * c_s / L)
            amplitudes.append(amplitude)
        
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(reader.files)}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(times_normalized, amplitudes, 'b.-', markersize=5, 
                linewidth=1.5, label='|δn| amplitude')
    
    # Try exponential fit
    if len(times_normalized) > 20:
        from scipy.optimize import curve_fit
        
        # Fit after initial transient
        idx_fit = np.array(times_normalized) > 0.5
        
        if np.sum(idx_fit) > 10:
            try:
                popt, _ = curve_fit(
                    lambda t, A, gamma: A * np.exp(-gamma * t),
                    np.array(times_normalized)[idx_fit],
                    np.array(amplitudes)[idx_fit],
                    p0=[amplitudes[0], 0.1]
                )
                A0, gamma = popt
                
                t_fit = np.linspace(times_normalized[np.where(idx_fit)[0][0]], 
                                   times_normalized[-1], 100)
                ax.semilogy(t_fit, A0 * np.exp(-gamma * t_fit),
                           'r--', linewidth=2, 
                           label=f'Fit: A·exp(-γt), γL/Cs = {gamma:.4f}')
                
                print(f"\n{'='*60}")
                print(f"ION ACOUSTIC WAVE DAMPING RATE")
                print(f"{'='*60}")
                print(f"Normalized damping rate: γ·L/Cs = {gamma:.6f}")
                print(f"Physical damping rate:   γ = {gamma*c_s/L:.6f}")
                print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"Fit failed: {e}")
    
    ax.set_xlabel('t·Cs/L', fontsize=14)
    ax.set_ylabel('|δn| Amplitude', fontsize=14)
    ax.set_title('Ion Acoustic Wave Damping', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('iaw_damping_rate.png', dpi=150, bbox_inches='tight')
    print("\nSaved: iaw_damping_rate.png")
    plt.show()


def plot_density_heatmap(time_index=None, time_value=None):
    """
    Plot 2D heatmap of density perturbation at a specific time
    
    Parameters:
    -----------
    time_index : int, optional
        Index into list of dumps
    time_value : float, optional
        Normalized time t*Cs/L (will find closest dump)
    """
    
    c_s = 1.0
    L = 16.0
    n0 = 1.0
    
    reader = VPICHydroReader()
    reader.read_info()
    reader.find_dumps()
    reader.print_available_times(c_s, L)
    
    # Determine which dump to load
    if time_index is not None:
        idx = time_index
    elif time_value is not None:
        times_norm = np.array(reader.timesteps) * reader.info['dt'] * c_s / L
        idx = np.argmin(np.abs(times_norm - time_value))
        print(f"Requested t·Cs/L={time_value:.3f}, closest at {times_norm[idx]:.3f}")
    else:
        idx = len(reader.timesteps) // 2
    
    # Load data
    data = reader.read_hydro_dump(reader.files[idx])
    
    if data is None:
        print("Failed to load data!")
        return
    
    # Compute normalized time
    time_phys = reader.timesteps[idx] * reader.info['dt']
    time_norm = time_phys * c_s / L
    
    # Get 1D density
    rho_1d = np.mean(data['rho'], axis=(0, 1))
    dn = rho_1d - n0
    
    # Normalized x
    x = np.linspace(-reader.info['Lx']/2, reader.info['Lx']/2, reader.info['nx'])
    x_norm = x / L
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Density perturbation
    ax1.plot(x_norm, dn, 'b-', linewidth=2)
    ax1.set_ylabel('δn/n₀', fontsize=13)
    ax1.set_title(f'Density Perturbation at t·Cs/L = {time_norm:.3f}', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(-0.5, color='r', linestyle=':', alpha=0.3, label='λ')
    ax1.axvline(0.5, color='r', linestyle=':', alpha=0.3)
    ax1.legend()
    
    # Plot 2: Absolute value
    ax2.plot(x_norm, np.abs(dn), 'r-', linewidth=2)
    ax2.set_xlabel('x/L', fontsize=13)
    ax2.set_ylabel('|δn/n₀|', fontsize=13)
    ax2.set_title('Absolute Density Perturbation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(-0.5, color='gray', linestyle=':', alpha=0.3)
    ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'iaw_density_t{time_norm:.3f}.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: iaw_density_t{time_norm:.3f}.png")
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("ION ACOUSTIC WAVE ANALYSIS")
    print("="*70)
    
    # Plot 1: Density evolution over time
    print("\n>>> Plotting density evolution...")
    plot_density_perturbation_vs_time(n_times=6)
    
    # Plot 2: Damping rate analysis
    print("\n>>> Computing damping rate...")
    plot_density_damping()
    
    # Plot 3: Single time snapshot
    print("\n>>> Plotting snapshot at t·Cs/L = 1.0...")
    plot_density_heatmap(time_value=1.0)