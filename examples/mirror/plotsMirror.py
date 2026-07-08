#!/usr/bin/env python3
"""
VPIC Magnetic Mirror Visualization in Cylindrical Coordinates
Reads VPIC field and hydro dumps and creates beautiful plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import struct
import glob
import os

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

class VPICReader:
    """Read VPIC binary output in cylindrical coordinates"""
    
    def __init__(self, run_dir="."):
        self.run_dir = run_dir
        self.read_info()
        
    def read_info(self):
        """Read info.bin file"""
        info_file = os.path.join(self.run_dir, "info.bin")
        
        with open(info_file, 'rb') as f:
            self.topology_x = struct.unpack('d', f.read(8))[0]
            self.topology_y = struct.unpack('d', f.read(8))[0]
            self.topology_z = struct.unpack('d', f.read(8))[0]
            
            self.Lr = struct.unpack('d', f.read(8))[0]
            self.Ltheta = struct.unpack('d', f.read(8))[0]
            self.Lz = struct.unpack('d', f.read(8))[0]
            
            self.nr = int(struct.unpack('d', f.read(8))[0])
            self.ntheta = int(struct.unpack('d', f.read(8))[0])
            self.nz = int(struct.unpack('d', f.read(8))[0])
            
            self.dt = struct.unpack('d', f.read(8))[0]
            self.r_min = struct.unpack('d', f.read(8))[0]
            self.mirror_ratio = struct.unpack('d', f.read(8))[0]
            
        self.r_max = self.r_min + self.Lr
        self.hr = self.Lr / self.nr
        self.htheta = self.Ltheta / self.ntheta
        self.hz = self.Lz / self.nz
        
        # Grid arrays
        self.r = np.linspace(self.r_min, self.r_max, self.nr)
        self.z = np.linspace(-self.Lz/2, self.Lz/2, self.nz)
        self.theta = np.linspace(0, self.Ltheta, self.ntheta)
        
        print(f"Grid: nr={self.nr}, ntheta={self.ntheta}, nz={self.nz}")
        print(f"Domain: r=[{self.r_min:.2f}, {self.r_max:.2f}]")
        print(f"        z=[{self.z[0]:.2f}, {self.z[-1]:.2f}]")
        print(f"Mirror ratio: {self.mirror_ratio:.2f}")
        
    def read_fields(self, timestep):
        """Read field dump at given timestep"""
        
        # VPIC field dumps are organized by topology
        ntot = self.nr * self.ntheta * self.nz
        
        # Initialize arrays
        ex = np.zeros(ntot)
        ey = np.zeros(ntot)
        ez = np.zeros(ntot)
        bx = np.zeros(ntot)  # Actually Br in cylindrical
        by = np.zeros(ntot)  # Actually Btheta
        bz = np.zeros(ntot)  # Actually Bz
        
        # Read from each topology domain
        ntop = int(self.topology_x * self.topology_y * self.topology_z)
        
        for itop in range(ntop):
            fname = os.path.join(self.run_dir, "fields", 
                                f"fields.{timestep}.{itop}")
            
            if not os.path.exists(fname):
                print(f"Warning: {fname} not found")
                continue
                
            with open(fname, 'rb') as f:
                # Read header info (varies by VPIC version)
                # Skip to data...
                # This is simplified - actual format may vary
                data = np.fromfile(f, dtype=np.float32)
                
                # Assume band-interleaved format as specified in deck
                npts = len(data) // 6  # ex, ey, ez, bx, by, bz
                
                if npts > 0:
                    ex_loc = data[0:npts]
                    ey_loc = data[npts:2*npts]
                    ez_loc = data[2*npts:3*npts]
                    bx_loc = data[3*npts:4*npts]
                    by_loc = data[4*npts:5*npts]
                    bz_loc = data[5*npts:6*npts]
                    
                    # Map to global array (simplified - needs proper indexing)
                    # This assumes data is in order
                    idx_start = itop * npts
                    idx_end = idx_start + npts
                    if idx_end <= ntot:
                        ex[idx_start:idx_end] = ex_loc
                        ey[idx_start:idx_end] = ey_loc
                        ez[idx_start:idx_end] = ez_loc
                        bx[idx_start:idx_end] = bx_loc
                        by[idx_start:idx_end] = by_loc
                        bz[idx_start:idx_end] = bz_loc
        
        # Reshape to 3D grid (r, theta, z)
        shape = (self.nr, self.ntheta, self.nz)
        
        return {
            'Er': ex.reshape(shape),     # Radial E
            'Etheta': ey.reshape(shape), # Azimuthal E
            'Ez': ez.reshape(shape),     # Axial E
            'Br': bx.reshape(shape),     # Radial B
            'Btheta': by.reshape(shape), # Azimuthal B
            'Bz': bz.reshape(shape),     # Axial B
        }
    
    def read_hydro(self, timestep, species='ion'):
        """Read hydro dump"""
        
        ntot = self.nr * self.ntheta * self.nz
        rho = np.zeros(ntot)
        jx = np.zeros(ntot)
        jy = np.zeros(ntot)
        jz = np.zeros(ntot)
        
        ntop = int(self.topology_x * self.topology_y * self.topology_z)
        
        for itop in range(ntop):
            fname = os.path.join(self.run_dir, "hydro",
                                f"Hhydro.{timestep}.{itop}")
            
            if not os.path.exists(fname):
                continue
                
            with open(fname, 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                
                # Band format: rho, jx, jy, jz, ...
                npts = len(data) // 4
                
                if npts > 0:
                    idx_start = itop * npts
                    idx_end = idx_start + npts
                    if idx_end <= ntot:
                        rho[idx_start:idx_end] = data[0:npts]
                        jx[idx_start:idx_end] = data[npts:2*npts]
                        jy[idx_start:idx_end] = data[2*npts:3*npts]
                        jz[idx_start:idx_end] = data[3*npts:4*npts]
        
        shape = (self.nr, self.ntheta, self.nz)
        
        return {
            'rho': rho.reshape(shape),
            'jr': jx.reshape(shape),
            'jtheta': jy.reshape(shape),
            'jz': jz.reshape(shape),
        }


class MirrorPlotter:
    """Create beautiful plots of magnetic mirror simulation"""
    
    def __init__(self, reader):
        self.reader = reader
        
    def plot_mirror_overview(self, timestep, savename=None):
        """Create comprehensive overview plot"""
        
        fields = self.reader.read_fields(timestep)
        hydro = self.reader.read_hydro(timestep)
        
        # Average over theta (axisymmetric)
        Br = np.mean(fields['Br'], axis=1)   # (nr, nz)
        Bz = np.mean(fields['Bz'], axis=1)
        rho = np.mean(hydro['rho'], axis=1)
        
        # Create meshgrid for plotting
        R, Z = np.meshgrid(self.reader.r, self.reader.z, indexing='ij')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Magnetic field lines
        ax = axes[0, 0]
        Bmag = np.sqrt(Br**2 + Bz**2)
        
        # Plot field magnitude as background
        im = ax.pcolormesh(Z, R, Bmag, shading='auto', cmap='viridis')
        
        # Streamlines for field lines
        # Seed points along axis
        seed_z = np.linspace(self.reader.z[0], self.reader.z[-1], 20)
        seed_r = np.ones_like(seed_z) * self.reader.r[5]
        seed_points = np.column_stack([seed_z, seed_r])
        
        ax.streamplot(Z.T, R.T, Bz.T, Br.T, color='white', 
                     density=1.5, linewidth=1, arrowsize=0.8,
                     start_points=seed_points)
        
        ax.set_xlabel('z')
        ax.set_ylabel('r')
        ax.set_title('Magnetic Field Lines')
        ax.set_aspect('auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='|B|')
        
        # 2. Density plot
        ax = axes[0, 1]
        im = ax.pcolormesh(Z, R, rho, shading='auto', cmap='plasma')
        ax.set_xlabel('z')
        ax.set_ylabel('r')
        ax.set_title('Ion Density')
        ax.set_aspect('auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='ρ')
        
        # 3. Axial profiles
        ax = axes[1, 0]
        
        # B_z along axis (r index close to axis)
        ir_axis = 2
        Bz_axis = Bz[ir_axis, :]
        rho_axis = rho[ir_axis, :]
        
        ax.plot(self.reader.z, Bz_axis, 'b-', linewidth=2, label='$B_z$')
        ax.set_xlabel('z')
        ax.set_ylabel('$B_z$', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(True, alpha=0.3)
        ax.set_title('On-Axis Profiles')
        
        ax2 = ax.twinx()
        ax2.plot(self.reader.z, rho_axis, 'r-', linewidth=2, label='ρ')
        ax2.set_ylabel('Density ρ', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # 4. Radial profiles at midplane
        ax = axes[1, 1]
        iz_mid = self.reader.nz // 2
        
        Br_mid = Br[:, iz_mid]
        Bz_mid = Bz[:, iz_mid]
        rho_mid = rho[:, iz_mid]
        
        ax.plot(self.reader.r, Bz_mid, 'b-', linewidth=2, label='$B_z$')
        ax.plot(self.reader.r, Br_mid, 'c--', linewidth=2, label='$B_r$')
        ax.set_xlabel('r')
        ax.set_ylabel('Magnetic Field', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('Midplane Radial Profiles')
        
        ax2 = ax.twinx()
        ax2.plot(self.reader.r, rho_mid, 'r-', linewidth=2)
        ax2.set_ylabel('Density ρ', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.suptitle(f'Magnetic Mirror: t = {timestep * self.reader.dt:.2f}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if savename:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_3d_field_lines(self, timestep, savename=None):
        """3D visualization of field lines"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fields = self.reader.read_fields(timestep)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Average over theta
        Br = np.mean(fields['Br'], axis=1)
        Bz = np.mean(fields['Bz'], axis=1)
        
        # Create 3D coordinates (cylindrical to Cartesian)
        R, Z = np.meshgrid(self.reader.r, self.reader.z, indexing='ij')
        
        # Trace field lines by integration
        nlines = 12
        for i in range(nlines):
            theta = 2*np.pi * i / nlines
            
            # Start point
            r0 = self.reader.r[5]
            z0 = self.reader.z[self.reader.nz//4]
            
            # Simple Euler integration
            r_line = [r0]
            z_line = [z0]
            
            for step in range(200):
                ir = np.argmin(np.abs(self.reader.r - r_line[-1]))
                iz = np.argmin(np.abs(self.reader.z - z_line[-1]))
                
                if ir >= self.reader.nr-1 or iz >= self.reader.nz-1:
                    break
                    
                br = Br[ir, iz]
                bz = Bz[ir, iz]
                bmag = np.sqrt(br**2 + bz**2)
                
                if bmag < 1e-10:
                    break
                
                ds = 0.05
                r_line.append(r_line[-1] + ds * br/bmag)
                z_line.append(z_line[-1] + ds * bz/bmag)
                
                if r_line[-1] < self.reader.r_min or r_line[-1] > self.reader.r_max:
                    break
            
            # Convert to Cartesian
            x_line = [r * np.cos(theta) for r in r_line]
            y_line = [r * np.sin(theta) for r in r_line]
            
            ax.plot(z_line, x_line, y_line, 'b-', linewidth=1, alpha=0.7)
        
        # Draw mirror coils (schematic)
        coil_r = self.reader.r_max * 0.8
        coil_z = [self.reader.z[0]*0.8, self.reader.z[-1]*0.8]
        theta_coil = np.linspace(0, 2*np.pi, 50)
        
        for zc in coil_z:
            x_coil = coil_r * np.cos(theta_coil)
            y_coil = coil_r * np.sin(theta_coil)
            z_coil = np.ones_like(theta_coil) * zc
            ax.plot(z_coil, x_coil, y_coil, 'r-', linewidth=3)
        
        ax.set_xlabel('z')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.set_title(f'3D Magnetic Field Lines\nt = {timestep * self.reader.dt:.2f}')
        
        if savename:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_phase_space(self, timestep, savename=None):
        """Plot phase space from hydro moments"""
        
        hydro = self.reader.read_hydro(timestep)
        
        # Average over theta
        rho = np.mean(hydro['rho'], axis=1)
        jz = np.mean(hydro['jz'], axis=1)
        
        # Velocity = j / rho (approximately)
        vz = np.divide(jz, rho, where=rho>1e-10, out=np.zeros_like(jz))
        
        R, Z = np.meshgrid(self.reader.r, self.reader.z, indexing='ij')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # z-vz phase space (along axis)
        ax = axes[0]
        ir_axis = 2
        z_axis = self.reader.z
        vz_axis = vz[ir_axis, :]
        rho_axis = rho[ir_axis, :]
        
        # Scatter plot colored by density
        scatter = ax.scatter(z_axis, vz_axis, c=rho_axis, 
                           cmap='plasma', s=50, alpha=0.7)
        ax.set_xlabel('z')
        ax.set_ylabel('$v_z$')
        ax.set_title('Phase Space (on-axis)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='ρ')
        
        # r-vz at midplane
        ax = axes[1]
        iz_mid = self.reader.nz // 2
        r_mid = self.reader.r
        vz_mid = vz[:, iz_mid]
        rho_mid = rho[:, iz_mid]
        
        scatter = ax.scatter(r_mid, vz_mid, c=rho_mid,
                           cmap='plasma', s=50, alpha=0.7)
        ax.set_xlabel('r')
        ax.set_ylabel('$v_z$')
        ax.set_title('Phase Space (midplane)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='ρ')
        
        plt.suptitle(f'Phase Space: t = {timestep * self.reader.dt:.2f}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if savename:
            plt.savefig(savename, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main analysis script"""
    
    # Read VPIC output
    print("Reading VPIC output...")
    reader = VPICReader("../../build")
    plotter = MirrorPlotter(reader)
    
    # Find available timesteps
    field_files = glob.glob("../../build/fields/T.*/fields.*.0")
    timesteps = sorted([int(f.split('.')[1]) for f in field_files])
    
    print(f"Found {len(timesteps)} timesteps")
    print(f"Timesteps: {timesteps[:5]} ... {timesteps[-5:]}")
    
    # Plot initial condition
    if len(timesteps) > 0:
        print("\nPlotting initial condition...")
        plotter.plot_mirror_overview(timesteps[0], 'mirror_t0.png')
    
    # Plot middle timestep
    if len(timesteps) > 10:
        tmid = timesteps[len(timesteps)//2]
        print(f"\nPlotting t={tmid}...")
        plotter.plot_mirror_overview(tmid, f'mirror_t{tmid}.png')
        plotter.plot_3d_field_lines(tmid, f'mirror_3d_t{tmid}.png')
        plotter.plot_phase_space(tmid, f'phase_t{tmid}.png')
    
    # # Create animation
    # print("\nCreating animation...")
    # make_movie(reader, plotter, timesteps)
    
    print("\nDone!")


def make_movie(reader, plotter, timesteps, skip=5):
    """Create movie from timesteps"""
    import matplotlib.animation as animation
    
    print(f"Animating {len(timesteps[::skip])} frames...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    def update(frame_idx):
        timestep = timesteps[frame_idx * skip]
        
        for ax in axes:
            ax.clear()
        
        fields = reader.read_fields(timestep)
        hydro = reader.read_hydro(timestep)
        
        Br = np.mean(fields['Br'], axis=1)
        Bz = np.mean(fields['Bz'], axis=1)
        rho = np.mean(hydro['rho'], axis=1)
        
        R, Z = np.meshgrid(reader.r, reader.z, indexing='ij')
        
        # Density
        ax = axes[0]
        im = ax.pcolormesh(Z, R, rho, shading='auto', cmap='plasma')
        ax.set_xlabel('z')
        ax.set_ylabel('r')
        ax.set_title('Ion Density')
        plt.colorbar(im, ax=ax)
        
        # Field lines
        ax = axes[1]
        Bmag = np.sqrt(Br**2 + Bz**2)
        ax.pcolormesh(Z, R, Bmag, shading='auto', cmap='viridis', alpha=0.5)