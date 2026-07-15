"""
Make movie of plasma property from hdf5 output

To view keys: $ h5dump -H data.h5

Data keys reference available in dump_strategy.h
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
import ffmpeg
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, List, Tuple

from utils import *


def make_figure(
    data_type_str: str,
    species_type_str: str,
    key_str: str,
    title_str: str,
    step: int,
    ixyz: Optional[List[List[int]]] = None,
    contour_range: Optional[Tuple[float, float]] = None,
    unit_label: Optional[str] = None,
    unit_conv: Optional[float] = None,
    show_fig: bool = False,
    tag: int = 0
) -> None:
    """
    Create a single figure from HDF5 data.
    
    Args:
        data_type_str: Type of data ('fields', 'hydro', 'fluid')
        species_type_str: Species type (empty for fields)
        key_str: Data key to plot
        step: Time step number
        ixyz: Data subset indices [[x_start, x_end], [y_idx], [z_start, z_end]]
        contour_range: Color scale range [min, max]
        unit_label: Label for units
        unit_conv: Conversion factor for units
        show_fig: Whether to display the figure
        tag: Figure sequence number for movie frames
    """
    # Create output directory
    fig_dir = f'figures/{data_type_str}'
    os.makedirs(fig_dir, exist_ok=True)
        
    # Load file and get data subset
    species_type_str = 'D_seed'
    fig_str = f'{fig_dir}/{species_type_str}_{key_str}'
    fname = f'{data_type_str}_hdf5/T.{step}/{data_type_str}_{species_type_str}_{step}.h5'

    with h5py.File(fname, "r") as fh:
        
        group = fh[f"Timestep_{step}"]
        data_shape = group['rho_m'].shape

        # If subdomain is not set, plot all data
        if ixyz is None:
            ixyz = [[0, data_shape[0]], [0], [0, data_shape[2]]]
        
        # calculate temperature
        rho_m = group['rho_m'][:,:,:]
        ke2 = (group['txx'][:,:,:] + group['tyy'][:,:,:] + group['tzz'][:,:,:]) / rho_m    

        vx = (group['px'][:,:,:]) / rho_m # v = (p*m*c) / (m*rho)
        vy = (group['py'][:,:,:]) / rho_m
        vz = (group['pz'][:,:,:]) / rho_m

        m_i = 2.014102 #* con.atomic_mass
        t = m_i * (ke2 - (vx**2.0 + vy**2.0 + vz**2.0)) / (3.0)
        t = np.nan_to_num(t)

        data = t[ixyz[0][0]:ixyz[0][1], ixyz[1][0], ixyz[2][0]:ixyz[2][1]] 
        data *= ref_E0 * 6.242e+11 * 1e-3 # from sim units -> erg -> eV -> keV

    species_type_str = 'D_seed'
    fig_str = f'{fig_dir}/{species_type_str}_{key_str}'
    fname = f'{data_type_str}_hdf5/T.{step}/{data_type_str}_{species_type_str}_{step}.h5'

    with h5py.File(fname, "r") as fh:
        
        group = fh[f"Timestep_{step}"]
        data_shape = group['rho_m'].shape

        # If subdomain is not set, plot all data
        if ixyz is None:
            ixyz = [[0, data_shape[0]], [0], [0, data_shape[2]]]
        
        # calculate temperature
        rho_m = group['rho_m'][:,:,:]
        ke2 = (group['txx'][:,:,:] + group['tyy'][:,:,:] + group['tzz'][:,:,:]) / rho_m    

        vx = (group['px'][:,:,:]) / rho_m # v = (p*m*c) / (m*rho)
        vy = (group['py'][:,:,:]) / rho_m
        vz = (group['pz'][:,:,:]) / rho_m

        m_i = 2.014102 #* con.atomic_mass
        t = m_i * (ke2 - (vx**2.0 + vy**2.0 + vz**2.0)) / (3.0)
        t = np.nan_to_num(t)
        
        data2 = t[ixyz[0][0]:ixyz[0][1], ixyz[1][0], ixyz[2][0]:ixyz[2][1]] 
        data2 *= ref_E0 * 6.242e+11 * 1e-3 # from sim units -> erg -> eV -> keV

    data += data2
    data=rho_m
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 3.25))

    t = dt * step / ref_wci * 1e6 # us
    ax.set_title(rf'{title_str}, $t=${t:4.2f} $\mu$s', loc='left', fontsize=18)
    # ax.set_title(rf'(a) $q=1$, $t=${t:4.2f} $\mu$s', loc='left', fontsize=18)
    # ax.set_title(rf'(a) $E=25$ keV, $t=${t:4.3f} $\mu$s', loc='left', fontsize=20)

    ax.set_ylabel('x (cm)')
    ax.set_xlabel('z (cm)')
    ax.tick_params(axis='both', pad=8)
    # ax.tick_params(axis='both', labelbottom=False, labelleft=False)

    ax.set_ylim([-35, 35])
    ax.set_xlim([-380, 380])

    lx = np.linspace(-Lx/2.0, Lx/2.0, np.shape(data)[0])[:] * ref_di
    lz = np.linspace(-Lz/2.0, Lz/2.0, np.shape(data)[1])[:] * ref_di

    # Plot data
    if contour_range is None:
        im = ax.pcolormesh(lz, lx, np.log10(data))
    else:
        # log_norm = colors.LogNorm(vmin=contour_range[0], vmax=contour_range[1])
        # im = ax.pcolormesh(lz, lx, data, norm=log_norm)
        im = ax.pcolormesh(lz, lx, data, vmin=contour_range[0], vmax=contour_range[1])
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_title(r'$n/n_0$', fontsize=18, pad=15)
    #cbar.ax.set_title(r'$T$ (keV)', fontsize=18, pad=15)

    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 1])
    plt.savefig(f'{fig_str}_{tag:05d}.png', bbox_inches='tight') # dpi=100, 

    if show_fig:
        plt.show()
    plt.close(fig)


def _process_single_step(
    step: int, 
    cntr: int,
    data_type_str: str,
    species_type_str: str,
    key_str: str,
    title_str: str,
    ixyz: Optional[List[List[int]]],
    contour_range: Optional[Tuple[float, float]],
    unit_label: Optional[str],
    unit_conv: Optional[float],
    show_fig: bool
) -> Tuple[int, int]:
    """Helper function to process a single step (for parallelization)."""
    make_figure(
        data_type_str, species_type_str, key_str, title_str, step, 
        ixyz, contour_range, unit_label, 
        unit_conv, show_fig, cntr
    )
    return cntr, step


def make_movie(
    data_type_str: str,
    species_type_str: str,
    key_str: str,
    title_str: str,
    steps: np.ndarray,
    frame_rate: int = 10,
    ixyz: Optional[List[List[int]]] = None,
    contour_range: Optional[Tuple[float, float]] = None,
    unit_label: Optional[str] = None,
    unit_conv: Optional[float] = None,
    show_fig: bool = False,
    max_workers: Optional[int] = None
) -> None:
    """
    Create movie from HDF5 data by stitching together figures with ffmpeg.
    
    Args:
        data_type_str: Type of data ('fields', 'hydro', 'fluid')
        species_type_str: Species type (empty for fields)
        key_str: Data key to plot
        steps: Array of time steps to process
        frame_rate: Frames per second for output movie
        ixyz: Data subset indices [[x_start, x_end], [y_idx], [z_start, z_end]]
        contour_range: Color scale range [min, max]
        unit_label: Label for units
        unit_conv: Conversion factor for units
        show_fig: Whether to display figures
        max_workers: Maximum number of parallel workers (None = CPU count)
    """
    # Create directories
    fig_dir = f'figures/{data_type_str}'
    movie_dir = 'figures/movies'
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(movie_dir, exist_ok=True)

    # Create figures in parallel
    print(f'Creating {len(steps)} figures using parallel processing...')
    
    # Create partial function with fixed parameters
    process_func = partial(
        _process_single_step,
        data_type_str=data_type_str,
        species_type_str=species_type_str,
        key_str=key_str,
        title_str=title_str,
        ixyz=ixyz,
        contour_range=contour_range,
        unit_label=unit_label,
        unit_conv=unit_conv,
        show_fig=show_fig
    )

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_func, step, cntr): (step, cntr)
            for cntr, step in enumerate(steps)
        }
        
        completed = 0
        for future in as_completed(futures):
            cntr, step = future.result()
            completed += 1
            print(f'Completed step {step} ({completed}/{len(steps)})')

    # Create movie with ffmpeg
    if data_type_str == 'fields':
        image_pattern = f'figures/{data_type_str}/{key_str}_%05d.png'
        movie_fname = f'{movie_dir}/movie_{key_str}.mp4'
    else:
        image_pattern = f'figures/{data_type_str}/{species_type_str}_{key_str}_%05d.png'
        movie_fname = f'{movie_dir}/movie_{species_type_str}_{key_str}.mp4'

    print(f'Writing movie to: {movie_fname}')
    (
        ffmpeg
        .input(image_pattern, framerate=frame_rate)
        .output(movie_fname, vcodec='libx264')
        .overwrite_output()
        .run(quiet=True)
    )
    print(f'Movie created successfully: {movie_fname}')


if __name__ == '__main__':
    # Configuration
    interval = 100
    num_step = 200
    steps = np.arange(0, num_step + interval, interval)
    frame_rate = 30

    # make_figure('hydro', 'D_seed', 'tmp', 'T (keV)', 60000, contour_range=[0.0, 40.0], show_fig=True) 

    # Hydro movies
    hydro_species = ['D_beam']
    hydro_keys = ['tmp']
    titles = ['Temperature (keV)']
    for species in hydro_species:
        for i_key in range(len(hydro_keys)):
            make_movie(
                'hydro', species, hydro_keys[i_key], titles[i_key], steps, frame_rate, 
                contour_range=[0.0, 40.0],
                max_workers=16
            )
