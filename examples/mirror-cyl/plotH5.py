"""
Make movie of plasma property from hdf5 output

To view keys: $ h5dump -H data.h5

Data keys (from dump_strategy.h):

    // field keys:
    "ex", "ey", "ez", "div_e_err",
    "cbx", "cby", "cbz", "pe",
    "tcax", "tcay", "tcaz", "rhob",
    "jfx", "jfy", "jfz", "rhof",
    "jfxold", "jfyold", "jfzold", "rhofold",
    "cbx0", "cby0", "cbz0", "te0",
    "tx", "ty", "tz", "te",
    "ox", "oy", "oz", "oe","div_b_err"

    typedef struct hydro {
    float jx, jy, jz, rho; // Current and charge density => <q v_i f>, <q f>
    float px, py, pz, rho_m; // Momentum and mass density (changed from ke_density)
    float txx, tyy, tzz;   // Stress diagonal            => <p_i v_j f>, i==j
    float tyz, tzx, txy;   // Stress off-diagonal        => <p_i v_j f>, i!=j
    #if VARIABLE_CHARGE
    float qmin, qmax;      // Minimum and maximum charge within a cell
    #else
    float _pad[2];         // 16-byte align
    #endif

    // fluid keys:
    "den", "prs", "tmp", "ux", "uy", "uz"


"""

# import matplotlib.pylab as plt
# import numpy as np
# import h5py, os, sys, string
# from scipy import constants as con
# import ffmpeg  # pip install ffmpeg-python

# import GenerateRun as run
from utils import *

m_i = 2.0 * con.atomic_mass


#################################################################
# Create single figure
#################################################################
def make_figure(
    data_type_str_,
    species_type_str_,
    key_str_, 
    step_,
    ixyz_=None,
    contour_range_=None,
    unit_label_=None,
    unit_conv_=None,
    show_fig_=False,
    tag_='0'):

    fig_dir = f'figures/{data_type_str_}'
    os.makedirs(fig_dir, exist_ok=True)
        
    # load file and get data subset
    if data_type_str_ == 'fields':
        fig_str = f'{fig_dir}/{key_str_}'
        fname = f'{data_type_str_}_hdf5/T.{step_}/{data_type_str_}_{step_}.h5'
    else:        
        fig_str = f'{fig_dir}/{species_type_str_}_{key_str_}'
        fname = f'{data_type_str_}_hdf5/T.{step_}/{data_type_str_}_{species_type_str_}_{step_}.h5'

    fh = h5py.File(fname, "r")
    group = fh["Timestep_" + str(step_)]

    
    if key_str_ == 'tmp':
        # calculate temperature
        rho_m = group['rho_m'][:,:,:]
        ke2 = (group['txx'][:,:,:] + group['tyy'][:,:,:] + group['tzz'][:,:,:]) / rho_m    

        vx = (group['px'][:,:,:] * con.c) / rho_m # v = (p*m*c) / (m*rho)
        vy = (group['py'][:,:,:] * con.c) / rho_m
        vz = (group['pz'][:,:,:] * con.c) / rho_m

        t = (ke2 - m_i * (vx**2.0 + vy**2.0 + vz**2.0)) / (3.0)

        # if subdomain is not set, plot all data
        data_shape = np.shape(t)
        if ixyz_ is None:
            ixyz_ = [[0, data_shape[0]], [0], [0, data_shape[2]]]
        data = t[ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]

    elif key_str_ == 'cbmag':
        data_shape = np.shape(group['cbx'])

        # if subdomain is not set, plot all data
        if ixyz_ is None:
            ixyz_ = [[0, data_shape[0]], [0], [0, data_shape[2]]]

        bx  = group['cbx'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]
        bx += group['cbx0'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]
        by  = group['cby'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]
        by += group['cby0'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]
        bz  = group['cbz'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]
        bz += group['cbz0'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]

        data = (bx**2 + by**2 + bz**2)**0.5

    else:
        data_shape = np.shape(group[key_str_])

        # if subdomain is not set, plot all data
        if ixyz_ is None:
            ixyz_ = [[0, data_shape[0]], [0], [0, data_shape[2]]]
        data = group[key_str_][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]

        # include background magentic field 
        # (second conditional lets user plot cbx0 independently)
        if key_str_[:2] == 'cb' and len(key_str_) == 3:
            data_backgrnd = group[key_str_ + '0'][ixyz_[0][0]:ixyz_[0][1], ixyz_[1][0], ixyz_[2][0]:ixyz_[2][1]]
            data += data_backgrnd

    # # change data to physical units
    # if unit_conv_ is None:
    #     data *= conv_unit_dict[key_str_]
    # else:
    #     data *= unit_conv_

    # if unit_label_ is None:
    #     unit_label = unit_labl_dict[key_str_]
    # else:
    #     unit_label = unit_label_

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))

    # title = f'{data_type_str_}, '
    # if data_type_str_ != 'fields':
    #     title += f'{species_type_str_}, '
    # title += f'{key_str_} ({unit_label}), t = {dt*step_*1.0e9:3.3f} ns'

    # title += f'{key_str_} ({unit_label}), t = {dt*step_:3.3f}'+r' $\omega_{ci}^{-1}$'
    # title = f't = {dt*step_:4.0f} $\omega_{{ci}}^{{-1}}$'

    # ax.set_title(title)

    # ax.set_title('(c) $q_0$, $100$ keV', loc='left', fontsize=16)


    ax.set_xlabel(r'$z$ (cm)') #(r'$z/d_{\mathrm{i}}$')
    ax.set_ylabel(r'$x$ (cm)') #(r'$x/d_{\mathrm{i}}$')
    ax.tick_params(axis='both', pad=8)

    lx = np.linspace(-Lx/2.0, Lx/2.0, np.shape(data)[0])[:] * ref_di
    lz = np.linspace(-Lz/2.0, Lz/2.0, np.shape(data)[1])[:] * ref_di

    if contour_range_ is None:
        log_norm = colors.LogNorm(vmin=10**(-5), vmax=10**(-3.5))
        im = ax.pcolormesh(lz, lx, data, norm=log_norm)
        # im = ax.pcolormesh(lz, lx, data)
        # im = ax.pcolormesh(lz, lx, np.log10(data))
    else:
        im = ax.pcolormesh(lz, lx, np.log10(data), 
                vmin=np.log10(contour_range_[0]), vmax=np.log10(contour_range_[1]))
    
    # if key_str_[:2] == 'cb':
    #     ax.contour(lz, lx, data, colors='w', linestyles='-', levels=[10**-1, 10**-0.4, 10**-0.2, 10**-0.1])

    fig.colorbar(im, ax=ax)#, format='%.0f')

    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 1])
    plt.savefig(f'{fig_str}_{tag_:05}.png')

    if show_fig_:
        plt.show()
    plt.close()

    return


#################################################################
# Stitch together figures with ffmpeg
#################################################################
def make_movie(
    data_type_str_, 
    species_type_str_,
    key_str_, 
    steps_,
    frame_rate_=10,
    ixyz_=None,
    contour_range_=None,    
    unit_label_=None,
    unit_conv_=None,
    show_fig_=False):

    # create directory for figures and movies
    fig_dir = f'figures/{data_type_str_}'
    movie_dir = 'figures/movies'
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(movie_dir, exist_ok=True)

    # create figures for each step
    cntr = 0
    for step in steps:
        print(f'Creating figure for step = {step}  ({cntr+1} / {np.size(steps)})')
        make_figure(data_type_str_, species_type_str_, key_str_, step, \
                    ixyz_, contour_range_, unit_label_, unit_conv_, show_fig_, cntr)
        cntr += 1

    # create movie with ffmpeg
    if data_type_str_ == 'fields':
        image_patter = f'figures/{data_type_str_}/{key_str_}_%05d.png'
        movie_fname = f'{movie_dir}/movie_{key_str_}.mp4'
    else:
        image_patter = f'figures/{data_type_str_}/{species_type_str_}_{key_str_}_%05d.png'
        movie_fname = f'{movie_dir}/movie_{species_type_str_}_{key_str_}.mp4'

    print(f'Writing movie to: {movie_fname}')
    (
        ffmpeg
        .input(image_patter, framerate=frame_rate_)
        .output(movie_fname, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )

    return


######################################################################################
######################################################################################

if __name__ == '__main__':

    interval =   1000
    num_step = 100000
    steps = np.arange(0, num_step + interval, interval)
    frame_rate = 10
    ixyz = [[250, 550], [0], [50, 500]] # x,y,z indices specify 2d subset of data to plot

    # Example 1: plot fields/rho with default options
    # make_figure('fields', '', 'cbx0', 0, show_fig_=True)
    # make_figure('fields', '', 'cbz0', 0, show_fig_=True)
    make_figure('hydro', 'D_beam', 'n_q1', 25000, show_fig_=True)
    # make_figure('hydro', 'beam', 'rho', 2000, show_fig_=True)

    # Example 2: plot fields/rho in units of cm^-3 with fixed colorbar
    # contour_range = [5.0e2, 1.0e9] # range in m
    # unit_label = r'cm$^{-3}$'
    # unit_conv = run.refval['n0_SI'] * 1.0e-6
    # make_figure('fields', '', 'rhof', 0, ixyz_=ixyz, contour_range_=contour_range, 
    #             unit_label_=unit_label, unit_conv_=unit_conv, show_fig_=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # field movies
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # field_keys = ['rhof']
    # for key in field_keys:
    #     # default options:
    #     # make_movie('fields', '', key, steps, frame_rate)

    #     # specify subdomain, units, and contour range
    #     make_movie('fields', '', key, steps, frame_rate ixyz_=ixyz,
    #                contour_range_=contour_range, unit_label_=unit_label, unit_conv_=unit_conv)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # hydro movies
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # hydro_species = ['D_beam', 'D_seed']
    # contour_range = [1.0e19, 1.0e20]
    # hydro_keys = ['n_q0', 'n_q1']
    # for species in hydro_species:
    #     for key in hydro_keys:
    #         # make_movie('hydro', species, key, steps, frame_rate)
    #         make_movie('hydro', species, key, steps, frame_rate)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # fluid movies
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # fluid_species = ['O', 'N2', 'HE']
    # fluid_species = run.msis['fluid_species'].keys()
    # fluid_keys = ['den', 'tmp']
    # for species in fluid_species:
    #     for key in fluid_keys:
    #         make_movie('fluid', species, key, steps, frame_rate, ixyz)
