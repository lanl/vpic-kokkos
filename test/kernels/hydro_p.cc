// Compare the Kokkos and legacy hydro_p against a double-precision sum of the
// same per-particle float contributions. The double sum has its own rounding
// error, but over a few hundred float terms it is many orders of magnitude
// below the float tolerance used here, so it serves as the reference.
// Written by Scott V. Luedtke, XCP-6, September 14, 2022
//
// Float summation error depends on the order of the additions, which differs
// between the legacy loop, OpenMP (duplicated scatter) and GPUs (atomics), so
// neither implementation can serve as the reference for the other. For a sum
// of n terms in any order, |error| <= (n-1)*u*sum|term| with u = FLT_EPSILON/2.
// Measuring against sum|term| rather than the sum itself keeps components that
// nearly cancel (momentum, off-diagonal stress) from looking inaccurate.

#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

void vpic_simulation::user_diagnostics() {}

// Per-particle hydro contributions, computed in float the same way as in
// accumulate_hydro_p, and summed in double
static void
reference_hydro( const species_t * sp, const interpolator_array_t * ia,
             std::vector<double>& sum, std::vector<double>& abs_sum,
             std::vector<int>& count )
{
    const grid_t * g = sp->g;
    const interpolator_t * f = ia->i;
    const float c        = g->cvac;
    const float qsp      = sp->q;
    const float mspc     = sp->m*c;
    const float qdt_2mc  = (qsp*g->dt)/(2*mspc);
    const float qdt_4mc2 = qdt_2mc / (2*c);
    const float r8V      = g->r8V;
    const int stride_10 = VOXEL(1,0,0, g->nx,g->ny,g->nz) - VOXEL(0,0,0, g->nx,g->ny,g->nz);
    const int stride_21 = VOXEL(0,1,0, g->nx,g->ny,g->nz) - VOXEL(1,0,0, g->nx,g->ny,g->nz);
    const int stride_43 = VOXEL(0,0,1, g->nx,g->ny,g->nz) - VOXEL(1,1,0, g->nx,g->ny,g->nz);

    for (size_t n = 0; n < (size_t)sp->np; n++)
    {
        float dx = sp->p[n].dx, dy = sp->p[n].dy, dz = sp->p[n].dz;
        float ux = sp->p[n].ux, uy = sp->p[n].uy, uz = sp->p[n].uz;
        float w  = sp->p[n].w;
        int   i  = sp->p[n].i;
        float vx, vy, vz, ke_mc, w0, w1, w2, w3, w4, w5, w6, w7;

        ux += qdt_2mc*((f[i].ex+dy*f[i].dexdy) + dz*(f[i].dexdz+dy*f[i].d2exdydz));
        uy += qdt_2mc*((f[i].ey+dz*f[i].deydz) + dx*(f[i].deydx+dz*f[i].d2eydzdx));
        uz += qdt_2mc*((f[i].ez+dx*f[i].dezdx) + dy*(f[i].dezdy+dx*f[i].d2ezdxdy));
        w5 = f[i].cbx + dx*f[i].dcbxdx;
        w6 = f[i].cby + dy*f[i].dcbydy;
        w7 = f[i].cbz + dz*f[i].dcbzdz;
        ke_mc = ux*ux + uy*uy + uz*uz;
        vz = sqrt(1+ke_mc);
        ke_mc *= c/(vz+1);
        vz = c/vz;
        w0 = qdt_4mc2*vz;
        w1 = w5*w5 + w6*w6 + w7*w7;
        w2 = w0*w0*w1;
        w3 = w0*(1+(1./3.)*w2*(1+0.4*w2));
        w4 = w3/(1 + w1*w3*w3); w4 += w4;
        w0 = ux + w3*( uy*w7 - uz*w6 );
        w1 = uy + w3*( uz*w5 - ux*w7 );
        w2 = uz + w3*( ux*w6 - uy*w5 );
        ux += w4*( w1*w7 - w2*w6 );
        uy += w4*( w2*w5 - w0*w7 );
        uz += w4*( w0*w6 - w1*w5 );
        vx  = ux*vz;
        vy  = uy*vz;
        vz *= uz;

        w0  = r8V*w;
        dx *= w0;
        w1  = w0+dx;
        w0 -= dx;
        w3  = 1+dy;
        w2  = w0*w3;
        w3 *= w1;
        dy  = 1-dy;
        w0 *= dy;
        w1 *= dy;
        w7  = 1+dz;
        w4  = w0*w7;
        w5  = w1*w7;
        w6  = w2*w7;
        w7 *= w3;
        dz  = 1-dz;
        w0 *= dz;
        w1 *= dz;
        w2 *= dz;
        w3 *= dz;

        const float wn[8] = { w0, w1, w2, w3, w4, w5, w6, w7 };
        const int   step[8] = { 0, stride_10, stride_21, stride_10,
                                stride_43, stride_10, stride_21, stride_10 };
        for (int k = 0; k < 8; k++)
        {
            i += step[k];
            float t = qsp*wn[k];
            float pt = mspc*wn[k];
            float px = pt*ux, py = pt*uy, pz = pt*uz;
            // Same order as the members of hydro_t
            const float term[14] = { t*vx, t*vy, t*vz, t,
                                     px, py, pz, pt*ke_mc,
                                     px*vx, py*vy, pz*vz,
                                     py*vz, pz*vx, px*vy };
            for (int m = 0; m < 14; m++)
            {
                sum[14*i + m]     += term[m];
                abs_sum[14*i + m] += std::abs(term[m]);
            }
            count[i]++;
        }
    }
}

void
vpic_simulation::user_initialization( int num_cmdline_arguments,
                                      char ** cmdline_argument )
{
    double L  = 1e2;
    int ncell = 8;
    int npart = 32*ncell*ncell*ncell;  // 32 per cell: ~256 contributions per interior grid node

    std::cout << "Initializing grid and fields" << std::endl;

    define_units( 1, 1 );
    define_timestep( 1 );
    define_periodic_grid( 0, 0, 0,   // Grid low corner
            L, L, L,   // Grid high corner
            ncell, ncell, ncell,   // Grid resolution
            1, 1, 1 ); // Processor configuration
    define_material( "vacuum", 1.0, 1.0, 0.0 );
    define_field_array();

    hydro_array_t *hydro_array_legacy = new_hydro_array(grid);
    hydro_array_t *hydro_array_kokkos = new_hydro_array(grid);

    field(1,1,1).ex  = 1;
    field(1,2,1).ex  = 1;
    field(1,1,2).ex  = 1;
    field(1,2,2).ex  = 1;

    field(1,1,1).ey  = 2;
    field(1,1,2).ey  = 2;
    field(2,1,1).ey  = 2;
    field(2,1,2).ey  = 2;

    field(1,1,1).ez  = 3;
    field(2,1,1).ez  = 3;
    field(1,2,1).ez  = 3;
    field(2,2,1).ez  = 3;

    std::cout << "Initializing particles" << std::endl;
    species_t * sp_temp;
    species_t * sp = define_species( "test_species", 1., 1., npart, npart, 0, 0 );

    for (int i = 0; i < npart; i++)
    {
        float x = uniform( rng(0), 0, L);
        float y = uniform( rng(0), 0, L);
        float z = uniform( rng(0), 0, L);

        // Some random relativistic momemtum
        float px = normal( rng(0), 0, 0.5);
        float py = normal( rng(0), 0, 1);
        float pz = normal( rng(0), 0, 2);

        inject_particle( sp , x, y, z, px, py, pz, 1., 0., 0);
    }

    // Make sure kokkos views have correct data
    field_array->copy_to_device();
    LIST_FOR_EACH( sp_temp, species_list ) {
      sp_temp->copy_to_device();
    }

    // Build the interpolators from the fields set above (on the device), and
    // copy them back for the legacy kernel and the reference
    load_interpolator_array( interpolator_array, field_array );
    interpolator_array->copy_to_host();

    // Call both functions
    std::cout << "Accumulating legacy" << std::endl;
    accumulate_hydro_p( hydro_array_legacy, sp, interpolator_array );

    std::cout << "Accumulating kokkos" << std::endl;
    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;

    auto hydro_view = hydro_array_kokkos->k_h_d;

    accumulate_hydro_p_kokkos(
        particles,
        particles_i,
        hydro_view,
        interpolators_k,
        sp
    );

    hydro_array_kokkos->copy_to_host();

    std::cout << "Accumulating double-precision reference" << std::endl;
    std::vector<double> reference(14*grid->nv, 0.0), reference_abs(14*grid->nv, 0.0);
    std::vector<int> count(grid->nv, 0);
    reference_hydro( sp, interpolator_array, reference, reference_abs, count );

    // Compare before synchronize_hydro_array, which folds ghost nodes into the
    // domain and so would mix in additions the reference does not model
    const char* names[14] = { "jx", "jy", "jz", "rho", "px", "py", "pz", "ke",
                              "txx", "tyy", "tzz", "tyz", "tzx", "txy" };
    int failed = 0;
    double worst_kokkos[14] = {0}, worst_legacy[14] = {0};

    for (int i = 0; i < grid->nv; i++)
    {
        const float* kokkos = &hydro_array_kokkos->h[i].jx;
        const float* legacy = &hydro_array_legacy->h[i].jx;

        for (int m = 0; m < 14; m++)
        {
            if (count[i] == 0)
            {
                if (kokkos[m] != 0 || legacy[m] != 0)
                {
                    std::cout << " Failed at " << i << " " << names[m]
                              << ": no particles contribute, but kokkos " << kokkos[m]
                              << " legacy " << legacy[m] << std::endl;
                    failed++;
                }
                continue;
            }

            // (n-1)*u bounds the summation, a few more ulps cover the terms
            const double tol = (count[i] + 8) * 0.5*FLT_EPSILON * reference_abs[14*i + m];
            const double err_kokkos = std::abs(kokkos[m] - reference[14*i + m]);
            const double err_legacy = std::abs(legacy[m] - reference[14*i + m]);
            worst_kokkos[m] = std::max(worst_kokkos[m], err_kokkos / tol);
            worst_legacy[m] = std::max(worst_legacy[m], err_legacy / tol);

            if (err_kokkos > tol || err_legacy > tol)
            {
                std::cout << " Failed at " << i << " " << names[m] << " (" << count[i]
                          << " contributions, tolerance " << tol << "): kokkos error "
                          << err_kokkos << ", legacy error " << err_legacy << std::endl;
                failed++;
            }
        }
    }

    std::cout << "Worst error as a fraction of the tolerance (kokkos/legacy):" << std::endl;
    for (int m = 0; m < 14; m++)
        std::cout << "  " << names[m] << " " << worst_kokkos[m] << " / "
                  << worst_legacy[m] << std::endl;

    // Unregister before halt, which refuses to run with objects outstanding
    delete_hydro_array( hydro_array_legacy );
    delete_hydro_array( hydro_array_kokkos );

    if( failed )
    {  std::cout << "FAIL" << std::endl;
    }
    REQUIRE_FALSE(failed);

    std::cout << "pass" << std::endl;
}

TEST_CASE( "Kokkos and legacy hydro_p match a double-precision sum", "[hydro_p]" )
{
    vpic_simulation* simulation = new vpic_simulation;
    simulation->initialize( 0, NULL );
    simulation->finalize();
    delete simulation;
}

// Boot before Catch parses the command line, so Kokkos options such as
// --kokkos-num-threads are consumed here
int main( int argc, char* argv[] )
{
    boot_services( &argc, &argv );

    int result = Catch::Session().run( argc, argv );

    halt_services();

    return result;
}
