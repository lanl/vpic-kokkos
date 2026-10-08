//==============================================================================
/* Numerical heating regression test, electrons only, 3D periodic box.

   A hot (k_B T_e = 1.5 MeV) uniform electron plasma in a periodic box is left
   to evolve with no driver.  Any change in the total (particle + field) energy
   is purely numerical (finite grid and finite particle count).  This test fits
   a line to the total energy and checks that the normalized growth rate

       rate = (dE/dt) / E(t=0)    [units of omega_pe]

   is within a tolerance of a reference value measured from many seeds.

   Only the total energy is checked.  The fields start at zero, so at early
   times there is a net flow of energy from the electrons into the fields,
   which makes the electron kinetic energy alone nonlinear in time until the
   two reach equilibrium.  That flow does not change the total, so the total
   can be fit from t = 0 with no transient to discard.

   This is a regression test, not a physics test: VPIC's low-order particle
   shape heats at some rate, and we check the rate has not changed.  The grid
   is deliberately coarse (dx/lambda_D of order 1.5) and the plasma is hot, so
   the heating is linear in time from the start.  This is not a regime you want
   to run in when correct physics matters.

   This source builds two tests:

   grid_heating_symmetric
       A 6x6x6 box of cubic cells, dx = 2.5 c/omega_pe, at the origin, with
       64 particles per cell of an ordinary electron (q = -1, m = 1).

   grid_heating_asymmetric (GRID_HEATING_ASYMMETRIC defined)
       Every simplifying symmetry is broken, to catch code that relies on it:
       - 7x5x6 cells of size 2.5 x 2.9 x 2.1 c/omega_pe, so the box is
         17.5 x 14.5 x 12.6.  Counts, sizes and lengths differ in every
         dimension and are ordered differently, so assuming nx = ny = nz or
         dx = dy = dz, or swapping dimensions, changes the result.  nx and ny
         are odd.  The 210 cells are close to the 216 of the symmetric test,
         so the two cost about the same.
       - The box does not start at the origin.
       - 63 particles per cell, so the particle count is not a round number
         and exercises remainder handling in chunked loops.
       - q = -2, m = 4, and T is 4 times higher.  q^2/m = 1 keeps omega_pe,
         and T/m unchanged keeps the normalized momentum, velocity and Debye
         length, so the physics is the same in normalized units.  Code that
         assumes |q| = 1 or m = 1 changes the result.
       - The particles are sorted every 20 steps.

   Ported from test/unit/grid_heating in the original VPIC.  That test used a
   2D 10x10 box at 200 keV with dx = 0.5 c/omega_pe for 27k steps, and fit the
   electron energy only after step 10000.

   The reference values are the mean and sample standard deviation of 100
   seeds on each platform, measured on Darwin in October 2026: skylake-gold
   (Xeon Gold 6152, OpenMP), amd-rome (EPYC 7702, OpenMP), volta-x86 (V100,
   CUDA) and shared-grace-hopper (GH200, CUDA).  The platform means agree
   within 1.2 standard errors.  (Plausibly different platforms should heat
   differently as a result of different backend algorithms.)  For the symmetric
   test amd-rome is left out of the pool, because it is bitwise identical to
   skylake-gold for every seed.  The asymmetric test is not reproducible run to
   run on CPU even with a fixed seed (the threaded particle sort is
   nondeterministic), so all four platforms are independent there.

       symmetric   1.1962e-4 +/- 2.39e-6 (2.00%), 300 runs, worst 3.4 std
       asymmetric  1.1065e-4 +/- 2.58e-6 (2.33%), 400 runs, worst 3.2 std

   The rates are close to normally distributed.  The test fails beyond 5 std
   from the reference and warns (but passes) beyond 3 std.

   For measuring reference values, GRID_HEATING_SEED overrides the seed and
   GRID_HEATING_HISTORY names a file to write the energy history to.
*/
//==============================================================================

#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include "deck/wrapper.h"

begin_globals {
    int energies_interval;
};

// Energy history recorded by user_diagnostics, in code units.  The energy
// functions sum over all ranks, so every rank holds the global history.
static std::vector<double> rec_t, rec_total;

begin_diagnostics {

    if( step() % global->energies_interval != 0 ) return;

    double en_f[6];
    field_array->kernel->energy_f_kokkos( en_f, field_array );
    double ke = energy_p_kokkos( find_species("electron"), interpolator_array );

    rec_t.push_back( step()*grid->dt );
    rec_total.push_back( ke + en_f[0] + en_f[1] + en_f[2]
                            + en_f[3] + en_f[4] + en_f[5] );
}

begin_initialization {

    // Code units: c = eps0 = m_e = |e| = omega_pe = 1, lengths in c/omega_pe
    double cfl_req = 0.98;              // Fraction of the Courant limit

    // Physical constants, SI
    const double e_SI   = 1.602176634e-19;   // C
    const double c_SI   = 2.99792458e8;      // m / s
    const double m_e_SI = 9.1093837015e-31;  // kg
    const double mec2   = m_e_SI*c_SI*c_SI;  // J

#ifndef GRID_HEATING_ASYMMETRIC
    double nx = 6,   ny = 6,   nz = 6;
    double hx = 2.5, hy = 2.5, hz = 2.5;
    double x0 = 0,   y0 = 0,   z0 = 0;      // Low corner
    double nppc = 64;
    double q_sp = -1, m_sp = 1;             // Species charge and mass
    int sort_interval = 0;
#else
    double nx = 7,    ny = 5,    nz = 6;
    double hx = 2.5,  hy = 2.9,  hz = 2.1;
    double x0 = 13.7, y0 = -5.1, z0 = 42.0;
    double nppc = 63;
    double q_sp = -2, m_sp = 4;             // q^2/m = 1, so omega_pe = 1
    int sort_interval = 20;
#endif

    double Lx = nx*hx, Ly = ny*hy, Lz = nz*hz;

    // Domain decomposition.  Only 1x1x1 is tested so far.
    int topology_x = 1;
    int topology_y = 1;
    int topology_z = 1;

    // Temperature.  VPIC momenta are normalized to the species m c.  The
    // temperature scales with m_sp so that T/(m c^2), and with it the
    // normalized momentum, velocity and Debye length, match the symmetric test.
    double T_e  = 1.5e6 * e_SI * m_sp;  // Technically k_B*T, in J
    double E_e  = 1.5 * T_e;            // 1/2 k_B T per dimension
    double mc2  = m_sp*mec2;
    double px_e = sqrt(1./3.)*sqrt(E_e*E_e + 2.*E_e*mc2) / mc2; // Normalized

    double dt     = cfl_req*courant_length( Lx, Ly, Lz, nx, ny, nz );
    double t_stop = 3000;               // In 1/omega_pe

    double Ne = nppc*nx*ny*nz;          // Macro electrons
    double we = Lx*Ly*Lz/Ne;            // Weight per macro electron (n_e = 1)

    int seed = 2995471;
    if( const char * s = getenv("GRID_HEATING_SEED") ) seed = atoi(s);
    seed_entropy( seed );

    num_step             = int(t_stop/dt);
    status_interval      = 0;
    sync_shared_interval = 0;
    clean_div_e_interval = 0;
    clean_div_b_interval = 0;
    global->energies_interval = 10;

    // lambda_D^2 = eps0 T / (n q^2) in code units, with T in units of m_e c^2
    // (q^2 = m keeps it the same in both tests)
    double lambda_D = sqrt( (T_e/mec2) / (q_sp*q_sp) );
    sim_log( "seed = " << seed );
    sim_log( "dt = " << dt << ", num_step = " << num_step );
    sim_log( "px_e = " << px_e << ", (dx, dy, dz)/lambda_D = " << hx/lambda_D
             << ", " << hy/lambda_D << ", " << hz/lambda_D );
    sim_log( "Ne = " << Ne );

    define_units( 1, 1 );
    define_timestep( dt );
    define_periodic_grid( x0,    y0,    z0,                  // Low corner
                          x0+Lx, y0+Ly, z0+Lz,               // High corner
                          nx,    ny,    nz,                  // Resolution
                          topology_x, topology_y, topology_z ); // Topology

    define_material( "vacuum", 1 );
    define_field_array( NULL, 0 );

    double nproc_grid = topology_x*topology_y*topology_z;
    species_t * electron = define_species( "electron", q_sp, m_sp,
                                           1.3*Ne/nproc_grid, -1,
                                           sort_interval, 1 );

    double xmin = grid->x0, xmax = grid->x0 + grid->nx*grid->dx;
    double ymin = grid->y0, ymax = grid->y0 + grid->ny*grid->dy;
    double zmin = grid->z0, zmax = grid->z0 + grid->nz*grid->dz;

    repeat( Ne/nproc_grid ) {
        double x = uniform( rng(0), xmin, xmax );
        double y = uniform( rng(0), ymin, ymax );
        double z = uniform( rng(0), zmin, zmax );
        inject_particle( electron, x, y, z,
                         normal( rng(0), 0, px_e ),
                         normal( rng(0), 0, px_e ),
                         normal( rng(0), 0, px_e ), we, 0, 0 );
    }
}

// Least-squares slope of y(t)
static double fit_slope( const std::vector<double> & t,
                         const std::vector<double> & y )
{
    double n = 0, st = 0, sy = 0, stt = 0, sty = 0;
    for( size_t i = 0; i < t.size(); i++ ) {
        n += 1; st += t[i]; sy += y[i]; stt += t[i]*t[i]; sty += t[i]*y[i];
    }
    return ( n*sty - st*sy ) / ( n*stt - st*st );
}

// Fail beyond nstd_fail standard deviations from the reference, and warn (but
// pass) beyond nstd_warn
static const double nstd_fail = 5;
static const double nstd_warn = 3;

static void check_rate( const char * name, double rate, double ref, double std )
{
    double nsig = std::abs( rate - ref ) / std;

    // Two-sided probability of a normal deviate at least nsig std from the
    // mean.  The measured rates are close to normally distributed.
    double p = std::erfc( nsig/std::sqrt(2.) );

    if( world_rank==0 ) {
        std::ostringstream one_in;
        if( p > 0 ) one_in << std::fixed << std::setprecision(0) << 1/p;
        else        one_in << std::numeric_limits<double>::infinity();

        std::cout << name << " = " << rate << " omega_pe, compared to reference"
            << " = " << ref << " +/- " << std << std::endl;
        std::cout << "This is " << nsig << " standard deviations away from the "
            << "precomputed mean.  A result this far away is expected in "
            << "approximately 1 in every " << one_in.str() << " runs."
            << std::endl;
    }

    if( nsig >= nstd_warn && nsig < nstd_fail )
        WARN( name << " is " << nsig << " std from the reference (warn at "
              << nstd_warn << ", fail at " << nstd_fail << "). The physics "
              "might have been subtly changed (not necessarily incorrectly), "
              "or you were unlucky with the random seed. The physics is not "
              "drastically different." );

    CHECK( nsig < nstd_fail );
}

TEST_CASE( "Total energy numerical heating rate matches reference",
           "[heating]" )
{
    vpic_simulation simulation = vpic_simulation();
    simulation.initialize( 0, NULL );

    auto start = std::chrono::steady_clock::now();
    while( simulation.advance() );
    std::chrono::duration<double> wall =
        std::chrono::steady_clock::now() - start;

    simulation.finalize();

    if( world_rank==0 ) {
        std::cout << "Advance loop wall time: " << wall.count() << " s"
                  << std::endl;

        if( const char * h = getenv("GRID_HEATING_HISTORY") ) {
            std::ofstream out( h );
            out.precision( 10 );
            out << "# t total_energy" << std::endl;
            for( size_t i = 0; i < rec_t.size(); i++ )
                out << rec_t[i] << " " << rec_total[i] << std::endl;
        }
    }

    double total_rate = fit_slope( rec_t, rec_total ) / rec_total.front();

    // Reference from many seeds; see the header comment
#ifndef GRID_HEATING_ASYMMETRIC
    const double ref_total_rate = 1.1962e-4;
    const double ref_total_std  = 2.39e-6;
#else
    const double ref_total_rate = 1.1065e-4;
    const double ref_total_std  = 2.58e-6;
#endif

    if( world_rank==0 ) {
        std::cout.precision( 6 );
        std::cout << "E(0) = " << rec_total.front() << ", final E/E(0) = "
                  << rec_total.back()/rec_total.front() << std::endl;
    }

    check_rate( "Normalized total energy growth rate", total_rate,
                ref_total_rate, ref_total_std );
}

begin_particle_injection {
}

begin_current_injection {
}

begin_field_injection {
}

begin_particle_collisions {
}

int main( int argc, char* argv[] )
{
    boot_services( &argc, &argv );

    int result = Catch::Session().run( argc, argv );

    halt_services();

    return result;
}
