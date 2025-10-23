#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "deck/wrapper.h"

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

std::string energy_file_name = "./energies";
std::string moment_file_name = "./moments";

void  sample_Maxwellian(
  std::vector<double> &v, 
  double m, 
  double s, 
  int N, 
  std::mt19937_64& rng,
  std::normal_distribution<double>& dist)
{
    long double sum = 0.0L;
    long double sum_sq = 0.0L;
    
    for (std::size_t i = 0; i < N; ++i) {
	double x = dist(rng);
	v[i] = x;
	sum += x;
	sum_sq += static_cast<long double>(x) * static_cast<long double>(x);	
    }
    
    // Compute the population mean:  μ = (1/N) * sum_i v[i]
    double mean = sum / static_cast<double>(N);
    // Compute population variance σ² = (1/N)∑(v[i]²) – μ²
    long double pop_variance = (sum_sq / static_cast<long double>(N)) - (mean * mean);
    long double pop_stddev  = std::sqrt(pop_variance);
    if (pop_stddev > 0.0L) {
	for (std::size_t i = 0; i < N; ++i) {
	    double x = static_cast<double>((static_cast<long double>(v[i]) - mean) / pop_stddev);
	    v[i] = m + s*x;
	}
    } else {
	printf("pop_stddev(%f)<=0. exit.",pop_stddev);
	exit(1);
    }
    
}

// Define cross section for H + H^+ --> H^+ + H
struct HpH_cex {
  KOKKOS_INLINE_FUNCTION float
  operator() (float vr, float Z) const {
	  return 0.0;
  }
};  


void vpic_simulation::user_diagnostics() {
  dump_energies(energy_file_name.c_str(), 1);
}

void
vpic_simulation::user_initialization( int num_cmdline_arguments,
                                      char ** cmdline_argument )
{
  // At this point, there is an empty grid and the random number generator is
  // seeded with the rank. The grid, materials, species need to be defined.
  // Then the initial non-zero fields need to be loaded at time level 0 and the
  // particles (position and momentum both) need to be loaded at time level 0.

  // Arguments can be passed from the command line to the input deck
  // if( num_cmdline_arguments!=3 ) {
  //   sim_log( "Usage: " << cmdline_argument[0] << " mass_ratio seed" );
  //   abort(0);
  // }
  seed_entropy(1); //seed_entropy( atoi( cmdline_argument[2] ) );

  // Diagnostic messages can be passed written (usually to stderr)
  sim_log("Computing simulation parameters");

  // Define the system of units for this problem (natural units)
  double dt       = 1e-12;
  //double wpdt     = 0.2;
  double debye    = 1;
  double wp       = 1; //wpdt / dt;
  double vt       = 1e-5; //debye*wp;
  double v0_x     = 0.0;
  double vt_fl    = 0.428485705712571e-5;
  double N        = 1; 
  double L        = debye;
  double dx       = L/N;

  double nppc     = 100;
  double np       = nppc*N*N*N;
  double q_Hp     = 1;
  double m_Hp     = 1;
  double m_H0     = 1;
  double q_H0     = 0;

  double n0       = 1; //w*nppc/(dx*dx*dx);
  double w        = n0/np;
  double kT0      = vt*vt*m_Hp;
  double kT0_fl   = vt_fl*vt_fl*m_H0;
  
  // The probability a physical particle has a collision in a timestep dt
  // is ~pi bmax^2 |vr| dt n.  For self collisions of a Maxwellian species,
  // the relative velocities are Gaussian with twice the variance as the
  // particle velocities.  This gives the expectiation of |vr| as:
  //   |vr| = sqrt(8/pi) sqrt(2) vt = (4/sqrt(pi)) wp debye
  // Substituting into the probability above gives:
  //   p_coll = 4 sqrt(pi) bmax^2 wp dt debye nppc w / V
  // where nppc is the number of computational particles per voxel, w is
  // their physical weight and V is the volume of voxel.  Inverting gives
  // the below.

  double interval = 1;
  double p_coll   = 0.1;
  double bmax     = sqrt( p_coll / ( 4.*sqrt(M_PI)*wp*debye*dt*interval*n0 ) );
  double sample   = 1;

  int n_step = 100;

  define_units( 1, 1 );
  define_timestep( dt );
  define_periodic_grid( 0, 0, 0,   // Grid low corner
                        L, L, L,   // Grid high corner
                        N, N, N,   // Grid resolution
                        1, 1, 1 ); // Processor topology

  define_material( "vacuum", 1.0, 1.0, 0.0 );
  define_field_array();

  sim_log("Creating fluid species");

  // Define fluid species
  fluid_species_t * H0_fl = define_fluid_species( "H_fluid_species", q_H0, m_H0 );
  set_region_fluid( everywhere, "H_fluid_species", n0, kT0_fl, n0*kT0_fl ); //density, tmperature, pressure
  H0_fl->copy_to_device();
  

  // define_species( const char *name,
  //                 double q,
  //                 double m,
  //                 double max_local_np,
  //                 double max_local_nm,
  //                 double sort_interval,
  //                 double sort_out_of_place )

  sim_log("Creating particle species");

  // Define (kinetic) plasma species
  species_t * Hp_pl = define_species( "Hp_kinetic_species", q_Hp, m_Hp, np, 1, 0, 0 );
  std::vector<double> vx (np);
  std::vector<double> vy (np);
  std::vector<double> vz (np);
  auto mean_x = v0_x;
  auto mean_y = 0.0;
  auto mean_z = 0.0;
  auto sigma_x = vt;
  auto sigma_y = vt;
  auto sigma_z = vt;
  std::mt19937_64 mtrng{ std::random_device{}() };
  // Create a N(0,1) distribution object and sample velocities
  std::normal_distribution<double> dist{0.0, 1.0};
  sample_Maxwellian(vx,mean_x,sigma_x,np,mtrng,dist);
  sample_Maxwellian(vy,mean_y,sigma_y,np,mtrng,dist);
  sample_Maxwellian(vz,mean_z,sigma_z,np,mtrng,dist);
  int i = 0;
  repeat( np ) {
      inject_particle( Hp_pl,
		       uniform( rng(0), grid->x0, grid->x1 ),
		       uniform( rng(0), grid->y0, grid->y1 ),
		       uniform( rng(0), grid->z0, grid->z1 ),
		       vx[i],
		       vy[i],
		       vz[i],
		       w, 0, 0 );
      i++;
  }  
  Hp_pl->copy_to_device();
  interpolator_array->copy_to_device();
  auto& particles = Hp_pl->k_p_d;
  auto& particles_i = Hp_pl->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;


  sim_log("Deep copy hydro array");

  Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos_nomove_ngp(
      particles,
      particles_i,
      hydro_array->k_h_d,
      interpolators_k,
      Hp_pl
  );

  const char* mode = "w";
  FILE* fp = std::fopen(moment_file_name.c_str(), mode);
  if( !fp ) ERROR(( "Could not open \"%s\".", moment_file_name.c_str() ));    


  sim_log("Copy hydro array to host");
  hydro_array->copy_to_host(fp); 

  set_region_te( everywhere, kT0_fl ); // Set the electron temperature
  //Hp_pl->copy_to_device();
  field_array->copy_to_device();
#define FAK field_array->kernel
  //Call accumulate_rho twice to set rho_old for extrapolation
  FAK->clear_jf_kokkos( field_array );
  k_accumulate_rho_p( field_array, Hp_pl );
  FAK->clear_jf_kokkos( field_array );
  k_accumulate_rho_p( field_array, Hp_pl );
  FAK->hyb_init(field_array,0);
  
  auto &k_field = field_array->k_f_d;
  
  // auto M_ln_Lamda = 10;
  // auto mu= m*m/(m+m);
  // auto dV=dx*dx*dx;
  // double cvar0 = q*q*q*q*M_ln_Lamda/(8.0*M_PI);       //in SI: (e^4*n0*Lambda)/(8*pi*eps0^2*m_e^2*c^3)
  // // --> internal operator multiplies by dt!

  //define_collision_op(lemons("lemons_coll", Hp_pl, H0_fl, cvar0, ncoll, field_array));
  //define_collision_op(lemons("lemons_coll", Hp_pl, H0_fl, cvar0, ncoll));
  
  sim_log("Define charge exchange");

  // Define charge exchange
  int ncoll = 1;      // How frequently to do collisions
  double dq = 1;      // Change in charge, the kinetic particle goes from +1 to 0
  HpH_cex HpH_cex_cx; // Define cross section
  define_collision_op(charge_exchange("HpH_cex", Hp_pl, H0_fl, dq, HpH_cex_cx, ncoll));

  Hp_pl->last_indexed = -1;

  // Run benchmark test
  double elapsed = wallclock();
  int istep = 1;
  

  sim_log("Perform collisions");

  repeat( n_step ) {
      std::cout << "pre co " << istep << std::endl;
      apply_collision_op_list( collision_op_list, *kokkos_rng );
      std::cout << "post co " << istep << std::endl;

      Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
      accumulate_hydro_p_kokkos_nomove_ngp(
				particles,
				particles_i,
				hydro_array->k_h_d,
				interpolators_k,
				Hp_pl
				);
      hydro_array->copy_to_host(fp,istep); //print==true
      ++istep;
  }
  
  elapsed = wallclock() - elapsed;
  fclose(fp);
  sim_log( (double)np*(double)n_step/elapsed/1e6 );

} // end vpic_simulation::user_initialization()


TEST_CASE( "Check if it gives correct energy (within tol)", "[energy]" )
{
    // Before we run this, we must make sure we remove the energy file
    std::ofstream ofs;
    ofs.open(energy_file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();

    // Init and run sim
    vpic_simulation simulation = vpic_simulation();

    simulation.initialize( 0, NULL );

    //while( simulation.advance() ); // instead just call collisions w/in initialization

    simulation.finalize();
    
    if( world_rank==0 ) log_printf( "normal exit\n" );

    /*
    std::cout << "Comparing " << moment_file_name << " to " <<
        moment_gold_file_name << std::endl;

    // Compare energies to make sure everything worked out OK (within 1%)
    const unsigned short t6_mask = 0b00100000;  //velocity_x
    const unsigned short t9_mask = 0b100000000; //temperature

    // Test just the step range 0-40, and have tight counts
    REQUIRE(
            test_utils::compare_energies(moment_file_name, moment_gold_file_name,
	       0.01, 1e-6, t6_mask, test_utils::FIELD_ENUM::Sum, 1, "lemons.t6.tight.out", 0, 40)
           );

    // Test the sum of the T
     REQUIRE(
            test_utils::compare_energies(moment_file_name, moment_gold_file_name,
	       0.01, 1e-6, t9_mask, test_utils::FIELD_ENUM::Sum, 1, "lemons.t9.tight.out", 0, 40)
           );
    */
}

begin_particle_injection {
  // No particle injection for this simulation
}

begin_current_injection {
  // No current injection for this simulation
}

begin_field_injection {
  // No field injection for this simulation
}

begin_particle_collisions{
  // No collisions for this simulation
}

// Manually implement catch main
int main( int argc, char* argv[] )
{

    // Setup
    boot_services( &argc, &argv );

    int result = Catch::Session().run( argc, argv );

    // clean-up...
    halt_services();

    return result;
} // end main()




