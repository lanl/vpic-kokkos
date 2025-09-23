//#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#define CATCH_CONFIG_RUNNER // We will provide a custom main
#include "catch.hpp"

// TODO: this import may ultimately be a bad idea, but it lets you paste an input deck in...

#include "deck/wrapper.h"

#include "src/species_advance/species_advance.h"
#include "src/vpic/vpic.h"

#include "compare_energies.h"


int pick_q(std::int64_t i) {
    // map i to [0,99] even if i is negative
    int r = static_cast<int>((i % 100 + 100) % 100);

    // q=0 :  5%
    // q=1 : 38%  (5+38 = 43)
    // q=2 : 10%  (43+10 = 53)
    // q=3 : 18%  (53+18 = 71)
    // q=4 : 29%  (71+29 = 100)
    if (r < 5)        return 0;
    else if (r < 43)  return 1;
    else if (r < 53)  return 2;
    else if (r < 71)  return 3;
    else              return 4;
    // if (r < 40)       return 1;
    // else if (r < 50)  return 2;
    // else if (r < 70)  return 3;
    // else              return 4;
    
}

double rand_uniform(double a, double b)
{
    double r = (double)rand() / (double)RAND_MAX;
    return a + (b - a) * r;
}

// Target PDF: Anisotropic 3D Gaussian with drift (mean) and diagonal sigmas.
// f(vx,vy,vz) =  1/[(2pi)^(3/2)*σ_x σ_y σ_z] *
//              exp{ -[(vx-μ_x)²/(2σ_x²) + (vy-μ_y)²/(2σ_y²) + (vz-μ_z)²/(2σ_z²)] }
double anisotropic_gauss_pdf_drift(double vx, double vy, double vz,
                                   double mu_x, double mu_y, double mu_z,
                                   double sx, double sy, double sz)
{
    double norm = 1.0 / ( pow(2.0 * M_PI, 1.5) * sx * sy * sz );
    double exponent = ((vx - mu_x)*(vx - mu_x))/(2.0*sx*sx)
                    + ((vy - mu_y)*(vy - mu_y))/(2.0*sy*sy)
                    + ((vz - mu_z)*(vz - mu_z))/(2.0*sz*sz);
    return norm * exp(-exponent);
}

// This function samples N particles (vx, vy, vz) from an anisotropic Gaussian
// with drift (mean) {mu_x, mu_y, mu_z} and diagonal sigmas (sx, sy, sz)
// using a piecewise-constant proposal on a 20x20x20 grid.  Then it applies
// a moment correction so that the weighted sample moments exactly match
// the desired drift and variances. Finally, it produces a diagnostic 1D histogram
// of the particle weights written to "w_hist.dat".
void vw_sample_Maxwellian_drift(double *vx, double *vy, double *vz, double *w,
                                double mu_x, double mu_y, double mu_z,
                                double sx, double sy, double sz, int N)
{
    // printf("mu_x=%f\n",mu_x);
    // Grid resolution
    const int Nx = 30, Ny = 30, Nz = 30;
    
    // Define bounding boxes for each direction (shifted by drift)
    double vmin_x = mu_x - 5.0 * sx;
    double vmax_x = mu_x + 5.0 * sx;
    double vmin_y = mu_y - 5.0 * sy;
    double vmax_y = mu_y + 5.0 * sy;
    double vmin_z = mu_z - 5.0 * sz;
    double vmax_z = mu_z + 5.0 * sz;
    
    // Cell sizes and volume
    double dx = (vmax_x - vmin_x) / Nx;
    double dy = (vmax_y - vmin_y) / Ny;
    double dz = (vmax_z - vmin_z) / Nz;
    double cellVolume = dx * dy * dz;
    
    // Build the piecewise-constant proposal.
    // Use order: [Nx][Ny][Nz] (i for x, j for y, k for z)
    static double cellMass[Nx][Ny][Nz];
    double totalMass = 0.0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                // Lower edges of cell
                double vx_lo = vmin_x + i*dx;
                double vy_lo = vmin_y + j*dy;
                double vz_lo = vmin_z + k*dz;
                // Midpoints in cell
                double vx_mid = vx_lo + 0.5 * dx;
                double vy_mid = vy_lo + 0.5 * dy;
                double vz_mid = vz_lo + 0.5 * dz;
                // Evaluate target PDF at the midpoint (with drift)
                double fval = anisotropic_gauss_pdf_drift(vx_mid, vy_mid, vz_mid,
                                                          mu_x, mu_y, mu_z, sx, sy, sz);
                double mass = fval * cellVolume;
                cellMass[i][j][k] = mass;
                totalMass += mass;
            }
        }
    }
    // Normalize cell masses so that the total probability is 1.
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                cellMass[i][j][k] /= totalMass;
            }
        }
    }
    
    // Build a cumulative distribution function (CDF) over the Nx*Ny*Nz cells.
    int Ncells = Nx * Ny * Nz;
    double *cdf = (double *)malloc(Ncells * sizeof(double));
    if (!cdf) {
        fprintf(stderr, "Memory allocation error (cdf).\n");
        exit(1);
    }
    double cum = 0.0;
    int idx = 0;
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                cum += cellMass[i][j][k];
                cdf[idx++] = cum;
            }
        }
    }
    
    // Generate N samples from the proposal.
    double sumW = 0.0;
    for (int n = 0; n < N; n++) {
        // Pick a cell index using the CDF.
        double r = rand_uniform(0.0, 1.0);
        int chosen_idx = 0;
        for (int m = 0; m < Ncells; m++) {
            if (r <= cdf[m]) {
                chosen_idx = m;
                break;
            }
        }
        // Decode chosen_idx into (i, j, k)
        int i = chosen_idx / (Ny * Nz);
        int rem = chosen_idx % (Ny * Nz);
        int j = rem / Nz;
        int k = rem % Nz;
        // Sample uniformly within the chosen cell.
        double vx_lo = vmin_x + i*dx;
        double vy_lo = vmin_y + j*dy;
        double vz_lo = vmin_z + k*dz;
        double vx_prop = rand_uniform(vx_lo, vx_lo + dx);
        double vy_prop = rand_uniform(vy_lo, vy_lo + dy);
        double vz_prop = rand_uniform(vz_lo, vz_lo + dz);
        vx[n] = vx_prop;
        vy[n] = vy_prop;
        vz[n] = vz_prop;
        // The proposal density in the cell is constant:
        // q = (cellMass[i][j][k]) / (cellVolume).
        double cell_pmass = cellMass[i][j][k];
        double q_cell = cell_pmass / cellVolume;
        // Compute importance weight = f / q (using the drift PDF).
        double f_val = anisotropic_gauss_pdf_drift(vx_prop, vy_prop, vz_prop,
                                                   mu_x, mu_y, mu_z, sx, sy, sz);
        double wt = f_val / q_cell;
        w[n] = wt;
        sumW += wt;
    }
    // Normalize weights so that sum(w) = 1.
    for (int n = 0; n < N; n++) {
        w[n] /= sumW;
    }
    free(cdf);
    
    // Correction of sample moments:
    // Compute weighted means and variances then adjust the samples so that the
    // weighted mean becomes exactly the drift and the weighted variance becomes exactly
    // sx^2, sy^2, and sz^2.
    double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;
    for (int n = 0; n < N; n++) {
        mean_x += w[n] * vx[n];
        mean_y += w[n] * vy[n];
        mean_z += w[n] * vz[n];
    }
    double var_x = 0.0, var_y = 0.0, var_z = 0.0;
    for (int n = 0; n < N; n++) {
        var_x += w[n] * (vx[n] - mean_x) * (vx[n] - mean_x);
        var_y += w[n] * (vy[n] - mean_y) * (vy[n] - mean_y);
        var_z += w[n] * (vz[n] - mean_z) * (vz[n] - mean_z);
    }
    double s_x_sample = sqrt(var_x);
    double s_y_sample = sqrt(var_y);
    double s_z_sample = sqrt(var_z);
    // Adjust each particle so that the weighted moments equal the target.
    for (int n = 0; n < N; n++) {
        vx[n] = (vx[n] - mean_x) * (sx / s_x_sample) + mu_x;
        vy[n] = (vy[n] - mean_y) * (sy / s_y_sample) + mu_y;
        vz[n] = (vz[n] - mean_z) * (sz / s_z_sample) + mu_z;
    }
    
    // Histogram diagnostics for particle weight.
    // We produce a 1D histogram of the normalized weight values using NBINS = 50.
    const int NBINS_w = 50;
    double w_min = w[0], w_max = w[0];
    sumW = 0;
    for (int n = 0; n < N; n++) {
        if (w[n] < w_min) w_min = w[n];
        if (w[n] > w_max) w_max = w[n];
	sumW += w[n];
    }
    //printf("total weight=%f\n",sumW);

    double dw = (w_max - w_min) / NBINS_w;
    double *hist_w = (double *) calloc(NBINS_w, sizeof(double));
    if (!hist_w) {
        fprintf(stderr, "Memory allocation error for weight histogram.\n");
        exit(1);
    }
    for (int n = 0; n < N; n++) {
        int bin = (int)((w[n] - w_min) / dw);
        if (bin >= NBINS_w)
            bin = NBINS_w - 1;
        hist_w[bin] += 1.0;
    }
    // Normalize histogram so that the area under it is 1.
    for (int b = 0; b < NBINS_w; b++) {
        hist_w[b] /= (N * dw);
    }

    /* persistent “have-we-written-before?” flag */
    static int first_call = 1;        /* 1 = yes, this is the first call */

    FILE *fpw;

    if (first_call){
	fpw = fopen("w_hist.dat", "w");
	if (!fpw) {
	    fprintf(stderr, "Failed to open w_hist.dat for writing.\n");
	    exit(1);
	}	
    }else{
	fpw = fopen("w_hist.dat", "a");   /* append */
        if (!fpw) { perror("w_hist.dat"); exit(1); }
    }
    
    for (int b = 0; b < NBINS_w; b++) {
        double center = w_min + (b + 0.5) * dw;
        fprintf(fpw, "%g %g\n", center, hist_w[b]);
    }
    fprintf(fpw,"\n\n");
    fclose(fpw);
    free(hist_w);

/* -----------------------------------------------------------------
 * 9)  Histogram of vx (weighted) + header with target parameters
 * ----------------------------------------------------------------- */
{
    const int NBINS_VX = 100;                 /* resolution of the histogram */
    double vx_min = vmin_x;                   /* same bounds used for sampling */
    double vx_max = vmax_x;
    double dvx    = (vx_max - vx_min) / NBINS_VX;

    double *hist_vx = (double *)calloc(NBINS_VX, sizeof(double));
    if (!hist_vx) {
        fprintf(stderr, "Memory allocation error for vx histogram.\n");
        exit(1);
    }

    /* accumulate weighted counts */
    for (int n = 0; n < N; n++) {
        int b = (int)((vx[n] - vx_min) / dvx);
        if (b < 0)                b = 0;
        if (b >= NBINS_VX)        b = NBINS_VX - 1;
        hist_vx[b] += w[n];                   /* ***weight***, not raw count */
    }

    /* convert to PDF estimate: divide by bin width */
    for (int b = 0; b < NBINS_VX; b++)
        hist_vx[b] /= dvx;

    /* write to file with a header block (comment lines begin with '#') */
    FILE *fp;
    if (first_call){
	fp = fopen("vx_hist.dat", "w");
	if (!fp) {
	    fprintf(stderr, "Cannot open vx_hist.dat for writing.\n");
	    exit(1);
	}
	/* --- header with target parameters --- */
	fprintf(fp, "# Drift-Maxwellian parameters\n");
	fprintf(fp, "# mean_x %g\n",  mu_x);
	fprintf(fp, "# mean_y %g\n",  mu_y);
	fprintf(fp, "# mean_z %g\n",  mu_z);
	fprintf(fp, "# sigma_x %g\n", sx);
	fprintf(fp, "# sigma_y %g\n", sy);
	fprintf(fp, "# sigma_z %g\n", sz);
	fprintf(fp, "# NBINS %d  dvx %g\n", NBINS_VX, dvx);
	fprintf(fp, "# columns:  bin_center   pdf_estimate\n");
    }else{
	fp = fopen("vx_hist.dat", "a");   /* append */
        if (!fp) { perror("vx_hist.dat"); exit(1); }	
    }	
    /* --- data section --- */
    for (int b = 0; b < NBINS_VX; b++) {
        double center = vx_min + (b + 0.5) * dvx;
        fprintf(fp, "%g  %g\n", center, hist_vx[b]);
    }
    fprintf(fp,"\n\n");
    fclose(fp);
    free(hist_vx);
    first_call = 0;                   /* subsequent calls will append */
}
    
}

struct IonParams {
  const char* name;  // e.g., "ion", "ion2", "C+", ...
  double n;          // physical number density (like ni, n2)
  double nppc;       // computational particles per cell used to build np
  double q;          // charge
  double m;          // mass
  double vt;         // thermal speed (stddev)
  double v0;         // drift in x (mean)
};

begin_globals {
  double energies_interval;
  double fields_interval;
  double ehydro_interval;
  double ihydro_interval;
  double eparticle_interval;
  double iparticle_interval;
  double restart_interval;
};

std::string energy_file_name = "./energies";
std::string moment_file_name = "./moments";
std::string moment_gold_file_name = GOLD_ENERGY_FILE;

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
  sim_log( "Computing simulation parameters");

  // Define the system of units for this problem (natural units)
  double dt       = 1e-15;
  //double wpdt     = 0.2;
  double debye    = 1;
  double wp       = 1; //wpdt / dt;
  double vt       = 1e-5; //debye*wp;
  double N        = 1; 
  double L        = debye;
  double dx       = L/N;

  int n_step = 5000;

  define_units( 1, 1 );
  define_timestep( dt );
  define_periodic_grid( 0, 0, 0,   // Grid low corner
                        L, L, L,   // Grid high corner
                        N, N, N,   // Grid resolution
                        1, 1, 1 ); // Processor topology

  define_material( "vacuum", 1.0, 1.0, 0.0 );
  define_field_array();

  double nppc     = 1000;
  double np       = nppc*N*N*N;
  double q        = 1;
  double m        = 1;


  double n0       = 1; //w*nppc/(dx*dx*dx);
  float  w0        = n0/np;
  double kT0      = vt*vt*m;

  double ni       = 1.0;
  double npi      = ni*nppc*N*N*N;
  double wi       = ni/npi;
  double qi       = 1.0;
  double mi       = 1.0;
  double vti      = 8e-5;
  double kTi      = vti*vti*mi;
  double v0i      = 10e-5;
  
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
  //double p_coll   = 0.1;
  //double bmax     = sqrt( p_coll / ( 4.*sqrt(M_PI)*wp*debye*dt*interval*n0 ) );
  double sample   = 1;

  
  species_t * sp = define_species( "test_species", q, m, np, 1, 0, 0 );
  double *vx = (double *) malloc(np * sizeof(double));
  double *vy = (double *) malloc(np * sizeof(double));
  double *vz = (double *) malloc(np * sizeof(double));  
  double *wt = (double *) malloc(np * sizeof(double));  // weights
  auto mean_x = 0.0;
  auto mean_y = 0.0;
  auto mean_z = 0.0;
  auto sigma_x = vt;
  auto sigma_y = vt;
  auto sigma_z = vt;

  vw_sample_Maxwellian_drift(vx,vy,vz,wt, mean_x, mean_y, mean_z, sigma_x, sigma_y, sigma_z, np);

  int i = 0;
  repeat( np ) {  
#ifdef VARIABLE_CHARGE
      inject_particle( sp,
		       uniform( rng(0), grid->x0, grid->x1 ),
		       uniform( rng(0), grid->y0, grid->y1 ),
		       uniform( rng(0), grid->z0, grid->z1 ),
		       vx[i],
		       vy[i],
		       vz[i],
		       wt[i], 0, 0, q );
#else
      printf("Err: This test is for variable charge. VARIABLE_CHARGE macro is not defined. Quit.\n");
      exit(1);
#endif      
      i++;
  }
  free(vx);
  free(vy);
  free(vz);
  free(wt);
  
  sp->copy_to_device();

  //define ion species
    std::vector<IonParams> ion_cfg = {
       {"ion0", 0.05, nppc, 0.0, 1.0, 8e-5, 10e-5}	      
      ,{"ion1", 0.38, nppc, 1.0, 1.0, 8e-5, 10e-5}
      ,{"ion2", 0.1 , nppc, 2.0, 1.0, 8e-5, 10e-5}
      ,{"ion3", 0.18, nppc, 3.0, 1.0, 8e-5, 10e-5}
      ,{"ion4", 0.29, nppc, 4.0, 1.0, 8e-5, 10e-5}
  };

  // ---- 2) Build ion species, hydro arrays, and inject particles ----
  species_t * ion      = define_species( "ion",    qi, mi,npi, 1, 0, 0 );
  hydro_array_t        * hydro_array_ion;        // define_hydro_array  for ion species
  hydro_array_ion        = new_hydro_array( grid );
  double wsum=0;
  const double N3 = N*N*N;
  
      // ---- variable-weight injection with Maxwellian + drift ----
      vx = (double*) malloc((size_t)npi * sizeof(double));
      vy = (double*) malloc((size_t)npi * sizeof(double));
      vz = (double*) malloc((size_t)npi * sizeof(double));
      wt = (double*) malloc((size_t)npi * sizeof(double));

      sigma_x = vti, sigma_y = vti, sigma_z = vti;
      mean_x  = v0i, mean_y  = 0.0,  mean_z = 0.0;

      vw_sample_Maxwellian_drift(vx,vy,vz,wt, mean_x,mean_y,mean_z, sigma_x,sigma_y,sigma_z, (int)npi);

  i = 0;
  for (const auto& s : ion_cfg) {
      const double np_i  = s.nppc * s.n * N3;             // computational particle count
      const double w_i   = s.n / np_i;              // weight per particle (fixed-wt path)
      double qt = s.q;
      int il = 0;
      
      repeat( (int)np_i ) {
	  qt = pick_q(i);
	  inject_particle( ion,
			   uniform(rng(0), grid->x0, grid->x1),
			   uniform(rng(0), grid->y0, grid->y1),
			   uniform(rng(0), grid->z0, grid->z1),
			   vx[i], vy[i], vz[i],
			   wt[i]*ni, 0, 0
			   ,qt
			   );
	  if(qt!=0) wsum += wt[i]*ni;
	  //if(il<10) printf("il=%d,i=%d,vxyz=%e,%e,%e,qt=%e, w=%e\n",il,i,vx[il], vy[il], vz[il],qt,wt[i]*ni);
	  ++il;
	  ++i;
      }
  }
  free(vx); free(vy); free(vz); free(wt);
  printf("#i=%d, wsum=%e\n",i,wsum);
  const double ni_neq0 = wsum;
  //exit(1);
  ion->copy_to_device();

  
  interpolator_array->copy_to_device();
  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;

  Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos_nomove_ngp(
      particles,
      particles_i,
      hydro_array->k_h_d,
      interpolators_k,
      sp
  );

  const char* mode = "w";
  FILE* fp = std::fopen(moment_file_name.c_str(), mode);
  if( !fp ) ERROR(( "Could not open \"%s\".", moment_file_name.c_str() ));
  // if( world_rank==0 && fp) {	
  //     std::fprintf(fp, "%% Layout\n");
  //     std::fprintf(fp, "%% step Tx Ty Tz vx vy vz cell_index \n");
  // }
    

  hydro_array->copy_to_host();

  float ve1 = (hydro_array->h[13]).px/m;
  float ve2 = (hydro_array->h[13]).py/m;
  float ve3 = (hydro_array->h[13]).pz/m;
  float ke2 = (hydro_array->h[13]).txx + (hydro_array->h[13]).tyy + (hydro_array->h[13]).tzz;
  float Te = (ke2 - m*(ve1*ve1 + ve2*ve2 + ve3*ve3))/3.0;

  auto& ions = ion->k_p_d;
  auto& ions_i = ion->k_p_i_d;

  Kokkos::deep_copy(hydro_array_ion->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos_nomove_ngp(
      ions,
      ions_i,
      hydro_array_ion->k_h_d,
      interpolators_k,
      ion
  );
  hydro_array_ion->copy_to_host();  
  float vi1 = (hydro_array_ion->h[13]).px/(ni_neq0*mi);
  float vi2 = (hydro_array_ion->h[13]).py/(ni_neq0*mi);
  float vi3 = (hydro_array_ion->h[13]).pz/(ni_neq0*mi);
  float ki2 = ((hydro_array_ion->h[13]).txx + (hydro_array_ion->h[13]).tyy + (hydro_array_ion->h[13]).tzz)/ni_neq0;
  float Ti = (ki2 - mi*(vi1*vi1 + vi2*vi2 + vi3*vi3))/3.0;
  float tot_momentum1_0 = n0*m*ve1 + ni_neq0*mi*vi1;
  float tot_momentum2_0 = n0*m*ve2 + ni_neq0*mi*vi2;
  float tot_momentum3_0 = n0*m*ve3 + ni_neq0*mi*vi3;
  float tot_en_0 = ke2*n0 + ki2*ni_neq0;
  float dmom1 = 0;
  float dmom2 = 0;
  float dmom3 = 0;
  float den   = 0;
  fprintf(fp,"0 %.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n",ve1,ve2,ve3,Te, vi1,vi2,vi3,Ti,dmom1, dmom2, dmom3, den);  
  
  auto M_ln_Lamda = 10;

  auto dV=dx*dx*dx;
  double cvar0 = q*q*q*q*M_ln_Lamda/(8.0*M_PI);       //in SI: (e^4*n0*Lambda)/(8*pi*eps0^2*m_e^2*c^3)
  // --> internal operator multiplies by dt!
  int sort_interval = 1;  
  int ncoll = (int) sort_interval;  // How frequently to do collisions
  bool var_wt = true;
  define_collision_op(takizuka_abe("ta_coll", sp, ion, cvar0, ncoll, var_wt));
  
  sp->last_indexed = -1;
  ion->last_indexed = -1;  
  // Do the benchmark
  double elapsed = wallclock();
  int istep = 1;
  
  repeat( n_step ) {
      apply_collision_op_list( collision_op_list, *kokkos_rng );

      Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
      accumulate_hydro_p_kokkos_nomove_ngp(
				particles,
				particles_i,
				hydro_array->k_h_d,
				interpolators_k,
				sp
				);
      hydro_array->copy_to_host();
  float ve1 = (hydro_array->h[13]).px/m;
  float ve2 = (hydro_array->h[13]).py/m;
  float ve3 = (hydro_array->h[13]).pz/m;
  float ke2 = (hydro_array->h[13]).txx + (hydro_array->h[13]).tyy + (hydro_array->h[13]).tzz;
  float Te = (ke2 - m*(ve1*ve1 + ve2*ve2 + ve3*ve3))/3.0;

  Kokkos::deep_copy(hydro_array_ion->k_h_d, 0.0f);
  accumulate_hydro_p_kokkos_nomove_ngp(
      ions,
      ions_i,
      hydro_array_ion->k_h_d,
      interpolators_k,
      ion
  );
  hydro_array_ion->copy_to_host();  
  float vi1 = (hydro_array_ion->h[13]).px/(ni_neq0*mi);
  float vi2 = (hydro_array_ion->h[13]).py/(ni_neq0*mi);
  float vi3 = (hydro_array_ion->h[13]).pz/(ni_neq0*mi);
  float ki2 = ((hydro_array_ion->h[13]).txx + (hydro_array_ion->h[13]).tyy + (hydro_array_ion->h[13]).tzz)/ni_neq0;
  float Ti = (ki2 - mi*(vi1*vi1 + vi2*vi2 + vi3*vi3))/3.0;

  float tot_momentum1 = n0*m*ve1 + ni_neq0*mi*vi1;
  float tot_momentum2 = n0*m*ve2 + ni_neq0*mi*vi2;
  float tot_momentum3 = n0*m*ve3 + ni_neq0*mi*vi3;
  float tot_en = ke2*n0 + ki2*ni_neq0;
  dmom1 = tot_momentum1 - tot_momentum1_0;
  dmom2 = tot_momentum2 - tot_momentum2_0;
  dmom3 = tot_momentum3 - tot_momentum3_0;
  den   = tot_en - tot_en_0;

  fprintf(fp,"%d %.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\t%.15e\n",istep, ve1,ve2,ve3,Te, vi1,vi2,vi3,Ti,dmom1,dmom2,dmom3,den);
      
      ++istep;
  }
  elapsed = wallclock() - elapsed;
  fclose(fp);
  sim_log( (double)np*(double)n_step/elapsed/1e6 );

  //exit(1);

  // Upon completion of the initialization, the following occurs:
  // - The synchronization error (tang E, norm B) is computed between domains
  //   and tang E / norm B are synchronized by averaging where discrepancies
  //   are encountered.
  // - The initial divergence error of the magnetic field is computed and
  //   one pass of cleaning is done (for good measure)
  // - The bound charge density necessary to give the simulation an initially
  //   clean divergence e is computed.
  // - The particle momentum is uncentered from u_0 to u_{-1/2}
  // - The user diagnostics are called on the initial state
  // - The physics loop is started
  //
  // The physics loop consists of:
  // - Advance particles from x_0,u_{-1/2} to x_1,u_{1/2}
  // - User particle injection at x_{1-age}, u_{1/2} (use inject_particles)
  // - User current injection (adjust field(x,y,z).jfx, jfy, jfz)
  // - Advance B from B_0 to B_{1/2}
  // - Advance E from E_0 to E_1
  // - User field injection to E_1 (adjust field(x,y,z).ex,ey,ez,cbx,cby,cbz)
  // - Advance B from B_{1/2} to B_1
  // - (periodically) Divergence clean electric field
  // - (periodically) Divergence clean magnetic field
  // - (periodically) Synchronize shared tang e and norm b
  // - Increment the time step
  // - Call user diagnostics
  // - (periodically) Print a status message
}

TEST_CASE( "Check if Weibel gives correct energy (within tol)", "[energy]" )
{
    // Before we run this, we must make sure we remove the energy file
    std::ofstream ofs;
    ofs.open(energy_file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();

    // Init and run sim
    vpic_simulation simulation = vpic_simulation();

    // TODO: We should do this in a safer manner
    simulation.initialize( 0, NULL );

    //while( simulation.advance() );

    simulation.finalize();

    if( world_rank==0 ) log_printf( "normal exit\n" );

    std::cout << "Comparing " << moment_file_name << " to " <<
        moment_gold_file_name << std::endl;

    // Compare energies to make sure everything worked out OK (within 1%)
    const unsigned short t2_mask = 0b00000010;
    const unsigned short t6_mask = 0b00100000; 

    // Test just the step range 0-5000, and have tight counts
    REQUIRE(
	    test_utils::compare_energies(moment_file_name, moment_gold_file_name, 
     					 0.02, 1e-5, t2_mask, test_utils::FIELD_ENUM::Sum, 1, "ta.t2.tight.out", 0, 5000)
            );

    // Test the sum of the T6
    REQUIRE(
            test_utils::compare_energies(moment_file_name, moment_gold_file_name,
    					 0.02, 1e-5, t6_mask, test_utils::FIELD_ENUM::Sum, 1, "ta.t6.tight.out", 0, 5000)
           );

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
}
