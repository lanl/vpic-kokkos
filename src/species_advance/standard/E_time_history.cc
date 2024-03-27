#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

// This function calculates the number of particles in each ionization state.
#ifdef FIELD_IONIZATION
double
E_time_history_kernel(const k_interpolator_t& k_interp, const k_particles_t& k_particles, const k_particles_i_t& k_particles_i, const int np) {

  double E_val = 0;
  // FIXME: this technically gets the total E-mag across all particles in this species
  // However there is a check in dump_E_time_history that ensures there is only 1 particle
    Kokkos::parallel_reduce(np, KOKKOS_LAMBDA(const int n, double& update) {
        float dx = k_particles(n, particle_var::dx);
        float dy = k_particles(n, particle_var::dy);
        float dz = k_particles(n, particle_var::dz);
        int   i  = k_particles_i(n);
        float ex = ( k_interp(i, interpolator_var::ex) + dy*k_interp(i, interpolator_var::dexdy) ) + dz*( k_interp(i, interpolator_var::dexdz) + dy*k_interp(i, interpolator_var::d2exdydz) );
        float ey = ( k_interp(i, interpolator_var::ey) + dz*k_interp(i, interpolator_var::deydz) ) + dx*( k_interp(i, interpolator_var::deydx) + dz*k_interp(i, interpolator_var::d2eydzdx) );
        float ez = ( k_interp(i, interpolator_var::ez) + dx*k_interp(i, interpolator_var::dezdx) ) + dy*( k_interp(i, interpolator_var::dezdy) + dx*k_interp(i, interpolator_var::d2ezdxdy) );
        float E_mag = sqrtf( ex*ex + ey*ey + ez*ez ); // code units
        update += static_cast<double>(E_mag);
    }, E_val);
    return E_val;
}



double
E_time_history( const species_t * RESTRICT sp, const interpolator_array_t * RESTRICT ia) {

    if(!sp || !ia) ERROR(("Bad args"));

    double local, global;

    local = E_time_history_kernel(ia->k_i_d, sp->k_p_d, sp->k_p_i_d, sp->np);
    Kokkos::fence();

    mp_allsum_d( &local, &global, 1 );

    return global;
}
#endif
