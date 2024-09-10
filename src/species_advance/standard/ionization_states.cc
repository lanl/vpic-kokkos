#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

// This function calculates the number of particles in each ionization state.
#ifdef FIELD_IONIZATION
Kokkos::View<int*, Kokkos::HostSpace>
ionization_states_kokkos(const species_t* RESTRICT sp) {

    if(!sp) ERROR(("Bad args"));

    const long long int np = sp->np;
    const k_particles_t& k_particles = sp->k_p_d;

    const int N_states = sp->n_energy+1; // Include charge state 0

    Kokkos::View<int*, Kokkos::HostSpace> charge_counts_h("charge_counts_h", N_states);
    Kokkos::View<int*> charge_counts("charge_counts", N_states);
    Kokkos::deep_copy(charge_counts,charge_counts_h);

    Kokkos::View<int*, Kokkos::HostSpace> global_charge_counts_h("global_charge_counts_h", N_states);

    // Calculate (local) number of particles in each charge state
    Kokkos::parallel_for("ionization_states_kokkos", np, KOKKOS_LAMBDA(const int n) {
        long long int charge = k_particles(n, particle_var::charge);
        Kokkos::atomic_increment(&charge_counts(charge));
    });
    Kokkos::fence();
    Kokkos::deep_copy(charge_counts_h,charge_counts);

    // Perform MPI reductuion over ranks
    for (int i = 0; i < N_states; ++i) {
      mp_allsum_i( &charge_counts_h(i), &global_charge_counts_h(i), 1 );
    };

    // Return the global charge state
    return global_charge_counts_h;
}
#endif
