#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

// This function calculates the number of particles in each ionization state.
#ifdef FIELD_IONIZATION
Kokkos::View<double*, Kokkos::LayoutLeft>
ionization_states_kokkos(const species_t* RESTRICT sp) {

    if(!sp) ERROR(("Bad args"));

    const int np = sp->np;
    const k_particles_t& k_particles = sp->k_p_d;

    auto epsilon_eV_list_h = Kokkos::create_mirror(sp->ionization_energy);
    const int N_states = epsilon_eV_list_h.extent(0)+1; // Include charge state 0

    Kokkos::View<double*, Kokkos::LayoutLeft> charge_counts("charge_counts", N_states);
    Kokkos::View<double*, Kokkos::LayoutLeft> global_charge_counts("charge_counts", N_states);

    // Calculate (local) number of particles in each charge state
    Kokkos::parallel_for("ionization_states_kokkos", np, KOKKOS_LAMBDA(const int n) {
	short int charge = k_particles(n, particle_var::charge);
	Kokkos::atomic_add(&charge_counts(charge), 1.0); //FIXME: GPU doesnt seem to like this.

    });
    Kokkos::fence();

    // Perform an all-reduce operation for each charge state
    Kokkos::parallel_for(N_states, KOKKOS_LAMBDA(const int i) {
	mp_allsum_d( &charge_counts(i), &global_charge_counts(i), 1 );
    });
    Kokkos::fence();

    // Return the global charge state
    return global_charge_counts;
}
#endif
