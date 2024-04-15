#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"

// This function calculates the number of particles in each ionization state.
#ifdef FIELD_IONIZATION
Kokkos::View<int*, Kokkos::LayoutLeft>
ionization_states_kokkos(const species_t* RESTRICT sp) {

    if(!sp) ERROR(("Bad args"));

    const int np = sp->np;
    const k_particles_t& k_particles = sp->k_p_d;

    auto epsilon_eV_list_h = Kokkos::create_mirror(sp->ionization_energy);
    Kokkos::deep_copy(epsilon_eV_list_h,sp->ionization_energy);
    const int N_states = epsilon_eV_list_h.extent(0)+1; // Include charge state 0

    Kokkos::View<int*, Kokkos::LayoutLeft> charge_counts("charge_counts", N_states);
    auto charge_counts_h = Kokkos::create_mirror(charge_counts);
    Kokkos::deep_copy(charge_counts_h,charge_counts);

    Kokkos::View<int*, Kokkos::LayoutLeft> global_charge_counts("global_charge_counts", N_states);
    auto global_charge_counts_h = Kokkos::create_mirror(global_charge_counts);
    Kokkos::deep_copy(global_charge_counts_h,global_charge_counts);

    // Calculate (local) number of particles in each charge state
    Kokkos::parallel_for("ionization_states_kokkos", np, KOKKOS_LAMBDA(const int n) {
        int charge = k_particles(n, particle_var::charge);
	Kokkos::atomic_increment(&charge_counts(charge));
    });
    Kokkos::fence();
    Kokkos::deep_copy(charge_counts_h,charge_counts);


    // Perform MPI reductuion over ranks
    for (int i = 0; i < N_states; ++i) {
      mp_allsum_i( &charge_counts_h(i), &global_charge_counts_h(i), 1 );
    };
    Kokkos::deep_copy(global_charge_counts,global_charge_counts_h);

    // Return the global charge state
    return global_charge_counts;
}
#endif
