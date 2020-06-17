#ifndef PARTICLE_SORT_POLICY_H
#define PARTICLE_SORT_POLICY_H

#include <Kokkos_Sort.hpp>
#include <Kokkos_DualView.hpp>
#include "../vpic/kokkos_helpers.h"

struct min_max_functor {
  typedef Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> minmax_scalar;
  Kokkos::View<int*> view;
  min_max_functor(const Kokkos::View<int*>& view_) : view(view_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, minmax_scalar& minmax) const {
    if(view(i) < minmax.min_val) minmax.min_val = view(i);
    if(view(i) > minmax.max_val) minmax.max_val = view(i);
  }
};

/**
 * @brief Simple bin sort using Kokkos inbuilt sort
 */
struct DefaultSort {
    // TODO: should the sort interface just take the sp?
    static void sort(
            k_particles_soa_t part,
            k_particles_t particles,
            k_particles_i_t particles_i,
            const int32_t np,
            const int32_t num_bins
    )
    {
        // Try grab the index's for a permute key
        //int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
        //auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);
//        auto keys = particles_i;
        auto keys = part.i;

        // TODO: we can tighten the bounds on this to avoid ghosts

        using key_type = decltype(keys);
        //using Comparator = Casted_BinOp1D<key_type>;
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();
//        bin_sort.sort(particles);
//        bin_sort.sort(particles_i);

        bin_sort.sort(part.dx);
        bin_sort.sort(part.dy);
        bin_sort.sort(part.dz);
        bin_sort.sort(part.ux);
        bin_sort.sort(part.uy);
        bin_sort.sort(part.uz);
        bin_sort.sort(part.w);
        bin_sort.sort(part.i);
    }

    static void sort(
            k_particles_soa_t part,
            const int32_t np,
            const int32_t num_bins
    )
    {
        // Try grab the index's for a permute key
        //int pi = particle_var::pi; // FIXME: can you really not pass an enum in??
        //auto keys = Kokkos::subview(particles, Kokkos::ALL, pi);
        auto keys = part.i;

        // TODO: we can tighten the bounds on this to avoid ghosts

        using key_type = decltype(keys);
        //using Comparator = Casted_BinOp1D<key_type>;
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(num_bins, 0, num_bins);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        bin_sort.sort(part.dx);
        bin_sort.sort(part.dy);
        bin_sort.sort(part.dz);
        bin_sort.sort(part.ux);
        bin_sort.sort(part.uy);
        bin_sort.sort(part.uz);
        bin_sort.sort(part.w);
        bin_sort.sort(part.i);
    }

    static void gpu_sort(
            k_particles_soa_t part,
            const int32_t np,
            const int32_t num_bins
    )
    {
        // Create permute view by taking index view and adding offsets such that we get
        // 1,2,3,1,2,3,1,2,3 instead of 1,1,1,2,2,2,3,3,3 
        Kokkos::MinMaxScalar<Kokkos::View<int*>::non_const_value_type> result;
        Kokkos::MinMax<Kokkos::View<int*>::non_const_value_type> reducer(result);
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,part.i.extent(0)), min_max_functor(part.i), reducer);
        Kokkos::View<int*> key_view("sorting keys", part.i.extent(0));
        Kokkos::View<int*> bin_counter("Counter for updating keys", num_bins);
        Kokkos::deep_copy(key_view, part.i);
        Kokkos::parallel_for("init bin counters", Kokkos::RangePolicy<>(0,num_bins), KOKKOS_LAMBDA(const int i) {
          bin_counter(i) = 0;
        });
        Kokkos::parallel_for("Update keys", Kokkos::RangePolicy<>(0, np), KOKKOS_LAMBDA(const int i) {
          int count = Kokkos::atomic_fetch_add(&(bin_counter(key_view(i))), 1);
          key_view(i) += count*result.max_val;
        });
        Kokkos::parallel_reduce("Get min/max bin", Kokkos::RangePolicy<>(0,part.i.extent(0)), min_max_functor(key_view), reducer);
        auto keys = key_view;

        // TODO: we can tighten the bounds on this to avoid ghosts

        using key_type = decltype(keys);
        using Comparator = Kokkos::BinOp1D<key_type>;
        Comparator comp(result.max_val-result.min_val+1, result.min_val, result.max_val);

        int sort_within_bins = 0;
        Kokkos::BinSort<key_type, Comparator> bin_sort(keys, 0, np, comp, sort_within_bins );
        bin_sort.create_permute_vector();

        bin_sort.sort(part.dx);
        bin_sort.sort(part.dy);
        bin_sort.sort(part.dz);
        bin_sort.sort(part.ux);
        bin_sort.sort(part.uy);
        bin_sort.sort(part.uz);
        bin_sort.sort(part.w);
        bin_sort.sort(part.i);
    }

};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
    using Policy::sort;
    using Policy::gpu_sort;
};

//template<class ExecSpace=Kokkos::DefaultExecutionSpace>
void gpu_memory_access_sort(k_particles_soa_t& particles, const int32_t np, const int32_t num_cells) {

  Kokkos::View<int*> ppc("Particles per cell", num_cells);
  Kokkos::View<int*> particle_counter("Particle counter", num_cells);

  ParticleSorter<> sorter;
  sorter.sort(particles, np, num_cells);

  k_particles_struct particle_copy(particles.i.extent(0));
  Kokkos::deep_copy(particle_copy.dx, particles.dx);
  Kokkos::deep_copy(particle_copy.dy, particles.dy);
  Kokkos::deep_copy(particle_copy.dz, particles.dz);
  Kokkos::deep_copy(particle_copy.ux, particles.ux);
  Kokkos::deep_copy(particle_copy.uy, particles.uy);
  Kokkos::deep_copy(particle_copy.uz, particles.uz);
  Kokkos::deep_copy(particle_copy.w, particles.w);
  Kokkos::deep_copy(particle_copy.i, particles.i);
  
//  Kokkos::parallel_for("Print sorted cell indices", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int x) {
//    printf("\n");
//    for(int i=0; i<1024; i++) {
//      printf("CPU Sorted: i: %d\n", particle_copy.i(i));
//    }
//    printf("\n");
//  });

  Kokkos::parallel_for("Get particle per cell counts", Kokkos::RangePolicy<>(0,num_cells), 
  KOKKOS_LAMBDA(const int idx){
    ppc(idx) = 0;
  });

  Kokkos::parallel_for("Get particle per cell counts", Kokkos::RangePolicy<>(0,np), 
  KOKKOS_LAMBDA(const int idx){
    Kokkos::atomic_increment(&(ppc(particles.i(idx))));
  });

//  Kokkos::parallel_for("Print sorted cell indices", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int x) {
//    printf("\n");
//    printf("num particles in cell 0: %d\n", ppc(0));
//    printf("\n");
//  });

//  Kokkos::parallel_for("Print cell particle counts", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int index) {
//    int count = 0;
//    for(int i=0; i<num_cells; i++) {
//      if(ppc(i) > 0) {
//        printf("cell: %d, ppc(cell): %d\n", i, ppc(i));
//        count++;
//      }
//    }
//    printf("count: %d\n", count);
//  });

  Kokkos::View<int[1]> first_cell("First cell with particles");
  Kokkos::View<int[1]> minloc("index of cell with min particles");
  Kokkos::View<int[1]> cell_count("Number of non zero cells");
  Kokkos::parallel_for("Find min ppc", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int index) {
    minloc(0) = -1;
    int min = INT_MAX;
    cell_count(0) = 0;
    first_cell(0) = -1;
    for(int j=0; j<num_cells; j++) {
      if(ppc(j) > 0 && first_cell(0) >= 0) {
        first_cell(0) = j;
      }
      if(ppc(j) > 0) {
        cell_count(0) += 1;
      }
      if(ppc(j) < min && ppc(j) > 0) {
        min = ppc(j);
        minloc(0) = j;
      }
    }
//    printf("min_loc: %d, min ppc: %d\n", minloc(0), ppc(minloc(0)));
//    printf("num_cells: %d, cell count: %d\n", num_cells, cell_count(0));
//    printf("First cell: %d\n", particles.i(0));
  });

  Kokkos::parallel_for("Fill particle coutner", Kokkos::RangePolicy<>(0,num_cells), KOKKOS_LAMBDA(const int i) {
    particle_counter(i) = 0;
  });

  Kokkos::parallel_for("Sort for GPU", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int index) {
    int offset = 0;
    int current_cell = particle_copy.i(0);
    int min_ppc = ppc(minloc(0));
    int end = cell_count(0)*min_ppc;
    for(int i=0; i<np; i++) {
      int cell = particle_copy.i(i);
      if(cell > current_cell) {
        offset++;
        current_cell = cell;
      }
      if(particle_counter(cell) < min_ppc) {
        particles.dx(particle_counter(cell)*cell_count(0) + offset) = particle_copy.dx(i);
        particles.dy(particle_counter(cell)*cell_count(0) + offset) = particle_copy.dy(i);
        particles.dz(particle_counter(cell)*cell_count(0) + offset) = particle_copy.dz(i);
        particles.ux(particle_counter(cell)*cell_count(0) + offset) = particle_copy.ux(i);
        particles.uy(particle_counter(cell)*cell_count(0) + offset) = particle_copy.uy(i);
        particles.uz(particle_counter(cell)*cell_count(0) + offset) = particle_copy.uz(i);
        particles.w(particle_counter(cell)*cell_count(0) + offset)  = particle_copy.w(i);
        particles.i(particle_counter(cell)*cell_count(0) + offset)  = particle_copy.i(i);
        particle_counter(cell) += 1;
      } else {
        particles.dx(end) = particle_copy.dx(i);
        particles.dy(end) = particle_copy.dy(i);
        particles.dz(end) = particle_copy.dz(i);
        particles.ux(end) = particle_copy.ux(i);
        particles.uy(end) = particle_copy.uy(i);
        particles.uz(end) = particle_copy.uz(i);
        particles.w(end)  = particle_copy.w(i);
        particles.i(end)  = particle_copy.i(i);
        end += 1;
      }
    }
  });

//  Kokkos::parallel_for("Print sorted cell indices for GPU", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int x) {
//    printf("\n");
//    for(int i=0; i<1024; i++) {
//      printf("GPU Sorted: i: %d\n", particles.i(i));
//    }
//    printf("\n");
//  });

//  Kokkos::parallel_for("\"Sort\" particles", Kokkos::RangePolicy<>(0,1), KOKKOS_LAMBDA(const int index) {
//    for(int i=0; i<np; i+=ppc(minloc(0))) {
//      for(int j=0; j<ppc(minloc(0)); j++) {
//        if(i+j < np) {
//          particles.i(i+j) = minloc(0) + j;
//        }
//      }
//    }
//  });
}

#endif //guard
