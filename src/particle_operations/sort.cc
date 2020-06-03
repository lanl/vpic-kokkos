#include "sort.h"

void BinSort::sort( species_t* sp,
                    const bool direct,
                    const bool use_cellindex )
{
    const int step = sp->g->step;
    const int num_bins = sp->g->nv;

    // Create partition.
    if( sp->k_partition_d.extent(0) < num_bins + 1) {
      sp->k_partition_d = k_particle_partition_t(
        Kokkos::ViewAllocateWithoutInitializing("k_partition_d"),
        num_bins + 1
      );
      sp->k_partition_h = Kokkos::create_mirror_view(sp->k_partition_d);
    }

    // Create sortindex.
    if( sp->k_sortindex_d.extent(0) < sp->np ) {
      sp->k_sortindex_d = k_particle_sortindex_t(
        Kokkos::ViewAllocateWithoutInitializing("k_sortindex_d"),
        sp->np
      );
      sp->k_sortindex_h = Kokkos::create_mirror_view(sp->k_sortindex_d);
    }

    // Create cellindex if needed.
    if( use_cellindex && sp->k_cellindex_d.extent(0) < sp->np ) {
      sp->k_cellindex_d = k_particle_sortindex_t(
        Kokkos::ViewAllocateWithoutInitializing("k_cellindex_d"),
        sp->np
      );
      sp->k_cellindex_h = Kokkos::create_mirror_view(sp->k_cellindex_d);
    }

    // Create temporary storage.
    Kokkos::View<Kokkos::DefaultExecutionSpace::size_type*> bincounts(
      Kokkos::ViewAllocateWithoutInitializing("bincounts"),
      num_bins + 1
    );

    k_particle_partition_t& partition = sp->k_partition_d;
    k_particle_sortindex_t& sortindex = sp->k_sortindex_d;
    k_particle_cellindex_t& cellindex = sp->k_cellindex_d;

    k_particles_i_t index_ra = sp->k_p_i_d;
    k_particle_partition_t_ra partition_ra = partition;
    k_particle_sortindex_t_ra sortindex_ra = sortindex;

    Kokkos::parallel_for("DefaultSort::sort::clear",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_bins+1),
      KOKKOS_LAMBDA (const int i) {
        partition(i) = 0;
        bincounts(i) = 0;
      });

    Kokkos::parallel_for("DefaultSort::sort::count",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, sp->np),
      KOKKOS_LAMBDA (const size_t i) {
        Kokkos::atomic_increment(&partition(index_ra(i)));
      });

    Kokkos::parallel_scan("DefaultSort::sort::partiiton",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, num_bins+1),
      KOKKOS_LAMBDA (const size_t i, size_t& update, const bool final) {
        const size_t count = partition_ra(i);
        if( final ) {
          partition(i) = update;
        }
        update += count;
      });

    Kokkos::parallel_for("DefaultSort::sort::index",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, sp->np),
      KOKKOS_LAMBDA (const size_t i) {
        const int bin = index_ra(i);
        const int count = Kokkos::atomic_fetch_add(&bincounts(bin), 1);
        sortindex(partition(bin) + count) = i;

        if( use_cellindex ) {
          cellindex(i) = count;
        }

      });

    if( direct ) {

      k_particles_t old_p = sp->k_p_d;
      k_particles_t new_p(
        Kokkos::ViewAllocateWithoutInitializing("k_particles"),
        sp->max_np
      );

      k_particles_i_t old_i = sp->k_p_i_d;
      k_particles_i_t new_i(
        Kokkos::ViewAllocateWithoutInitializing("k_particles_i"),
        sp->max_np
      );

      Kokkos::parallel_for("DefaultSort::sort::reorder",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, sp->np),
        KOKKOS_LAMBDA (const int i) {
          const int j  = sortindex_ra(i);
          const auto v = old_i(j);
          sortindex(i) = i;

          new_p(i, particle_var::ux) = old_p(j, particle_var::ux);
          new_p(i, particle_var::uy) = old_p(j, particle_var::uy);
          new_p(i, particle_var::uz) = old_p(j, particle_var::uz);
          new_p(i, particle_var::dx) = old_p(j, particle_var::dx);
          new_p(i, particle_var::dy) = old_p(j, particle_var::dy);
          new_p(i, particle_var::dz) = old_p(j, particle_var::dz);
          new_p(i, particle_var::w)  = old_p(j, particle_var::w);
          new_i(i)                   = v;

          if( use_cellindex ) {
            cellindex(i) = i - partition_ra(v);
          }

        });

      sp->k_p_d   = new_p;
      sp->k_p_i_d = new_i;

      sp->k_p_h   = Kokkos::create_mirror_view(sp->k_p_d);
      sp->k_p_i_h = Kokkos::create_mirror_view(sp->k_p_i_d);

      sp->last_sorted = step;

    }

    sp->last_indexed = step;

}
