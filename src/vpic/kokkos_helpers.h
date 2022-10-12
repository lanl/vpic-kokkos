#ifndef _kokkos_helpers_h_
#define _kokkos_helpers_h_

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <iostream>

#include "../material/material.h" // Need material_t

// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
#define PARTICLE_VAR_COUNT 7
#define PARTICLE_MOVER_VAR_COUNT 3
#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4
#define INTERPOLATOR_VAR_COUNT 18
#define MATERIAL_COEFFICIENT_VAR_COUNT 13
#define HYDRO_VAR_COUNT 14
#define NUM_J_DIMS 3

#ifdef KOKKOS_ENABLE_CUDA
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterNonDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutLeft
#else
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterNonAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutRight
#endif

typedef int16_t material_id;

// TODO: we dont need the [1] here
// TODO: this can likely be unsigned, but that tends to upset Kokkos
using k_counter_t = Kokkos::View<int[1]>;

using k_field_t = Kokkos::View<float *[FIELD_VAR_COUNT]>;
// TODO: This scatter access is needed only for jfxyz, not all field vars.
// This is probably terrible on CPU.
using k_field_sa_t = Kokkos::Experimental::ScatterView<float *[FIELD_VAR_COUNT]>;
using k_field_edge_t = Kokkos::View<material_id* [FIELD_EDGE_COUNT]>;
using k_field_accum_t = Kokkos::View<float *>;

using k_jf_accum_t = Kokkos::View<float *[NUM_J_DIMS]>;

using k_particles_t = Kokkos::View<float *[PARTICLE_VAR_COUNT], Kokkos::LayoutLeft>;
using k_particles_i_t = Kokkos::View<int*>;

// TODO: think about the layout here
using k_particle_copy_t = Kokkos::View<float *[PARTICLE_VAR_COUNT], Kokkos::LayoutRight>;
using k_particle_i_copy_t = Kokkos::View<int*>;

using k_particle_movers_t = Kokkos::View<float *[PARTICLE_MOVER_VAR_COUNT]>;
using k_particle_i_movers_t = Kokkos::View<int*>;

using k_neighbor_t = Kokkos::View<int64_t*>;

using k_interpolator_t = Kokkos::View<float *[INTERPOLATOR_VAR_COUNT]>;

// TODO: Delete these
using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

// TODO: why is this _sa_ not _sv_?
using k_accumulators_sa_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using k_hydro_d_t = Kokkos::View<float* [HYDRO_VAR_COUNT]>;
using k_hydro_sv_t = Kokkos::Experimental::ScatterView<float* [HYDRO_VAR_COUNT]>;

//using k_accumulators_sah_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated, Kokkos::Experimental::ScatterNonAtomic>;

using static_sched = Kokkos::Schedule<Kokkos::Static>;
using host_execution_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, static_sched, int>;

using k_material_coefficient_t = Kokkos::View<float* [MATERIAL_COEFFICIENT_VAR_COUNT]>;


#define KOKKOS_TEAM_POLICY_DEVICE  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>
#define KOKKOS_TEAM_POLICY_HOST  Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>

namespace Kokkos {
  /** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each
   * i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
   */
  template <template <typename iType, class ThreadsExecTeamMember> class ThreadVectorRangeBoundariesStruct, 
            typename iType, class ThreadsExecTeamMember, class Lambda>
  KOKKOS_INLINE_FUNCTION void parallel_for_simd(
      const ThreadVectorRangeBoundariesStruct<
          iType, ThreadsExecTeamMember>& loop_boundaries,
      const Lambda& lambda) {
    #pragma omp simd
    for (iType i = loop_boundaries.start; i < loop_boundaries.end;
         i += loop_boundaries.increment)
      lambda(i);
  }

  /** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i,
   * ValueType & val) for each i=0..N-1.
   *
   * The range i=0..N-1 is mapped to all vector lanes of the the calling thread
   * and a summation of val is performed and put into result.
   */
  template <template <typename iType, class ThreadsExecTeamMember> class ThreadVectorRangeBoundariesStruct, 
            typename iType, class ThreadsExecTeamMember, class Lambda, typename ValueType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<!Kokkos::is_reducer<ValueType>::value>::type
      parallel_reduce_simd_sum(const ThreadVectorRangeBoundariesStruct<
                          iType, ThreadsExecTeamMember>& loop_boundaries,
                      const Lambda& lambda, ValueType& result) {
    result = ValueType();
    #pragma omp simd reduction(+:result)
    for (iType i = loop_boundaries.start; i < loop_boundaries.end;
         i += loop_boundaries.increment) {
      lambda(i, result);
    }
  }
}

namespace field_var {
  enum f_v {
    ex        = 0,
    ey        = 1,
    ez        = 2,
    div_e_err = 3,
    cbx       = 4,
    cby       = 5,
    cbz       = 6,
    div_b_err = 7,
    tcax      = 8,
    tcay      = 9,
    tcaz      = 10,
    rhob      = 11,
    jfx       = 12,
    jfy       = 13,
    jfz       = 14,
    rhof      = 15
  };
};
namespace field_edge_var { \
  enum f_e_v {
    ematx = 0,
    ematy = 1,
    ematz = 2,
    nmat  = 3,
    fmatx = 4,
    fmaty = 5,
    fmatz = 6,
    cmat  = 7
  };
};

namespace interpolator_var {
  enum i_r {
    ex       = 0,
    dexdy    = 1,
    dexdz    = 2,
    d2exdydz = 3,
    ey       = 4,
    deydz    = 5,
    deydx    = 6,
    d2eydzdx = 7,
    ez       = 8,
    dezdx    = 9,
    dezdy    = 10,
    d2ezdxdy = 11,
    cbx      = 12,
    dcbxdx   = 13,
    cby      = 14,
    dcbydy   = 15,
    cbz      = 16,
    dcbzdz   = 17
  };
};

namespace particle_var {
  enum p_v {
    dx = 0,
    dy,
    dz,
    //pi = 3,
    ux,
    uy,
    uz,
    w,
  };
};

namespace particle_mover_var {
  enum p_m_v {
     dispx = 0,
     dispy = 1,
     dispz = 2,
     //pmi   = 3,
  };
};

namespace accumulator_var {
  enum a_v {
    jx = 0,
    jy = 1,
    jz = 2,
  };
};

namespace material_coeff_var {
    enum mc_v {
        decayx        = 0,
        drivex        = 1,
        decayy        = 2,
        drivey        = 3,
        decayz        = 4,
        drivez        = 5,
        rmux          = 6,
        rmuy          = 7,
        rmuz          = 8,
        nonconductive = 9,
        epsx          = 10,
        epsy          = 11,
        epsz          = 12,
    };
};

namespace hydro_var {
    enum h_v {
        jx  = 0,
        jy  = 1,
        jz  = 2,
        rho = 3,
        px  = 4,
        py  = 5,
        pz  = 6,
        ke  = 7,
        txx = 8,
        tyy = 9,
        tzz = 10,
        tyz = 11,
        tzx = 12,
        txy = 13,
    };
};

void print_particles_d(
        k_particles_t particles,
        int np
        );

// The templating here is to defer the type until later in the head include chain
template <class P>
bool compareParticleMovers(P& a, P& b) {
    return a.i < b.i;
}

#endif // _kokkos_helpers_h_
