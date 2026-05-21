#ifndef _kokkos_helpers_h_
#define _kokkos_helpers_h_

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <iostream>

#include "../material/material.h" // Need material_t

// This module implements kokkos macros

#define FIELD_VAR_COUNT 44
#define FIELD_EDGE_COUNT 8

#ifdef VARIABLE_CHARGE
  #define PARTICLE_VAR_COUNT 8
#else
  #define PARTICLE_VAR_COUNT 7
#endif

#define PARTICLE_MOVER_VAR_COUNT 3
#define ACCUMULATOR_VAR_COUNT 4
#define ACCUMULATOR_ARRAY_LENGTH 4
#define MATERIAL_COEFFICIENT_VAR_COUNT 13
#ifdef VARIABLE_CHARGE
  #define HYDRO_VAR_COUNT 16
#else
  #define HYDRO_VAR_COUNT 14
#endif
#define NUM_J_DIMS 4
#define FLUID_VAR_COUNT 6+4

#ifdef SHAPE_NGP
  #define INTERPOLATOR_VAR_COUNT 18
#else
#ifdef SHAPE_QS
  #define INTERPOLATOR_VAR_COUNT 42
#endif
#endif

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

using k_particle_partition_t = Kokkos::View<Kokkos::DefaultExecutionSpace::size_type*>;
using k_particle_partition_t_ra = Kokkos::View<const Kokkos::DefaultExecutionSpace::size_type*,
                                               Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using k_particle_sortindex_t = Kokkos::View<Kokkos::DefaultExecutionSpace::size_type*>;
using k_particle_sortindex_t_ra = Kokkos::View<const Kokkos::DefaultExecutionSpace::size_type*,
                                               Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using k_neighbor_t = Kokkos::View<int64_t*>;

using k_interpolator_t = Kokkos::View<float *[INTERPOLATOR_VAR_COUNT]>;

// TODO: Delete these
using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

// TODO: why is this _sa_ not _sv_?
using k_accumulators_sa_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using k_hydro_d_t = Kokkos::View<float* [HYDRO_VAR_COUNT]>;
using k_hydro_sv_t = Kokkos::Experimental::ScatterView<float* [HYDRO_VAR_COUNT]>;


using k_accumulators_sah_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated, Kokkos::Experimental::ScatterNonAtomic>;

using k_fluid_t = Kokkos::View<float *[FLUID_VAR_COUNT], Kokkos::LayoutRight>;
// 1D View: shape [FLUID_VAR_COUNT]
using k_fluid_1d = Kokkos::View<float*>;

using static_sched = Kokkos::Schedule<Kokkos::Static>;
using host_execution_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, static_sched, int>;

using k_material_coefficient_t = Kokkos::View<float* [MATERIAL_COEFFICIENT_VAR_COUNT]>;

using k_field_sa_t = Kokkos::Experimental::ScatterView<float *[FIELD_VAR_COUNT]>;

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
    pe        = 7,
    cbx0      = 8,
    cby0      = 9,
    cbz0      = 10,
    te0       = 11,
    tcax      = 12,
    tcay      = 13,
    tcaz      = 14,
    rhob      = 15,
    jfx       = 16,
    jfy       = 17,
    jfz       = 18,
    rhof      = 19,
    jfxold    = 20,
    jfyold    = 21,
    jfzold    = 22,
    rhofold   = 23,
    tx        = 24,
    ty        = 25,
    tz        = 26,
    te        = 27,
    ox        = 28,
    oy        = 29,
    oz        = 30,
    oe        = 31,
    pex       = 32,
    pey       = 33,
    pez       = 34,
    div_b_err = 35,
    ux        = 36,
    uy        = 37,
    uz        = 38,
    ue        = 39,
    sx        = 40,
    sy        = 41,
    sz        = 42,
    se        = 43
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
    cmat  = 7,
  };
};

namespace interpolator_var {
  enum i_r {
#ifdef SHAPE_NGP
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
    dcbzdz   = 17,
#else
#ifdef SHAPE_QS
    ex       = 0,
    dexdx    = 1,
    dexdy    = 2,
    dexdz    = 3,
    d2exdx   = 4,
    d2exdy   = 5,
    d2exdz   = 6,
    ey       = 7,
    deydx    = 8,
    deydy    = 9,
    deydz    = 10,
    d2eydx   = 11,
    d2eydy   = 12,
    d2eydz   = 13,
    ez       = 14,
    dezdx    = 15,
    dezdy    = 16,
    dezdz    = 17,
    d2ezdx   = 18,
    d2ezdy   = 19,
    d2ezdz   = 20,
    cbx      = 21,
    dcbxdx   = 22,
    dcbxdy   = 23,
    dcbxdz   = 24,
    d2cbxdx  = 25,
    d2cbxdy  = 26,
    d2cbxdz  = 27,
    cby      = 28,
    dcbydx   = 29,
    dcbydy   = 30,
    dcbydz   = 31,
    d2cbydx  = 32,
    d2cbydy  = 33,
    d2cbydz  = 34,
    cbz      = 35,
    dcbzdx   = 36,
    dcbzdy   = 37,
    dcbzdz   = 38,
    d2cbzdx  = 39,
    d2cbzdy  = 40,
    d2cbzdz  = 41,
#endif
#endif
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
#ifdef VARIABLE_CHARGE
    qp,
#endif
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
    rho= 3,
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
	//        ke  = 7,
	rho_m = 7,
        txx = 8,
        tyy = 9,
        tzz = 10,
        tyz = 11,
        tzx = 12,
        txy = 13,
#ifdef VARIABLE_CHARGE
	min_q = 14,
	max_q = 15,
#endif
    };
};

namespace fluid_var {
    enum fl_v {
        den  = 0,
        tmp  = 1,
        prs  = 2,
        ux  = 3,
        uy  = 4,
        uz  = 5,
	msx = 6,
	msy = 7,
	msz = 8,
	ens = 9,
    };
};

void print_particles_d(
        k_particles_t particles,
        int np
        );
void print_accumulator(k_accumulators_t fields, int n);

// The templating here is to defer the type until later in the head include chain
template <class P>
bool compareParticleMovers(P& a, P& b) {
    return a.i < b.i;
}

#endif // _kokkos_helpers_h_
