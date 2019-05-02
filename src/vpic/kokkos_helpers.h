#ifndef _kokkos_helpers_h_
#define _kokkos_helpers_h_

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <iostream>

#include "../material/material.h" // Need material_t

// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
#define PARTICLE_VAR_COUNT 8
#define PARTICLE_MOVER_VAR_COUNT 4
#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4
#define INTERPOLATOR_VAR_COUNT 18

/*
#ifdef KOKKOS_ENABLE_CUDA
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterNonDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutLeft
#else
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterNonAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutRight
#endif
*/

// TODO: we dont need the [1] here
using k_iterator_t = Kokkos::View<int[1]>;

using k_field_t = Kokkos::View<float *[FIELD_VAR_COUNT]>;
using k_field_edge_t = Kokkos::View<material_id* [FIELD_EDGE_COUNT]>;

using k_particles_t = Kokkos::View<float *[PARTICLE_VAR_COUNT]>;
using k_particle_movers_t = Kokkos::View<float *[PARTICLE_MOVER_VAR_COUNT]>;

using k_neighbor_t = Kokkos::View<int64_t*>;

using k_interpolator_t = Kokkos::View<float *[INTERPOLATOR_VAR_COUNT]>;

using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

//using k_accumulators_sa_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], KOKKOS_LAYOUT, Kokkos::DefaultExecutionSpace, Kokkos::Experimental::ScatterSum, KOKKOS_SCATTER_DUPLICATED, KOKKOS_SCATTER_ATOMIC>;
using k_accumulators_sa_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using static_sched = Kokkos::Schedule<Kokkos::Static>;
using host_execution_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, static_sched, int>;

#define KOKKOS_TEAM_POLICY_DEVICE  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>
#define KOKKOS_TEAM_POLICY_HOST  Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>

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
    dy = 1,
    dz = 2,
    pi = 3,
    ux = 4,
    uy = 5,
    uz = 6,
    w  = 7
  };
};

namespace particle_mover_var {
  enum p_m_v {
     dispx = 0,
     dispy = 1,
     dispz = 2,
     pmi   = 3,
  };
};

namespace accumulator_var {
  enum a_v { \
    jx = 0, \
    jy = 1, \
    jz = 2, \
  };
};

#define KOKKOS_FIELD_VARIABLES() \
  int n_fields = field_array->g->nv; \
  k_field_t::HostMirror k_field; \
  k_field_edge_t::HostMirror k_field_edge; \

#define KOKKOS_COPY_FIELD_MEM_TO_DEVICE() \
  k_field = field_array->k_f_h; \
  k_field_edge = field_array->k_fe_h; \
  Kokkos::parallel_for("copy field to device", host_execution_policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
          k_field(i, field_var::ex) = field_array->f[i].ex; \
          k_field(i, field_var::ey) = field_array->f[i].ey; \
          k_field(i, field_var::ez) = field_array->f[i].ez; \
          k_field(i, field_var::div_e_err) = field_array->f[i].div_e_err; \
          \
          k_field(i, field_var::cbx) = field_array->f[i].cbx; \
          k_field(i, field_var::cby) = field_array->f[i].cby; \
          k_field(i, field_var::cbz) = field_array->f[i].cbz; \
          k_field(i, field_var::div_b_err) = field_array->f[i].div_b_err; \
          \
          k_field(i, field_var::tcax) = field_array->f[i].tcax; \
          k_field(i, field_var::tcay) = field_array->f[i].tcay; \
          k_field(i, field_var::tcaz) = field_array->f[i].tcaz; \
          k_field(i, field_var::rhob) = field_array->f[i].rhob; \
          \
          k_field(i, field_var::jfx) = field_array->f[i].jfx; \
          k_field(i, field_var::jfy) = field_array->f[i].jfy; \
          k_field(i, field_var::jfz) = field_array->f[i].jfz; \
          k_field(i, field_var::rhof) = field_array->f[i].rhof; \
          \
          k_field_edge(i, field_edge_var::ematx) = field_array->f[i].ematx; \
          k_field_edge(i, field_edge_var::ematy) = field_array->f[i].ematy; \
          k_field_edge(i, field_edge_var::ematz) = field_array->f[i].ematz; \
          k_field_edge(i, field_edge_var::nmat) = field_array->f[i].nmat; \
          \
          k_field_edge(i, field_edge_var::fmatx) = field_array->f[i].fmatx; \
          k_field_edge(i, field_edge_var::fmaty) = field_array->f[i].fmaty; \
          k_field_edge(i, field_edge_var::fmatz) = field_array->f[i].fmatz; \
          k_field_edge(i, field_edge_var::cmat) = field_array->f[i].cmat; \
  });     \
  Kokkos::deep_copy(field_array->k_f_d, field_array->k_f_h); \
  Kokkos::deep_copy(field_array->k_fe_d, field_array->k_fe_h);


#define KOKKOS_COPY_FIELD_MEM_TO_HOST() \
  Kokkos::deep_copy(field_array->k_f_h, field_array->k_f_d); \
  Kokkos::deep_copy(field_array->k_fe_h, field_array->k_fe_d); \
  k_field = field_array->k_f_h; \
  k_field_edge = field_array->k_fe_h; \
  Kokkos::parallel_for("copy field to host", host_execution_policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
          field_array->f[i].ex = k_field(i, field_var::ex); \
          field_array->f[i].ey = k_field(i, field_var::ey); \
          field_array->f[i].ez = k_field(i, field_var::ez); \
          field_array->f[i].div_e_err = k_field(i, field_var::div_e_err); \
          \
          field_array->f[i].cbx = k_field(i, field_var::cbx); \
          field_array->f[i].cby = k_field(i, field_var::cby); \
          field_array->f[i].cbz = k_field(i, field_var::cbz); \
          field_array->f[i].div_b_err = k_field(i, field_var::div_b_err); \
          \
          field_array->f[i].tcax = k_field(i, field_var::tcax); \
          field_array->f[i].tcay = k_field(i, field_var::tcay); \
          field_array->f[i].tcaz = k_field(i, field_var::tcaz); \
          field_array->f[i].rhob = k_field(i, field_var::rhob); \
          \
          field_array->f[i].jfx = k_field(i, field_var::jfx); \
          field_array->f[i].jfy = k_field(i, field_var::jfy); \
          field_array->f[i].jfz = k_field(i, field_var::jfz); \
          field_array->f[i].rhof = k_field(i, field_var::rhof); \
          \
          field_array->f[i].ematx = k_field_edge(i, field_edge_var::ematx); \
          field_array->f[i].ematy = k_field_edge(i, field_edge_var::ematy); \
          field_array->f[i].ematz = k_field_edge(i, field_edge_var::ematz); \
          field_array->f[i].nmat = k_field_edge(i, field_edge_var::nmat); \
          \
          field_array->f[i].fmatx = k_field_edge(i, field_edge_var::fmatx); \
          field_array->f[i].fmaty = k_field_edge(i, field_edge_var::fmaty); \
          field_array->f[i].fmatz = k_field_edge(i, field_edge_var::fmatz); \
          field_array->f[i].cmat = k_field_edge(i, field_edge_var::cmat); \
  });


#define KOKKOS_PARTICLE_VARIABLES() \
  int n_particles; \
  int max_pmovers; \
  \
  k_particles_t::HostMirror k_particles_h; \
  k_particle_movers_t::HostMirror k_particle_movers_h; \
  k_iterator_t::HostMirror k_nm_h;


// TODO: I don't like that these hard code use `sp`
// perhaps it should cache old sp and put it back to what it was?
#define KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE() \
  LIST_FOR_EACH( sp, species_list ) {\
    n_particles = sp->np; \
    max_pmovers = sp->max_nm; \
    \
    k_particles_h = sp->k_p_h; \
    k_particle_movers_h = sp->k_pm_h; \
    k_nm_h = sp->k_nm_h; \
    k_nm_h(0) = sp->nm; \
    Kokkos::parallel_for("copy particles to device", host_execution_policy(0, n_particles) , KOKKOS_LAMBDA (int i) { \
      k_particles_h(i, particle_var::dx) = sp->p[i].dx; \
      k_particles_h(i, particle_var::dy) = sp->p[i].dy; \
      k_particles_h(i, particle_var::dz) = sp->p[i].dz; \
      k_particles_h(i, particle_var::ux) = sp->p[i].ux; \
      k_particles_h(i, particle_var::uy) = sp->p[i].uy; \
      k_particles_h(i, particle_var::uz) = sp->p[i].uz; \
      k_particles_h(i, particle_var::w)  = sp->p[i].w;  \
      k_particles_h(i, particle_var::pi) = sp->p[i].i;  \
    });\
    \
    Kokkos::parallel_for("copy movers to device", host_execution_policy(0, max_pmovers) , KOKKOS_LAMBDA (int i) { \
      k_particle_movers_h(i, particle_mover_var::dispx) = sp->pm[i].dispx; \
      k_particle_movers_h(i, particle_mover_var::dispy) = sp->pm[i].dispy; \
      k_particle_movers_h(i, particle_mover_var::dispz) = sp->pm[i].dispz; \
      k_particle_movers_h(i, particle_mover_var::pmi)   = sp->pm[i].i;     \
    });\
    Kokkos::deep_copy(sp->k_p_d, sp->k_p_h);  \
    Kokkos::deep_copy(sp->k_pm_d, sp->k_pm_h); \
    Kokkos::deep_copy(sp->k_nm_d, sp->k_nm_h); \
  };


#define KOKKOS_COPY_PARTICLE_MEM_TO_HOST() \
  LIST_FOR_EACH( sp, species_list ) {\
    Kokkos::deep_copy(sp->k_p_h, sp->k_p_d);  \
    Kokkos::deep_copy(sp->k_pm_h, sp->k_pm_d); \
    Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d); \
    n_particles = sp->np; \
    max_pmovers = sp->max_nm; \
    k_particles_h = sp->k_p_h; \
    k_particle_movers_h = sp->k_pm_h; \
    k_nm_h = sp->k_nm_h; \
    sp->nm = k_nm_h(0); \
    Kokkos::parallel_for("copy particles to host", host_execution_policy(0, n_particles) , KOKKOS_LAMBDA (int i) { \
      sp->p[i].dx = k_particles_h(i, particle_var::dx); \
      sp->p[i].dy = k_particles_h(i, particle_var::dy); \
      sp->p[i].dz = k_particles_h(i, particle_var::dz); \
      sp->p[i].ux = k_particles_h(i, particle_var::ux); \
      sp->p[i].uy = k_particles_h(i, particle_var::uy); \
      sp->p[i].uz = k_particles_h(i, particle_var::uz); \
      sp->p[i].w  = k_particles_h(i, particle_var::w);  \
      sp->p[i].i  = k_particles_h(i, particle_var::pi); \
    });\
    \
    Kokkos::parallel_for("copy movers to host", host_execution_policy(0, max_pmovers) , KOKKOS_LAMBDA (int i) { \
      sp->pm[i].dispx = k_particle_movers_h(i, particle_mover_var::dispx); \
      sp->pm[i].dispy = k_particle_movers_h(i, particle_mover_var::dispy); \
      sp->pm[i].dispz = k_particle_movers_h(i, particle_mover_var::dispz); \
      sp->pm[i].i     = k_particle_movers_h(i, particle_mover_var::pmi);   \
    });\
  };

#define KOKKOS_INTERPOLATOR_VARIABLES() \
  int nv; \
  k_interpolator_t::HostMirror k_interpolator_h; \


#define KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE() \
  nv = interpolator_array->g->nv; \
  \
  k_interpolator_h = interpolator_array->k_i_h; \
  Kokkos::parallel_for("Copy interpolators to device", host_execution_policy(0, nv) , KOKKOS_LAMBDA (int i) { \
    k_interpolator_h(i, interpolator_var::ex)       = interpolator_array->i[i].ex; \
    k_interpolator_h(i, interpolator_var::ey)       = interpolator_array->i[i].ey; \
    k_interpolator_h(i, interpolator_var::ez)       = interpolator_array->i[i].ez; \
    k_interpolator_h(i, interpolator_var::dexdy)    = interpolator_array->i[i].dexdy; \
    k_interpolator_h(i, interpolator_var::dexdz)    = interpolator_array->i[i].dexdz; \
    k_interpolator_h(i, interpolator_var::d2exdydz) = interpolator_array->i[i].d2exdydz; \
    k_interpolator_h(i, interpolator_var::deydz)    = interpolator_array->i[i].deydz; \
    k_interpolator_h(i, interpolator_var::deydx)    = interpolator_array->i[i].deydx; \
    k_interpolator_h(i, interpolator_var::d2eydzdx) = interpolator_array->i[i].d2eydzdx; \
    k_interpolator_h(i, interpolator_var::dezdx)    = interpolator_array->i[i].dezdx; \
    k_interpolator_h(i, interpolator_var::dezdy)    = interpolator_array->i[i].dezdy; \
    k_interpolator_h(i, interpolator_var::d2ezdxdy) = interpolator_array->i[i].d2ezdxdy; \
    k_interpolator_h(i, interpolator_var::cbx)      = interpolator_array->i[i].cbx; \
    k_interpolator_h(i, interpolator_var::cby)      = interpolator_array->i[i].cby; \
    k_interpolator_h(i, interpolator_var::cbz)      = interpolator_array->i[i].cbz; \
    k_interpolator_h(i, interpolator_var::dcbxdx)   = interpolator_array->i[i].dcbxdx; \
    k_interpolator_h(i, interpolator_var::dcbydy)   = interpolator_array->i[i].dcbydy; \
    k_interpolator_h(i, interpolator_var::dcbzdz)   = interpolator_array->i[i].dcbzdz; \
  });\
  Kokkos::deep_copy(interpolator_array->k_i_d, interpolator_array->k_i_h);


#define KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST() \
  Kokkos::deep_copy(interpolator_array->k_i_h, interpolator_array->k_i_d);  \
  nv = interpolator_array->g->nv;; \
  k_interpolator_h = interpolator_array->k_i_h; \
  Kokkos::parallel_for("Copy interpolators to device", host_execution_policy(0, nv) , KOKKOS_LAMBDA (int i) { \
    interpolator_array->i[i].ex       = k_interpolator_h(i, interpolator_var::ex); \
    interpolator_array->i[i].ey       = k_interpolator_h(i, interpolator_var::ey); \
    interpolator_array->i[i].ez       = k_interpolator_h(i, interpolator_var::ez); \
    interpolator_array->i[i].dexdy    = k_interpolator_h(i, interpolator_var::dexdy); \
    interpolator_array->i[i].dexdz    = k_interpolator_h(i, interpolator_var::dexdz); \
    interpolator_array->i[i].d2exdydz = k_interpolator_h(i, interpolator_var::d2exdydz); \
    interpolator_array->i[i].deydz    = k_interpolator_h(i, interpolator_var::deydz); \
    interpolator_array->i[i].deydx    = k_interpolator_h(i, interpolator_var::deydx); \
    interpolator_array->i[i].d2eydzdx = k_interpolator_h(i, interpolator_var::d2eydzdx); \
    interpolator_array->i[i].dezdx    = k_interpolator_h(i, interpolator_var::dezdx); \
    interpolator_array->i[i].dezdy    = k_interpolator_h(i, interpolator_var::dezdy); \
    interpolator_array->i[i].d2ezdxdy = k_interpolator_h(i, interpolator_var::d2ezdxdy); \
    interpolator_array->i[i].cbx      = k_interpolator_h(i, interpolator_var::cbx); \
    interpolator_array->i[i].cby      = k_interpolator_h(i, interpolator_var::cby); \
    interpolator_array->i[i].cbz      = k_interpolator_h(i, interpolator_var::cbz); \
    interpolator_array->i[i].dcbxdx   = k_interpolator_h(i, interpolator_var::dcbxdx); \
    interpolator_array->i[i].dcbydy   = k_interpolator_h(i, interpolator_var::dcbydy); \
    interpolator_array->i[i].dcbzdz   = k_interpolator_h(i, interpolator_var::dcbzdz); \
  });


#define KOKKOS_ACCUMULATOR_VARIABLES() \
  int na; \
  k_accumulators_t::HostMirror k_accumulators_h;

#define KOKKOS_COPY_ACCUMULATOR_MEM_TO_DEVICE() \
  na = accumulator_array->na; \
  \
  k_accumulators_h = accumulator_array->k_a_h; \
  Kokkos::parallel_for("copy accumulator to device", KOKKOS_TEAM_POLICY_HOST \
      (na, Kokkos::AUTO),                          \
      KOKKOS_LAMBDA                                \
      (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) { \
    const unsigned int i = team_member.league_rank();              \
    /* TODO: Do we really need a 2d loop here*/ \
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) { \
      k_accumulators_h(i, accumulator_var::jx, j)       = accumulator_array->a[i].jx[j]; \
      k_accumulators_h(i, accumulator_var::jy, j)       = accumulator_array->a[i].jy[j]; \
      k_accumulators_h(i, accumulator_var::jz, j)       = accumulator_array->a[i].jz[j]; \
    }); \
  });\
  Kokkos::deep_copy(accumulator_array->k_a_d, accumulator_array->k_a_h);


#define KOKKOS_COPY_ACCUMULATOR_MEM_TO_HOST() \
  na = accumulator_array->na; \
  \
  k_accumulators_h = accumulator_array->k_a_h; \
  \
  Kokkos::deep_copy(accumulator_array->k_a_h, accumulator_array->k_a_d); \
  Kokkos::parallel_for("copy accumulator to host", KOKKOS_TEAM_POLICY_HOST \
      (na, Kokkos::AUTO),                          \
      KOKKOS_LAMBDA                                \
      (const KOKKOS_TEAM_POLICY_HOST::member_type &team_member) { \
    const unsigned int i = team_member.league_rank();              \
    \
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, ACCUMULATOR_ARRAY_LENGTH), [=] (int j) { \
      accumulator_array->a[i].jx[j] = k_accumulators_h(i, accumulator_var::jx, j); \
      accumulator_array->a[i].jy[j] = k_accumulators_h(i, accumulator_var::jy, j); \
      accumulator_array->a[i].jz[j] = k_accumulators_h(i, accumulator_var::jz, j); \
    }); \
  });

#endif // _kokkos_helpers_h_
