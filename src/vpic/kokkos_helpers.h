#include <Kokkos_Core.hpp>

// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
#define PARTICLE_VAR_COUNT 8
#define PARTICLE_MOVER_VAR_COUNT 4
#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4
#define INTERPOLATOR_VAR_COUNT 18


using k_field_t = Kokkos::View<float *[FIELD_VAR_COUNT]>;
using k_field_edge_t = Kokkos::View<material_id*[FIELD_EDGE_COUNT]>;

using k_particles_t = Kokkos::View<float *[PARTICLE_VAR_COUNT]>;
using k_particle_movers_t = Kokkos::View<float *[PARTICLE_MOVER_VAR_COUNT]>;

using k_interpolator_t = Kokkos::View<float *[INTERPOLATOR_VAR_COUNT]>;

using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using static_sched = Kokkos::Schedule<Kokkos::Static>; 
using host_execution_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, static_sched, int>; 

typedef typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type k_member_t;

#define KOKKOS_ENUMS() \
enum field_var { \
  ex = 0, \
  ey = 1, \
  ez = 2, \
  cbx = 3, \
  cby = 4, \
  cbz = 5, \
  div_e_err = 6, \
  div_b_err = 7, \
  tcax = 8, \
  tcay = 9, \
  tcaz = 10, \
  rhob = 11, \
  jfx = 12, \
  jfy = 13, \
  jfz = 14, \
  rhof = 15 \
}; \
\
/* Use first 6 params from 
 field_var for interpolator */ \
enum interpolator { \
  dexdy = 6, \
  dexdz = 7, \
  d2exdydz = 8, \
  deydz = 9, \
  deydx = 10, \
  d2eydzdx = 11, \
  dezdx = 12, \
  dezdy = 13, \
  d2ezdxdy = 14, \
  dcbxdx = 15, \
  dcbydy = 16, \
  dcbzdz = 17 \
}; \
\
enum field_edge_var { \
  ematx = 0, \
  ematy = 1, \
  ematz = 2, \
  nmat = 3, \
  fmatx = 4, \
  fmaty = 5, \
  fmatz = 6, \
  cmat = 7 \
}; \
\
enum particles_var { \
  dx = 0, \
  dy = 1, \
  dz = 2, \
  ux = 3, \
  uy = 4, \
  uz = 5, \
  w = 6, \
  pi = 7 \
}; \
\
enum particle_mover_var { \
  dispx = 0, \
  dispy = 1, \
  dispz = 2, \
  pmi = 3, \
}; \
\
enum accumulator_var { \
  jx = 0, \
  jy = 1, \
  jz = 2, \
};

#define KOKKOS_FIELD_VARIABLES() \
  int n_fields = field_array->g->nv; \
  k_field_t::HostMirror k_field; \
  k_field_edge_t::HostMirror k_field_edge; \

  
#define KOKKOS_COPY_FIELD_MEM_TO_DEVICE() \
  k_field = field_array->k_f_h; \
  k_field_edge = field_array->k_fe_h; \
  Kokkos::parallel_for(host_execution_policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
          k_field(i,ex) = field_array->f[i].ex; \
          k_field(i,ey) = field_array->f[i].ey; \
          k_field(i,ez) = field_array->f[i].ez; \
          k_field(i,div_e_err) = field_array->f[i].div_e_err; \
          \
          k_field(i,cbx) = field_array->f[i].cbx; \
          k_field(i,cby) = field_array->f[i].cby; \
          k_field(i,cbz) = field_array->f[i].cbz; \
          k_field(i,div_b_err) = field_array->f[i].div_b_err; \
          \
          k_field(i,tcax) = field_array->f[i].tcax; \
          k_field(i,tcay) = field_array->f[i].tcay; \
          k_field(i,tcaz) = field_array->f[i].tcaz; \
          k_field(i,rhob) = field_array->f[i].rhob; \
          \
          k_field(i,jfx) = field_array->f[i].jfx; \
          k_field(i,jfy) = field_array->f[i].jfy; \
          k_field(i,jfz) = field_array->f[i].jfz; \
          k_field(i,rhof) = field_array->f[i].rhof; \
          \
          k_field_edge(i, ematx) = field_array->f[i].ematx; \
          k_field_edge(i, ematy) = field_array->f[i].ematy; \
          k_field_edge(i, ematz) = field_array->f[i].ematz; \
          k_field_edge(i, nmat) = field_array->f[i].nmat; \
          \
          k_field_edge(i, fmatx) = field_array->f[i].fmatx; \
          k_field_edge(i, fmaty) = field_array->f[i].fmaty; \
          k_field_edge(i, fmatz) = field_array->f[i].fmatz; \
          k_field_edge(i, cmat) = field_array->f[i].cmat; \
  });     \
  Kokkos::deep_copy(field_array->k_f_d, field_array->k_f_h); \
  Kokkos::deep_copy(field_array->k_fe_d, field_array->k_fe_h);


#define KOKKOS_COPY_FIELD_MEM_TO_HOST() \
  Kokkos::deep_copy(field_array->k_f_h, field_array->k_f_d); \
  Kokkos::deep_copy(field_array->k_fe_h, field_array->k_fe_d); \
  k_field = field_array->k_f_h; \
  k_field_edge = field_array->k_fe_h; \
  Kokkos::parallel_for(host_execution_policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
          field_array->f[i].ex = k_field(i,ex); \
          field_array->f[i].ey = k_field(i,ey); \
          field_array->f[i].ez = k_field(i,ez); \
          field_array->f[i].div_e_err = k_field(i,div_e_err); \
          \
          field_array->f[i].cbx = k_field(i,cbx); \
          field_array->f[i].cby = k_field(i,cby); \
          field_array->f[i].cbz = k_field(i,cbz); \
          field_array->f[i].div_b_err = k_field(i,div_b_err); \
          \
          field_array->f[i].tcax = k_field(i,tcax); \
          field_array->f[i].tcay = k_field(i,tcay); \
          field_array->f[i].tcaz = k_field(i,tcaz); \
          field_array->f[i].rhob = k_field(i,rhob); \
          \
          field_array->f[i].jfx = k_field(i,jfx); \
          field_array->f[i].jfy = k_field(i,jfy); \
          field_array->f[i].jfz = k_field(i,jfz); \
          field_array->f[i].rhof = k_field(i,rhof); \
          \
          field_array->f[i].ematx = k_field_edge(i, ematx); \
          field_array->f[i].ematy = k_field_edge(i, ematy); \
          field_array->f[i].ematz = k_field_edge(i, ematz); \
          field_array->f[i].nmat = k_field_edge(i, nmat); \
          \
          field_array->f[i].fmatx = k_field_edge(i, fmatx); \
          field_array->f[i].fmaty = k_field_edge(i, fmaty); \
          field_array->f[i].fmatz = k_field_edge(i, fmatz); \
          field_array->f[i].cmat = k_field_edge(i, cmat); \
  });


#define KOKKOS_PARTICLE_VARIABLES() \
  int n_particles; \
  int max_pmovers; \
  \
  k_particles_t::HostMirror k_particles_h; \
  k_particle_movers_t::HostMirror k_particle_movers_h; 


#define KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE() \
  LIST_FOR_EACH( sp, species_list ) {\
    n_particles = sp->max_np; \
    max_pmovers = sp->max_nm; \
    \
    k_particles_h = sp->k_p_h; \
    k_particle_movers_h = sp->k_pm_h; \
    Kokkos::parallel_for(host_execution_policy(0, n_particles) , KOKKOS_LAMBDA (int i) { \
      k_particles_h(i,dx) = sp->p[i].dx; \
      k_particles_h(i,dy) = sp->p[i].dy; \
      k_particles_h(i,dz) = sp->p[i].dz; \
      k_particles_h(i,ux) = sp->p[i].ux; \
      k_particles_h(i,uy) = sp->p[i].uy; \
      k_particles_h(i,uz) = sp->p[i].uz; \
      k_particles_h(i,w)  = sp->p[i].w;  \
      k_particles_h(i,pi) = sp->p[i].i;  \
    });\
    \
    Kokkos::parallel_for(host_execution_policy(0, max_pmovers) , KOKKOS_LAMBDA (int i) { \
      k_particle_movers_h(i,dispx) = sp->pm[i].dispx; \
      k_particle_movers_h(i,dispy) = sp->pm[i].dispy; \
      k_particle_movers_h(i,dispz) = sp->pm[i].dispz; \
      k_particle_movers_h(i,pmi)   = sp->pm[i].i;     \
    });\
    Kokkos::deep_copy(sp->k_p_d, sp->k_p_h);  \
    Kokkos::deep_copy(sp->k_pm_d, sp->k_pm_h); \
  };


#define KOKKOS_COPY_PARTICLE_MEM_TO_HOST() \
  LIST_FOR_EACH( sp, species_list ) {\
    Kokkos::deep_copy(sp->k_p_h, sp->k_p_d);  \
    Kokkos::deep_copy(sp->k_pm_h, sp->k_pm_d); \
    n_particles = sp->max_np; \
    max_pmovers = sp->max_nm; \
    k_particles_h = sp->k_p_h; \
    k_particle_movers_h = sp->k_pm_h; \
    Kokkos::parallel_for(host_execution_policy(0, n_particles) , KOKKOS_LAMBDA (int i) { \
      sp->p[i].dx = k_particles_h(i,dx); \
      sp->p[i].dy = k_particles_h(i,dy); \
      sp->p[i].dz = k_particles_h(i,dz); \
      sp->p[i].ux = k_particles_h(i,ux); \
      sp->p[i].uy = k_particles_h(i,uy); \
      sp->p[i].uz = k_particles_h(i,uz); \
      sp->p[i].w  = k_particles_h(i,w);  \
      sp->p[i].i  = k_particles_h(i,pi); \
    });\
    \
    Kokkos::parallel_for(host_execution_policy(0, max_pmovers) , KOKKOS_LAMBDA (int i) { \
      sp->pm[i].dispx = k_particle_movers_h(i,dispx); \
      sp->pm[i].dispy = k_particle_movers_h(i,dispy); \
      sp->pm[i].dispz = k_particle_movers_h(i,dispz); \
      sp->pm[i].i     = k_particle_movers_h(i,pmi);   \
    });\
  };

#define KOKKOS_INTERPOLATOR_VARIABLES() \
  int nv; \
  k_interpolator_t::HostMirror k_interpolator_h; \


#define KOKKOS_COPY_INTERPOLATOR_MEM_TO_DEVICE() \
  nv = interpolator_array->g->nv; \
  \
  k_interpolator_h = interpolator_array->k_i_h; \
  Kokkos::parallel_for(host_execution_policy(0, nv) , KOKKOS_LAMBDA (int i) { \
    k_interpolator_h(i, ex)       = interpolator_array->i[i].ex; \
    k_interpolator_h(i, ey)       = interpolator_array->i[i].ey; \
    k_interpolator_h(i, ez)       = interpolator_array->i[i].ez; \
    k_interpolator_h(i, dexdy)    = interpolator_array->i[i].dexdy; \
    k_interpolator_h(i, dexdz)    = interpolator_array->i[i].dexdz; \
    k_interpolator_h(i, d2exdydz) = interpolator_array->i[i].d2exdydz; \
    k_interpolator_h(i, deydz)    = interpolator_array->i[i].deydz; \
    k_interpolator_h(i, deydx)    = interpolator_array->i[i].deydx; \
    k_interpolator_h(i, d2eydzdx) = interpolator_array->i[i].d2eydzdx; \
    k_interpolator_h(i, dezdx)    = interpolator_array->i[i].dezdx; \
    k_interpolator_h(i, dezdy)    = interpolator_array->i[i].dezdy; \
    k_interpolator_h(i, d2ezdxdy) = interpolator_array->i[i].d2ezdxdy; \
    k_interpolator_h(i, cbx)      = interpolator_array->i[i].cbx; \
    k_interpolator_h(i, cby)      = interpolator_array->i[i].cby; \
    k_interpolator_h(i, cbz)      = interpolator_array->i[i].cbz; \
    k_interpolator_h(i, dcbxdx)   = interpolator_array->i[i].dcbxdx; \
    k_interpolator_h(i, dcbydy)   = interpolator_array->i[i].dcbydy; \
    k_interpolator_h(i, dcbzdz)   = interpolator_array->i[i].dcbzdz; \
  });\
  Kokkos::deep_copy(interpolator_array->k_i_d, interpolator_array->k_i_h);  


#define KOKKOS_COPY_INTERPOLATOR_MEM_TO_HOST() \
  Kokkos::deep_copy(interpolator_array->k_i_h, interpolator_array->k_i_d);  \
  nv = interpolator_array->g->nv;; \
  k_interpolator_h = interpolator_array->k_i_h; \
  Kokkos::parallel_for(host_execution_policy(0, nv) , KOKKOS_LAMBDA (int i) { \
    interpolator_array->i[i].ex       = k_interpolator_h(i, ex); \
    interpolator_array->i[i].ey       = k_interpolator_h(i, ey); \
    interpolator_array->i[i].ez       = k_interpolator_h(i, ez); \
    interpolator_array->i[i].dexdy    = k_interpolator_h(i, dexdy); \
    interpolator_array->i[i].dexdz    = k_interpolator_h(i, dexdz); \
    interpolator_array->i[i].d2exdydz = k_interpolator_h(i, d2exdydz); \
    interpolator_array->i[i].deydz    = k_interpolator_h(i, deydz); \
    interpolator_array->i[i].deydx    = k_interpolator_h(i, deydx); \
    interpolator_array->i[i].d2eydzdx = k_interpolator_h(i, d2eydzdx); \
    interpolator_array->i[i].dezdx    = k_interpolator_h(i, dezdx); \
    interpolator_array->i[i].dezdy    = k_interpolator_h(i, dezdy); \
    interpolator_array->i[i].d2ezdxdy = k_interpolator_h(i, d2ezdxdy); \
    interpolator_array->i[i].cbx      = k_interpolator_h(i, cbx); \
    interpolator_array->i[i].cby      = k_interpolator_h(i, cby); \
    interpolator_array->i[i].cbz      = k_interpolator_h(i, cbz); \
    interpolator_array->i[i].dcbxdx   = k_interpolator_h(i, dcbxdx); \
    interpolator_array->i[i].dcbydy   = k_interpolator_h(i, dcbydy); \
    interpolator_array->i[i].dcbzdz   = k_interpolator_h(i, dcbzdz); \
  });

