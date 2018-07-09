#include <list>
#include <Kokkos_Core.hpp>

// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
#define PARTICLE_VAR_COUNT 8
#define PARTICLE_MOVER_VAR_COUNT 4
#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4

using k_field_d_t = Kokkos::View<float *[FIELD_VAR_COUNT], Kokkos::DefaultExecutionSpace>;
using k_field_edge_d_t = Kokkos::View<material_id*[FIELD_EDGE_COUNT], Kokkos::DefaultExecutionSpace>;

using k_particles_d_t = Kokkos::View<float *[PARTICLE_VAR_COUNT], Kokkos::DefaultExecutionSpace>;
using k_particle_movers_d_t = Kokkos::View<float *[PARTICLE_MOVER_VAR_COUNT], Kokkos::DefaultExecutionSpace>;

#ifdef ENABLE_KOKKOS_CUDA 
  using k_particles_h_t = Kokkos::View<float *[PARTICLE_VAR_COUNT], Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>;
  using k_particle_movers_h_t = Kokkos::View<float *[PARTICLE_MOVER_VAR_COUNT],  Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace>;
#else
  using k_particles_h_t = k_particles_d_t;
  using k_particle_movers_h_t = k_particle_movers_d_t;
#endif
  
using k_accumulators_d_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], Kokkos::DefaultExecutionSpace>;

using static_sched = Kokkos::Schedule<Kokkos::Static>; 
using host_execution_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, static_sched, int>; 

typedef typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type k_member_t;

#define KOKKOS_ENUMS() \
enum field_var { \
  ex = 0, \
  ey = 1, \
  ez = 2, \
  div_e_err = 3, \
  cbx = 4, \
  cby = 5, \
  cbz = 6, \
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


#ifdef ENABLE_KOKKOS_CUDA 

  #define KOKKOS_FIELD_VIEW_INIT() \
    Kokkos::View<float*[FIELD_VAR_COUNT], Kokkos::LayoutLeft,\
        Kokkos::DefaultHostExecutionSpace> k_field_h \
        (Kokkos::ViewAllocateWithoutInitializing("k_field_h"), n_fields); \
    \
    Kokkos::View<material_id*[FIELD_EDGE_COUNT], Kokkos::LayoutLeft, \
        Kokkos::DefaultHostExecutionSpace> k_field_edge_h \
        (Kokkos::ViewAllocateWithoutInitializing("k_field_edge_h"), n_fields); \
    \
    k_field_d_t k_field_d = Kokkos::create_mirror_view_and_copy( \
        Kokkos::DefaultExecutionSpace(), k_field_h, "k_field_d"); \
    k_field_edge_d_t k_field_edge_d = Kokkos::create_mirror_view_and_copy( \
        Kokkos::DefaultExecutionSpace(), k_field_edge_h, "k_field_edge_d"); 


  #define KOKKOS_MEMORY_COPY_FIELD_TO_DEVICE() \
    Kokkos::deep_copy(k_field_d, k_field_h); \
    Kokkos::deep_copy(k_field_edge_d, k_field_edge_h);


  #define KOKKOS_MEMORY_COPY_FIELD_FROM_DEVICE() \
    Kokkos::deep_copy(k_field_h, k_field_d); \
    Kokkos::deep_copy(k_field_edge_h, k_field_edge_d); 

  
  #define KOKKOS_PARTICLE_VIEW_INIT() \
    sp->k_p_h = new k_particles_h_t(Kokkos::ViewAllocateWithoutInitializing("k_particles_h"), n_particles); \
    sp->k_p = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), k_particles_h, "k_particles_d"); \
    \
    sp->k_pm_h = new k_particle_movers_h_t(Kokkos::ViewAllocateWithoutInitializing("k_particle_movers_h"), max_pmovers); \
    sp->k_pm = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), k_particle_movers_h, "k_particle_movers_d");


  #define KOKKOS_MEMORY_COPY_PARTICLE_TO_DEVICE() \
    Kokkos::deep_copy(*(sp->k_p), *(sp->k_p_h));  \
    Kokkos::deep_copy(*(sp->k_pm), *(sp->k_pm_h)); 


  #define KOKKOS_MEMORY_COPY_PARTICLE_FROM_DEVICE() \
    Kokkos::deep_copy(*(sp->k_p_h), *(sp->k_p));  \
    Kokkos::deep_copy(*(sp->k_pm_h), *(sp->k_pm)); 


#else 

  #define KOKKOS_FIELD_VIEW_INIT() \
    k_field_d_t k_field_d ("k_field_d", n_fields); \
    k_field_edge_d_t k_field_edge_d ("k_field_edge_d", n_fields); \
    \
    k_field_d_t k_field_h = k_field_d; \
    k_field_edge_d_t k_field_edge_h = k_field_edge_d;


  #define KOKKOS_MEMORY_COPY_FIELD_TO_DEVICE()   


  #define KOKKOS_MEMORY_COPY_FIELD_FROM_DEVICE()  


  #define KOKKOS_PARTICLE_VIEW_INIT() \
    sp->k_p = new k_particles_d_t("k_particles_d", n_particles); \
    sp->k_pm = new k_particle_movers_d_t ("k_particle_movers_d", max_pmovers); \
    sp->k_p_h = sp->k_p; \
    sp->k_pm_h = sp->k_pm; 


  #define KOKKOS_MEMORY_COPY_PARTICLE_TO_DEVICE() 


  #define KOKKOS_MEMORY_COPY_PARTICLE_FROM_DEVICE()  


#endif


#define KOKKOS_FIELD_VARIABLES() \
  int n_fields = field_array->g->nv; \
  \
  KOKKOS_FIELD_VIEW_INIT() 

  
#define KOKKOS_COPY_FIELD_MEM_TO_DEVICE() \
  Kokkos::parallel_for(host_execution_policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
          k_field_h(i,ex) = field_array->f[i].ex; \
          k_field_h(i,ey) = field_array->f[i].ey; \
          k_field_h(i,ez) = field_array->f[i].ez; \
          k_field_h(i,div_e_err) = field_array->f[i].div_e_err; \
          \
          k_field_h(i,cbx) = field_array->f[i].cbx; \
          k_field_h(i,cby) = field_array->f[i].cby; \
          k_field_h(i,cbz) = field_array->f[i].cbz; \
          k_field_h(i,div_b_err) = field_array->f[i].div_b_err; \
          \
          k_field_h(i,tcax) = field_array->f[i].tcax; \
          k_field_h(i,tcay) = field_array->f[i].tcay; \
          k_field_h(i,tcaz) = field_array->f[i].tcaz; \
          k_field_h(i,rhob) = field_array->f[i].rhob; \
          \
          k_field_h(i,jfx) = field_array->f[i].jfx; \
          k_field_h(i,jfy) = field_array->f[i].jfy; \
          k_field_h(i,jfz) = field_array->f[i].jfz; \
          k_field_h(i,rhof) = field_array->f[i].rhof; \
          \
          k_field_edge_h(i, ematx) = field_array->f[i].ematx; \
          k_field_edge_h(i, ematy) = field_array->f[i].ematy; \
          k_field_edge_h(i, ematz) = field_array->f[i].ematz; \
          k_field_edge_h(i, nmat) = field_array->f[i].nmat; \
          \
          k_field_edge_h(i, fmatx) = field_array->f[i].fmatx; \
          k_field_edge_h(i, fmaty) = field_array->f[i].fmaty; \
          k_field_edge_h(i, fmatz) = field_array->f[i].fmatz; \
          k_field_edge_h(i, cmat) = field_array->f[i].cmat; \
  });     \
  KOKKOS_MEMORY_COPY_FIELD_TO_DEVICE()


#define KOKKOS_COPY_FIELD_MEM_TO_HOST() \
  KOKKOS_MEMORY_COPY_FIELD_FROM_DEVICE() \
  Kokkos::parallel_for(host_execution_policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
          field_array->f[i].ex = k_field_h(i,ex); \
          field_array->f[i].ey = k_field_h(i,ey); \
          field_array->f[i].ez = k_field_h(i,ez); \
          field_array->f[i].div_e_err = k_field_h(i,div_e_err); \
          \
          field_array->f[i].cbx = k_field_h(i,cbx); \
          field_array->f[i].cby = k_field_h(i,cby); \
          field_array->f[i].cbz = k_field_h(i,cbz); \
          field_array->f[i].div_b_err = k_field_h(i,div_b_err); \
          \
          field_array->f[i].tcax = k_field_h(i,tcax); \
          field_array->f[i].tcay = k_field_h(i,tcay); \
          field_array->f[i].tcaz = k_field_h(i,tcaz); \
          field_array->f[i].rhob = k_field_h(i,rhob); \
          \
          field_array->f[i].jfx = k_field_h(i,jfx); \
          field_array->f[i].jfy = k_field_h(i,jfy); \
          field_array->f[i].jfz = k_field_h(i,jfz); \
          field_array->f[i].rhof = k_field_h(i,rhof); \
          \
          field_array->f[i].ematx = k_field_edge_h(i, ematx); \
          field_array->f[i].ematy = k_field_edge_h(i, ematy); \
          field_array->f[i].ematz = k_field_edge_h(i, ematz); \
          field_array->f[i].nmat = k_field_edge_h(i, nmat); \
          \
          field_array->f[i].fmatx = k_field_edge_h(i, fmatx); \
          field_array->f[i].fmaty = k_field_edge_h(i, fmaty); \
          field_array->f[i].fmatz = k_field_edge_h(i, fmatz); \
          field_array->f[i].cmat = k_field_edge_h(i, cmat); \
  });


#define KOKKOS_PARTICLE_VARIABLES() \
  int n_particles; \
  int max_pmovers; \
  int n_pmovers; \
  \
  k_particles_h_t k_particles_h; \
  k_particle_movers_h_t k_particle_movers_h; \
  \
  LIST_FOR_EACH( sp, species_list ) {\
    n_particles = sp->np; \
    if (n_particles % 32) n_particles += (32 - (n_particles % 32)); \
    max_pmovers = sp->max_nm; \
    n_pmovers = sp->nm; \
    \
    KOKKOS_PARTICLE_VIEW_INIT() \
  };


#define KOKKOS_COPY_PARTICLE_MEM_TO_DEVICE() \
  LIST_FOR_EACH( sp, species_list ) {\
    n_particles = sp->np; \
    n_pmovers = sp->nm; \
    max_pmovers = sp->max_nm; \
    \
    k_particles_h = *sp->k_p_h; \
    k_particle_movers_h = *sp->k_pm_h; \
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
    KOKKOS_MEMORY_COPY_PARTICLE_TO_DEVICE(); \
  };


#define KOKKOS_COPY_PARTICLE_MEM_TO_HOST() \
  LIST_FOR_EACH( sp, species_list ) {\
    KOKKOS_MEMORY_COPY_PARTICLE_FROM_DEVICE(); \
    n_particles = sp->np; \
    n_pmovers = sp->nm; \
    max_pmovers = sp->max_nm; \
    k_particles_h = *sp->k_p_h; \
    k_particle_movers_h = *sp->k_pm_h; \
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

