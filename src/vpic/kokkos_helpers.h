#include <Kokkos_Core.hpp>

#define USE_KOKKOS
// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
typedef Kokkos::View<float *[FIELD_VAR_COUNT], Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> k_field_d_t;
typedef Kokkos::View<material_id*[FIELD_EDGE_COUNT], Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> k_field_edge_d_t;

#define KOKKOS_ENUMS \
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
};

#define KOKKOS_VARIABLES \
  using StaticSched = Kokkos::Schedule<Kokkos::Static>; \
  using Policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, StaticSched, int>; \
  int n_fields = (field_array->g)->nv; \
  int initial_view_bytes = 100000; \
  \
  Kokkos::View<float*[FIELD_VAR_COUNT], Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace> k_field_h (Kokkos::ViewAllocateWithoutInitializing("k_field_h"), initial_view_bytes); \
  Kokkos::View<material_id*[FIELD_EDGE_COUNT], Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace> k_field_edge_h (Kokkos::ViewAllocateWithoutInitializing("k_field_edge_h"), initial_view_bytes); \
  k_field_d_t k_field_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), k_field_h, "k_field_d"); \
  k_field_edge_d_t k_field_edge_d = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), k_field_edge_h, "k_field_edge_d"); \
/*
  field_array->f[0].ex = 10.0; \
  field_array->f[0].ey = 11.0; \
  field_array->f[0].ez = 12.0; 
*/
  
#define KOKKOS_COPY_MEM_TO_DEVICE() \
  Kokkos::parallel_for(Policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
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
  Kokkos::deep_copy(k_field_d, k_field_h); \
  Kokkos::deep_copy(k_field_edge_d, k_field_edge_h); \


          //printf("before deep copy host %f, %f, %f\n", k_field_h(i,ex), k_field_h(i,ey), k_field_h(i,ez)); 
          //printf("after deep copy host %f, %f, %f\n", k_field_h(i,ex), k_field_h(i,ey), k_field_h(i,ez)); 
          // printf("final field array %f, %f, %f\n", field_array->f[i].ex, field_array->f[i].ey, field_array->f[i].ez); 
          
#define KOKKOS_COPY_MEM_TO_HOST() \
  Kokkos::deep_copy(k_field_h, k_field_d); \
  Kokkos::deep_copy(k_field_edge_h, k_field_edge_d); \
  \
  Kokkos::parallel_for(Policy(0, n_fields - 1) , KOKKOS_LAMBDA (int i) { \
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
