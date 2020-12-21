#define IN_field_advance
#include "field_advance_private.h"

void
checkpt_field_array_internal( field_array_t * fa ) {

  fa->copy_to_host();

  CHECKPT( fa, 1 );
  CHECKPT_ALIGNED( fa->f, fa->g->nv, 128 );
  CHECKPT_PTR( fa->g );
  CHECKPT_PTR( fa->fb );

  CHECKPT_SYM( fa->kernel->delete_fa                 );
  CHECKPT_SYM( fa->kernel->advance_b                 );
  CHECKPT_SYM( fa->kernel->advance_e                 );
  CHECKPT_SYM( fa->kernel->energy_f                  );
  CHECKPT_SYM( fa->kernel->clear_jf                  );
  CHECKPT_SYM( fa->kernel->synchronize_jf            );
  CHECKPT_SYM( fa->kernel->clear_rhof                );
  CHECKPT_SYM( fa->kernel->synchronize_rho           );
  CHECKPT_SYM( fa->kernel->compute_rhob              );
  CHECKPT_SYM( fa->kernel->compute_curl_b            );
  CHECKPT_SYM( fa->kernel->synchronize_tang_e_norm_b );
  CHECKPT_SYM( fa->kernel->compute_div_e_err         );
  CHECKPT_SYM( fa->kernel->compute_rms_div_e_err     );
  CHECKPT_SYM( fa->kernel->clean_div_e               );
  CHECKPT_SYM( fa->kernel->compute_div_b_err         );
  CHECKPT_SYM( fa->kernel->compute_rms_div_b_err     );
  CHECKPT_SYM( fa->kernel->clean_div_b               );

  CHECKPT_VIEW( fa->k_f_d );
  CHECKPT_VIEW( fa->k_f_h );
  CHECKPT_VIEW( fa->k_fe_h );
  CHECKPT_VIEW( fa->k_fe_d );
  CHECKPT_VIEW( fa->k_f_rhob_accum_d );
  CHECKPT_VIEW( fa->k_f_rhob_accum_h );

}

field_array_t *
restore_field_array_internal( void * params ) {

  field_array_t * fa;
  RESTORE( fa );

  RESTORE_ALIGNED( fa->f );
  RESTORE_PTR( fa->g );
  RESTORE_PTR( fa->fb );
  fa->params = params;

  RESTORE_SYM( fa->kernel->delete_fa                 );
  RESTORE_SYM( fa->kernel->advance_b                 );
  RESTORE_SYM( fa->kernel->advance_e                 );
  RESTORE_SYM( fa->kernel->energy_f                  );
  RESTORE_SYM( fa->kernel->clear_jf                  );
  RESTORE_SYM( fa->kernel->synchronize_jf            );
  RESTORE_SYM( fa->kernel->clear_rhof                );
  RESTORE_SYM( fa->kernel->synchronize_rho           );
  RESTORE_SYM( fa->kernel->compute_rhob              );
  RESTORE_SYM( fa->kernel->compute_curl_b            );
  RESTORE_SYM( fa->kernel->synchronize_tang_e_norm_b );
  RESTORE_SYM( fa->kernel->compute_div_e_err         );
  RESTORE_SYM( fa->kernel->compute_rms_div_e_err     );
  RESTORE_SYM( fa->kernel->clean_div_e               );
  RESTORE_SYM( fa->kernel->compute_div_b_err         );
  RESTORE_SYM( fa->kernel->compute_rms_div_b_err     );
  RESTORE_SYM( fa->kernel->clean_div_b               );

  RESTORE_VIEW( &fa->k_f_d );
  RESTORE_VIEW( &fa->k_f_h );
  RESTORE_VIEW( &fa->k_fe_d );
  RESTORE_VIEW( &fa->k_fe_h );
  RESTORE_VIEW( &fa->k_f_rhob_accum_d );
  RESTORE_VIEW( &fa->k_f_rhob_accum_h );

  fa->copy_to_device();

  return fa;

}

field_array::field_array(
  int n_fields,
  int xyz_sz,
  int yzx_sz,
  int zxy_sz
)
{

  k_f_d = k_field_t("k_fields", n_fields);
  k_fe_d = k_field_edge_t("k_field_edges", n_fields);
  k_f_h = Kokkos::create_mirror_view(k_f_d);
  k_fe_h = Kokkos::create_mirror_view(k_fe_d);

  k_f_rhob_accum_d = k_field_accum_t("k_rhob_accum", n_fields);
  k_f_rhob_accum_h = Kokkos::create_mirror_view(k_f_rhob_accum_d);

  fb = new field_buffers_t(xyz_sz, yzx_sz, zxy_sz);

}

void
delete_field_array( field_array_t * fa ) {
  if( !fa ) return;
  delete fa->fb;
  fa->kernel->delete_fa( fa );
}

void
field_array_t::copy_to_host() {

  Kokkos::deep_copy(k_f_h, k_f_d);
  Kokkos::deep_copy(k_fe_h, k_fe_d);

  // Avoid capturing this
  auto& k_field = k_f_h;
  auto& k_field_edge = k_fe_h;
  field_t * host_field = f;

  Kokkos::parallel_for("copy field to host",
    host_execution_policy(0, g->nv - 1) ,
    KOKKOS_LAMBDA (int i) {

      host_field[i].ex = k_field(i, field_var::ex);
      host_field[i].ey = k_field(i, field_var::ey);
      host_field[i].ez = k_field(i, field_var::ez);
      host_field[i].div_e_err = k_field(i, field_var::div_e_err);

      host_field[i].cbx = k_field(i, field_var::cbx);
      host_field[i].cby = k_field(i, field_var::cby);
      host_field[i].cbz = k_field(i, field_var::cbz);
      host_field[i].div_b_err = k_field(i, field_var::div_b_err);

      host_field[i].tcax = k_field(i, field_var::tcax);
      host_field[i].tcay = k_field(i, field_var::tcay);
      host_field[i].tcaz = k_field(i, field_var::tcaz);
      host_field[i].rhob = k_field(i, field_var::rhob);

      host_field[i].jfx = k_field(i, field_var::jfx);
      host_field[i].jfy = k_field(i, field_var::jfy);
      host_field[i].jfz = k_field(i, field_var::jfz);
      host_field[i].rhof = k_field(i, field_var::rhof);

      host_field[i].ematx = k_field_edge(i, field_edge_var::ematx);
      host_field[i].ematy = k_field_edge(i, field_edge_var::ematy);
      host_field[i].ematz = k_field_edge(i, field_edge_var::ematz);
      host_field[i].nmat = k_field_edge(i, field_edge_var::nmat);

      host_field[i].fmatx = k_field_edge(i, field_edge_var::fmatx);
      host_field[i].fmaty = k_field_edge(i, field_edge_var::fmaty);
      host_field[i].fmatz = k_field_edge(i, field_edge_var::fmatz);
      host_field[i].cmat = k_field_edge(i, field_edge_var::cmat);

    });

}

void
field_array_t::copy_to_device() {

  // Avoid capturing this
  auto& k_field = k_f_h;
  auto& k_field_edge = k_fe_h;
  field_t * host_field = f;

  Kokkos::parallel_for("copy field to device",
    host_execution_policy(0, g->nv - 1) ,
    KOKKOS_LAMBDA (int i) {

      k_field(i, field_var::ex) = host_field[i].ex;
      k_field(i, field_var::ey) = host_field[i].ey;
      k_field(i, field_var::ez) = host_field[i].ez;
      k_field(i, field_var::div_e_err) = host_field[i].div_e_err;

      k_field(i, field_var::cbx) = host_field[i].cbx;
      k_field(i, field_var::cby) = host_field[i].cby;
      k_field(i, field_var::cbz) = host_field[i].cbz;
      k_field(i, field_var::div_b_err) = host_field[i].div_b_err;

      k_field(i, field_var::tcax) = host_field[i].tcax;
      k_field(i, field_var::tcay) = host_field[i].tcay;
      k_field(i, field_var::tcaz) = host_field[i].tcaz;
      k_field(i, field_var::rhob) = host_field[i].rhob;

      k_field(i, field_var::jfx) = host_field[i].jfx;
      k_field(i, field_var::jfy) = host_field[i].jfy;
      k_field(i, field_var::jfz) = host_field[i].jfz;
      k_field(i, field_var::rhof) = host_field[i].rhof;

      k_field_edge(i, field_edge_var::ematx) = host_field[i].ematx;
      k_field_edge(i, field_edge_var::ematy) = host_field[i].ematy;
      k_field_edge(i, field_edge_var::ematz) = host_field[i].ematz;
      k_field_edge(i, field_edge_var::nmat) = host_field[i].nmat;

      k_field_edge(i, field_edge_var::fmatx) = host_field[i].fmatx;
      k_field_edge(i, field_edge_var::fmaty) = host_field[i].fmaty;
      k_field_edge(i, field_edge_var::fmatz) = host_field[i].fmatz;
      k_field_edge(i, field_edge_var::cmat) = host_field[i].cmat;

    });

  Kokkos::deep_copy(k_f_d, k_f_h);
  Kokkos::deep_copy(k_fe_d, k_fe_h);

}
