#define IN_field_advance
#include "field_advance_private.h"

void
delete_field_array( field_array_t * fa ) {
  if( !fa ) return;
  fa->kernel->delete_fa( fa );
}

void
checkpt_field_advance_kernels( const field_advance_kernels_t * kernel ) {
  CHECKPT_SYM( kernel->delete_fa                 );
  CHECKPT_SYM( kernel->advance_b                 );
  CHECKPT_SYM( kernel->advance_e                 );
  CHECKPT_SYM( kernel->energy_f                  );
  CHECKPT_SYM( kernel->clear_jf                  );
  CHECKPT_SYM( kernel->synchronize_jf            );
  CHECKPT_SYM( kernel->clear_rhof                );
  CHECKPT_SYM( kernel->synchronize_rho           );
  CHECKPT_SYM( kernel->compute_rhob              );
  CHECKPT_SYM( kernel->compute_curl_b            );
  CHECKPT_SYM( kernel->synchronize_tang_e_norm_b );
  CHECKPT_SYM( kernel->compute_div_e_err         );
  CHECKPT_SYM( kernel->compute_rms_div_e_err     );
  CHECKPT_SYM( kernel->clean_div_e               );
  CHECKPT_SYM( kernel->compute_div_b_err         );
  CHECKPT_SYM( kernel->compute_rms_div_b_err     );
  CHECKPT_SYM( kernel->clean_div_b               );
}

void
restore_field_advance_kernels( field_advance_kernels_t * kernel ) {
  RESTORE_SYM( kernel->delete_fa                 );
  RESTORE_SYM( kernel->advance_b                 );
  RESTORE_SYM( kernel->advance_e                 );
  RESTORE_SYM( kernel->energy_f                  );
  RESTORE_SYM( kernel->clear_jf                  );
  RESTORE_SYM( kernel->synchronize_jf            );
  RESTORE_SYM( kernel->clear_rhof                );
  RESTORE_SYM( kernel->synchronize_rho           );
  RESTORE_SYM( kernel->compute_rhob              );
  RESTORE_SYM( kernel->compute_curl_b            );
  RESTORE_SYM( kernel->synchronize_tang_e_norm_b );
  RESTORE_SYM( kernel->compute_div_e_err         );
  RESTORE_SYM( kernel->compute_rms_div_e_err     );
  RESTORE_SYM( kernel->clean_div_e               );
  RESTORE_SYM( kernel->compute_div_b_err         );
  RESTORE_SYM( kernel->compute_rms_div_b_err     );
  RESTORE_SYM( kernel->clean_div_b               );
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

  Kokkos::deep_copy(k_f_h, k_f_d);
  Kokkos::deep_copy(k_fe_h, k_fe_d);

}
