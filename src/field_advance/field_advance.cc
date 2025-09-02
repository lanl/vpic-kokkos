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
  CHECKPT_SYM( kernel->hyb_smooth_b              );
  CHECKPT_SYM( kernel->hyb_smooth_eb_interp      );
  CHECKPT_SYM( kernel->energy_f                  );
  CHECKPT_SYM( kernel->clear_jf                  );
  CHECKPT_SYM( kernel->synchronize_jf            );
  CHECKPT_SYM( kernel->clear_rhof                );
  CHECKPT_SYM( kernel->synchronize_rho           );
  CHECKPT_SYM( kernel->compute_rhob              );
  CHECKPT_SYM( kernel->compute_curl_b            );
#ifdef HYB_USE_SEPARATE_PE
  CHECKPT_SYM( kernel->hyb_init                );
#endif
  CHECKPT_SYM( kernel->synchronize_tang_e_norm_b );
  CHECKPT_SYM( kernel->compute_div_e_err         );
  CHECKPT_SYM( kernel->compute_rms_div_e_err     );
  CHECKPT_SYM( kernel->clean_div_e               );
  CHECKPT_SYM( kernel->compute_div_b_err         );
  CHECKPT_SYM( kernel->compute_rms_div_b_err     );
  CHECKPT_SYM( kernel->clean_div_b               );

  // Checkpoint Kokkos Specific kernels
  CHECKPT_SYM( kernel->advance_e_kokkos                 );
  CHECKPT_SYM( kernel->energy_f_kokkos                  );
  CHECKPT_SYM( kernel->clear_jf_kokkos                  );
  CHECKPT_SYM( kernel->clear_rhof_kokkos                );

  CHECKPT_SYM( kernel->k_synchronize_jf                 );
  CHECKPT_SYM( kernel->k_reduce_jf                      );
  CHECKPT_SYM( kernel->k_synchronize_rho                );

  CHECKPT_SYM( kernel->synchronize_tang_e_norm_b_kokkos );

  CHECKPT_SYM( kernel->compute_div_e_err_kokkos         );
  CHECKPT_SYM( kernel->compute_rms_div_e_err_kokkos     );
  CHECKPT_SYM( kernel->clean_div_e_kokkos               );

  CHECKPT_SYM( kernel->compute_div_b_err_kokkos         );
  CHECKPT_SYM( kernel->compute_rms_div_b_err_kokkos     );
  CHECKPT_SYM( kernel->clean_div_b_kokkos               );
}

void
restore_field_advance_kernels( field_advance_kernels_t * kernel ) {
  RESTORE_SYM( kernel->delete_fa                 );
  RESTORE_SYM( kernel->advance_b                 );
  RESTORE_SYM( kernel->advance_e                 );
  RESTORE_SYM( kernel->hyb_smooth_b              );
  RESTORE_SYM( kernel->hyb_smooth_eb_interp      );
  RESTORE_SYM( kernel->energy_f                  );
  RESTORE_SYM( kernel->clear_jf                  );
  RESTORE_SYM( kernel->synchronize_jf            );
  RESTORE_SYM( kernel->clear_rhof                );
  RESTORE_SYM( kernel->synchronize_rho           );
  RESTORE_SYM( kernel->compute_rhob              );
  RESTORE_SYM( kernel->compute_curl_b            );
#ifdef HYB_USE_SEPARATE_PE
  RESTORE_SYM( kernel->hyb_init                  );
#endif
  RESTORE_SYM( kernel->synchronize_tang_e_norm_b );
  RESTORE_SYM( kernel->compute_div_e_err         );
  RESTORE_SYM( kernel->compute_rms_div_e_err     );
  RESTORE_SYM( kernel->clean_div_e               );
  RESTORE_SYM( kernel->compute_div_b_err         );
  RESTORE_SYM( kernel->compute_rms_div_b_err     );
  RESTORE_SYM( kernel->clean_div_b               );


  // Restore Kokkos Kernels
  RESTORE_SYM( kernel->advance_e_kokkos                 );
  RESTORE_SYM( kernel->energy_f_kokkos                  );
  RESTORE_SYM( kernel->clear_jf_kokkos                  );
  RESTORE_SYM( kernel->clear_rhof_kokkos                );

  RESTORE_SYM( kernel->k_synchronize_jf                 );
  RESTORE_SYM( kernel->k_reduce_jf                      );
  RESTORE_SYM( kernel->k_synchronize_rho                );

  RESTORE_SYM( kernel->synchronize_tang_e_norm_b_kokkos );

  RESTORE_SYM( kernel->compute_div_e_err_kokkos         );
  RESTORE_SYM( kernel->compute_rms_div_e_err_kokkos     );
  RESTORE_SYM( kernel->clean_div_e_kokkos               );

  RESTORE_SYM( kernel->compute_div_b_err_kokkos         );
  RESTORE_SYM( kernel->compute_rms_div_b_err_kokkos     );
  RESTORE_SYM( kernel->clean_div_b_kokkos               );
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
      host_field[i].pe  = k_field(i, field_var::pe);
      
      host_field[i].cbx0 = k_field(i, field_var::cbx0);
      host_field[i].cby0 = k_field(i, field_var::cby0);
      host_field[i].cbz0 = k_field(i, field_var::cbz0);
      host_field[i].te0  = k_field(i, field_var::te0);

      host_field[i].tcax = k_field(i, field_var::tcax);
      host_field[i].tcay = k_field(i, field_var::tcay);
      host_field[i].tcaz = k_field(i, field_var::tcaz);
      host_field[i].rhob = k_field(i, field_var::rhob);

      host_field[i].jfx = k_field(i, field_var::jfx);
      host_field[i].jfy = k_field(i, field_var::jfy);
      host_field[i].jfz = k_field(i, field_var::jfz);
      host_field[i].rhof = k_field(i, field_var::rhof);

      host_field[i].jfxold = k_field(i, field_var::jfxold);
      host_field[i].jfyold = k_field(i, field_var::jfyold);
      host_field[i].jfzold = k_field(i, field_var::jfzold);
      host_field[i].rhofold = k_field(i, field_var::rhofold);

      host_field[i].tx = k_field(i, field_var::tx);
      host_field[i].ty = k_field(i, field_var::ty);
      host_field[i].tz = k_field(i, field_var::tz);
      host_field[i].te = k_field(i, field_var::te);

      host_field[i].ox = k_field(i, field_var::ox);
      host_field[i].oy = k_field(i, field_var::oy);
      host_field[i].oz = k_field(i, field_var::oz);
      host_field[i].oe = k_field(i, field_var::oe);

      host_field[i].pex = k_field(i, field_var::pex);
      host_field[i].pey = k_field(i, field_var::pey);
      host_field[i].pez = k_field(i, field_var::pez);
      host_field[i].div_b_err = k_field(i, field_var::div_b_err);
      
      host_field[i].ux = k_field(i, field_var::ux);
      host_field[i].uy = k_field(i, field_var::uy);
      host_field[i].uz = k_field(i, field_var::uz);
      host_field[i].ue = k_field(i, field_var::ue);
      
      host_field[i].sx = k_field(i, field_var::sx);
      host_field[i].sy = k_field(i, field_var::sy);
      host_field[i].sz = k_field(i, field_var::sz);
      host_field[i].se = k_field(i, field_var::se);

      host_field[i].zx = k_field(i, field_var::zx);
      host_field[i].zy = k_field(i, field_var::zy);
      host_field[i].zz = k_field(i, field_var::zz);
      host_field[i].ze = k_field(i, field_var::ze);
      
      host_field[i].zxold = k_field(i, field_var::zxold);
      host_field[i].zyold = k_field(i, field_var::zyold);
      host_field[i].zzold = k_field(i, field_var::zzold);
      host_field[i].zeold = k_field(i, field_var::zeold);
      
      host_field[i].ematx = k_field_edge(i, field_edge_var::ematx);
      host_field[i].ematy = k_field_edge(i, field_edge_var::ematy);
      host_field[i].ematz = k_field_edge(i, field_edge_var::ematz);
      host_field[i].nmat = k_field_edge(i, field_edge_var::nmat);

      host_field[i].fmatx = k_field_edge(i, field_edge_var::fmatx);
      host_field[i].fmaty = k_field_edge(i, field_edge_var::fmaty);
      host_field[i].fmatz = k_field_edge(i, field_edge_var::fmatz);
      host_field[i].cmat = k_field_edge(i, field_edge_var::cmat);



    });

  last_copied = g->step;

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
      k_field(i, field_var::pe)  = host_field[i].pe;
      
      k_field(i, field_var::cbx0) = host_field[i].cbx0;
      k_field(i, field_var::cby0) = host_field[i].cby0;
      k_field(i, field_var::cbz0) = host_field[i].cbz0;
      k_field(i, field_var::te0)  = host_field[i].te0;

      k_field(i, field_var::tcax) = host_field[i].tcax;
      k_field(i, field_var::tcay) = host_field[i].tcay;
      k_field(i, field_var::tcaz) = host_field[i].tcaz;
      k_field(i, field_var::rhob) = host_field[i].rhob;

      k_field(i, field_var::jfx) = host_field[i].jfx;
      k_field(i, field_var::jfy) = host_field[i].jfy;
      k_field(i, field_var::jfz) = host_field[i].jfz;
      k_field(i, field_var::rhof) = host_field[i].rhof;

      k_field(i, field_var::jfxold) = host_field[i].jfxold;
      k_field(i, field_var::jfyold) = host_field[i].jfyold;
      k_field(i, field_var::jfzold) = host_field[i].jfzold;
      k_field(i, field_var::rhofold) = host_field[i].rhofold;

      k_field(i, field_var::te) = host_field[i].te;
      k_field(i, field_var::tx) = host_field[i].tx;
      k_field(i, field_var::ty) = host_field[i].ty;
      k_field(i, field_var::tz) = host_field[i].tz;

      k_field(i, field_var::ox) = host_field[i].ox;
      k_field(i, field_var::oy) = host_field[i].oy;
      k_field(i, field_var::oz) = host_field[i].oz;
      k_field(i, field_var::oe) = host_field[i].oe;

      k_field(i, field_var::pex) = host_field[i].pex;
      k_field(i, field_var::pey) = host_field[i].pey;
      k_field(i, field_var::pez) = host_field[i].pez;
      k_field(i, field_var::div_b_err) = host_field[i].div_b_err;
      
      k_field(i, field_var::ux) = host_field[i].ux;
      k_field(i, field_var::uy) = host_field[i].uy;
      k_field(i, field_var::uz) = host_field[i].uz;
      k_field(i, field_var::ue) = host_field[i].ue;
      
      k_field(i, field_var::sx) = host_field[i].sx;
      k_field(i, field_var::sy) = host_field[i].sy;
      k_field(i, field_var::sz) = host_field[i].sz;
      k_field(i, field_var::se) = host_field[i].se;
      
      k_field(i, field_var::zx) = host_field[i].zx;
      k_field(i, field_var::zy) = host_field[i].zy;
      k_field(i, field_var::zz) = host_field[i].zz;
      k_field(i, field_var::ze) = host_field[i].ze;
      
      k_field(i, field_var::zxold) = host_field[i].zxold;
      k_field(i, field_var::zyold) = host_field[i].zyold;
      k_field(i, field_var::zzold) = host_field[i].zzold;
      k_field(i, field_var::zeold) = host_field[i].zeold;
      
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
