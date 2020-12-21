#define IN_sfa
#include "sfa_private.h"

static field_advance_kernels_t sfa_kernels = {

  // Destructor
  delete_standard_field_array,

  // Time stepping interfaces

  advance_b,
  advance_e,

  // Diagnostic interfaces

  energy_f,

  // Accumulator interfaces

  clear_jf,
  synchronize_jf,
  clear_rhof,
  synchronize_rho,

  // Initialize interface

  compute_rhob,
  compute_curl_b,

  // Shared face cleaning interface

  synchronize_tang_e_norm_b,

  // Electric field divergence cleaning interface

  compute_div_e_err,
  compute_rms_div_e_err,
  clean_div_e,

  // Magnetic field divergence cleaning interface

  compute_div_b_err,
  compute_rms_div_b_err,
  clean_div_b,

};

static float
minf( float a,
      float b ) {
  return a<b ? a : b;
}

void
sfa_params::copy_to_device() {

  // Avoid capturing this
  auto kmc = k_mc_h;

  Kokkos::parallel_for("Copy material coefficients to device",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_mc),
    KOKKOS_LAMBDA (const int i) {
      kmc(i, material_coeff_var::decayx) = mc[i].decayx;
      kmc(i, material_coeff_var::drivex) = mc[i].drivex;
      kmc(i, material_coeff_var::decayy) = mc[i].decayy;
      kmc(i, material_coeff_var::drivey) = mc[i].drivey;
      kmc(i, material_coeff_var::decayz) = mc[i].decayz;
      kmc(i, material_coeff_var::drivez) = mc[i].drivez;
      kmc(i, material_coeff_var::rmux) = mc[i].rmux;
      kmc(i, material_coeff_var::rmuy) = mc[i].rmuy;
      kmc(i, material_coeff_var::rmuz) = mc[i].rmuz;
      kmc(i, material_coeff_var::nonconductive) = mc[i].nonconductive;
      kmc(i, material_coeff_var::epsx) = mc[i].epsx;
      kmc(i, material_coeff_var::epsy) = mc[i].epsy;
      kmc(i, material_coeff_var::epsz) = mc[i].epsz;
    });

  Kokkos::deep_copy(k_mc_d, k_mc_h);

}

void
sfa_params::copy_to_host() {

  Kokkos::deep_copy(k_mc_h, k_mc_d);

  // Avoid capturing this
  auto kmc = k_mc_h;

  Kokkos::parallel_for("Copy material coefficients to host",
    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_mc),
    KOKKOS_LAMBDA (const int i) {
      mc[i].decayx = kmc(i, material_coeff_var::decayx);
      mc[i].drivex = kmc(i, material_coeff_var::drivex);
      mc[i].decayy = kmc(i, material_coeff_var::decayy);
      mc[i].drivey = kmc(i, material_coeff_var::drivey);
      mc[i].decayz = kmc(i, material_coeff_var::decayz);
      mc[i].drivez = kmc(i, material_coeff_var::drivez);
      mc[i].rmux = kmc(i, material_coeff_var::rmux);
      mc[i].rmuy = kmc(i, material_coeff_var::rmuy);
      mc[i].rmuz = kmc(i, material_coeff_var::rmuz);
      mc[i].nonconductive = kmc(i, material_coeff_var::nonconductive);
      mc[i].epsx = kmc(i, material_coeff_var::epsx);
      mc[i].epsy = kmc(i, material_coeff_var::epsy);
      mc[i].epsz = kmc(i, material_coeff_var::epsz);
    });

}

static sfa_params_t *
create_sfa_params( grid_t           * g,
                   const material_t * m_list,
                   float              damp ) {
  sfa_params_t * p;
  float ax, ay, az, cg2;
  material_coefficient_t *mc;
  const material_t *m;
  int n_mc;

  // Run sanity checks on the material list

  ax = g->nx>1 ? g->cvac*g->dt*g->rdx : 0; ax *= ax;
  ay = g->ny>1 ? g->cvac*g->dt*g->rdy : 0; ay *= ay;
  az = g->nz>1 ? g->cvac*g->dt*g->rdz : 0; az *= az;
  n_mc = 0;
  LIST_FOR_EACH(m,m_list) {
    if( m->sigmax/m->epsx<0 )
      WARNING(("\"%s\" is an active medium along x", m->name));
    if( m->epsy*m->muz<0 )
      WARNING(("\"%s\" has an imaginary x speed of light (ey)", m->name));
    if( m->epsz*m->muy<0 )
      WARNING(("\"%s\" has an imaginary x speed of light (ez)", m->name));
    if( m->sigmay/m->epsy<0 )
      WARNING(("\"%s\" is an active medium along y", m->name));
    if( m->epsz*m->mux<0 )
      WARNING(("\"%s\" has an imaginary y speed of light (ez)", m->name));
    if( m->epsx*m->muz<0 )
      WARNING(("\"%s\" has an imaginary y speed of light (ex)", m->name));
    if( m->sigmaz/m->epsz<0 )
      WARNING(("\"%s\" is an an active medium along z", m->name));
    if( m->epsx*m->muy<0 )
      WARNING(("\"%s\" has an imaginary z speed of light (ex)", m->name));
    if( m->epsy*m->mux<0 )
      WARNING(("\"%s\" has an imaginary z speed of light (ey)", m->name));
    cg2 = ax/minf(m->epsy*m->muz,m->epsz*m->muy) +
          ay/minf(m->epsz*m->mux,m->epsx*m->muz) +
          az/minf(m->epsx*m->muy,m->epsy*m->mux);
    if( cg2>=1 )
      WARNING(( "\"%s\" Courant condition estimate = %e", m->name, sqrt(cg2) ));
    if( m->zetax!=0 || m->zetay!=0 || m->zetaz!=0 )
      WARNING(( "\"%s\" magnetic conductivity is not supported" ));
    n_mc++;
  }

  // Allocate the sfa parameters
  p = new sfa_params_t();

  MALLOC_ALIGNED( p->mc, n_mc, 128 );
  p->k_mc_d = k_material_coefficient_t("k_mc_d", n_mc);
  p->k_mc_h = Kokkos::create_mirror_view(p->k_mc_d);
  p->n_mc = n_mc;
  p->damp = damp;

  // Fill up the material coefficient array
  // FIXME: THIS IMPLICITLY ASSUMES MATERIALS ARE NUMBERED CONSECUTIVELY FROM
  // O.

  LIST_FOR_EACH( m, m_list ) {
    mc = p->mc + m->id;

    // Advance E coefficients
    // Note: m ->sigma{x,y,z} = 0 -> Non conductive
    //       mc->decay{x,y,z} = 0 -> Perfect conductor to numerical precision
    //       otherwise            -> Conductive
    ax = ( m->sigmax*g->dt ) / ( m->epsx*g->eps0 );
    ay = ( m->sigmay*g->dt ) / ( m->epsy*g->eps0 );
    az = ( m->sigmaz*g->dt ) / ( m->epsz*g->eps0 );
    mc->decayx = exp(-ax);
    mc->decayy = exp(-ay);
    mc->decayz = exp(-az);
    if( ax==0 )              mc->drivex = 1./m->epsx;
    else if( mc->decayx==0 ) mc->drivex = 0;
    else mc->drivex = 2.*exp(-0.5*ax)*sinh(0.5*ax) / (ax*m->epsx);
    if( ay==0 )              mc->drivey = 1./m->epsy;
    else if( mc->decayy==0 ) mc->drivey = 0;
    else mc->drivey = 2.*exp(-0.5*ay)*sinh(0.5*ay) / (ay*m->epsy);
    if( az==0 )              mc->drivez = 1./m->epsz;
    else if( mc->decayz==0 ) mc->drivez = 0;
    else mc->drivez = 2.*exp(-0.5*az)*sinh(0.5*az) / (az*m->epsz);
    mc->rmux = 1./m->mux;
    mc->rmuy = 1./m->muy;
    mc->rmuz = 1./m->muz;

    // Clean div E coefficients.  Note: The charge density due to J =
    // sigma E currents is not computed.  Consequently, the divergence
    // error inside conductors cannot computed.  The divergence error
    // multiplier is thus set to zero to ignore divergence errors
    // inside conducting materials.

    mc->nonconductive = ( ax==0 && ay==0 && az==0 ) ? 1. : 0.;
    mc->epsx = m->epsx;
    mc->epsy = m->epsy;
    mc->epsz = m->epsz;
  }

  p->copy_to_device();

  return p;
}

void
destroy_sfa_params( sfa_params_t * p ) {
  FREE_ALIGNED( p->mc );
  delete(p);
}

/*****************************************************************************/

void
checkpt_standard_field_array( field_array_t * fa ) {
  sfa_params_t * p = (sfa_params_t *)fa->params;
  p->copy_to_host();

  CHECKPT( p, 1 );
  CHECKPT_ALIGNED( p->mc, p->n_mc, 128 );
  CHECKPT_VIEW( p->k_mc_d );
  CHECKPT_VIEW( p->k_mc_h );

  checkpt_field_array_internal( fa );
}

field_array_t *
restore_standard_field_array( void ) {
  sfa_params_t * p;
  RESTORE( p );
  RESTORE_ALIGNED( p->mc );
  RESTORE_VIEW( &p->k_mc_d );
  RESTORE_VIEW( &p->k_mc_h );

  p->copy_to_device();
  return restore_field_array_internal( (void*) p);
}

field_array_t *
new_standard_field_array( grid_t           * RESTRICT g,
                          const material_t * RESTRICT m_list,
                          float                       damp ) {
  field_array_t * fa;
  if( !g || !m_list || damp<0 ) ERROR(( "Bad args" ));

  int nx = g->nx;
  int ny = g->ny;
  int nz = g->nz;
  int xyz_sz = 2*ny*(nz+1) + 2*nz*(ny+1) + ny*nz;
  int yzx_sz = 2*nz*(nx+1) + 2*nx*(nz+1) + nz*nx;
  int zxy_sz = 2*nx*(ny+1) + 2*ny*(nx+1) + nx*ny;

  //MALLOC( fa, 1 );
  fa = new field_array_t(g->nv, xyz_sz, yzx_sz, zxy_sz);

  // Zero host accum array
  Kokkos::parallel_for("Clear rhob accumulation array on host",
    host_execution_policy(0, g->nv),
    KOKKOS_LAMBDA (int i) {
      fa->k_f_rhob_accum_h(i) = 0;
    });

  MALLOC_ALIGNED( fa->f, g->nv, 128 );
  CLEAR( fa->f, g->nv );
  fa->g = g;
  fa->params = create_sfa_params( g, m_list, damp );
  fa->kernel[0] = sfa_kernels;

  REGISTER_OBJECT( fa, checkpt_standard_field_array,
                       restore_standard_field_array, NULL );
  return fa;
}

void
delete_standard_field_array( field_array_t * fa ) {
  if( !fa ) return;
  UNREGISTER_OBJECT( fa );
  destroy_sfa_params( (sfa_params_t *)fa->params );
  FREE_ALIGNED( fa->f );
  delete(fa);
}

/*****************************************************************************/

// Kokkos versions
void clear_jf(
  field_array_t* RESTRICT fa
)
{
    if( !fa )
    {
      ERROR(("Bad args" ));
    }

    k_field_t kfield = fa->k_f_d;
    const int nv = fa->g->nv;
    Kokkos::parallel_for("clear_jf", Kokkos::RangePolicy<>(0,nv),
    KOKKOS_LAMBDA(const int v) {
        kfield(v, field_var::jfx) = 0;
        kfield(v, field_var::jfy) = 0;
        kfield(v, field_var::jfz) = 0;
    });
}

void clear_rhof(
  field_array_t* RESTRICT fa
)
{
    if( !fa )
    {
      ERROR(("Bad args" ));
    }

    k_field_t kfield = fa->k_f_d;
    const int nv = fa->g->nv;
    Kokkos::parallel_for("clear_rhof", Kokkos::RangePolicy<>(0,nv),
    KOKKOS_LAMBDA(const int v) {
        kfield(v, field_var::rhof) = 0;
    });
}

// FIXME: ADD clear_jf_and_rhof CALL AND/OR ELIMINATE SOME OF THE ABOVE
// (MORE EFFICIENT TOO).
