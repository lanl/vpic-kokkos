#define IN_spa
#define HAS_V4_PIPELINE
#include "spa_private.h"
#include "../species_advance.h"

double
momentum_p_kernel(const k_particles_soa_t& k_part, const float msp, const int np, const float sp_w, double* mom_x, double* mom_y, double* mom_z) {

  double momentumx = 0.0;
  double momentumy = 0.0;
  double momentumz = 0.0;

    Kokkos::parallel_reduce("momentum_p x-dim", np, KOKKOS_LAMBDA(const int n, double& update) {
      float ux = static_cast<float>(k_part.get_ux(n));
#if defined PARTICLE_WEIGHT_FLOAT
      update += static_cast<double>(ux) * k_part.w(n) * msp;
#elif defined PARTICLE_WEIGHT_SHORT
      update += static_cast<double>(ux) * k_part.w(n) * sp_w * msp;
#elif defined PARTICLE_WEIGHT_CONSTANT
      update += static_cast<double>(ux) * sp_w * msp;
#endif
    }, momentumx);
    Kokkos::parallel_reduce("momentum_p y-dim", np, KOKKOS_LAMBDA(const int n, double& update) {
      float uy = static_cast<float>(k_part.get_uy(n));
#if defined PARTICLE_WEIGHT_FLOAT
      update += static_cast<double>(uy) * k_part.w(n) * msp;
#elif defined PARTICLE_WEIGHT_SHORT
      update += static_cast<double>(uy) * k_part.w(n) * sp_w * msp;
#elif defined PARTICLE_WEIGHT_CONSTANT
      update += static_cast<double>(uy) * sp_w * msp;
#endif
    }, momentumy);
    Kokkos::parallel_reduce("momentum_p z-dim", np, KOKKOS_LAMBDA(const int n, double& update) {
      float uz = static_cast<float>(k_part.get_uz(n));
#if defined PARTICLE_WEIGHT_FLOAT
      update += static_cast<double>(uz) * k_part.w(n) * msp;
#elif defined PARTICLE_WEIGHT_SHORT
      update += static_cast<double>(uz) * k_part.w(n) * sp_w * msp;
#elif defined PARTICLE_WEIGHT_CONSTANT
      update += static_cast<double>(uz) * sp_w * msp;
#endif
    }, momentumz);

    *mom_x = momentumx*msp;
    *mom_y = momentumy*msp;
    *mom_z = momentumz*msp;

    double v0 = momentumx*momentumx + momentumy*momentumy + momentumz*momentumz;
//    float lorentz = sqrtf(1.0+v0); 
//#if defined PARTICLE_WEIGHT_FLOAT
//        v0 = (msp * k_part.w(0)) * lorentz * sqrtf(v0)/lorentz;
//#elif defined PARTICLE_WEIGHT_SHORT
//        v0 = (msp * k_part.w(0)*sp_w) * lorentz * sqrtf(v0);
//#elif defined PARTICLE_WEIGHT_CONSTANT
//        v0 = (msp * sp_w) * lorentz * sqrtf(v0);
//#endif
    return v0;

//#if defined PARTICLE_WEIGHT_FLOAT
//        v0 = (msp * k_part.w(0)) * lorentz * sqrtf(v0)/lorentz;
//#elif defined PARTICLE_WEIGHT_SHORT
//        v0 = (msp * k_part.w(0)*sp_w) * lorentz * sqrtf(v0);
//#elif defined PARTICLE_WEIGHT_CONSTANT
//        v0 = (msp * sp_w) * lorentz * sqrtf(v0);
//#endif
//    return v0;
}

double
momentum_p_kokkos(const species_t* RESTRICT sp, double* mom_x, double* mom_y, double* mom_z) {

    double local=0.0, global=0.0;

    if(!sp) ERROR(("Bad args"));


    double momentum_x=0.0, momentum_y=0.0, momentum_z=0.0;
    double global_m_x=0.0, global_m_y=0.0, global_m_z=0.0;
    local = momentum_p_kernel(sp->k_p_soa_d, sp->m, sp->np, sp->w, &momentum_x, &momentum_y, &momentum_z);
    Kokkos::fence();

    mp_allsum_d(&momentum_x, &global_m_x, 1);
    mp_allsum_d(&momentum_y, &global_m_y, 1);
    mp_allsum_d(&momentum_z, &global_m_z, 1);
    *mom_x = global_m_x;
    *mom_y = global_m_y;
    *mom_z = global_m_z;
//    mp_allsum_d( &local, &global, 1 );
    return global;
}
