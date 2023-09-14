// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
//#include <atomic>
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/kokkos_tuning.hpp"

#ifdef FIELD_IONIZATION	
#include <iostream>
using namespace std;
#include <fstream>
#include <Kokkos_Random.hpp>
#endif




// Write current values to either an accumulator or directly to the fields
template<class CurrentScatterAccess>
void KOKKOS_INLINE_FUNCTION
accumulate_current(CurrentScatterAccess& current_sa, int ii,
                   const int nx, const int ny, const int nz, 
                   const float cx, const float cy, const float cz, 
                   const float v0, const float v1, const float v2, const float v3,
                   const float v4, const float v5, const float v6, const float v7,
                   const float v8, const float v9, const float v10, const float v11) {
#ifdef VPIC_ENABLE_ACCUMULATORS
  current_sa(ii, 0)  += cx*v0;
  current_sa(ii, 1)  += cx*v1;
  current_sa(ii, 2)  += cx*v2;
  current_sa(ii, 3)  += cx*v3;
  
  current_sa(ii, 4)  += cy*v4;
  current_sa(ii, 5)  += cy*v5;
  current_sa(ii, 6)  += cy*v6;
  current_sa(ii, 7)  += cy*v7;
  
  current_sa(ii, 8)  += cz*v8;
  current_sa(ii, 9)  += cz*v9;
  current_sa(ii, 10) += cz*v10;
  current_sa(ii, 11) += cz*v11;
#else
  int iii = ii;
  int zi = iii/((nx+2)*(ny+2));
  iii -= zi*(nx+2)*(ny+2);
  int yi = iii/(nx+2);
  int xi = iii - yi*(nx+2);
  
  current_sa(ii, field_var::jfx)                           += cx*v0;
  current_sa(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx)   += cx*v1;
  current_sa(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx)   += cx*v2;
  current_sa(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
  
  current_sa(ii, field_var::jfy)                           += cy*v4;
  current_sa(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy)   += cy*v5;
  current_sa(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy)   += cy*v6;
  current_sa(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v7;
  
  current_sa(ii, field_var::jfz)                           += cz*v8;
  current_sa(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz)   += cz*v9;
  current_sa(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz)   += cz*v10;
  current_sa(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v11;
#endif
}

// Reduce the current for all active threads/lanes to reduce the number of writes to memory
template<class TeamMember, class CurrentScatterAccess>
void KOKKOS_INLINE_FUNCTION
reduce_and_accumulate_current(TeamMember& team_member, CurrentScatterAccess& access, 
                              const int num_iters, const int ii, 
                              const int nx, const int ny, const int nz, 
                              const float cx, const float cy, const float cz, 
                              float *v0, float *v1,  float *v2,  float *v3,
                              float *v4, float *v5,  float *v6,  float *v7,
                              float *v8, float *v9,  float *v10, float *v11) {

#ifdef VPIC_ENABLE_VECTORIZATION
  alignas(16) float valx[4] = {0.0, 0.0, 0.0, 0.0};
  alignas(16) float valy[4] = {0.0, 0.0, 0.0, 0.0};
  alignas(16) float valz[4] = {0.0, 0.0, 0.0, 0.0};
  #pragma omp simd reduction(+:valx[0:4],valy[0:4],valz[0:4])
  for(int lane=0; lane<num_iters; lane++) {
    valx[0] += v0[lane];
    valx[1] += v1[lane];
    valx[2] += v2[lane];
    valx[3] += v3[lane];
    valy[0] += v4[lane];
    valy[1] += v5[lane];
    valy[2] += v6[lane];
    valy[3] += v7[lane];
    valz[0] += v8[lane];
    valz[1] += v9[lane];
    valz[2] += v10[lane];
    valz[3] += v11[lane];
  }
  accumulate_current(access, ii, nx, ny, nz, cx, cy, cz, 
                     valx[0], valx[1], valx[2], valx[3], 
                     valy[0], valy[1], valy[2], valy[3], 
                     valz[0], valz[1], valz[2], valz[3]);
#elif defined( __CUDA_ARCH__ )
  int mask = 0xffffffff;
  for(int i=16; i>0; i=i/2) {
    v0[0]  += __shfl_down_sync(mask, v0[0],  i);
    v1[0]  += __shfl_down_sync(mask, v1[0],  i);
    v2[0]  += __shfl_down_sync(mask, v2[0],  i);
    v3[0]  += __shfl_down_sync(mask, v3[0],  i);
    v4[0]  += __shfl_down_sync(mask, v4[0],  i);
    v5[0]  += __shfl_down_sync(mask, v5[0],  i);
    v6[0]  += __shfl_down_sync(mask, v6[0],  i);
    v7[0]  += __shfl_down_sync(mask, v7[0],  i);
    v8[0]  += __shfl_down_sync(mask, v8[0],  i);
    v9[0]  += __shfl_down_sync(mask, v9[0],  i);
    v10[0] += __shfl_down_sync(mask, v10[0], i);
    v11[0] += __shfl_down_sync(mask, v11[0], i);
  }
  if(team_member.team_rank()%32 == 0) {
    accumulate_current(access, ii, nx, ny, nz, cx, cy, cz, 
                       v0[0], v1[0], v2[0], v3[0], 
                       v4[0], v5[0], v6[0], v7[0], 
                       v8[0], v9[0], v10[0], v11[0]);
  }
#else
  team_member.team_reduce(Kokkos::Sum<float>(v0[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v1[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v2[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v3[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v4[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v5[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v6[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v7[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v8[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v9[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v10[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v11[0]));
  if(team_member.team_rank() == 0) {
    accumulate_current(access, ii, nx, ny, nz, cx, cy, cz, 
                       v0[0], v1[0], v2[0], v3[0], 
                       v4[0], v5[0], v6[0], v7[0], 
                       v8[0], v9[0], v10[0], v11[0]);
  }
#endif
}

template<class TeamMember, class field_sa_t, class field_var>
void KOKKOS_INLINE_FUNCTION
contribute_current(TeamMember& team_member, field_sa_t& access, int i0, int i1, int i2, int i3, field_var j, float v0, float v1,  float v2, float v3) {
#ifdef __CUDA_ARCH__
  int mask = 0xffffffff;
  int team_rank = team_member.team_rank();
  for(int i=16; i>0; i=i/2) {
    v0 += __shfl_down_sync(mask, v0, i);
    v1 += __shfl_down_sync(mask, v1, i);
    v2 += __shfl_down_sync(mask, v2, i);
    v3 += __shfl_down_sync(mask, v3, i);
  }
  if(team_rank%32 == 0) {
    access(i0, j) += v0;
    access(i1, j) += v1;
    access(i2, j) += v2;
    access(i3, j) += v3;
  }
#else
  team_member.team_reduce(Kokkos::Sum<float>(v0));
  team_member.team_reduce(Kokkos::Sum<float>(v1));
  team_member.team_reduce(Kokkos::Sum<float>(v2));
  team_member.team_reduce(Kokkos::Sum<float>(v3));
  if(team_member.team_rank() == 0) {
    access(i0, j) += v0;
    access(i1, j) += v1;
    access(i2, j) += v2;
    access(i3, j) += v3;
  }
#endif
}

// Detect whether all threads/vector lanes are processing particles belonging to the same cell
template<class TeamMember, class IndexView, class BoundsView>
int KOKKOS_INLINE_FUNCTION particles_in_same_cell(TeamMember& team_member, IndexView& ii, BoundsView& inbnds, const int num_lanes) {
#ifdef USE_GPU
  int min_inbnds = inbnds[0];
  int max_inbnds = inbnds[0];
  team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
  team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
  int min_index = ii[0];
  int max_index = ii[0];
  team_member.team_reduce(Kokkos::Max<int>(max_index));
  team_member.team_reduce(Kokkos::Min<int>(min_index));
  return min_inbnds == max_inbnds && min_index == max_index;
#else
  for(int lane=0; lane<num_lanes; lane++) {
    if(ii[0] != ii[lane] || inbnds[0] != inbnds[lane])
      return 0;
  }
  return 1;
#endif
}

// Load the interpolator for cell ii
KOKKOS_INLINE_FUNCTION
void simd_load_interpolator_var(float* v0, const int ii, const k_interpolator_t& k_interp, int len) {
  #pragma omp simd
  for(int i=0; i<len; i++) {
    v0[i] = k_interp(ii, i);
  }
}

// Template for unrolling a loop in reverse order
// Necessary to avoid the looping/pointer overhead when loading interpolator data
template<int N>
KOKKOS_INLINE_FUNCTION
void unrolled_simd_load(float* vals, const int* ii, const k_interpolator_t& k_interp, int len) {
  unrolled_simd_load<N-1>(vals, ii, k_interp, len);
  simd_load_interpolator_var(vals+(N-1)*18, ii[N-1], k_interp, len);
}
template<>
KOKKOS_INLINE_FUNCTION
void unrolled_simd_load<0>(float* vals, const int* ii, const k_interpolator_t& k_interp, int len) {}

// Non forced unrolled version. Potentially less performance than the template version
// This will work with arbitrary number of particles rather than having to use a multiple of the number of simd lanes
void unrolled_simd_load(float* vals, const int* ii, const k_interpolator_t& k_interp, int num_var, int num_part) {
  for(int i=0; i<num_part; i++) {
    simd_load_interpolator_var(vals+i*num_var, ii[i], k_interp, num_var);
  }
}

// Load interpolators
template<int NumLanes>
KOKKOS_INLINE_FUNCTION
void load_interpolators(
                        float* fex,
                        float* fdexdy,
                        float* fdexdz,
                        float* fd2exdydz,
                        float* fey,
                        float* fdeydz,
                        float* fdeydx,
                        float* fd2eydzdx,
                        float* fez,
                        float* fdezdx,
                        float* fdezdy,
                        float* fd2ezdxdy,
                        float* fcbx,
                        float* fdcbxdx,
                        float* fcby,
                        float* fdcbydy,
                        float* fcbz,
                        float* fdcbzdz,
                        const int* ii,
			const int num_part,
                        const k_interpolator_t& k_interp
                        ) {
#if defined(VPIC_ENABLE_VECTORIZATION) && !defined(USE_GPU)
  int same_cell = 1;
  for(int lane=0; lane<NumLanes; lane++) {
    if(ii[0] != ii[lane]) {
      same_cell = 0;
      break;
    }
  }

  // Try to reduce the number of loads if all particles are in the same cell
  if(same_cell) {
    float vals[18];

    simd_load_interpolator_var(vals, ii[0], k_interp, 18);
    #pragma omp simd
    for(int i=0; i<NumLanes; i++) {
      fex[i]       = vals[0];
      fdexdy[i]    = vals[1];
      fdexdz[i]    = vals[2];
      fd2exdydz[i] = vals[3];
      fey[i]       = vals[4];
      fdeydz[i]    = vals[5];
      fdeydx[i]    = vals[6];
      fd2eydzdx[i] = vals[7];
      fez[i]       = vals[8];
      fdezdx[i]    = vals[9];
      fdezdy[i]    = vals[10];
      fd2ezdxdy[i] = vals[11];
      fcbx[i]      = vals[12];
      fdcbxdx[i]   = vals[13];
      fcby[i]      = vals[14];
      fdcbydy[i]   = vals[15];
      fcbz[i]      = vals[16];
      fdcbzdz[i]   = vals[17];
    }
  } else {

    // Efficient vectorized load
    float vals[18*NumLanes];
    unrolled_simd_load(vals, ii, k_interp, 18, num_part);
//    unrolled_simd_load<NumLanes>(vals, ii, k_interp, 18);

    // Essentially a transpose
    #pragma omp simd
    for(int i=0; i<num_part; i++) {
      fex[i]       = vals[18*i];
      fdexdy[i]    = vals[1+18*i];
      fdexdz[i]    = vals[2+18*i];
      fd2exdydz[i] = vals[3+18*i];
      fey[i]       = vals[4+18*i];
      fdeydz[i]    = vals[5+18*i];
      fdeydx[i]    = vals[6+18*i];
      fd2eydzdx[i] = vals[7+18*i];
      fez[i]       = vals[8+18*i];
      fdezdx[i]    = vals[9+18*i];
      fdezdy[i]    = vals[10+18*i];
      fd2ezdxdy[i] = vals[11+18*i];
      fcbx[i]      = vals[12+18*i];
      fdcbxdx[i]   = vals[13+18*i];
      fcby[i]      = vals[14+18*i];
      fdcbydy[i]   = vals[15+18*i];
      fcbz[i]      = vals[16+18*i];
      fdcbzdz[i]   = vals[17+18*i];
    }
  }
#else
  for(int lane=0; lane<NumLanes; lane++) {
    // Load interpolators
    fex[LANE]       = k_interp(ii[LANE], interpolator_var::ex);     
    fdexdy[LANE]    = k_interp(ii[LANE], interpolator_var::dexdy);  
    fdexdz[LANE]    = k_interp(ii[LANE], interpolator_var::dexdz);  
    fd2exdydz[LANE] = k_interp(ii[LANE], interpolator_var::d2exdydz);
    fey[LANE]       = k_interp(ii[LANE], interpolator_var::ey);     
    fdeydz[LANE]    = k_interp(ii[LANE], interpolator_var::deydz);  
    fdeydx[LANE]    = k_interp(ii[LANE], interpolator_var::deydx);  
    fd2eydzdx[LANE] = k_interp(ii[LANE], interpolator_var::d2eydzdx);
    fez[LANE]       = k_interp(ii[LANE], interpolator_var::ez);     
    fdezdx[LANE]    = k_interp(ii[LANE], interpolator_var::dezdx);  
    fdezdy[LANE]    = k_interp(ii[LANE], interpolator_var::dezdy);  
    fd2ezdxdy[LANE] = k_interp(ii[LANE], interpolator_var::d2ezdxdy);
    fcbx[LANE]      = k_interp(ii[LANE], interpolator_var::cbx);    
    fdcbxdx[LANE]   = k_interp(ii[LANE], interpolator_var::dcbxdx); 
    fcby[LANE]      = k_interp(ii[LANE], interpolator_var::cby);    
    fdcbydy[LANE]   = k_interp(ii[LANE], interpolator_var::dcbydy); 
    fcbz[LANE]      = k_interp(ii[LANE], interpolator_var::cbz);    
    fdcbzdz[LANE]   = k_interp(ii[LANE], interpolator_var::dcbzdz); 
  }
#endif
}




void
advance_p_kokkos_unified(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sa_t k_f_sa,
        k_interpolator_t& k_interp,
        //k_particle_movers_t k_local_particle_movers,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
#ifdef FIELD_IONIZATION	
        const float dt_2mc,
#else
        const float qdt_2mc,
#endif	
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
#ifndef	FIELD_IONIZATION
        const float qsp,
#endif	
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
#ifdef FIELD_IONIZATION	
        const int nz,
	species_t * RESTRICT sp,
	species_t * sp_e)
#else
        const int nz) 
#endif
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  k_field_t k_field = fa->k_f_d;
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

  float timestep = g->step;

  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux)
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define p_w     k_particles(p_index, particle_var::w)
#ifdef FIELD_IONIZATION
  #define p_q     k_particles(p_index, particle_var::charge)
#endif  
  #define pii     k_particles_i(p_index)

  #define f_cbx k_interp(ii[LANE], interpolator_var::cbx)
  #define f_cby k_interp(ii[LANE], interpolator_var::cby)
  #define f_cbz k_interp(ii[LANE], interpolator_var::cbz)
  #define f_ex  k_interp(ii[LANE], interpolator_var::ex)
  #define f_ey  k_interp(ii[LANE], interpolator_var::ey)
  #define f_ez  k_interp(ii[LANE], interpolator_var::ez)

  #define f_dexdy    k_interp(ii[LANE], interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii[LANE], interpolator_var::dexdz)

  #define f_d2exdydz k_interp(ii[LANE], interpolator_var::d2exdydz)
  #define f_deydx    k_interp(ii[LANE], interpolator_var::deydx)
  #define f_deydz    k_interp(ii[LANE], interpolator_var::deydz)

  #define f_d2eydzdx k_interp(ii[LANE], interpolator_var::d2eydzdx)
  #define f_dezdx    k_interp(ii[LANE], interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii[LANE], interpolator_var::dezdy)

  #define f_d2ezdxdy k_interp(ii[LANE], interpolator_var::d2ezdxdy)
  #define f_dcbxdx   k_interp(ii[LANE], interpolator_var::dcbxdx)
  #define f_dcbydy   k_interp(ii[LANE], interpolator_var::dcbydy)
  #define f_dcbzdz   k_interp(ii[LANE], interpolator_var::dcbzdz)

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

#ifdef FIELD_IONIZATION
  // constants
  int q_e_c     = -1; // code units
  float E_to_SI = 1.44303994037981860e+19; // code to SI
  float t_to_SI = 1.18119324097025572e-22; // code to SI
  float l_to_SI = 3.54112825083459245e-14; // code to SI
  float q_to_SI = 1.60217663399999989e-19; // code to SI
  float m_to_SI = 9.10938370150000079e-31; // code to SI
  float time_to_SI = 1.18119324097025572e-22; // code to SI
  float energy_to_SI = m_to_SI * (l_to_SI/time_to_SI) * (l_to_SI/time_to_SI); // code to SI
  float dt = g->dt*time_to_SI; // [s], timestep
  float q_e       = 1.60217663e-19;  // coulombs
  float q_e_au    = 1.0;             // au
  float m_e       = 9.1093837e-31;   // kilograms
  float m_e_au    = 1.0;             // au 
  float c         = 299792458;       // m/s
  float c_au      = 137.02;          // atomic units
  float epsilon_0_au = 1/(4*M_PI);   // au
  float alpha     = 0.00729735;      // fine structure constant
  float h_bar     = 1.054571817e-34; // J *s
  float E_field_conversion = 5.1422e+11;      // (alpha^3*m_e^2*c^3)/(q_e*h_bar), multiply AU to get SI
  float Gamma_conversion   = 1.0/(h_bar/(pow(alpha,2)*m_e*pow(c,2))); // multiply au to get sec^-1 
  
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  Kokkos::Random_XorShift64_Pool<> random_pool(seed);
  Kokkos::View<double*> epsilon_eV_list = sp->ionization_energy;
  float n = sp->qn; // principal quantum number
  float m = sp->qm; // magnetic quantum number
  float l = sp->ql; // angular momentum quantum number
  float lambda = g->lambda;
#endif  

// Determine whether to use accumulators
#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::View<float*[12]> accumulator("Accumulator", k_field.extent(0));
  Kokkos::deep_copy(accumulator, 0);
  auto current_sv = Kokkos::Experimental::create_scatter_view(accumulator);
#else
  k_field_sa_t current_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
#endif

// Setting up work distribution settings
#if defined( VPIC_ENABLE_VECTORIZATION ) && !defined( USE_GPU )
  constexpr int num_lanes = 32;
  int chunk_size = num_lanes;
  int num_chunks = np/num_lanes;
  if(num_chunks*num_lanes < np)
    num_chunks += 1;
  auto policy = Kokkos::TeamPolicy<>(num_chunks, 1, num_lanes);
#elif defined( VPIC_ENABLE_HIERARCHICAL )
  auto policy = Kokkos::TeamPolicy<>(LEAGUE_SIZE, TEAM_SIZE);
  int chunk_size = np/LEAGUE_SIZE;
  if(chunk_size*LEAGUE_SIZE < np)
    chunk_size += 1;
  constexpr int num_lanes = 1;
  int num_chunks = LEAGUE_SIZE;
#else
  constexpr int num_lanes = 1;
#endif

// Outermost parallel loop
#if defined(VPIC_ENABLE_HIERARCHICAL) || defined(VPIC_ENABLE_VECTORIZATION)
  Kokkos::parallel_for("advance_p", policy, 
  KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
      auto current_sa = current_sv.access();
      int chunk = team_member.league_rank();
      int num_iters = chunk_size;
      if((chunk+1)*chunk_size > np)
        num_iters = np - chunk*chunk_size;
      size_t pi_offset = chunk*chunk_size;
#else  
  auto policy = Kokkos::RangePolicy<>(0,np);
  Kokkos::parallel_for("advance_p", policy, KOKKOS_LAMBDA (const size_t pi_offset) {
      auto current_sa = current_sv.access();
#endif

	
     
// Inner parallelization loop
#if defined ( VPIC_ENABLE_HIERARCHICAL ) && !defined( VPIC_ENABLE_VECTORIZATION )
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_iters), [&] (const size_t index) {
      size_t pi_offset = chunk*chunk_size + index;
#endif
      int num_particles = num_lanes;
      if(pi_offset+num_particles > np)
        num_particles = np - pi_offset;
      float v0[num_lanes];
      float v1[num_lanes];
      float v2[num_lanes];
      float v3[num_lanes];
      float v4[num_lanes];
      float v5[num_lanes];
      float dx[num_lanes];
      float dy[num_lanes];
      float dz[num_lanes];
      float ux[num_lanes];
      float uy[num_lanes];
      float uz[num_lanes];
      float hax[num_lanes];
      float hay[num_lanes];
      float haz[num_lanes];
      float cbx[num_lanes];
      float cby[num_lanes];
      float cbz[num_lanes];
      float q[num_lanes];
   #ifdef FIELD_IONIZATION
      float charge[num_lanes];
   #endif
      int   ii[num_lanes];
      int   inbnds[num_lanes];

      float fcbx[num_lanes];
      float fcby[num_lanes];
      float fcbz[num_lanes];
      float fex[num_lanes];
      float fey[num_lanes];
      float fez[num_lanes];
      float fdexdy[num_lanes];
      float fdexdz[num_lanes];
      float fd2exdydz[num_lanes];
      float fdeydx[num_lanes];
      float fdeydz[num_lanes];
      float fd2eydzdx[num_lanes];
      float fdezdx[num_lanes];
      float fdezdy[num_lanes];
      float fd2ezdxdy[num_lanes];
      float fdcbxdx[num_lanes];
      float fdcbydy[num_lanes];
      float fdcbzdz[num_lanes];
      float *v6 = fex;
      float *v7 = fdexdy;
      float *v8 = fdexdz;
      float *v9 = fd2exdydz;
      float *v10 = fey;
      float *v11 = fdeydz;
      float *v12 = fdeydx;
      float *v13 = fd2eydzdx;

      size_t p_index = pi_offset;
      
      BEGIN_VECTOR_BLOCK {
        p_index = pi_offset + LANE;
        // Load position
        dx[LANE] = p_dx;
        dy[LANE] = p_dy;
        dz[LANE] = p_dz;
        // Load momentum
        ux[LANE] = p_ux;
        uy[LANE] = p_uy;
        uz[LANE] = p_uz;
        // Load weight
        q[LANE]  = p_w;
    #ifdef FIELD_IONIZATION	
	// Load charge
	charge[LANE] = p_q;
    #endif	
        // Load index
        ii[LANE] = pii;
      } END_VECTOR_BLOCK;

      
      load_interpolators<num_lanes>( fex, fdexdy, fdexdz, fd2exdydz,
                                     fey, fdeydz, fdeydx, fd2eydzdx,
                                     fez, fdezdx, fdezdy, fd2ezdxdy,
                                     fcbx, fdcbxdx,
                                     fcby, fdcbydy,
                                     fcbz, fdcbzdz,
                                     ii, num_particles, k_interp);

      BEGIN_VECTOR_BLOCK {
#ifdef FIELD_IONIZATION
        const float qdt_2mc = charge[LANE] * dt_2mc;
#endif
        
        // Interpolate E
        hax[LANE] = qdt_2mc*( (fex[LANE] + dy[LANE]*fdexdy[LANE] ) + dz[LANE]*(fdexdz[LANE] + dy[LANE]*fd2exdydz[LANE]) );
        hay[LANE] = qdt_2mc*( (fey[LANE] + dz[LANE]*fdeydz[LANE] ) + dx[LANE]*(fdeydx[LANE] + dz[LANE]*fd2eydzdx[LANE]) );
        haz[LANE] = qdt_2mc*( (fez[LANE] + dx[LANE]*fdezdx[LANE] ) + dy[LANE]*(fdezdy[LANE] + dx[LANE]*fd2ezdxdy[LANE]) );
		
#ifdef FIELD_IONIZATION	
	// ***** Field Ioization *****
	// Declate varviables
	bool multiphoton_ionised = false;
	float K;
	
	if (sp != sp_e){ // FIXME: need to check which species the user wants ionization enabled on
          float lambda_SI    = lambda*l_to_SI;  // meters

	  // Check if the particle is fully ionized already
          int N_ionization        = int(abs(charge[LANE])); // Current ionization state of the particle
          int N_ionization_before = N_ionization; // save variable to compare with ionization state after ionization algorithm
	  int N_ionization_levels = epsilon_eV_list.extent(0);

          // code units
  	  float hax_c = hax[LANE]/qdt_2mc;
  	  float hay_c = hay[LANE]/qdt_2mc;
  	  float haz_c = haz[LANE]/qdt_2mc;
  	  float ha_mag_c = sqrtf(pow(hax_c,2.0)+pow(hay_c,2.0)+pow(haz_c,2.0));
          // SI units
  	  float E_mag_SI = E_to_SI * ha_mag_c;
  	  // particle index
  	  int particle_index = LANE+pi_offset;
        
          // Calculate stuff
          float E_au       = E_mag_SI/E_field_conversion; // field strength, atomic units
          float nu         = c/lambda_SI; // Hz
          float omega_SI   = 2*M_PI*nu;   // Hz
          float omega_eV   = 1.2398/(lambda_SI/1e-6); // eV
          float omega_au   = omega_eV * 0.036749; // energy, Hartree units
          float I_au       = 0.5*c_au*epsilon_0_au*pow(E_au,2.0); // intensity from the field, atomic units

	   // initialize variables for the while loop
          int ionization_flag = 1;
          float t_ionize      = 0;
          float Gamma = 0;
          // loop for multiple ionization events in a single timestep
          while (ionization_flag == 1 && t_ionize <= dt && N_ionization < N_ionization_levels) {
        
            // Get the appropriate ionization energy
            float epsilon_eV = epsilon_eV_list(int(N_ionization)); // [eV], ionization energy
            float epsilon_au = epsilon_eV/27.2;         // atomic units, ionization energy
            float epsilon_SI = epsilon_au*4.3597463e-18;// [J],  ionization energy
           
            // Calculate stuff
	    K                    = floor(epsilon_au/omega_au)+1; // number of photons required for multiphoton ionization          
            float Z              = N_ionization + 1;          // ion charge number after ionization
            float Z_star         = N_ionization; // initial charge state
            float n_star         = (Z_star + 1.0)/sqrt(2*epsilon_au); // effective principle quantum number
            float l_star         = n_star - 1.0; // angular momentum
            float T_0            = M_PI*Z/(abs(epsilon_au) * sqrt(2*abs(epsilon_au))); // period of classical radial trajectories
            float gamma_keldysh  = omega_au*sqrt(2*m_e_au*epsilon_au)/(q_e_au*E_au);
          
            // Ionization events are tested for every particle with a bound electron at every timestep
            
            // Choose the ionization process based on the E-field at the particle
            // Specifically, Gamma =
            // min( Gamma_MPI, Gamma_ADK ) for        E <= E_M
            // Gamma_ADK                   for E_M <= E <= E_T
            // min( Gamma_ADK, Gamma_BSI ) for E_T <= E <= E_B
            // Gamma_BSI                   for        E >  E_B
            // ** NOTE: E_B is defined such that dGamma_ADK(E_B)/dE = 0 so that we have a monotonically increasing rate. 
            
            // Choose the ionization process based on |E| at each particle
            // Note E_T = epsilon^2/(4*Z) is the correct version but EPOCH uses epsilon^2/Z for some reason (maybe a typo in their paper?)
            float E_M_SI = 2*omega_SI*sqrt(2*m_e*epsilon_SI)/q_e;
            float E_M_au = omega_au*sqrt(8*epsilon_au); // atomic units
            float E_T_au = pow(epsilon_au,2.0)/(4*Z);      // atomic units
            float E_B_au = (6*m*pow(n,3.0) + 4*pow(Z,3.0))/(12*pow(n,4.0) - 9*pow(n,3.0)); // atomic units

            if (E_au<=E_M_au){
              // MPI Ionization
              // ionization rate per atom: Gamma^(K)
              // sigma^(K) = (h_bar*omega)^K*Gamma^(K)/I^K : K-photon cross section, units [cm^2K * s^(K-1)]
              // I: intensity of the laser field, units [W/cm^2]
              //cout << "MPI Ionization" << endl;
              // FIXME: need to add the case of circularly polarized field
              float T_K = 4.80*pow(1.30,2*K)*pow(2*K+1,-1)*pow(K,-1.0/2.0); // in the case of linearly polarized field
              float sigma_K_au = pow(c_au*pow(tgamma(K+1),2)*pow(n,5)* pow(omega_au,(10*K-1)/3), -1)*T_K*pow(E_au,2*K-2); // atomic units, [cm^2K * s^(K-1)]
	      float flux = c_au*pow(E_au, 2.0)/(8*M_PI*omega_au);
              float Gamma_MPI = sigma_K_au * pow(flux, K);
          
              // Tunneling Regime
              //cout << "Tunneling Ionization" << endl;
              float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
              float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
              float Gamma_ADK_au = C_nstar_lstar_squared * f_n_l * epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_au));
              float Gamma_ADK_SI = Gamma_ADK_au*Gamma_conversion;
          
              Gamma = min(Gamma_MPI,Gamma_ADK_SI);
	      if(Gamma_MPI < Gamma_ADK_SI){ multiphoton_ionised = true; }
            }
          
            else if (E_au>E_M_au && E_au<=E_T_au) {
              // Tunneling Regime
              //cout << "Tunneling Ionization" << endl;
              float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
              float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
              float Gamma_ADK_au = C_nstar_lstar_squared * f_n_l * epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_au));
              float Gamma_ADK_SI = Gamma_ADK_au*Gamma_conversion;
              Gamma = Gamma_ADK_SI;
          
            }
          
            else if (E_au>E_T_au && E_au<=E_B_au){
              // Either classical ADK or with BSI correction
              // Whichever has the minimum ionization rate
              //cout << "Tunneling or BSI Regime" << endl;
              // ADK (no correction)
              float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
              float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
              float Gamma_ADK_au = C_nstar_lstar_squared * f_n_l * epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_au));
              float Gamma_ADK_SI = Gamma_ADK_au*Gamma_conversion;
          
              // With BSI correction: Gamma = Gamma_classical + Gamma_ADK(I_classical)
              // Gamma_ADK(I_classical): ADK at the classical appearanace (threshold) intensity, i.e, the intensity that corresponds to the threshold E-field magnitude
              float Gamma_ADK_au_threshold = C_nstar_lstar_squared* f_n_l* epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_T_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_T_au));
              float Gamma_ADK_SI_threshold = Gamma_ADK_au_threshold*Gamma_conversion;
              // uniform field, FIXME: enable this
              // gamma_cl_uniform_atomic = (1 - E_n_atomic^2./(4*Z*E_atomic))/(2*T_0);
              // gamma_cl_uniform_SI = gamma_cl_uniform_atomic / (h_bar/(alpha^2*m_e*c^2));
              // oscillating field
              float Gamma_cl_au  = 1.0/(M_PI*T_0) * (  M_PI/2.0 - asin( pow(epsilon_au,2.0)/(4*Z*E_au)) + pow(epsilon_au,2.0)/(4*Z*E_au) * log( ( 4*Z*E_au - sqrt( 16*pow(Z,2.0)*pow(E_au,2.0) - pow(epsilon_au,4.0) ) )/pow(epsilon_au,2.0) ) );
              float Gamma_cl_SI  = Gamma_cl_au* Gamma_conversion;
              float Gamma_BSI_SI = Gamma_cl_SI + Gamma_ADK_SI_threshold;
          
              // Decide if the BSI correction is applicable
              Gamma = min(Gamma_ADK_SI, Gamma_BSI_SI);
            }
          
            else if (E_au>E_B_au) {
              // BSI Ionization
              //cout << "BSI Ionization" << endl;
              // BSI Ionization: Gamma = Gamma_classical + Gamma_ADK(I_classical)
              // Gamma_ADK(I_classical): ADK at the classical appearanace (threshold) intensity
              float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
              float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
              float Gamma_ADK_au_threshold = C_nstar_lstar_squared*f_n_l * sqrt(3*E_T_au*pow(n_star,3.0)/(M_PI*pow(Z,3.0))) * pow(Z,2.0)/(2*pow(n_star,2.0)) * pow(2*pow(Z,3.0)/(E_T_au*pow(n_star,3.0)),2*n_star-abs(m)-1)*exp(-2*pow(Z,3.0)/(3*pow(n_star,3.0)*E_T_au));
              float Gamma_ADK_SI_threshold = Gamma_ADK_au_threshold*Gamma_conversion;
              // uniform field, FIXME: enable this
              // gamma_cl_uniform_atomic = (1 - E_n_atomic^2./(4*Z*E_atomic))/(2*T_0);
              // oscillating field
              float Gamma_cl_au  = 1.0/(M_PI*T_0) * (  M_PI/2.0 - asin( pow(epsilon_au,2.0)/(4*Z*E_au)) + pow(epsilon_au,2.0)/(4*Z*E_au) * log( ( 4*Z*E_au - sqrt( 16*pow(Z,2.0)*pow(E_au,2.0) - pow(epsilon_au,4.0) ) )/pow(epsilon_au,2.0) ) );
              float Gamma_cl_SI  = Gamma_cl_au* Gamma_conversion;
              float Gamma_BSI_SI = Gamma_cl_SI + Gamma_ADK_SI_threshold;
              Gamma = Gamma_BSI_SI;
            }     
            
            // Ionization occurs if U_1 < 1 - exp(-Gamma * delta_t), for a uniform number U_1~[0,1]
            auto generator = random_pool.get_state();
            double U = generator.drand(0,1);
            random_pool.free_state(generator);
            if ( U < 1 - exp(-Gamma * (dt-t_ionize) ) ) {
              // ionization occurs
              N_ionization++;
              ionization_flag = 1;
              //cout << ionization << " Ionization(s) occurs" << endl;
          
              // deal with multiple ionizations
              t_ionize = -1.0/Gamma * log(1-U); // use previous U to calc
            } 
            else {
              // ionization doesnt occur
              ionization_flag = 0;
              //cout << "Ionization doesnt occur" << endl;
            } 
          
          } // end while loop
  
  	
  	  // Check if ionization event occured
  	  if(N_ionization_before < N_ionization){

	    if(multiphoton_ionised){cout << "Multiphoton Ionization used" << endl;}
	    
  	    // Change the charge of the particle
  	    k_particles(particle_index, particle_var::charge) = N_ionization * abs(q_e_c); // code units
  
  	    // Inject the macro electron with the same
  	    // momentum and position as the ionized particle
  	    // Multiple ionization events are enabled so the injected
  	    // electron weight needs to account for that
  	    int electron_index = sp_e->np++;
	    
            #define p_dx_e    sp_e->k_p_h(electron_index, particle_var::dx)
            #define p_dy_e    sp_e->k_p_h(electron_index, particle_var::dy)
            #define p_dz_e    sp_e->k_p_h(electron_index, particle_var::dz)
            #define p_ux_e    sp_e->k_p_h(electron_index, particle_var::ux)
            #define p_uy_e    sp_e->k_p_h(electron_index, particle_var::uy)
            #define p_uz_e    sp_e->k_p_h(electron_index, particle_var::uz)
  	    #define p_q_e     sp_e->k_p_h(electron_index, particle_var::charge)
            #define p_w_e     sp_e->k_p_h(electron_index, particle_var::w)
            #define pii_e     sp_e->k_p_i_h(electron_index)
  
  	    // get the positions and momentum from ionized species
            p_dx_e = dx[LANE];
  	    p_dy_e = dy[LANE];
  	    p_dz_e = dz[LANE];
  	    pii_e  = ii[LANE];
  	    p_ux_e = ux[LANE];
  	    p_uy_e = uy[LANE];
  	    p_uz_e = uz[LANE];
  	    p_q_e  = q_e_c; // electrons charge in code units
  	    p_w_e  = (N_ionization - N_ionization_before) * k_particles(particle_index, particle_var::w); // weight is dependent on the number of ionization events and weight of ionized particle

	    // With multiphoton ionization, the additional energy from photons
            // accelerates the electron in the direction of the electric field.
            // This is an approximation as the ejection angle ranges widely
            // with a maxima at theta = 0 with respect to the field
	    float epsilon_t_au = 0; // initialize the total energy from multiple ionizations
    	    float epsilon_t_c  = 0; 
    	    float epsilon_t_SI = 0;
	    if(multiphoton_ionised){
	      for(int i=0; i<int(N_ionization-N_ionization_before); i++) {
    	        epsilon_t_SI += epsilon_eV_list[i] * q_e; // joules
              }
	      double p_correction = sqrt( 2*m_e*(K*h_bar*omega_SI - epsilon_t_SI) )/energy_to_SI; // code units
	      p_ux_e += p_correction * hax_c/(ha_mag_c*ha_mag_c);
	      p_uy_e += p_correction * hay_c/(ha_mag_c*ha_mag_c);
	      p_uz_e += p_correction * haz_c/(ha_mag_c*ha_mag_c);	
	    }

	    //cout << "pii_e: " << pii_e << endl; // FIXME: remove
	    
  	    #undef p_dx_e
            #undef p_dy_e
            #undef p_dz_e
            #undef p_ux_e
            #undef p_uy_e
            #undef p_uz_e
            #undef p_w_e
            #undef pii_e
 



            // Energy conservation is accounted for by a current density correction through Poynting’s theorem (J_ionize = (N*epsilon_t)/(dt*E) E_hat atomic units)
            // in tunnelling ionisation and BSI the energy loss from the field is the ionisation energy of the electron
            // in multiphoton ionisation it is the total energy for the number of photons absorbed.
            // The total energy loss from multiple ionisations is summed and a current density correction is weighted back to the grid points
    	    float N_ions = k_particles(particle_index, particle_var::w); // this is the number of physical ions created  
	    if(multiphoton_ionised) {
	      epsilon_t_c = (K*h_bar*omega_SI)/energy_to_SI; // code units
	    } 
	    else {
              for(int i=0; i<int(N_ionization-N_ionization_before); i++) {
                epsilon_t_au += epsilon_eV_list[i]/27.2; // atomic units, ionization energy
    	        epsilon_t_SI += epsilon_eV_list[i] * q_e; // joules
    	        epsilon_t_c  += (epsilon_eV_list[i] * q_e) / energy_to_SI; // code units
              }
	    } 
    
            float cell_volume_c  = g->dV;
            float cell_volume_SI = cell_volume_c * l_to_SI*l_to_SI*l_to_SI;
            float cell_volume_au = cell_volume_SI/pow(5.29177210903e-11,3.0);
    	
    	    float dt_au = dt / 2.41884e-17; // convsersion factor is h_bar/Eh where Eh is hartree energy

	    float j_ionize_SI_x = epsilon_t_SI * N_ions * (hax_c * E_to_SI) / (dt * cell_volume_SI * E_mag_SI*E_mag_SI);
	    float j_ionize_SI_y = epsilon_t_SI * N_ions * (hay_c * E_to_SI) / (dt * cell_volume_SI * E_mag_SI*E_mag_SI);
	    float j_ionize_SI_z = epsilon_t_SI * N_ions * (haz_c * E_to_SI) / (dt * cell_volume_SI * E_mag_SI*E_mag_SI);
	    float j_ionize_SI_mag = sqrt( j_ionize_SI_x*j_ionize_SI_x + j_ionize_SI_y*j_ionize_SI_y + j_ionize_SI_z*j_ionize_SI_z );

    	    float j_ionize_mag_c  = (N_ions * epsilon_t_c )/(g->dt    * ha_mag_c * cell_volume_c ); // code
	    float jx_ionize = (epsilon_t_c * N_ions * hax_c) / (g->dt * cell_volume_c * ha_mag_c*ha_mag_c);
	    float jy_ionize = (epsilon_t_c * N_ions * hay_c) / (g->dt * cell_volume_c * ha_mag_c*ha_mag_c);
	    float jz_ionize = (epsilon_t_c * N_ions * haz_c) / (g->dt * cell_volume_c * ha_mag_c*ha_mag_c);

	    //float j_to_SI = 1.08169825311233e30; //q_to_SI/(t_to_SI * l_to_SI*l_to_SI); // code units to SI

	    // FIXME: remove
	    //cout << "*************" << endl;
	    //cout << "electron_index: " << electron_index << ", timestep: " << timestep << endl;  
	    //cout << "epsilon_t_SI: " << epsilon_t_SI << "," << "weight: " << N_ions << "," << "dt: " << dt << "," << "V_cell" << cell_volume_SI << "," << endl;
	    //cout << "E_mag_SI: " << E_mag_SI << "hax_c * E_to_SI: " << hax_c * E_to_SI << "," << "hay_c * E_to_SI: " << hay_c * E_to_SI << "," << "haz_c * E_to_SI: " << haz_c * E_to_SI << endl;
	    //cout << "j_ionize_SI_mag: " << j_ionize_SI_mag << "," <<  "j_ionize_SI_x: " << j_ionize_SI_x << "," <<  "j_ionize_SI_y: " << j_ionize_SI_y << "," <<  "j_ionize_SI_z: " << j_ionize_SI_z << endl;
	    //cout << "j_ionize_mag_c: " << j_ionize_mag_c << "," << "jx_ionize: " << jx_ionize << "," << "jy_ionize: " << jy_ionize << "," << "jz_ionize: " << jz_ionize << endl;
	    //cout << "ii[LANE]: " << ii[LANE] << "," << "dx[LANE]: " << dx[LANE] << "," << "dy[LANE]: " << dy[LANE] << "," << "dz[LANE]: " << dz[LANE] << endl;

            //Declaration of local variables
            int ip, id, jp, jd, kp, kd;
            double xpn, xpmxip, xpmxid;
            double ypn, ypmyjp, ypmyjd;
            double zpn, zpmzkp, zpmzkd;
            double Sxp[2], Sxd[2], Syp[2], Syd[2], Szp[2], Szd[2];
	    int N_voxel_x,N_voxel_y,N_voxel_z,N_voxel_xy;
	    int voxel_indx,voxel_indy,voxel_indz;
            int xgrid,ygrid,zgrid;
            int y_start,z_start,y_end,z_end;
	    double xfrac,yfrac,zfrac;
	    double xpos,ypos,zpos;
	    double xcorner,ycorner,zcorner;

	    // Calculate grid coordinates from voxel index
            // (this logic has been checked to be true with the grid coordinate to VOXEL macro)
	    int nghost = 2; // tophat shape
            N_voxel_x   = (g->nx + nghost); // number of voxels in the x direction
            N_voxel_y   = (g->ny + nghost);
            N_voxel_z   = (g->nz + nghost);
            N_voxel_xy = N_voxel_x*N_voxel_y; // calculate the number of voxels in the xy plane
            zgrid = ceil( float(ii[LANE] + 1)/float( N_voxel_xy  ) - 1.0 ); // calculate the z-coordinate
            z_start = zgrid*N_voxel_xy;           // beginning voxel index for the z-coordinate
            z_end   = (zgrid+1) * N_voxel_xy - 1; // final voxel index for the z-coordinate
            ygrid = floor( float(ii[LANE]-z_start)/(float)N_voxel_x ); // calculate y-coordinate
            y_start = ((nx)+2)*((ygrid) + ((ny)+2)*(zgrid));// beginning voxel index for the y,z-coordinates
            y_end   = y_start + (N_voxel_x - 1); // final voxel index for the y,z-coordinates
            xgrid = ii[LANE]-y_start; // calculate x-coordinate

	    // calculate the normalized particle position (global)
	    xfrac = (dx[LANE] + 1)/2.0; // fraction from left edge of cell (middle of cell is 0.5)
	    yfrac = (dy[LANE] + 1)/2.0;
	    zfrac = (dz[LANE] + 1)/2.0;
	    xpos  = xgrid + xfrac;      // global position of particle (normalized)
	    ypos  = ygrid + yfrac;
	    zpos  = zgrid + zfrac;

            //top corner position of particle
            xcorner = xpos + 0.5;
            ycorner = ypos + 0.5;
	    zcorner = zpos + 0.5;
            
            // primal grid
            ip = round(xcorner); // closest node to top particle corner 
            jp = round(ycorner);
	    kp = round(zcorner);
            xpmxip  = xcorner - ( double )ip; // normalized distance from particle corner to node 
            ypmyjp  = ycorner - ( double )jp;
	    zpmzkp  = zcorner - ( double )kp;
            
            Sxp[1] = 0.5 + xpmxip; // weight
            Sxp[0] = 0.5 - xpmxip;
            Syp[1] = 0.5 + ypmyjp;
            Syp[0] = 0.5 - ypmyjp;
	    Szp[1] = 0.5 + zpmzkp;
            Szp[0] = 0.5 - zpmzkp;
            
            // staggered grid (staggered by 1/2 cell)
	    // Jx staggered in x, Jy staggered in y, Jz staggered in z
            id = round(xcorner + 0.5);
            jd = round(ycorner + 0.5);
	    kd = round(zcorner + 0.5);
            xpmxid  = (xcorner + 0.5) - ( double )id;
            ypmyjd  = (ycorner + 0.5) - ( double )jd;
	    zpmzkd  = (zcorner + 0.5) - ( double )kd;
            
            Sxd[1] = 0.5 + xpmxid;
            Sxd[0] = 0.5 - xpmxid;
            Syd[1] = 0.5 + ypmyjd;
            Syd[0] = 0.5 - ypmyjd;
	    Szd[1] = 0.5 + zpmzkd;
            Szd[0] = 0.5 - zpmzkd;

	    // FIXME: remove
	    //cout << "xgrid,ygrid,zgrid: " << xgrid << "," << ygrid << "," << zgrid << endl;
	    //cout << "xfrac,yfrac,zfrac: " << xfrac << "," << yfrac << "," << zfrac << endl;
	    //cout << "xpos,ypos,zpos: " << xpos<<"," << ypos<<"," << zpos << endl;
	    //cout << "xcorner,ycorner,zcorner: " << xcorner<<"," << ycorner<<"," << zcorner << endl;
            //cout << endl;
            //cout << "non-staggered" << endl;
            //cout << "ip,jp,kp: " << ip<<"," << jp<<"," << kp << endl;
            //cout << "xpmxip,ypmyjp,zpmzkp: " << xpmxip<<","<< ypmyjp<<","<< zpmzkp << endl;
            //cout << "Sxp[0], Sxp[1]: " << Sxp[0]<<"," <<Sxp[1] << endl;
            //cout << "Syp[0], Syp[1]: " << Syp[0]<<"," <<Syp[1] << endl;
            //cout << "Szp[0], Szp[1]: " << Szp[0]<<"," <<Szp[1] << endl;
            //cout << endl;   
            //cout << "staggered" << endl;
            //cout << "id,jd,kd: " << id<<"," << jd<<"," << kd << endl;
            //cout << "xpmxid,ypmyjd,zpmzkd: " << xpmxid<<"," <<ypmyjd<<"," <<zpmzkd << endl;
            //cout << "Sxd[0], Sxd[1]: " << Sxd[0]<<"," <<Sxd[1] << endl;
            //cout << "Syd[0], Syd[1]: " << Syd[0]<<"," <<Syd[1] << endl;
            //cout << "Szd[0], Szd[1]: " << Szd[0]<<"," <<Szd[1] << endl;
            //cout << endl;   
            
            //double TEMP = 0; // FIXME: remove
	    //int iter = 1; // FIXME: remove
            for (unsigned int i=0 ; i<2 ; i++) {
                int iploc=ip+i-1;
                int idloc=id+i-1;
                for (unsigned int j=0 ; j<2 ; j++) {
                    int jploc=jp+j-1;
                    int jdloc=jd+j-1;
		    for (unsigned int k=0 ; k<2 ; k++) {
                      int kploc=kp+k-1;
                      int kdloc=kd+k-1;
            	      
                      //cout << "iter: " << iter++<< endl; // FIXME: remove
		      
		      voxel_indx = VOXEL(idloc, jploc, kploc, g->nx,g->ny,g->nz); // Jx is staggered in x 
          	      voxel_indy = VOXEL(iploc, jdloc, kploc, g->nx,g->ny,g->nz); // Jy is staggered in y
          	      voxel_indz = VOXEL(iploc, jploc, kdloc, g->nx,g->ny,g->nz); // Jz is staggered in z
		      
		      Kokkos::atomic_fetch_add(&k_field(voxel_indx, field_var::jfx), jx_ionize * Sxd[i] * Syp[j] * Szp[k]); 
          	      Kokkos::atomic_fetch_add(&k_field(voxel_indy, field_var::jfy), jy_ionize * Sxp[i] * Syd[j] * Szp[k]); 
          	      Kokkos::atomic_fetch_add(&k_field(voxel_indz, field_var::jfz), jz_ionize * Sxp[i] * Syp[j] * Szd[k]);
                          
                      // x // FIXME: remove
                      //cout << "idloc, jploc, kploc: " << idloc<<","<< jploc<<","<< kploc << endl;
                      //cout << "Sxd[i]*Syp[j]*Szp[k]: " << Sxd[i]*Syp[j]*Szp[k] << endl;
                      //cout << "i,j,k: " << i <<","<<j<<","<<k<<endl;
                      //TEMP += Sxd[i]*Syp[j]*Szp[k];
		      
		      // y // FIXME: remove
                      //cout << "iploc, jdloc, kploc: " << iploc<<","<< jdloc<<","<< kploc << endl;
                      //cout << "Sxp[i]*Syd[j]*Szp[k]: " << Sxp[i]*Syd[j]*Szp[k] << endl;
                      //cout << "i,j,k: " << i <<","<<j<<","<<k<<endl;
                      //TEMP += Sxp[i]*Syd[j]*Szp[k];
		      
		      // z // FIXME: remove
                      //cout << "iter: " << iter << endl;
                      //cout << "iploc, jploc, kdloc: " << iploc<<","<< jploc<<","<< kdloc << endl;
                      //cout << "Sxp[i]*Syp[j]*Szd[k]: " << Sxp[i]*Syp[j]*Szd[k] << endl;
                      //cout << "i,j,k: " << i <<","<<j<<","<<k<<endl;
                      //TEMP += Sxp[i]*Syp[j]*Szd[k];
		      
		    }//k  
                }//j
            }//i

                
            //cout << "TEMP: " << TEMP << endl; // FIXME: remove

         } // if ionization event occured


	




	
	  


	  // FIXME: This needs to be replaced with a calulation on the hydro data
	  if (N_ionization_levels == 6) {
	    if (int(timestep)>=0 && int(timestep)<=2000 && int(timestep) % 4 == 0){
	     // N Ionizations: Open file, write value, close file
	     char gn [100];
	     snprintf(gn, sizeof gn, "N_ionizations_t_%g.txt",timestep);
	     std::ofstream outfile1;
	     if ((int)pi_offset == 0 && LANE == 0) {
               outfile1.open(gn); // overwrite old files
	       outfile1 << "# N ionizations Before" << "," << "# N ionizations After" << "," << "Charge After" << "," << "species name" << endl; // Header
	     }
	     else {
	       outfile1.open(gn, std::ios_base::app); // append to file
	     }
	     outfile1 << N_ionization_before << "," << N_ionization << "," << k_particles(particle_index, particle_var::charge) << "," << sp->name << endl;
             outfile1.close();
	   }
	  }
	   
	  
	} // if not electrons
	  
#endif // FIELD_IONIZATION


	
        // Interpolate B
        cbx[LANE] = fcbx[LANE] + dx[LANE]*fdcbxdx[LANE];
        cby[LANE] = fcby[LANE] + dy[LANE]*fdcbydy[LANE];
        cbz[LANE] = fcbz[LANE] + dz[LANE]*fdcbzdz[LANE];
  
        // Half advance e
        ux[LANE] += hax[LANE];
        uy[LANE] += hay[LANE];
        uz[LANE] += haz[LANE];
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
#ifdef FIELD_IONIZATION
        const float qdt_2mc = charge[LANE] * dt_2mc; 
#endif	
        v0[LANE] = qdt_2mc/sqrtf(one + (ux[LANE]*ux[LANE] + (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE])));
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
        p_index = pi_offset + LANE;

        // Boris - scalars
        v1[LANE] = cbx[LANE]*cbx[LANE] + (cby[LANE]*cby[LANE] + cbz[LANE]*cbz[LANE]);
        v2[LANE] = (v0[LANE]*v0[LANE])*v1[LANE];
        v3[LANE] = v0[LANE]*(one+v2[LANE]*(one_third+v2[LANE]*two_fifteenths));
        v4[LANE] = v3[LANE]/(one+v1[LANE]*(v3[LANE]*v3[LANE]));
        v4[LANE] += v4[LANE];
        // Boris - uprime
        v0[LANE] = ux[LANE] + v3[LANE]*(uy[LANE]*cbz[LANE] - uz[LANE]*cby[LANE]);
        v1[LANE] = uy[LANE] + v3[LANE]*(uz[LANE]*cbx[LANE] - ux[LANE]*cbz[LANE]);
        v2[LANE] = uz[LANE] + v3[LANE]*(ux[LANE]*cby[LANE] - uy[LANE]*cbx[LANE]);
        // Boris - rotation
        ux[LANE] += v4[LANE]*(v1[LANE]*cbz[LANE] - v2[LANE]*cby[LANE]);
        uy[LANE] += v4[LANE]*(v2[LANE]*cbx[LANE] - v0[LANE]*cbz[LANE]);
        uz[LANE] += v4[LANE]*(v0[LANE]*cby[LANE] - v1[LANE]*cbx[LANE]);
        // Half advance e
        ux[LANE] += hax[LANE];
        uy[LANE] += hay[LANE];
        uz[LANE] += haz[LANE];
        // Store momentum
        p_ux = ux[LANE];
        p_uy = uy[LANE];
        p_uz = uz[LANE];
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
        v0[LANE]   = one/sqrtf(one + (ux[LANE]*ux[LANE]+ (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE])));
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {

        /**/                                      // Get norm displacement
        ux[LANE]  *= cdt_dx;
        uy[LANE]  *= cdt_dy;
        uz[LANE]  *= cdt_dz;
        ux[LANE]  *= v0[LANE];
        uy[LANE]  *= v0[LANE];
        uz[LANE]  *= v0[LANE];
        v0[LANE]   = dx[LANE] + ux[LANE];                           // Streak midpoint (inbnds)
        v1[LANE]   = dy[LANE] + uy[LANE];
        v2[LANE]   = dz[LANE] + uz[LANE];
        v3[LANE]   = v0[LANE] + ux[LANE];                           // New position
        v4[LANE]   = v1[LANE] + uy[LANE];
        v5[LANE]   = v2[LANE] + uz[LANE];
  
        inbnds[LANE] = v3[LANE]<=one &&  v4[LANE]<=one &&  v5[LANE]<=one &&
                      -v3[LANE]<=one && -v4[LANE]<=one && -v5[LANE]<=one;
      } END_VECTOR_BLOCK;
    
#ifdef VPIC_ENABLE_TEAM_REDUCTION
      int in_cell = particles_in_same_cell(team_member, ii, inbnds, num_iters);
#endif

      BEGIN_VECTOR_BLOCK {
        p_index = pi_offset + LANE;

        v3[LANE] = static_cast<float>(inbnds[LANE])*v3[LANE] + (1.0-static_cast<float>(inbnds[LANE]))*p_dx;
        v4[LANE] = static_cast<float>(inbnds[LANE])*v4[LANE] + (1.0-static_cast<float>(inbnds[LANE]))*p_dy;
        v5[LANE] = static_cast<float>(inbnds[LANE])*v5[LANE] + (1.0-static_cast<float>(inbnds[LANE]))*p_dz;
#ifdef FIELD_IONIZATION
	q[LANE]  = static_cast<float>(inbnds[LANE])*q[LANE]*charge[LANE]; 
#else	
        q[LANE]  = static_cast<float>(inbnds[LANE])*q[LANE]*qsp;
#endif
        p_dx = v3[LANE];
        p_dy = v4[LANE];
        p_dz = v5[LANE];
        dx[LANE] = v0[LANE];
        dy[LANE] = v1[LANE];
        dz[LANE] = v2[LANE];
        v5[LANE] = q[LANE]*ux[LANE]*uy[LANE]*uz[LANE]*one_third;

#       define ACCUMULATE_J(X,Y,Z,v0,v1,v2,v3)                                              \
        v4[LANE]  = q[LANE]*u##X[LANE];   /* v2 = q ux                            */        \
        v1[LANE]  = v4[LANE]*d##Y[LANE];  /* v1 = q ux dy                         */        \
        v0[LANE]  = v4[LANE]-v1[LANE];    /* v0 = q ux (1-dy)                     */        \
        v1[LANE] += v4[LANE];             /* v1 = q ux (1+dy)                     */        \
        v4[LANE]  = one+d##Z[LANE];       /* v4 = 1+dz                            */        \
        v2[LANE]  = v0[LANE]*v4[LANE];    /* v2 = q ux (1-dy)(1+dz)               */        \
        v3[LANE]  = v1[LANE]*v4[LANE];    /* v3 = q ux (1+dy)(1+dz)               */        \
        v4[LANE]  = one-d##Z[LANE];       /* v4 = 1-dz                            */        \
        v0[LANE] *= v4[LANE];             /* v0 = q ux (1-dy)(1-dz)               */        \
        v1[LANE] *= v4[LANE];             /* v1 = q ux (1+dy)(1-dz)               */        \
        v0[LANE] += v5[LANE];             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
        v1[LANE] -= v5[LANE];             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
        v2[LANE] -= v5[LANE];             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
        v3[LANE] += v5[LANE];             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

        ACCUMULATE_J( x,y,z, v6,v7,v8,v9 );

        ACCUMULATE_J( y,z,x, v10,v11,v12,v13 );

        ACCUMULATE_J( z,x,y, v0,v1,v2,v3 );
      } END_VECTOR_BLOCK;

#ifdef VPIC_ENABLE_TEAM_REDUCTION
      if(in_cell) {
        int first = ii[0];
        reduce_and_accumulate_current(team_member, current_sa, num_iters, first, 
                                      nx, ny, nz, cx, cy, cz,
                                      v6, v7, v8, v9,
                                      v10, v11, v12, v13,
                                      v0, v1, v2, v3);
      } else {
#endif
        BEGIN_VECTOR_BLOCK {
          accumulate_current(current_sa, ii[LANE],
                       nx, ny, nz, cx, cy, cz, 
                       v6[LANE], v7[LANE], v8[LANE], v9[LANE],
                       v10[LANE], v11[LANE], v12[LANE], v13[LANE],
                       v0[LANE], v1[LANE], v2[LANE], v3[LANE]);
        } END_VECTOR_BLOCK;
#ifdef VPIC_ENABLE_TEAM_REDUCTION
      }
#endif
#       undef ACCUMULATE_J
      BEGIN_THREAD_BLOCK {
        if(!inbnds[LANE]) {
          p_index = pi_offset + LANE;

          DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
          local_pm->dispx = ux[LANE];
          local_pm->dispy = uy[LANE];
          local_pm->dispz = uz[LANE];
          local_pm->i     = p_index;

          if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
#ifdef FIELD_IONIZATION
			     current_sv, g, k_neighbors, rangel, rangeh, charge[LANE], cx, cy, cz, nx, ny, nz ) ) 
#else			     
                             current_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
#endif	    
          {
            if( k_nm(0)<max_nm ) {
              const unsigned int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
              if (nm >= max_nm) Kokkos::abort("overran max_nm");

              k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
              k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
              k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
              k_particle_movers_i(nm)   = local_pm->i;

              // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
              k_particle_copy(nm, particle_var::dx) = p_dx;
              k_particle_copy(nm, particle_var::dy) = p_dy;
              k_particle_copy(nm, particle_var::dz) = p_dz;
              k_particle_copy(nm, particle_var::ux) = p_ux;
              k_particle_copy(nm, particle_var::uy) = p_uy;
              k_particle_copy(nm, particle_var::uz) = p_uz;
              k_particle_copy(nm, particle_var::w) = p_w;
       #ifdef FIELD_IONIZATION    
	      k_particle_copy(nm, particle_var::charge) = p_q;
       #endif      
              k_particle_i_copy(nm) = pii;
            }
          }
        }
      } END_THREAD_BLOCK;
#if defined( VPIC_ENABLE_HIERARCHICAL ) && !defined( VPIC_ENABLE_VECTORIZATION )
      });
#endif
  });

#if defined( VPIC_ENABLE_ACCUMULATORS )
  Kokkos::Experimental::contribute(accumulator, current_sv);
  Kokkos::MDRangePolicy<Kokkos::Rank<3>> unload_policy({1, 1, 1}, {nz+2, ny+2, nx+2});
  Kokkos::parallel_for("unload accumulator array", unload_policy, 
  KOKKOS_LAMBDA(const int z, const int y, const int x) {
      int f0  = VOXEL(1, y, z, nx, ny, nz) + x-1;
      int a0  = VOXEL(1, y, z, nx, ny, nz) + x-1;
      int ax  = VOXEL(0, y, z, nx, ny, nz) + x-1;
      int ay  = VOXEL(1, y-1, z, nx, ny, nz) + x-1;
      int az  = VOXEL(1, y, z-1, nx, ny, nz) + x-1;
      int ayz = VOXEL(1, y-1, z-1, nx, ny, nz) + x-1;
      int azx = VOXEL(0, y, z-1, nx, ny, nz) + x-1;
      int axy = VOXEL(0, y-1, z, nx, ny, nz) + x-1;
      k_field(f0, field_var::jfx) += ( accumulator(a0, 0) +
                                       accumulator(ay, 1) +
                                       accumulator(az, 2) +
                                       accumulator(ayz, 3) );
      k_field(f0, field_var::jfy) += ( accumulator(a0, 4) +
                                       accumulator(az, 5) +
                                       accumulator(ax, 6) +
                                       accumulator(azx, 7) );
      k_field(f0, field_var::jfz) += ( accumulator(a0, 8) +
                                       accumulator(ax, 9) +
                                       accumulator(ay, 10) +
                                       accumulator(axy, 11) );
  });
#else
  Kokkos::Experimental::contribute(k_field, current_sv);
#endif

#undef p_dx
#undef p_dy
#undef p_dz
#undef p_ux
#undef p_uy
#undef p_uz
#undef p_w 
#undef pii

#ifdef FIELD_IONIZATION
#undef p_q_e

// FIXME: this is temporary
if(strcmp(sp->name, "electron") != 0){
  char kn [100];
  snprintf(kn, sizeof kn, "Photoelectrons.txt");
  std::ofstream outfile;
  if (g->step == 0) {
    outfile.open(kn); // overwrite old files
    outfile << g->step << "," << sp->np << "," << sp_e->np << "," << sp->name << endl;
  } else {
    outfile.open(kn, std::ios_base::app); // append to file
    outfile << g->step << "," << sp->np << "," << sp_e->np << "," << sp->name << endl;
  }
  outfile.close();
}
#endif

#undef f_cbx
#undef f_cby
#undef f_cbz
#undef f_ex 
#undef f_ey 
#undef f_ez 

#undef f_dexdy
#undef f_dexdz

#undef f_d2exdydz
#undef f_deydx   
#undef f_deydz   

#undef f_d2eydzdx
#undef f_dezdx   
#undef f_dezdy   

#undef f_d2ezdxdy
#undef f_dcbxdx  
#undef f_dcbydy  
#undef f_dcbzdz  
}

		       

void
advance_p_kokkos_gpu(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_particle_copy_t& k_particle_copy,
        k_particle_i_copy_t& k_particle_i_copy,
        k_particle_movers_t& k_particle_movers,
        k_particle_i_movers_t& k_particle_movers_i,
        k_field_sa_t k_f_sa,
        k_interpolator_t& k_interp,
        k_counter_t& k_nm,
        k_neighbor_t& k_neighbors,
        field_array_t* RESTRICT fa,
        const grid_t *g,
#ifdef FIELD_IONIZATION	
        const float dt_2mc,
#else
        const float qdt_2mc,
#endif	
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
#ifndef	FIELD_IONIZATION
        const float qsp,
#endif
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
#ifdef	FIELD_IONIZATION	
        const int nz,
	species_t * RESTRICT sp,
	species_t * sp_e)
#else
        const int nz)
#endif
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;
  k_field_t k_field = fa->k_f_d;
  k_field_sa_t k_f_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

  // Process particles for this pipeline
  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux)
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define p_w     k_particles(p_index, particle_var::w)
#ifdef FIELD_IONIZATION
  #define p_q     k_particles(p_index, particle_var::charge)
#endif    
  #define pii     k_particles_i(p_index)

  #define f_cbx k_interp(ii, interpolator_var::cbx)
  #define f_cby k_interp(ii, interpolator_var::cby)
  #define f_cbz k_interp(ii, interpolator_var::cbz)
  #define f_ex  k_interp(ii, interpolator_var::ex)
  #define f_ey  k_interp(ii, interpolator_var::ey)
  #define f_ez  k_interp(ii, interpolator_var::ez)

  #define f_dexdy    k_interp(ii, interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii, interpolator_var::dexdz)

  #define f_d2exdydz k_interp(ii, interpolator_var::d2exdydz)
  #define f_deydx    k_interp(ii, interpolator_var::deydx)
  #define f_deydz    k_interp(ii, interpolator_var::deydz)

  #define f_d2eydzdx k_interp(ii, interpolator_var::d2eydzdx)
  #define f_dezdx    k_interp(ii, interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii, interpolator_var::dezdy)

  #define f_d2ezdxdy k_interp(ii, interpolator_var::d2ezdxdy)
  #define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
  #define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
  #define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)

  // copy local memmbers from grid
  //auto nfaces_per_voxel = 6;
  //auto nvoxels = g->nv;
  //Kokkos::View<int64_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      //h_neighbors(g->neighbor, nfaces_per_voxel * nvoxels);
  //auto d_neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), h_neighbors);

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // zero out nm, we could probably do this earlier if we're worried about it
  // slowing things down
  Kokkos::deep_copy(k_nm, 0);

#ifdef FIELD_IONIZATION
  // constants, FIXME: need to get from deck
  int q_e_c     = -1; // code units
  float E_to_SI = 1.44303994037981860e+19; // code to SI
  float t_to_SI = 1.18119324097025572e-22; // code to SI
  float l_to_SI = 3.54112825083459245e-14; // code to SI
  float q_to_SI = 1.60217663399999989e-19; // code to SI
  float m_to_SI = 9.10938370150000079e-31; // code to SI
  float time_to_SI = 1.18119324097025572e-22; // code to SI
  float energy_to_SI = m_to_SI * (l_to_SI/time_to_SI) * (l_to_SI/time_to_SI); // code to SI
  float dt = g->dt*time_to_SI; // [s], timestep
  float q_e       = 1.60217663e-19;  // coulombs
  float q_e_au    = 1.0;             // au
  float m_e       = 9.1093837e-31;   // kilograms
  float m_e_au    = 1.0;             // au 
  float c         = 299792458;       // m/s
  float c_au      = 137.02;          // atomic units
  float epsilon_0_au = 1/(4*M_PI);   // au
  float alpha     = 0.00729735;      // fine structure constant
  float h_bar     = 1.054571817e-34; // J *s
  float E_field_conversion = 5.1422e+11;      // (alpha^3*m_e^2*c^3)/(q_e*h_bar), multiply AU to get SI
  float Gamma_conversion   = 1.0/(h_bar/(pow(alpha,2)*m_e*pow(c,2))); // multiply au to get sec^-1 
    
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  Kokkos::Random_XorShift64_Pool<> random_pool(seed);
  auto sp_name = sp->name;
  k_particles_t& k_electrons = sp_e->k_p_d; 
  k_particles_i_t& k_electrons_i = sp_e->k_p_i_d;  
  Kokkos::View<int> count("count");
  Kokkos::deep_copy(count, sp_e->np);
  int timestep = g->step;
  Kokkos::View<double*> epsilon_eV_list = sp->ionization_energy;
  float n = sp->qn; // principal quantum number
  float m = sp->qm; // magnetic quantum number
  float l = sp->ql; // angular momentum quantum number
  float lambda = g->lambda;
#endif
  
#ifdef VPIC_ENABLE_HIERARCHICAL
  auto team_policy = Kokkos::TeamPolicy<>(LEAGUE_SIZE, TEAM_SIZE);
  int per_league = np/LEAGUE_SIZE;
  if(np%LEAGUE_SIZE > 0)
    per_league += 1;
  Kokkos::parallel_for("advance_p", team_policy, KOKKOS_LAMBDA(const KOKKOS_TEAM_POLICY_DEVICE::member_type team_member) {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, per_league), [=] (size_t pindex) {
      int p_index = team_member.league_rank()*per_league + pindex;
      if(p_index < np) {
#else
  auto range_policy = Kokkos::RangePolicy<>(0,np);
  Kokkos::parallel_for("advance_p", range_policy, KOKKOS_LAMBDA (size_t p_index) {
#endif

      
    float v0, v1, v2, v3, v4, v5;
    auto  k_field_scatter_access = k_f_sv.access();

    float dx   = p_dx;                             // Load position
    float dy   = p_dy;
    float dz   = p_dz;
    int   ii   = pii;
#ifdef FIELD_IONIZATION    
    float charge = p_q;
    const float qdt_2mc = charge * dt_2mc;
#endif    
    float hax  = qdt_2mc*(    ( f_ex    + dy*f_dexdy    ) + // Interpolate E
                           dz*( f_dexdz + dy*f_d2exdydz ) );
    float hay  = qdt_2mc*(    ( f_ey    + dz*f_deydz    ) +
                           dx*( f_deydx + dz*f_d2eydzdx ) );
    float haz  = qdt_2mc*(    ( f_ez    + dx*f_dezdx    ) +
                           dy*( f_dezdy + dx*f_d2ezdxdy ) );

#ifdef FIELD_IONIZATION
    // ***** Field Ioization *****
    // Declate varviables
    bool multiphoton_ionised = false;
    float K;
	
    // FIXME: need to check which species the user wants ionization enabled on
    if (sp != sp_e){
    // Check if the particle is fully ionized already
    int N_ionization        = int(abs(charge)); // Current ionization state of the particle
    int N_ionization_before = N_ionization; // save variable to compare with ionization state after ionization algorithm
    int N_ionization_levels = epsilon_eV_list.extent(0);
    
    // code units
    float hax_c = hax/qdt_2mc;
    float hay_c = hay/qdt_2mc;
    float haz_c = haz/qdt_2mc;
    float ha_mag_c = sqrtf(pow(hax_c,2.0)+pow(hay_c,2.0)+pow(haz_c,2.0));
    // SI units
    float E_mag_SI = E_to_SI * ha_mag_c;

    // Calculate the ionization rate and number of ionizations
    float lambda_SI    = lambda*l_to_SI;  // meters	
  
    // Calculate stuff
    float E_au       = E_mag_SI/E_field_conversion; // field strength, atomic units
    float nu         = c/lambda_SI; // Hz
    float omega_SI   = 2*M_PI*nu;   // Hz
    float omega_eV   = 1.2398/(lambda_SI/1e-6); // eV
    float omega_au   = omega_eV * 0.036749; // energy, Hartree units
    float I_au       = 0.5*c_au*epsilon_0_au*pow(E_au,2.0); // intensity from the field, atomic units

    // initialize variables for while loop
    int ionization_flag = 1;
    float t_ionize      = 0;
    float Gamma = 0;
    // loop for multiple ionization events in a single timestep
    while (ionization_flag == 1 && t_ionize <= dt && N_ionization < N_ionization_levels) {
        
      // Get the appropriate ionization energy
      float epsilon_eV = epsilon_eV_list(int(N_ionization)); // [eV], ionization energy
      float epsilon_au = epsilon_eV/27.2;         // atomic units, ionization energy
      float epsilon_SI = epsilon_au*4.3597463e-18;// [J],  ionization energy
      
      // Calculate stuff
      K                    = floor(epsilon_au/omega_au)+1; // number of photons required for multiphoton ionization       
      float Z              = N_ionization + 1;          // ion charge number after ionization
      float Z_star         = N_ionization; // initial charge state
      float n_star         = (Z_star + 1.0)/sqrt(2*epsilon_au); // effective principle quantum number
      float l_star         = n_star - 1.0; // angular momentum
      float T_0            = M_PI*Z/(abs(epsilon_au) * sqrt(2*abs(epsilon_au))); // period of classical radial trajectories
      float gamma_keldysh  = omega_au*sqrt(2*m_e_au*epsilon_au)/(q_e_au*E_au);
      
      // Ionization events are tested for every particle with a bound electron at every timestep
      
      // Choose the ionization process based on the E-field at the particle
      // Specifically, Gamma =
      // min( Gamma_MPI, Gamma_ADK ) for        E <= E_M
      // Gamma_ADK                   for E_M <= E <= E_T
      // min( Gamma_ADK, Gamma_BSI ) for E_T <= E <= E_B
      // Gamma_BSI                   for        E >  E_B
      // ** NOTE: E_B is defined such that dGamma_ADK(E_B)/dE = 0 so that we have a monotonically increasing rate. 
      
      // Choose the ionization process based on |E| at each particle
      // Note E_T = epsilon^2/(4*Z) is the correct version but EPOCH uses epsilon^2/Z for some reason (maybe a typo in their paper?)
      float E_M_SI = 2*omega_SI*sqrt(2*m_e*epsilon_SI)/q_e;
      float E_M_au = omega_au*sqrt(8*epsilon_au); // atomic units
      float E_T_au = pow(epsilon_au,2.0)/(4*Z);      // atomic units
      float E_B_au = (6*m*pow(n,3.0) + 4*pow(Z,3.0))/(12*pow(n,4.0) - 9*pow(n,3.0)); // atomic units 
      
      if (E_au<=E_M_au){
	// MPI Ionization
        // ionization rate per atom: Gamma^(K)
        // sigma^(K) = (h_bar*omega)^K*Gamma^(K)/I^K : K-photon cross section, units [cm^2K * s^(K-1)]
        // I: intensity of the laser field, units [W/cm^2]
        //cout << "MPI Ionization" << endl;
	multiphoton_ionised = true;
        // FIXME: need to add the case of circularly polarized field
        float T_K = 4.80*pow(1.30,2*K)*pow(2*K+1,-1)*pow(K,-1.0/2.0); // in the case of linearly polarized field
        float sigma_K_au = pow(c_au*pow(tgamma(K+1),2)*pow(n,5)* pow(omega_au,(10*K-1)/3), -1)*T_K*pow(E_au,2*K-2); // atomic units, [cm^2K * s^(K-1)]
	float flux = c_au*pow(E_au, 2.0)/(8*M_PI*omega_au);
        float Gamma_MPI = sigma_K_au * pow(flux, K);
        
        // Tunneling Regime
        //cout << "Tunneling Ionization" << endl;
        float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
        float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
        float Gamma_ADK_au = C_nstar_lstar_squared * f_n_l * epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_au));
        float Gamma_ADK_SI = Gamma_ADK_au*Gamma_conversion;
        
        Gamma = min(Gamma_MPI,Gamma_ADK_SI); 
	if(Gamma_MPI < Gamma_ADK_SI){ multiphoton_ionised = true; }
      }
      
      else if (E_au>E_M_au && E_au<=E_T_au) {
        // Tunneling Regime
        //cout << "Tunneling Ionization" << endl;
        float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
        float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
        float Gamma_ADK_au = C_nstar_lstar_squared * f_n_l * epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_au));
        float Gamma_ADK_SI = Gamma_ADK_au*Gamma_conversion;
        Gamma = Gamma_ADK_SI;
      }
      
      else if (E_au>E_T_au && E_au<=E_B_au){
        // Either classical ADK or with BSI correction
        // Whichever has the minimum ionization rate
        //cout << "Tunneling or BSI Regime" << endl;
        // ADK (no correction)
        float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
        float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
        float Gamma_ADK_au = C_nstar_lstar_squared * f_n_l * epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_au));
        float Gamma_ADK_SI = Gamma_ADK_au*Gamma_conversion;
      
        // With BSI correction: Gamma = Gamma_classical + Gamma_ADK(I_classical)
        // Gamma_ADK(I_classical): ADK at the classical appearanace (threshold) intensity, i.e, the intensity that corresponds to the threshold E-field magnitude
        float Gamma_ADK_au_threshold = C_nstar_lstar_squared* f_n_l* epsilon_au * pow( 2*pow(2*epsilon_au, 3.0/2.0)/E_T_au, 2*n_star-abs(m)-1) * exp(-2*pow(2*epsilon_au,3.0/2.0)/(3*E_T_au));
        float Gamma_ADK_SI_threshold = Gamma_ADK_au_threshold*Gamma_conversion;
        // uniform field, FIXME: enable this
        // gamma_cl_uniform_atomic = (1 - E_n_atomic^2./(4*Z*E_atomic))/(2*T_0);
        // gamma_cl_uniform_SI = gamma_cl_uniform_atomic / (h_bar/(alpha^2*m_e*c^2));
        // oscillating field
        float Gamma_cl_au  = 1.0/(M_PI*T_0) * (  M_PI/2.0 - asin( pow(epsilon_au,2.0)/(4*Z*E_au)) + pow(epsilon_au,2.0)/(4*Z*E_au) * log( ( 4*Z*E_au - sqrt( 16*pow(Z,2.0)*pow(E_au,2.0) - pow(epsilon_au,4.0) ) )/pow(epsilon_au,2.0) ) );
        float Gamma_cl_SI  = Gamma_cl_au* Gamma_conversion;
        float Gamma_BSI_SI = Gamma_cl_SI + Gamma_ADK_SI_threshold;
      
        // Decide if the BSI correction is applicable
        Gamma = min(Gamma_ADK_SI, Gamma_BSI_SI);
      
        /*
        if (Gamma_ADK_SI < Gamma_BSI_SI) {
          cout << "Tunneling Ionization" << endl;
        }
        else {
          cout << "BSI Correction" << endl;
        }
        */
          
      }
      
      else if (E_au>E_B_au) {
        // BSI Ionization
        //cout << "BSI Ionization" << endl;
        // BSI Ionization: Gamma = Gamma_classical + Gamma_ADK(I_classical)
        // Gamma_ADK(I_classical): ADK at the classical appearanace (threshold) intensity
        float f_n_l = ( ( 2*l+1 ) * tgamma( l+abs(m)+1 ) ) / ( pow(2,abs(m))*tgamma(abs(m)+1)*tgamma(l-abs(m)+1) );
        float C_nstar_lstar_squared = pow(2,2*n_star)/(n_star*tgamma(n_star+l_star+1)*tgamma(n_star-l_star));
        float Gamma_ADK_au_threshold = C_nstar_lstar_squared*f_n_l * sqrt(3*E_T_au*pow(n_star,3.0)/(M_PI*pow(Z,3.0))) * pow(Z,2.0)/(2*pow(n_star,2.0)) * pow(2*pow(Z,3.0)/(E_T_au*pow(n_star,3.0)),2*n_star-abs(m)-1)*exp(-2*pow(Z,3.0)/(3*pow(n_star,3.0)*E_T_au));
        float Gamma_ADK_SI_threshold = Gamma_ADK_au_threshold*Gamma_conversion;
        // uniform field, FIXME: enable this
        // gamma_cl_uniform_atomic = (1 - E_n_atomic^2./(4*Z*E_atomic))/(2*T_0);
        // oscillating field
        float Gamma_cl_au  = 1.0/(M_PI*T_0) * (  M_PI/2.0 - asin( pow(epsilon_au,2.0)/(4*Z*E_au)) + pow(epsilon_au,2.0)/(4*Z*E_au) * log( ( 4*Z*E_au - sqrt( 16*pow(Z,2.0)*pow(E_au,2.0) - pow(epsilon_au,4.0) ) )/pow(epsilon_au,2.0) ) );
        float Gamma_cl_SI  = Gamma_cl_au* Gamma_conversion;
        float Gamma_BSI_SI = Gamma_cl_SI + Gamma_ADK_SI_threshold;
        Gamma = Gamma_BSI_SI;
      }
      
      
      // Ionization occurs if U_1 < 1 - exp(-Gamma * delta_t), for a uniform number U_1~[0,1]
      auto generator = random_pool.get_state();
      double U = generator.drand(0,1);
      random_pool.free_state(generator);

      if ( U < 1 - exp(-Gamma * (dt-t_ionize) ) ) {
        // ionization occurs
        N_ionization++; 
        ionization_flag = 1;
        // deal with multiple ionizations
	t_ionize = -1.0/Gamma * log(1-U); // use previous U to calc
      } 
      else {
        // ionization doesnt occur
        ionization_flag = 0;
      }
      
    } // end while loop

    // Check if ionization event occured
    if (N_ionization_before < N_ionization){
        // Change the charge of the particle
        k_particles(p_index, particle_var::charge) = N_ionization * abs(q_e_c); // code units

        // Inject the macro electron with the same
        // momentum and position as the ionized particle
        // Multiple ionization events are enabled so the injected
        // electron weight needs to account for that
        #define p_dx_e    k_electrons(electron_index, particle_var::dx)
        #define p_dy_e    k_electrons(electron_index, particle_var::dy)
        #define p_dz_e    k_electrons(electron_index, particle_var::dz)
        #define p_ux_e    k_electrons(electron_index, particle_var::ux)
        #define p_uy_e    k_electrons(electron_index, particle_var::uy)
        #define p_uz_e    k_electrons(electron_index, particle_var::uz)
        #define p_q_e     k_electrons(electron_index, particle_var::charge)
        #define p_w_e     k_electrons(electron_index, particle_var::w)
        #define pii_e     k_electrons_i(electron_index)
	
	int electron_index = Kokkos::atomic_fetch_add(&count(),1);
        // get the positions and momentum from ionized species
	p_dx_e = dx;
        p_dy_e = dy;
        p_dz_e = dz;
        pii_e  = ii;
        p_ux_e = p_ux;
        p_uy_e = p_uy;
        p_uz_e = p_uz;
        p_q_e  = q_e_c; // electrons charge in code units
        p_w_e  = (N_ionization - N_ionization_before) * k_particles(p_index, particle_var::w); // weight is dependent on the number of ionization events and weight of ionized particle

	// With multiphoton ionization, the additional energy from photons
        // accelerates the electron in the direction of the electric field.
        // This is an approximation as the ejection angle ranges widely
        // with a maxima at theta = 0 with respect to the field
	float epsilon_t_au = 0; // initialize the total energy from multiple ionizations
    	float epsilon_t_c  = 0; 
    	float epsilon_t_SI = 0;
	if(multiphoton_ionised){
	  for(int i=0; i<int(N_ionization-N_ionization_before); i++) {
    	    epsilon_t_SI += epsilon_eV_list[i] * q_e; // joules
          }
	  double p_correction = sqrt( 2*m_e*(K*h_bar*omega_SI - epsilon_t_SI) )/energy_to_SI; // code units
	  p_ux_e += p_correction * hax_c/(ha_mag_c*ha_mag_c);
	  p_uy_e += p_correction * hay_c/(ha_mag_c*ha_mag_c);
	  p_uz_e += p_correction * haz_c/(ha_mag_c*ha_mag_c);	
	}

        #undef p_dx_e
        #undef p_dy_e
        #undef p_dz_e
        #undef p_ux_e
        #undef p_uy_e
        #undef p_uz_e
        #undef p_w_e
        #undef pii_e
      #ifdef FIELD_IONIZATION
	#undef p_q_e
      #endif


	// Energy conservation is accounted for by a current density correction through Poynting’s theorem (J_ionize = (N*epsilon_t)/(dt*E) E_hat atomic units)
        // in tunnelling ionisation and BSI the energy loss from the field is the ionisation energy of the electron
        // in multiphoton ionisation it is the total energy for the number of photons absorbed.
        // The total energy loss from multiple ionisations is summed and a current density correction is weighted back to the grid points
    	float N_ions = k_particles(p_index, particle_var::w); // this is the number of physical ions created  
	if(multiphoton_ionised) {
	  epsilon_t_c = (K*h_bar*omega_SI)/energy_to_SI; // code units
	} 
	else {
          for(int i=0; i<int(N_ionization-N_ionization_before); i++) {
            epsilon_t_au += epsilon_eV_list[i]/27.2; // atomic units, ionization energy
    	    epsilon_t_SI += epsilon_eV_list[i] * q_e; // joules
    	    epsilon_t_c  += (epsilon_eV_list[i] * q_e) / energy_to_SI; // code units
          }
	} 
    
        float cell_volume_c  = g->dV;
        float cell_volume_SI = cell_volume_c * l_to_SI*l_to_SI*l_to_SI;
        float cell_volume_au = cell_volume_SI/pow(5.29177210903e-11,3.0);
    	
    	float dt_au = dt / 2.41884e-17; // convsersion factor is h_bar/Eh where Eh is hartree energy

	float j_ionize_SI_x = epsilon_t_SI * N_ions * (hax_c * E_to_SI) / (dt * cell_volume_SI * E_mag_SI*E_mag_SI);
	float j_ionize_SI_y = epsilon_t_SI * N_ions * (hay_c * E_to_SI) / (dt * cell_volume_SI * E_mag_SI*E_mag_SI);
	float j_ionize_SI_z = epsilon_t_SI * N_ions * (haz_c * E_to_SI) / (dt * cell_volume_SI * E_mag_SI*E_mag_SI);
	float j_ionize_SI_mag = sqrt( j_ionize_SI_x*j_ionize_SI_x + j_ionize_SI_y*j_ionize_SI_y + j_ionize_SI_z*j_ionize_SI_z );

    	float j_ionize_mag_c  = (N_ions * epsilon_t_c )/(g->dt    * ha_mag_c * cell_volume_c ); // code
	float jx_ionize = (epsilon_t_c * N_ions * hax_c) / (g->dt * cell_volume_c * ha_mag_c*ha_mag_c);
	float jy_ionize = (epsilon_t_c * N_ions * hay_c) / (g->dt * cell_volume_c * ha_mag_c*ha_mag_c);
	float jz_ionize = (epsilon_t_c * N_ions * haz_c) / (g->dt * cell_volume_c * ha_mag_c*ha_mag_c);

	//float j_to_SI = 1.08169825311233e30; //q_to_SI/(t_to_SI * l_to_SI*l_to_SI); // code units to SI

	// FIXME: remove
	//cout << "*************" << endl;
	//cout << "electron_index: " << electron_index << ", timestep: " << timestep << endl;  
	//cout << "epsilon_t_SI: " << epsilon_t_SI << "," << "weight: " << N_ions << "," << "dt: " << dt << "," << "V_cell" << cell_volume_SI << "," << endl;
	//cout << "E_mag_SI: " << E_mag_SI << "hax_c * E_to_SI: " << hax_c * E_to_SI << "," << "hay_c * E_to_SI: " << hay_c * E_to_SI << "," << "haz_c * E_to_SI: " << haz_c * E_to_SI << endl;
	//cout << "j_ionize_SI_mag: " << j_ionize_SI_mag << "," <<  "j_ionize_SI_x: " << j_ionize_SI_x << "," <<  "j_ionize_SI_y: " << j_ionize_SI_y << "," <<  "j_ionize_SI_z: " << j_ionize_SI_z << endl;
	//cout << "j_ionize_mag_c: " << j_ionize_mag_c << "," << "jx_ionize: " << jx_ionize << "," << "jy_ionize: " << jy_ionize << "," << "jz_ionize: " << jz_ionize << endl;
	//cout << "ii: " << ii << "," << "dx: " << dx << "," << "dy: " << dy << "," << "dz: " << dz << endl;

        //Declaration of local variables
        int ip, id, jp, jd, kp, kd;
        double xpn, xpmxip, xpmxid;
        double ypn, ypmyjp, ypmyjd;
        double zpn, zpmzkp, zpmzkd;
        double Sxp[2], Sxd[2], Syp[2], Syd[2], Szp[2], Szd[2];
	int N_voxel_x,N_voxel_y,N_voxel_z,N_voxel_xy;
	int voxel_indx,voxel_indy,voxel_indz;
        int xgrid,ygrid,zgrid;
        int y_start,z_start,y_end,z_end;
	double xfrac,yfrac,zfrac;
	double xpos,ypos,zpos;
	double xcorner,ycorner,zcorner;

	// Calculate grid coordinates from voxel index
        // (this logic has been checked to be true with the grid coordinate to VOXEL macro)
	int nghost = 2; // tophat shape
        N_voxel_x   = (g->nx + nghost); // number of voxels in the x direction
        N_voxel_y   = (g->ny + nghost);
        N_voxel_z   = (g->nz + nghost);
        N_voxel_xy = N_voxel_x*N_voxel_y; // calculate the number of voxels in the xy plane
        zgrid = ceil( float(ii + 1)/float( N_voxel_xy  ) - 1.0 ); // calculate the z-coordinate
        z_start = zgrid*N_voxel_xy;           // beginning voxel index for the z-coordinate
        z_end   = (zgrid+1) * N_voxel_xy - 1; // final voxel index for the z-coordinate
        ygrid = floor( float(ii-z_start)/(float)N_voxel_x ); // calculate y-coordinate
        y_start = ((nx)+2)*((ygrid) + ((ny)+2)*(zgrid));// beginning voxel index for the y,z-coordinates
        y_end   = y_start + (N_voxel_x - 1); // final voxel index for the y,z-coordinates
        xgrid = ii-y_start; // calculate x-coordinate

	// calculate the normalized particle position (global)
	xfrac = (dx + 1)/2.0; // fraction from left edge of cell (middle of cell is 0.5)
	yfrac = (dy + 1)/2.0;
	zfrac = (dz + 1)/2.0;
	xpos  = xgrid + xfrac;      // global position of particle (normalized)
	ypos  = ygrid + yfrac;
	zpos  = zgrid + zfrac;

        //top corner position of particle
        xcorner = xpos + 0.5;
        ycorner = ypos + 0.5;
	zcorner = zpos + 0.5;
        
        // primal grid
        ip = round(xcorner); // closest node to top particle corner 
        jp = round(ycorner);
	kp = round(zcorner);
        xpmxip  = xcorner - ( double )ip; // normalized distance from particle corner to node 
        ypmyjp  = ycorner - ( double )jp;
	zpmzkp  = zcorner - ( double )kp;
        
        Sxp[1] = 0.5 + xpmxip; // weight
        Sxp[0] = 0.5 - xpmxip;
        Syp[1] = 0.5 + ypmyjp;
        Syp[0] = 0.5 - ypmyjp;
	Szp[1] = 0.5 + zpmzkp;
        Szp[0] = 0.5 - zpmzkp;
        
        // staggered grid (staggered by 1/2 cell)
	// Jx staggered in x, Jy staggered in y, Jz staggered in z
        id = round(xcorner + 0.5);
        jd = round(ycorner + 0.5);
	kd = round(zcorner + 0.5);
        xpmxid  = (xcorner + 0.5) - ( double )id;
        ypmyjd  = (ycorner + 0.5) - ( double )jd;
	zpmzkd  = (zcorner + 0.5) - ( double )kd;
        
        Sxd[1] = 0.5 + xpmxid;
        Sxd[0] = 0.5 - xpmxid;
        Syd[1] = 0.5 + ypmyjd;
        Syd[0] = 0.5 - ypmyjd;
	Szd[1] = 0.5 + zpmzkd;
        Szd[0] = 0.5 - zpmzkd;

	// FIXME: remove
	//cout << "xgrid,ygrid,zgrid: " << xgrid << "," << ygrid << "," << zgrid << endl;
	//cout << "xfrac,yfrac,zfrac: " << xfrac << "," << yfrac << "," << zfrac << endl;
	//cout << "xpos,ypos,zpos: " << xpos<<"," << ypos<<"," << zpos << endl;
	//cout << "xcorner,ycorner,zcorner: " << xcorner<<"," << ycorner<<"," << zcorner << endl;
        //cout << endl;
        //cout << "non-staggered" << endl;
        //cout << "ip,jp,kp: " << ip<<"," << jp<<"," << kp << endl;
        //cout << "xpmxip,ypmyjp,zpmzkp: " << xpmxip<<","<< ypmyjp<<","<< zpmzkp << endl;
        //cout << "Sxp[0], Sxp[1]: " << Sxp[0]<<"," <<Sxp[1] << endl;
        //cout << "Syp[0], Syp[1]: " << Syp[0]<<"," <<Syp[1] << endl;
        //cout << "Szp[0], Szp[1]: " << Szp[0]<<"," <<Szp[1] << endl;
        //cout << endl;   
        //cout << "staggered" << endl;
        //cout << "id,jd,kd: " << id<<"," << jd<<"," << kd << endl;
        //cout << "xpmxid,ypmyjd,zpmzkd: " << xpmxid<<"," <<ypmyjd<<"," <<zpmzkd << endl;
        //cout << "Sxd[0], Sxd[1]: " << Sxd[0]<<"," <<Sxd[1] << endl;
        //cout << "Syd[0], Syd[1]: " << Syd[0]<<"," <<Syd[1] << endl;
        //cout << "Szd[0], Szd[1]: " << Szd[0]<<"," <<Szd[1] << endl;
        //cout << endl;   
        
        //double TEMP = 0; // FIXME: remove
	//int iter = 1; // FIXME: remove
        for (unsigned int i=0 ; i<2 ; i++) {
            int iploc=ip+i-1;
            int idloc=id+i-1;
            for (unsigned int j=0 ; j<2 ; j++) {
                int jploc=jp+j-1;
                int jdloc=jd+j-1;
		    for (unsigned int k=0 ; k<2 ; k++) {
                      int kploc=kp+k-1;
                      int kdloc=kd+k-1;
        	      
                      //cout << "iter: " << iter++<< endl; // FIXME: remove
		      
		      voxel_indx = VOXEL(idloc, jploc, kploc, g->nx,g->ny,g->nz); // Jx is staggered in x 
        	      voxel_indy = VOXEL(iploc, jdloc, kploc, g->nx,g->ny,g->nz); // Jy is staggered in y
        	      voxel_indz = VOXEL(iploc, jploc, kdloc, g->nx,g->ny,g->nz); // Jz is staggered in z
		      
		      Kokkos::atomic_fetch_add(&k_field(voxel_indx, field_var::jfx), jx_ionize * Sxd[i] * Syp[j] * Szp[k]); 
        	      Kokkos::atomic_fetch_add(&k_field(voxel_indy, field_var::jfy), jy_ionize * Sxp[i] * Syd[j] * Szp[k]); 
        	      Kokkos::atomic_fetch_add(&k_field(voxel_indz, field_var::jfz), jz_ionize * Sxp[i] * Syp[j] * Szd[k]);
                      
                      // x // FIXME: remove
                      //cout << "idloc, jploc, kploc: " << idloc<<","<< jploc<<","<< kploc << endl;
                      //cout << "Sxd[i]*Syp[j]*Szp[k]: " << Sxd[i]*Syp[j]*Szp[k] << endl;
                      //cout << "i,j,k: " << i <<","<<j<<","<<k<<endl;
                      //TEMP += Sxd[i]*Syp[j]*Szp[k];
		          
		      // y // FIXME: remove
                      //cout << "iploc, jdloc, kploc: " << iploc<<","<< jdloc<<","<< kploc << endl;
                      //cout << "Sxp[i]*Syd[j]*Szp[k]: " << Sxp[i]*Syd[j]*Szp[k] << endl;
                      //cout << "i,j,k: " << i <<","<<j<<","<<k<<endl;
                      //TEMP += Sxp[i]*Syd[j]*Szp[k];
		          
		          // z // FIXME: remove
                      //cout << "iter: " << iter << endl;
                      //cout << "iploc, jploc, kdloc: " << iploc<<","<< jploc<<","<< kdloc << endl;
                      //cout << "Sxp[i]*Syp[j]*Szd[k]: " << Sxp[i]*Syp[j]*Szd[k] << endl;
                      //cout << "i,j,k: " << i <<","<<j<<","<<k<<endl;
                      //TEMP += Sxp[i]*Syp[j]*Szd[k];
		    }//k       
            }//j
        }//i
            
        //cout << "TEMP: " << TEMP << endl; // FIXME: remove

       
      } // if ionization event occured
    
      } // if not electrons
#endif // FIELD_IONIZATION

    float cbx  = f_cbx + dx*f_dcbxdx;             // Interpolate B
    float cby  = f_cby + dy*f_dcbydy;
    float cbz  = f_cbz + dz*f_dcbzdz;
    float ux   = p_ux;                             // Load momentum
    float uy   = p_uy;
    float uz   = p_uz;
    float q    = p_w;
    ux  += hax;                               // Half advance Er
    uy  += hay;
    uz  += haz;
    v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
    /**/                                      // Boris - scalars
    v1   = cbx*cbx + (cby*cby + cbz*cbz);
    v2   = (v0*v0)*v1;
    v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
    v4   = v3/(one+v1*(v3*v3));
    v4  += v4;
    v0   = ux + v3*( uy*cbz - uz*cby );       // Boris - uprime
    v1   = uy + v3*( uz*cbx - ux*cbz );
    v2   = uz + v3*( ux*cby - uy*cbx );
    ux  += v4*( v1*cbz - v2*cby );            // Boris - rotation
    uy  += v4*( v2*cbx - v0*cbz );
    uz  += v4*( v0*cby - v1*cbx );
    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;
    p_ux = ux;                               // Store momentum
    p_uy = uy;
    p_uz = uz;

    v0   = one/sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));

    /**/                                      // Get norm displacement
    ux  *= cdt_dx;
    uy  *= cdt_dy;
    uz  *= cdt_dz;
    ux  *= v0;
    uy  *= v0;
    uz  *= v0;
    v0   = dx + ux;                           // Streak midpoint (inbnds)
    v1   = dy + uy;
    v2   = dz + uz;
    v3   = v0 + ux;                           // New position
    v4   = v1 + uy;
    v5   = v2 + uz;

#ifdef VPIC_ENABLE_TEAM_REDUCTION
    int reduce = 0;
    int inbnds = v3<=one && v4<=one && v5<=one && -v3<=one && -v4<=one && -v5<=one;
    int min_inbnds = inbnds;
    int max_inbnds = inbnds;
    team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
    team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
    int min_index = ii;
    int max_index = ii;
    team_member.team_reduce(Kokkos::Max<int>(max_index));
    team_member.team_reduce(Kokkos::Min<int>(min_index));
    reduce = min_inbnds == max_inbnds && min_index == max_index;
#endif

    // FIXME-KJB: COULD SHORT CIRCUIT ACCUMULATION IN THE CASE WHERE QSP==0!
    if(  v3<=one &&  v4<=one &&  v5<=one &&   // Check if inbnds
        -v3<=one && -v4<=one && -v5<=one ) {

      // Common case (inbnds).  Note: accumulator values are 4 times
      // the total physical charge that passed through the appropriate
      // current quadrant in a time-step
#ifdef FIELD_IONIZATION
      q *= p_q;  //FIXME: check that this is doing the correct thing
#else      
      q *= qsp;
#endif      
      p_dx = v3;                             // Store new position
      p_dy = v4;
      p_dz = v5;
      dx = v0;                                // Streak midpoint
      dy = v1;
      dz = v2;
      v5 = q*ux*uy*uz*one_third;              // Compute correction

#     define ACCUMULATE_J(X,Y,Z)                                 \
      v4  = q*u##X;   /* v2 = q ux                            */        \
      v1  = v4*d##Y;  /* v1 = q ux dy                         */        \
      v0  = v4-v1;    /* v0 = q ux (1-dy)                     */        \
      v1 += v4;       /* v1 = q ux (1+dy)                     */        \
      v4  = one+d##Z; /* v4 = 1+dz                            */        \
      v2  = v0*v4;    /* v2 = q ux (1-dy)(1+dz)               */        \
      v3  = v1*v4;    /* v3 = q ux (1+dy)(1+dz)               */        \
      v4  = one-d##Z; /* v4 = 1-dz                            */        \
      v0 *= v4;       /* v0 = q ux (1-dy)(1-dz)               */        \
      v1 *= v4;       /* v1 = q ux (1+dy)(1-dz)               */        \
      v0 += v5;       /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */        \
      v1 -= v5;       /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */        \
      v2 -= v5;       /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */        \
      v3 += v5;       /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */

#ifdef VPIC_ENABLE_TEAM_REDUCTION
      if(reduce) {
        int iii = ii;
        int zi = iii/((nx+2)*(ny+2));
        iii -= zi*(nx+2)*(ny+2);
        int yi = iii/(nx+2);
        int xi = iii - yi*(nx+2);

        int i0 = ii;
        int i1 = VOXEL(xi,yi+1,zi,nx,ny,nz);
        int i2 = VOXEL(xi,yi,zi+1,nx,ny,nz);
        int i3 = VOXEL(xi,yi+1,zi+1,nx,ny,nz);
        ACCUMULATE_J( x,y,z );
        contribute_current(team_member, k_field_scatter_access, i0, i1, i2, i3, 
                            field_var::jfx, cx*v0, cx*v1, cx*v2, cx*v3);

        i1 = VOXEL(xi,yi,zi+1,nx,ny,nz);
        i2 = VOXEL(xi+1,yi,zi,nx,ny,nz);
        i3 = VOXEL(xi+1,yi,zi+1,nx,ny,nz);
        ACCUMULATE_J( y,z,x );
        contribute_current(team_member, k_field_scatter_access, i0, i1, i2, i3, 
                            field_var::jfy, cy*v0, cy*v1, cy*v2, cy*v3);

        i1 = VOXEL(xi+1,yi,zi,nx,ny,nz);
        i2 = VOXEL(xi,yi+1,zi,nx,ny,nz);
        i3 = VOXEL(xi+1,yi+1,zi,nx,ny,nz);
        ACCUMULATE_J( z,x,y );
        contribute_current(team_member, k_field_scatter_access, i0, i1, i2, i3, 
                            field_var::jfz, cz*v0, cz*v1, cz*v2, cz*v3);
      } else {
#endif
        // TODO: That 2 needs to be 2*NGHOST eventually
        int iii = ii;
        int zi = iii/((nx+2)*(ny+2));
        iii -= zi*(nx+2)*(ny+2);
        int yi = iii/(nx+2);
        int xi = iii - yi*(nx+2);
        ACCUMULATE_J( x,y,z );
        //Kokkos::atomic_add(&k_field(ii, field_var::jfx), cx*v0);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx), cx*v1);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx), cx*v2);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx), cx*v3);
        k_field_scatter_access(ii, field_var::jfx) += cx*v0;
        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
        k_field_scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;

        ACCUMULATE_J( y,z,x );
        //Kokkos::atomic_add(&k_field(ii, field_var::jfy), cy*v0);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v1);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy), cy*v2);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy), cy*v3);
        k_field_scatter_access(ii, field_var::jfy) += cy*v0;
        k_field_scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
        k_field_scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;

        ACCUMULATE_J( z,x,y );
        //Kokkos::atomic_add(&k_field(ii, field_var::jfz), cz*v0);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz), cz*v1);
        //Kokkos::atomic_add(&k_field(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v2);
        //Kokkos::atomic_add(&k_field(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz), cz*v3);
        k_field_scatter_access(ii, field_var::jfz) += cz*v0;
        k_field_scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
        k_field_scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
        k_field_scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;

//	if (tempy >0) {
//	  printf("tempx, %e, tempy, %e, tempz, %e, cx*v0, %e, cy*v0, %e, cz*v0, %e, v0, %e, cx, %e, cy, %e, cz, %e\n",tempx,tempy,tempz,cx*v0,cy*v0,cz*v0,v0,cx,cy,cz);
//	}
//        k_field_scatter_access(pii, field_var::jfx) += tempx;
//        k_field_scatter_access(pii, field_var::jfy) += tempy;
//        k_field_scatter_access(pii, field_var::jfz) += tempz;
#ifdef VPIC_ENABLE_TEAM_REDUCTION
      }
#endif

#     undef ACCUMULATE_J
    } else {
      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
      local_pm->dispx = ux;
      local_pm->dispy = uy;
      local_pm->dispz = uz;
      local_pm->i     = p_index;

      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
#ifdef FIELD_IONIZATION
      if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                         k_f_sv, g, k_neighbors, rangel, rangeh, p_q, cx, cy, cz, nx, ny, nz ) )
#else
      if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
                         k_f_sv, g, k_neighbors, rangel, rangeh, qsp, cx, cy, cz, nx, ny, nz ) )
#endif	
      {
        if( k_nm(0) < max_nm )
        {
            const int nm = Kokkos::atomic_fetch_add( &k_nm(0), 1 );
            if (nm >= max_nm) Kokkos::abort("overran max_nm");

            k_particle_movers(nm, particle_mover_var::dispx) = local_pm->dispx;
            k_particle_movers(nm, particle_mover_var::dispy) = local_pm->dispy;
            k_particle_movers(nm, particle_mover_var::dispz) = local_pm->dispz;
            k_particle_movers_i(nm)   = local_pm->i;

            // Keep existing mover structure, but also copy the particle data so we have a reduced set to move to host
            k_particle_copy(nm, particle_var::dx) = p_dx;
            k_particle_copy(nm, particle_var::dy) = p_dy;
            k_particle_copy(nm, particle_var::dz) = p_dz;
            k_particle_copy(nm, particle_var::ux) = p_ux;
            k_particle_copy(nm, particle_var::uy) = p_uy;
            k_particle_copy(nm, particle_var::uz) = p_uz;
            k_particle_copy(nm, particle_var::w) = p_w;
     #ifdef FIELD_IONIZATION     
	    k_particle_copy(nm, particle_var::charge) = p_q;
     #endif	    
            k_particle_i_copy(nm) = pii;

            // Tag this one as having left
            //k_particles(p_index, particle_var::pi) = 999999;

            // Copy local local_pm back
            //local_pm_dispx = local_pm->dispx;
            //local_pm_dispy = local_pm->dispy;
            //local_pm_dispz = local_pm->dispz;
            //local_pm_i = local_pm->i;
            //printf("rank copying %d to nm %d \n", local_pm_i, nm);
            //copy_local_to_pm(nm);
        }
      }
    }

#ifdef VPIC_ENABLE_HIERARCHICAL
  }
  });
#endif
  });
  Kokkos::Experimental::contribute(k_field, k_f_sv);


  // TODO: abstract this manual data copy
  //Kokkos::deep_copy(h_nm, k_nm);

  //args->seg[pipeline_rank].pm        = pm;
  //args->seg[pipeline_rank].max_nm    = max_nm;
  //args->seg[pipeline_rank].nm        = h_nm(0);
  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
  //delete(k_local_particle_movers_p);
  //return h_nm(0);
#ifdef FIELD_IONIZATION
  Kokkos:deep_copy(sp_e->np,count);

  // FIXME: this is temporary
  if(strcmp(sp->name, "electron") != 0){
    char kn [100];
    snprintf(kn, sizeof kn, "Photoelectrons.txt");
    std::ofstream outfile;
    if (g->step == 0) {
      outfile.open(kn); // overwrite old files
      outfile << g->step << "," << sp->np << "," << sp_e->np << "," << sp->name << endl;
    } else {
      outfile.open(kn, std::ios_base::app); // append to file
      outfile << g->step << "," << sp->np << "," << sp_e->np << "," << sp->name << endl;
    }
    outfile.close();
  }
#endif  
}

void
advance_p( /**/  species_t            * RESTRICT sp,
//           accumulator_array_t * RESTRICT aa,
           interpolator_array_t * RESTRICT ia,
      #ifndef FIELD_IONIZATION
           field_array_t* RESTRICT fa) {
      #else
           field_array_t* RESTRICT fa,
	   species_t * RESTRICT species_list) {
      #endif
  //DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );
  //DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE+1 );
  //int rank;

  //species_t * sp_e = find_species_name("electron", species_list);
  // species_list is the last defined species in the deck
  // thus electrons need to be defined last in the input deck for this to work
#ifdef FIELD_IONIZATION
  species_t * sp_e = species_list;
  if(strcmp(species_list->name, "electron") != 0)
  {
   ERROR(( "Electrons need to be defined last in input deck when field ionization is enabled" ));
  }
#endif  


  if( !sp )
  {
    ERROR(( "Bad args" ));
  }
  if( !ia  )
  {
    ERROR(( "Bad args" ));
  }
  if( sp->g!=ia->g )
  {
    ERROR(( "Bad args" ));
  }


#ifdef FIELD_IONIZATION
  float dt_2mc  = (sp->g->dt)/(2*sp->m*sp->g->cvac); // need to take the charge from the particle when field ionization is enabled
#else  
  float qdt_2mc  = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac); // take charge from species when ionization is off
#endif
  float cdt_dx   = sp->g->cvac*sp->g->dt*sp->g->rdx;
  float cdt_dy   = sp->g->cvac*sp->g->dt*sp->g->rdy;
  float cdt_dz   = sp->g->cvac*sp->g->dt*sp->g->rdz;
  
  #ifdef USE_GPU
    // Use the gpu kernel for slightly better performance
    #define ADVANCE_P advance_p_kokkos_gpu
  #else
    // Portable kernel with additional vectorization options
    #define ADVANCE_P advance_p_kokkos_unified
  #endif
  KOKKOS_TIC();
  ADVANCE_P(
          sp->k_p_d,
          sp->k_p_i_d,
          sp->k_pc_d,
          sp->k_pc_i_d,
          sp->k_pm_d,
          sp->k_pm_i_d,
          fa->k_field_sa_d,
          ia->k_i_d,
          sp->k_nm_d,
          sp->g->k_neighbor_d,
          fa,
          sp->g,
#ifdef FIELD_IONIZATION
          dt_2mc,
#else
          qdt_2mc,
#endif	  
          cdt_dx,
          cdt_dy,
          cdt_dz,
#ifndef FIELD_IONIZATION       
          sp->q,
#endif	  
          sp->np,
          sp->max_nm,
          sp->g->nx,
          sp->g->ny,
#ifdef FIELD_IONIZATION	  
          sp->g->nz,
	  sp,
	  sp_e);
#else
          sp->g->nz);
#endif	  
  KOKKOS_TOC( advance_p, 1);

  KOKKOS_TIC();
  // I need to know the number of movers that got populated so I can call the
  // compress. Let's copy it back
  Kokkos::deep_copy(sp->k_nm_h, sp->k_nm_d);
  // TODO: which way round should this copy be?

  //  int nm = sp->k_nm_h(0);

  //  printf("nm = %d \n", nm);

  // Copy particle mirror movers back so we have their data safe. Ready for
  // boundary_p_kokkos
  auto pc_d_subview = Kokkos::subview(sp->k_pc_d, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
  auto pci_d_subview = Kokkos::subview(sp->k_pc_i_d, std::make_pair(0, sp->k_nm_h(0)));
  auto pc_h_subview = Kokkos::subview(sp->k_pc_h, std::make_pair(0, sp->k_nm_h(0)), Kokkos::ALL);
  auto pci_h_subview = Kokkos::subview(sp->k_pc_i_h, std::make_pair(0, sp->k_nm_h(0)));

  Kokkos::deep_copy(pc_h_subview, pc_d_subview);
  Kokkos::deep_copy(pci_h_subview, pci_d_subview);
  //  Kokkos::deep_copy(sp->k_pc_h, sp->k_pc_d);
  //  Kokkos::deep_copy(sp->k_pc_i_h, sp->k_pc_i_d);

  KOKKOS_TOC( PARTICLE_DATA_MOVEMENT, 1);
}
