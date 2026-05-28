// FIXME: PARTICLE MOVERS NEED TO BE OVERALLOCATED IN STRUCTORS TO
// ACCOUNT FOR SPLITTING THE MOVER ARRAY BETWEEN HOST AND PIPELINES

#define IN_spa
#define HAS_V4_PIPELINE
#include <stdio.h>
#include "spa_private.h"
#include "../../vpic/kokkos_helpers.h"
#include "../../vpic/kokkos_tuning.hpp"

// Write current values to either an accumulator or directly to the fields
template<class CurrentScatterAccess>
void KOKKOS_INLINE_FUNCTION
accumulate_current(CurrentScatterAccess& current_sa, int ii,
                   const int nx, const int ny, const int nz, 
                   const float rV, 
                   const float v0, const float v1, const float v2, const float v3,
                   const float v4, const float v5, const float v6, const float v7,
                   const float v8, const float v9, const float v10, const float v11) {
#ifdef SHAPE_NGP
#   ifdef VPIC_ENABLE_ACCUMULATORS
      current_sa(ii, 0)  += v0;
      //current_sa(ii, 1)  += cx*v1;
      //current_sa(ii, 2)  += cx*v2;
      //current_sa(ii, 3)  += cx*v3;
      current_sa(ii, 4)  += v1;
      current_sa(ii, 8)  += v2;
#   else
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);
      current_sa(ii, field_var::jfx)                           += v0;
      current_sa(ii, field_var::jfy)                           += v1;
      current_sa(ii, field_var::jfz)                           += v2;
      current_sa(ii, field_var::rhof)                          += v3;
#   endif
#else
#ifdef SHAPE_QS
#   ifdef VPIC_ENABLE_ACCUMULATORS
      // not handling VPIC_ENABLE_ACCUMULATORS case
      // so fail loudly by not depositing anything
      Kokkos::abort("shape_qs lacks support for VPIC_ENABLE_ACCUMULATORS");
#   else
      // Voxel indices
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);
      // Neighboring voxel 1D (flattened) indices
      int iix = VOXEL(xi+1,yi,zi,nx,ny,nz);
      int iiy = VOXEL(xi,yi+1,zi,nx,ny,nz);
      int iiz = VOXEL(xi,yi,zi+1,nx,ny,nz);
      int iimx = VOXEL(xi-1,yi,zi,nx,ny,nz);
      int iimy = VOXEL(xi,yi-1,zi,nx,ny,nz);
      int iimz = VOXEL(xi,yi,zi-1,nx,ny,nz);

      current_sa(ii, field_var::jfx)  += v3*v6; // w0*ux;
      current_sa(ii, field_var::jfy)  += v3*v7; // w0*uy;
      current_sa(ii, field_var::jfz)  += v3*v8; // w0*uz;
      current_sa(ii, field_var::rhof) += v3;    // w0;

      current_sa(iix, field_var::jfx)  += v4*v6; // wx*ux;
      current_sa(iix, field_var::jfy)  += v4*v7; // wx*uy;
      current_sa(iix, field_var::jfz)  += v4*v8; // wx*uz;
      current_sa(iix, field_var::rhof) += v4;    // wx;

      current_sa(iiy, field_var::jfx)  += v5*v6; // wy*ux;
      current_sa(iiy, field_var::jfy)  += v5*v7; // wy*uy;
      current_sa(iiy, field_var::jfz)  += v5*v8; // wy*uz;
      current_sa(iiy, field_var::rhof) += v5;    // wy;

      current_sa(iiz, field_var::jfx)  += v9*v6; // wz*ux;
      current_sa(iiz, field_var::jfy)  += v9*v7; // wz*uy;
      current_sa(iiz, field_var::jfz)  += v9*v8; // wz*uz;
      current_sa(iiz, field_var::rhof) += v9;    // wz;

      current_sa(iimx, field_var::jfx)  += v0*v6; // wmx*ux;
      current_sa(iimx, field_var::jfy)  += v0*v7; // wmx*uy;
      current_sa(iimx, field_var::jfz)  += v0*v8; // wmx*uz;
      current_sa(iimx, field_var::rhof) += v0;    // wmx;

      current_sa(iimy, field_var::jfx)  += v1*v6; // wmy*ux;
      current_sa(iimy, field_var::jfy)  += v1*v7; // wmy*uy;
      current_sa(iimy, field_var::jfz)  += v1*v8; // wmy*uz;
      current_sa(iimy, field_var::rhof) += v1;    // wmy;

      current_sa(iimz, field_var::jfx)  += v2*v6; // wmz*ux;
      current_sa(iimz, field_var::jfy)  += v2*v7; // wmz*uy;
      current_sa(iimz, field_var::jfz)  += v2*v8; // wmz*uz;
      current_sa(iimz, field_var::rhof) += v2;    // wmz;
#   endif
#endif // defined(SHAPE_QS)
#endif // defined(SHAPE_NGP)
}

// Reduce the current for all active threads/lanes to reduce the number of writes to memory
template<class TeamMember, class CurrentScatterAccess>
void KOKKOS_INLINE_FUNCTION
reduce_and_accumulate_current(TeamMember& team_member, CurrentScatterAccess& access, 
                              const int num_iters, const int ii, 
                              const int nx, const int ny, const int nz, 
                              const float rV, 
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
  accumulate_current(access, ii, nx, ny, nz, rV, 
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
    accumulate_current(access, ii, nx, ny, rV, 
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
    accumulate_current(access, ii, nx, ny, nz, rV, 
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
  simd_load_interpolator_var(vals+(N-1)*INTERPOLATOR_VAR_COUNT, ii[N-1], k_interp, len);
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

#ifdef SHAPE_NGP

// Load interpolators
template<int NumLanes>
KOKKOS_INLINE_FUNCTION
void load_interpolators(
                        float* fex,
                        float* fey,
                        float* fez,
                        float* fcbx,
                        float* fcby,
                        float* fcbz,
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
    float vals[INTERPOLATOR_VAR_COUNT];

    simd_load_interpolator_var(vals, ii[0], k_interp, INTERPOLATOR_VAR_COUNT);
    #pragma omp simd
    for(int i=0; i<NumLanes; i++) {
      fex[i]       = vals[0];
      fey[i]       = vals[1];
      fez[i]       = vals[2];
      fcbx[i]      = vals[3];
      fcby[i]      = vals[4];
      fcbz[i]      = vals[5];
    }
  } else {

    // Efficient vectorized load
    float vals[INTERPOLATOR_VAR_COUNT*NumLanes];
    unrolled_simd_load(vals, ii, k_interp, INTERPOLATOR_VAR_COUNT, num_part);
//    unrolled_simd_load<NumLanes>(vals, ii, k_interp, INTERPOLATOR_VAR_COUNT);

    // Essentially a transpose
    #pragma omp simd
    for(int i=0; i<num_part; i++) {
      fex[i]       = vals[  INTERPOLATOR_VAR_COUNT*i];
      fey[i]       = vals[1+INTERPOLATOR_VAR_COUNT*i];
      fez[i]       = vals[2+INTERPOLATOR_VAR_COUNT*i];
      fcbx[i]      = vals[3+INTERPOLATOR_VAR_COUNT*i];
      fcby[i]      = vals[4+INTERPOLATOR_VAR_COUNT*i];
      fcbz[i]      = vals[5+INTERPOLATOR_VAR_COUNT*i];
    }
  }
#else
  for(int lane=0; lane<NumLanes; lane++) {
    // Load interpolators
    fex[LANE]       = k_interp(ii[LANE], interpolator_var::ex);     
    fey[LANE]       = k_interp(ii[LANE], interpolator_var::ey);     
    fez[LANE]       = k_interp(ii[LANE], interpolator_var::ez);     
    fcbx[LANE]      = k_interp(ii[LANE], interpolator_var::cbx);    
    fcby[LANE]      = k_interp(ii[LANE], interpolator_var::cby);    
    fcbz[LANE]      = k_interp(ii[LANE], interpolator_var::cbz);    
  }
#endif // defined(VPIC_ENABLE_VECTORIZATION) && !defined(USE_GPU)
} // void load_interpolators(...) for SHAPE_NGP

#else
#ifdef SHAPE_QS

// Load interpolators
template<int NumLanes>
KOKKOS_INLINE_FUNCTION
void load_interpolators(
                        float* fex,
                        float* fdexdx,
                        float* fdexdy,
                        float* fdexdz,
                        float* fd2exdx,
                        float* fd2exdy,
                        float* fd2exdz,
                        float* fey,
                        float* fdeydx,
                        float* fdeydy,
                        float* fdeydz,
                        float* fd2eydx,
                        float* fd2eydy,
                        float* fd2eydz,
                        float* fez,
                        float* fdezdx,
                        float* fdezdy,
                        float* fdezdz,
                        float* fd2ezdx,
                        float* fd2ezdy,
                        float* fd2ezdz,
                        float* fcbx,
                        float* fdcbxdx,
                        float* fdcbxdy,
                        float* fdcbxdz,
                        float* fd2cbxdx,
                        float* fd2cbxdy,
                        float* fd2cbxdz,
                        float* fcby,
                        float* fdcbydx,
                        float* fdcbydy,
                        float* fdcbydz,
                        float* fd2cbydx,
                        float* fd2cbydy,
                        float* fd2cbydz,
                        float* fcbz,
                        float* fdcbzdx,
                        float* fdcbzdy,
                        float* fdcbzdz,
                        float* fd2cbzdx,
                        float* fd2cbzdy,
                        float* fd2cbzdz,
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
    float vals[INTERPOLATOR_VAR_COUNT];

    simd_load_interpolator_var(vals, ii[0], k_interp, INTERPOLATOR_VAR_COUNT);
    #pragma omp simd
    for(int i=0; i<NumLanes; i++) {
      fex[i]      = vals[ 0];
      fdexdx[i]   = vals[ 1];
      fdexdy[i]   = vals[ 2];
      fdexdz[i]   = vals[ 3];
      fd2exdx[i]  = vals[ 4];
      fd2exdy[i]  = vals[ 5];
      fd2exdz[i]  = vals[ 6];
      fey[i]      = vals[ 7];
      fdeydx[i]   = vals[ 8];
      fdeydy[i]   = vals[ 9];
      fdeydz[i]   = vals[10];
      fd2eydx[i]  = vals[11];
      fd2eydy[i]  = vals[12];
      fd2eydz[i]  = vals[13];
      fez[i]      = vals[14];
      fdezdx[i]   = vals[15];
      fdezdy[i]   = vals[16];
      fdezdz[i]   = vals[17];
      fd2ezdx[i]  = vals[18];
      fd2ezdy[i]  = vals[19];
      fd2ezdz[i]  = vals[20];
      fcbx[i]     = vals[21];
      fdcbxdx[i]  = vals[22];
      fdcbxdy[i]  = vals[23];
      fdcbxdz[i]  = vals[24];
      fd2cbxdx[i] = vals[25];
      fd2cbxdy[i] = vals[26];
      fd2cbxdz[i] = vals[27];
      fcby[i]     = vals[28];
      fdcbydx[i]  = vals[29];
      fdcbydy[i]  = vals[30];
      fdcbydz[i]  = vals[31];
      fd2cbydx[i] = vals[32];
      fd2cbydy[i] = vals[33];
      fd2cbydz[i] = vals[34];
      fcbz[i]     = vals[35];
      fdcbzdx[i]  = vals[36];
      fdcbzdy[i]  = vals[37];
      fdcbzdz[i]  = vals[38];
      fd2cbzdx[i] = vals[39];
      fd2cbzdy[i] = vals[40];
      fd2cbzdz[i] = vals[41];
    }
  } else {

    // Efficient vectorized load
    float vals[INTERPOLATOR_VAR_COUNT * NumLanes];
    unrolled_simd_load(vals, ii, k_interp, INTERPOLATOR_VAR_COUNT, num_part);
//    unrolled_simd_load<NumLanes>(vals, ii, k_interp, INTERPOLATOR_VAR_COUNT);

    // Essentially a transpose
    #pragma omp simd
    for(int i=0; i<num_part; i++) {
      fex[i]      = vals[ 0 + INTERPOLATOR_VAR_COUNT*i];
      fdexdx[i]   = vals[ 1 + INTERPOLATOR_VAR_COUNT*i];
      fdexdy[i]   = vals[ 2 + INTERPOLATOR_VAR_COUNT*i];
      fdexdz[i]   = vals[ 3 + INTERPOLATOR_VAR_COUNT*i];
      fd2exdx[i]  = vals[ 4 + INTERPOLATOR_VAR_COUNT*i];
      fd2exdy[i]  = vals[ 5 + INTERPOLATOR_VAR_COUNT*i];
      fd2exdz[i]  = vals[ 6 + INTERPOLATOR_VAR_COUNT*i];
      fey[i]      = vals[ 7 + INTERPOLATOR_VAR_COUNT*i];
      fdeydx[i]   = vals[ 8 + INTERPOLATOR_VAR_COUNT*i];
      fdeydy[i]   = vals[ 9 + INTERPOLATOR_VAR_COUNT*i];
      fdeydz[i]   = vals[10 + INTERPOLATOR_VAR_COUNT*i];
      fd2eydx[i]  = vals[11 + INTERPOLATOR_VAR_COUNT*i];
      fd2eydy[i]  = vals[12 + INTERPOLATOR_VAR_COUNT*i];
      fd2eydz[i]  = vals[13 + INTERPOLATOR_VAR_COUNT*i];
      fez[i]      = vals[14 + INTERPOLATOR_VAR_COUNT*i];
      fdezdx[i]   = vals[15 + INTERPOLATOR_VAR_COUNT*i];
      fdezdy[i]   = vals[16 + INTERPOLATOR_VAR_COUNT*i];
      fdezdz[i]   = vals[17 + INTERPOLATOR_VAR_COUNT*i];
      fd2ezdx[i]  = vals[18 + INTERPOLATOR_VAR_COUNT*i];
      fd2ezdy[i]  = vals[19 + INTERPOLATOR_VAR_COUNT*i];
      fd2ezdz[i]  = vals[20 + INTERPOLATOR_VAR_COUNT*i];
      fcbx[i]     = vals[21 + INTERPOLATOR_VAR_COUNT*i];
      fdcbxdx[i]  = vals[22 + INTERPOLATOR_VAR_COUNT*i];
      fdcbxdy[i]  = vals[23 + INTERPOLATOR_VAR_COUNT*i];
      fdcbxdz[i]  = vals[24 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbxdx[i] = vals[25 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbxdy[i] = vals[26 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbxdz[i] = vals[27 + INTERPOLATOR_VAR_COUNT*i];
      fcby[i]     = vals[28 + INTERPOLATOR_VAR_COUNT*i];
      fdcbydx[i]  = vals[29 + INTERPOLATOR_VAR_COUNT*i];
      fdcbydy[i]  = vals[30 + INTERPOLATOR_VAR_COUNT*i];
      fdcbydz[i]  = vals[31 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbydx[i] = vals[32 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbydy[i] = vals[33 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbydz[i] = vals[34 + INTERPOLATOR_VAR_COUNT*i];
      fcbz[i]     = vals[35 + INTERPOLATOR_VAR_COUNT*i];
      fdcbzdx[i]  = vals[36 + INTERPOLATOR_VAR_COUNT*i];
      fdcbzdy[i]  = vals[37 + INTERPOLATOR_VAR_COUNT*i];
      fdcbzdz[i]  = vals[38 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbzdx[i] = vals[39 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbzdy[i] = vals[40 + INTERPOLATOR_VAR_COUNT*i];
      fd2cbzdz[i] = vals[41 + INTERPOLATOR_VAR_COUNT*i];
    }
  }
#else
  for(int lane=0; lane<NumLanes; lane++) {
    // Load interpolators
    fex[LANE]      = k_interp(ii[LANE], interpolator_var::ex);
    fdexdx[LANE]   = k_interp(ii[LANE], interpolator_var::dexdx);
    fdexdy[LANE]   = k_interp(ii[LANE], interpolator_var::dexdy);
    fdexdz[LANE]   = k_interp(ii[LANE], interpolator_var::dexdz);
    fd2exdx[LANE]  = k_interp(ii[LANE], interpolator_var::d2exdx);
    fd2exdy[LANE]  = k_interp(ii[LANE], interpolator_var::d2exdy);
    fd2exdz[LANE]  = k_interp(ii[LANE], interpolator_var::d2exdz);
    fey[LANE]      = k_interp(ii[LANE], interpolator_var::ey);
    fdeydx[LANE]   = k_interp(ii[LANE], interpolator_var::deydx);
    fdeydy[LANE]   = k_interp(ii[LANE], interpolator_var::deydy);
    fdeydz[LANE]   = k_interp(ii[LANE], interpolator_var::deydz);
    fd2eydx[LANE]  = k_interp(ii[LANE], interpolator_var::d2eydx);
    fd2eydy[LANE]  = k_interp(ii[LANE], interpolator_var::d2eydy);
    fd2eydz[LANE]  = k_interp(ii[LANE], interpolator_var::d2eydz);
    fez[LANE]      = k_interp(ii[LANE], interpolator_var::ez);
    fdezdx[LANE]   = k_interp(ii[LANE], interpolator_var::dezdx);
    fdezdy[LANE]   = k_interp(ii[LANE], interpolator_var::dezdy);
    fdezdz[LANE]   = k_interp(ii[LANE], interpolator_var::dezdz);
    fd2ezdx[LANE]  = k_interp(ii[LANE], interpolator_var::d2ezdx);
    fd2ezdy[LANE]  = k_interp(ii[LANE], interpolator_var::d2ezdy);
    fd2ezdz[LANE]  = k_interp(ii[LANE], interpolator_var::d2ezdz);
    fcbx[LANE]     = k_interp(ii[LANE], interpolator_var::cbx);
    fdcbxdx[LANE]  = k_interp(ii[LANE], interpolator_var::dcbxdx);
    fdcbxdy[LANE]  = k_interp(ii[LANE], interpolator_var::dcbxdy);
    fdcbxdz[LANE]  = k_interp(ii[LANE], interpolator_var::dcbxdz);
    fd2cbxdx[LANE] = k_interp(ii[LANE], interpolator_var::d2cbxdx);
    fd2cbxdy[LANE] = k_interp(ii[LANE], interpolator_var::d2cbxdy);
    fd2cbxdz[LANE] = k_interp(ii[LANE], interpolator_var::d2cbxdz);
    fcby[LANE]     = k_interp(ii[LANE], interpolator_var::cby);
    fdcbydx[LANE]  = k_interp(ii[LANE], interpolator_var::dcbydx);
    fdcbydy[LANE]  = k_interp(ii[LANE], interpolator_var::dcbydy);
    fdcbydz[LANE]  = k_interp(ii[LANE], interpolator_var::dcbydz);
    fd2cbydx[LANE] = k_interp(ii[LANE], interpolator_var::d2cbydx);
    fd2cbydy[LANE] = k_interp(ii[LANE], interpolator_var::d2cbydy);
    fd2cbydz[LANE] = k_interp(ii[LANE], interpolator_var::d2cbydz);
    fcbz[LANE]     = k_interp(ii[LANE], interpolator_var::cbz);
    fdcbzdx[LANE]  = k_interp(ii[LANE], interpolator_var::dcbzdx);
    fdcbzdy[LANE]  = k_interp(ii[LANE], interpolator_var::dcbzdy);
    fdcbzdz[LANE]  = k_interp(ii[LANE], interpolator_var::dcbzdz);
    fd2cbzdx[LANE] = k_interp(ii[LANE], interpolator_var::d2cbzdx);
    fd2cbzdy[LANE] = k_interp(ii[LANE], interpolator_var::d2cbzdy);
    fd2cbzdz[LANE] = k_interp(ii[LANE], interpolator_var::d2cbzdz);
  }
#endif // defined(VPIC_ENABLE_VECTORIZATION) && !defined(USE_GPU)
} // void load_interpolators(...) for SHAPE_QS

#endif // defined(SHAPE_QS)
#endif // defined(SHAPE_NGP)

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
#ifdef VARIABLE_CHARGE
	const float dt_2mc,
#else
        const float qdt_2mc,
#endif
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{

  constexpr float three          = 3.;
  constexpr float two            = 2.;
  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;
  constexpr float one_twelfth    = 1./12.;

  k_field_t k_field = fa->k_f_d;
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;
  float rV = g->rdx * g->rdy * g->rdz;
  float gdx=g->dx, gdy=g->dy, gdz=g->dz, gdt=g->dt;

#ifdef EXTERNAL_FORCE
  // Don't futz with interpolator loading code that we won't even use
  // --ATr,2026feb25
  Kokkos::abort("External force is not implemented for CPU particle advance");
#endif

  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux)
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define p_w     k_particles(p_index, particle_var::w)
#ifdef VARIABLE_CHARGE
  #define p_q     k_particles(p_index, particle_var::qp)
#endif
  #define pii     k_particles_i(p_index)

  auto rangel = g->rangel;
  auto rangeh = g->rangeh;

  // TODO: is this the right place to do this?
  Kokkos::deep_copy(k_nm, 0);

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
#ifdef VARIABLE_CHARGE
      float qp[num_lanes];
#endif
      int   ii[num_lanes];
      int   inbnds[num_lanes];
      //int   midbnds[num_lanes];

#ifdef SHAPE_NGP
      float fcbx[num_lanes];
      float fcby[num_lanes];
      float fcbz[num_lanes];
      float fex[num_lanes];
      float fey[num_lanes];
      float fez[num_lanes];
      float tmp0[num_lanes];
      float tmp1[num_lanes];
      float *v6 = fex;
      float *v7 = fey;
      float *v8 = fez;
      float *v9 = fcbx;
      float *v10 = fcby;
      float *v11 = fcbz;
      float *v12 = tmp0;
      float *v13 = tmp1;
#else
#ifdef SHAPE_QS
      float fex[num_lanes];
      float fdexdx[num_lanes];
      float fdexdy[num_lanes];
      float fdexdz[num_lanes];
      float fd2exdx[num_lanes];
      float fd2exdy[num_lanes];
      float fd2exdz[num_lanes];
      float fey[num_lanes];
      float fdeydx[num_lanes];
      float fdeydy[num_lanes];
      float fdeydz[num_lanes];
      float fd2eydx[num_lanes];
      float fd2eydy[num_lanes];
      float fd2eydz[num_lanes];
      float fez[num_lanes];
      float fdezdx[num_lanes];
      float fdezdy[num_lanes];
      float fdezdz[num_lanes];
      float fd2ezdx[num_lanes];
      float fd2ezdy[num_lanes];
      float fd2ezdz[num_lanes];
      float fcbx[num_lanes];
      float fdcbxdx[num_lanes];
      float fdcbxdy[num_lanes];
      float fdcbxdz[num_lanes];
      float fd2cbxdx[num_lanes];
      float fd2cbxdy[num_lanes];
      float fd2cbxdz[num_lanes];
      float fcby[num_lanes];
      float fdcbydx[num_lanes];
      float fdcbydy[num_lanes];
      float fdcbydz[num_lanes];
      float fd2cbydx[num_lanes];
      float fd2cbydy[num_lanes];
      float fd2cbydz[num_lanes];
      float fcbz[num_lanes];
      float fdcbzdx[num_lanes];
      float fdcbzdy[num_lanes];
      float fdcbzdz[num_lanes];
      float fd2cbzdx[num_lanes];
      float fd2cbzdy[num_lanes];
      float fd2cbzdz[num_lanes];
      float *v6 = fex;
      float *v7 = fdexdx;
      float *v8 = fdexdy;
      float *v9 = fdexdz;
      float *v10 = fey;
      float *v11 = fdeydx;
      float *v12 = fdeydy;
      float *v13 = fdeydz;
#endif // defined(SHAPE_QS)
#endif // defined(SHAPE_NGP)

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
#ifdef VARIABLE_CHARGE
	qp[LANE] = p_q;
#endif
      
        // Load index
        ii[LANE] = pii;
      } END_VECTOR_BLOCK;

#ifdef SHAPE_NGP
      load_interpolators<num_lanes>( fex, fey, fez, fcbx, fcby, fcbz,
                                     ii, num_particles, k_interp);
#else
#ifdef SHAPE_QS
      load_interpolators<num_lanes>( fex, fdexdx, fdexdy, fdexdz, fd2exdx, fd2exdy, fd2exdz,
                                     fey, fdeydx, fdeydy, fdeydz, fd2eydx, fd2eydy, fd2eydz,
                                     fez, fdezdx, fdezdy, fdezdz, fd2ezdx, fd2ezdy, fd2ezdz,
                                     fcbx, fdcbxdx, fdcbxdy, fdcbxdz, fd2cbxdx, fd2cbxdy, fd2cbxdz,
                                     fcby, fdcbydx, fdcbydy, fdcbydz, fd2cbydx, fd2cbydy, fd2cbydz,
                                     fcbz, fdcbzdx, fdcbzdy, fdcbzdz, fd2cbzdx, fd2cbzdy, fd2cbzdz,
                                     ii, num_particles, k_interp);
#endif // defined(SHAPE_QS)
#endif // defined(SHAPE_NGP)

      BEGIN_VECTOR_BLOCK {
#ifdef SHAPE_NGP
        // Interpolate E
#ifdef VARIABLE_CHARGE
        hax[LANE] = dt_2mc*qp[LANE]*( (fex[LANE] ) );
        hay[LANE] = dt_2mc*qp[LANE]*( (fey[LANE] ) );
        haz[LANE] = dt_2mc*qp[LANE]*( (fez[LANE] ) );
#else
        hax[LANE] = qdt_2mc*( (fex[LANE] ) );
        hay[LANE] = qdt_2mc*( (fey[LANE] ) );
        haz[LANE] = qdt_2mc*( (fez[LANE] ) );
#endif
        // Interpolate B
        cbx[LANE] = fcbx[LANE];// + dx[LANE]*fdcbxdx[LANE];
        cby[LANE] = fcby[LANE];// + dy[LANE]*fdcbydy[LANE];
        cbz[LANE] = fcbz[LANE];// + dz[LANE]*fdcbzdz[LANE];
#else
#ifdef SHAPE_QS
        // Interpolate E
#ifdef VARIABLE_CHARGE
        hax[LANE]  = dt_2mc*qp[LANE]*( fex[LANE]
                        + dx[LANE]*( fdexdx[LANE] + dx[LANE]*fd2exdx[LANE] )
                        + dy[LANE]*( fdexdy[LANE] + dy[LANE]*fd2exdy[LANE] )
                        + dz[LANE]*( fdexdz[LANE] + dz[LANE]*fd2exdz[LANE] ) );
        hay[LANE]  = dt_2mc*qp[LANE]*( fey[LANE]
                        + dx[LANE]*( fdeydx[LANE] + dx[LANE]*fd2eydx[LANE] )
                        + dy[LANE]*( fdeydy[LANE] + dy[LANE]*fd2eydy[LANE] )
                        + dz[LANE]*( fdeydz[LANE] + dz[LANE]*fd2eydz[LANE] ) );
        haz[LANE]  = dt_2mc*qp[LANE]*( fez[LANE]
                        + dx[LANE]*( fdezdx[LANE] + dx[LANE]*fd2ezdx[LANE] )
                        + dy[LANE]*( fdezdy[LANE] + dy[LANE]*fd2ezdy[LANE] )
                        + dz[LANE]*( fdezdz[LANE] + dz[LANE]*fd2ezdz[LANE] ) );
#else
        hax[LANE]  = qdt_2mc*( fex[LANE] + dx[LANE]*( fdexdx[LANE] + dx[LANE]*fd2exdx[LANE] )
                                         + dy[LANE]*( fdexdy[LANE] + dy[LANE]*fd2exdy[LANE] )
                                         + dz[LANE]*( fdexdz[LANE] + dz[LANE]*fd2exdz[LANE] ) );
        hay[LANE]  = qdt_2mc*( fey[LANE] + dx[LANE]*( fdeydx[LANE] + dx[LANE]*fd2eydx[LANE] )
                                         + dy[LANE]*( fdeydy[LANE] + dy[LANE]*fd2eydy[LANE] )
                                         + dz[LANE]*( fdeydz[LANE] + dz[LANE]*fd2eydz[LANE] ) );
        haz[LANE]  = qdt_2mc*( fez[LANE] + dx[LANE]*( fdezdx[LANE] + dx[LANE]*fd2ezdx[LANE] )
                                         + dy[LANE]*( fdezdy[LANE] + dy[LANE]*fd2ezdy[LANE] )
                                         + dz[LANE]*( fdezdz[LANE] + dz[LANE]*fd2ezdz[LANE] ) );
#endif
        // Interpolate B
        cbx[LANE]  = fcbx[LANE] + dx[LANE]*( fdcbxdx[LANE] + dx[LANE]*fd2cbxdx[LANE] )
                                + dy[LANE]*( fdcbxdy[LANE] + dy[LANE]*fd2cbxdy[LANE] )
                                + dz[LANE]*( fdcbxdz[LANE] + dz[LANE]*fd2cbxdz[LANE] );
        cby[LANE]  = fcby[LANE] + dx[LANE]*( fdcbydx[LANE] + dx[LANE]*fd2cbydx[LANE] )
                                + dy[LANE]*( fdcbydy[LANE] + dy[LANE]*fd2cbydy[LANE] )
                                + dz[LANE]*( fdcbydz[LANE] + dz[LANE]*fd2cbydz[LANE] );
        cbz[LANE]  = fcbz[LANE] + dx[LANE]*( fdcbzdx[LANE] + dx[LANE]*fd2cbzdx[LANE] )
                                + dy[LANE]*( fdcbzdy[LANE] + dy[LANE]*fd2cbzdy[LANE] )
                                + dz[LANE]*( fdcbzdz[LANE] + dz[LANE]*fd2cbzdz[LANE] );
#endif // defined(SHAPE_QS)
#endif // defined(SHAPE_NGP)
  
        // Half advance e
        ux[LANE] += hax[LANE];
        uy[LANE] += hay[LANE];
        uz[LANE] += haz[LANE];
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {
#ifdef VARIABLE_CHARGE
	v0[LANE] = dt_2mc*qp[LANE]; ///sqrtf(one + (ux[LANE]*ux[LANE] + (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE]))); 
#else
        v0[LANE] = qdt_2mc;///sqrtf(one + (ux[LANE]*ux[LANE] + (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE])));
#endif
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
        v0[LANE]   = one;///sqrtf(one + (ux[LANE]*ux[LANE]+ (uy[LANE]*uy[LANE] + uz[LANE]*uz[LANE])));
      } END_VECTOR_BLOCK;

      BEGIN_VECTOR_BLOCK {

        /**/                                      // Get norm displacement
	ux[LANE]  *= v0[LANE];
        uy[LANE]  *= v0[LANE];
        uz[LANE]  *= v0[LANE];
	v6[LANE]   = ux[LANE];
	v7[LANE]   = uy[LANE];
	v8[LANE]   = uz[LANE];
        ux[LANE]  *= cdt_dx;
        uy[LANE]  *= cdt_dy;
        uz[LANE]  *= cdt_dz;

        v0[LANE]   = dx[LANE] + ux[LANE];                           // Streak midpoint (inbnds)
        v1[LANE]   = dy[LANE] + uy[LANE];
        v2[LANE]   = dz[LANE] + uz[LANE];
        v3[LANE]   = v0[LANE] + ux[LANE];                           // New position
        v4[LANE]   = v1[LANE] + uy[LANE];
        v5[LANE]   = v2[LANE] + uz[LANE];
  
        //midbnds[LANE] = v0[LANE]<=one &&  v1[LANE]<=one &&  v2[LANE]<=one &&
          //            -v0[LANE]<=one && -v1[LANE]<=one && -v2[LANE]<=one;

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
#ifdef VARIABLE_CHARGE
	q[LANE]  = static_cast<float>(inbnds[LANE])*q[LANE]*qp[LANE]*rV; // q is previously weight
#else
        q[LANE]  = static_cast<float>(inbnds[LANE])*q[LANE]*qsp*rV;
#endif
	
        p_dx = v3[LANE];
        p_dy = v4[LANE];
        p_dz = v5[LANE];
        //dx[LANE] = v0[LANE];
        //dy[LANE] = v1[LANE];
        //dz[LANE] = v2[LANE];
        //v5[LANE] = q[LANE]*ux[LANE]*uy[LANE]*uz[LANE]*one_third;

#ifdef SHAPE_NGP
        v0[LANE]  = q[LANE]*v6[LANE]; // q*ux
        v1[LANE]  = q[LANE]*v7[LANE]; // q*uy
        v2[LANE]  = q[LANE]*v8[LANE]; // q*uz
        v3[LANE]  = q[LANE];
#else
#ifdef SHAPE_QS
        q[LANE] *= one_twelfth;
        v3[LANE] = q[LANE]*two*( three - v0[LANE]*v0[LANE] - v1[LANE]*v1[LANE] - v2[LANE]*v2[LANE] );  // w0
        v4[LANE] = q[LANE]*( v0[LANE] + one )*( v0[LANE] + one );  // wx
        v5[LANE] = q[LANE]*( v1[LANE] + one )*( v1[LANE] + one );  // wy
        v9[LANE] = q[LANE]*( v2[LANE] + one )*( v2[LANE] + one );  // wz
        v0[LANE] = q[LANE]*( v0[LANE] - one )*( v0[LANE] - one );  // wmx  // re-use due to limited number of slots
        v1[LANE] = q[LANE]*( v1[LANE] - one )*( v1[LANE] - one );  // wmy  // in accumulate_current(...) signature
        v2[LANE] = q[LANE]*( v2[LANE] - one )*( v2[LANE] - one );  // wmz
#endif
#endif

      } END_VECTOR_BLOCK;

#ifdef VPIC_ENABLE_TEAM_REDUCTION
      if(in_cell) {
        int first = ii[0];
        reduce_and_accumulate_current(team_member, current_sa, num_iters, first, 
                                      nx, ny, nz, rV,
				      v0, v1, v2, v3,
                                      v6, v7, v8, v9,
                                      v10, v11, v12, v13);
      } else {
#endif
        BEGIN_VECTOR_BLOCK {
          accumulate_current(current_sa, ii[LANE],
                       nx, ny, nz, rV,
		       v0[LANE], v1[LANE], v2[LANE], v3[LANE],
                       v4[LANE], v5[LANE], v6[LANE], v7[LANE],
                       v8[LANE], v9[LANE], v10[LANE], v11[LANE]);
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
                             current_sv, g, k_neighbors, rangel, rangeh, qsp, gdx,gdy,gdz,gdt, nx, ny, nz ) )
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
#ifdef VARIABLE_CHARGE
	      k_particle_copy(nm, particle_var::qp) = p_q;
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
  
#ifdef VARIABLE_CHARGE
  #undef p_q
#endif
  
#undef pii 
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
#ifdef VARIABLE_CHARGE
	const float dt_2mc,
#else
        const float qdt_2mc,
#endif
        const float cdt_dx,
        const float cdt_dy,
        const float cdt_dz,
        const float qsp,
        const int np,
        const int max_nm,
        const int nx,
        const int ny,
        const int nz)
{

  constexpr float three          = 3.;
  constexpr float two            = 2.;
  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;
  constexpr float one_twelfth    = 1./12.;
  k_field_t k_field = fa->k_f_d;
  k_field_sa_t k_f_sv = Kokkos::Experimental::create_scatter_view<>(k_field);
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;
  float rV = g->rdx*g->rdy*g->rdz;
  float rV12 = rV*one_twelfth;
  float gdx=g->dx, gdy=g->dy, gdz = g->dz, gdt = g->dt;

#ifdef EXTERNAL_FORCE
  const float dt_2c = (g->dt)/(2*g->cvac);
#endif

  // Process particles for this pipeline

  #define p_dx    k_particles(p_index, particle_var::dx)
  #define p_dy    k_particles(p_index, particle_var::dy)
  #define p_dz    k_particles(p_index, particle_var::dz)
  #define p_ux    k_particles(p_index, particle_var::ux)
  #define p_uy    k_particles(p_index, particle_var::uy)
  #define p_uz    k_particles(p_index, particle_var::uz)
  #define p_w     k_particles(p_index, particle_var::w)
#ifdef VARIABLE_CHARGE
  #define p_q     k_particles(p_index, particle_var::qp)
#endif
  #define pii     k_particles_i(p_index)

  #define f_ex       k_interp(ii, interpolator_var::ex)
  #define f_dexdx    k_interp(ii, interpolator_var::dexdx)
  #define f_dexdy    k_interp(ii, interpolator_var::dexdy)
  #define f_dexdz    k_interp(ii, interpolator_var::dexdz)
  #define f_d2exdx   k_interp(ii, interpolator_var::d2exdx)
  #define f_d2exdy   k_interp(ii, interpolator_var::d2exdy)
  #define f_d2exdz   k_interp(ii, interpolator_var::d2exdz)
  #define f_ey       k_interp(ii, interpolator_var::ey)
  #define f_deydx    k_interp(ii, interpolator_var::deydx)
  #define f_deydy    k_interp(ii, interpolator_var::deydy)
  #define f_deydz    k_interp(ii, interpolator_var::deydz)
  #define f_d2eydx   k_interp(ii, interpolator_var::d2eydx)
  #define f_d2eydy   k_interp(ii, interpolator_var::d2eydy)
  #define f_d2eydz   k_interp(ii, interpolator_var::d2eydz)
  #define f_ez       k_interp(ii, interpolator_var::ez)
  #define f_dezdx    k_interp(ii, interpolator_var::dezdx)
  #define f_dezdy    k_interp(ii, interpolator_var::dezdy)
  #define f_dezdz    k_interp(ii, interpolator_var::dezdz)
  #define f_d2ezdx   k_interp(ii, interpolator_var::d2ezdx)
  #define f_d2ezdy   k_interp(ii, interpolator_var::d2ezdy)
  #define f_d2ezdz   k_interp(ii, interpolator_var::d2ezdz)
  #define f_cbx      k_interp(ii, interpolator_var::cbx)
  #define f_dcbxdx   k_interp(ii, interpolator_var::dcbxdx)
  #define f_dcbxdy   k_interp(ii, interpolator_var::dcbxdy)
  #define f_dcbxdz   k_interp(ii, interpolator_var::dcbxdz)
  #define f_d2cbxdx  k_interp(ii, interpolator_var::d2cbxdx)
  #define f_d2cbxdy  k_interp(ii, interpolator_var::d2cbxdy)
  #define f_d2cbxdz  k_interp(ii, interpolator_var::d2cbxdz)
  #define f_cby      k_interp(ii, interpolator_var::cby)
  #define f_dcbydx   k_interp(ii, interpolator_var::dcbydx)
  #define f_dcbydy   k_interp(ii, interpolator_var::dcbydy)
  #define f_dcbydz   k_interp(ii, interpolator_var::dcbydz)
  #define f_d2cbydx  k_interp(ii, interpolator_var::d2cbydx)
  #define f_d2cbydy  k_interp(ii, interpolator_var::d2cbydy)
  #define f_d2cbydz  k_interp(ii, interpolator_var::d2cbydz)
  #define f_cbz      k_interp(ii, interpolator_var::cbz)
  #define f_dcbzdx   k_interp(ii, interpolator_var::dcbzdx)
  #define f_dcbzdy   k_interp(ii, interpolator_var::dcbzdy)
  #define f_dcbzdz   k_interp(ii, interpolator_var::dcbzdz)
  #define f_d2cbzdx  k_interp(ii, interpolator_var::d2cbzdx)
  #define f_d2cbzdy  k_interp(ii, interpolator_var::d2cbzdy)
  #define f_d2cbzdz  k_interp(ii, interpolator_var::d2cbzdz)

  #define f_Ex0       k_interp(ii, interpolator_var::Ex0)
  #define f_dEx0dx    k_interp(ii, interpolator_var::dEx0dx)
  #define f_dEx0dy    k_interp(ii, interpolator_var::dEx0dy)
  #define f_dEx0dz    k_interp(ii, interpolator_var::dEx0dz)
  #define f_d2Ex0dx   k_interp(ii, interpolator_var::d2Ex0dx)
  #define f_d2Ex0dy   k_interp(ii, interpolator_var::d2Ex0dy)
  #define f_d2Ex0dz   k_interp(ii, interpolator_var::d2Ex0dz)
  #define f_Ey0       k_interp(ii, interpolator_var::Ey0)
  #define f_dEy0dx    k_interp(ii, interpolator_var::dEy0dx)
  #define f_dEy0dy    k_interp(ii, interpolator_var::dEy0dy)
  #define f_dEy0dz    k_interp(ii, interpolator_var::dEy0dz)
  #define f_d2Ey0dx   k_interp(ii, interpolator_var::d2Ey0dx)
  #define f_d2Ey0dy   k_interp(ii, interpolator_var::d2Ey0dy)
  #define f_d2Ey0dz   k_interp(ii, interpolator_var::d2Ey0dz)
  #define f_Ez0       k_interp(ii, interpolator_var::Ez0)
  #define f_dEz0dx    k_interp(ii, interpolator_var::dEz0dx)
  #define f_dEz0dy    k_interp(ii, interpolator_var::dEz0dy)
  #define f_dEz0dz    k_interp(ii, interpolator_var::dEz0dz)
  #define f_d2Ez0dx   k_interp(ii, interpolator_var::d2Ez0dx)
  #define f_d2Ez0dy   k_interp(ii, interpolator_var::d2Ez0dy)
  #define f_d2Ez0dz   k_interp(ii, interpolator_var::d2Ez0dz)

  #define f_Gx0       k_interp(ii, interpolator_var::Gx0)
  #define f_dGx0dx    k_interp(ii, interpolator_var::dGx0dx)
  #define f_dGx0dy    k_interp(ii, interpolator_var::dGx0dy)
  #define f_dGx0dz    k_interp(ii, interpolator_var::dGx0dz)
  #define f_d2Gx0dx   k_interp(ii, interpolator_var::d2Gx0dx)
  #define f_d2Gx0dy   k_interp(ii, interpolator_var::d2Gx0dy)
  #define f_d2Gx0dz   k_interp(ii, interpolator_var::d2Gx0dz)
  #define f_Gy0       k_interp(ii, interpolator_var::Gy0)
  #define f_dGy0dx    k_interp(ii, interpolator_var::dGy0dx)
  #define f_dGy0dy    k_interp(ii, interpolator_var::dGy0dy)
  #define f_dGy0dz    k_interp(ii, interpolator_var::dGy0dz)
  #define f_d2Gy0dx   k_interp(ii, interpolator_var::d2Gy0dx)
  #define f_d2Gy0dy   k_interp(ii, interpolator_var::d2Gy0dy)
  #define f_d2Gy0dz   k_interp(ii, interpolator_var::d2Gy0dz)
  #define f_Gz0       k_interp(ii, interpolator_var::Gz0)
  #define f_dGz0dx    k_interp(ii, interpolator_var::dGz0dx)
  #define f_dGz0dy    k_interp(ii, interpolator_var::dGz0dy)
  #define f_dGz0dz    k_interp(ii, interpolator_var::dGz0dz)
  #define f_d2Gz0dx   k_interp(ii, interpolator_var::d2Gz0dx)
  #define f_d2Gz0dy   k_interp(ii, interpolator_var::d2Gz0dy)
  #define f_d2Gz0dz   k_interp(ii, interpolator_var::d2Gz0dz)

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
      
    float v0, v1, v2, v3, v4, v5, v6;
    float w0, wx, wy, wz, wmx, wmy, wmz;
    auto  k_field_scatter_access = k_f_sv.access();

#ifdef VARIABLE_CHARGE
    float qp = p_q;
    float qdt_2mc = qp*dt_2mc;
#endif
    
    float dx   = p_dx;                             // Load position
    float dy   = p_dy;
    float dz   = p_dz;
    int   ii   = pii;
#ifdef SHAPE_NGP
  #ifdef EXTERNAL_FORCE
    float hax  = qdt_2mc*( f_ex + f_Ex0 ) + dt_2c * f_Gx0;
    float hay  = qdt_2mc*( f_ey + f_Ey0 ) + dt_2c * f_Gy0;
    float haz  = qdt_2mc*( f_ez + f_Ez0 ) + dt_2c * f_Gz0;
  #else
    float hax  = qdt_2mc*(    ( f_ex ) );
    float hay  = qdt_2mc*(    ( f_ey ) );
    float haz  = qdt_2mc*(    ( f_ez  ) );
  #endif
    float cbx  = f_cbx;// + dx*f_dcbxdx;             // Interpolate B
    float cby  = f_cby;// + dy*f_dcbydy;
    float cbz  = f_cbz;// + dz*f_dcbzdz;
#else
#ifdef SHAPE_QS
  #ifdef EXTERNAL_FORCE
    // Interpolate E, E0, G0
    float hax  = qdt_2mc*( f_ex + dx*( f_dexdx + dx*f_d2exdx )
                                + dy*( f_dexdy + dy*f_d2exdy )
                                + dz*( f_dexdz + dz*f_d2exdz )
                           + f_Ex0 + dx*( f_dEx0dx + dx*f_d2Ex0dx )
                                   + dy*( f_dEx0dy + dy*f_d2Ex0dy )
                                   + dz*( f_dEx0dz + dz*f_d2Ex0dz ) );
    float hay  = qdt_2mc*( f_ey + dx*( f_deydx + dx*f_d2eydx )
                                + dy*( f_deydy + dy*f_d2eydy )
                                + dz*( f_deydz + dz*f_d2eydz )
                           + f_Ey0 + dx*( f_dEy0dx + dx*f_d2Ey0dx )
                                   + dy*( f_dEy0dy + dy*f_d2Ey0dy )
                                   + dz*( f_dEy0dz + dz*f_d2Ey0dz ) );
    float haz  = qdt_2mc*( f_ez + dx*( f_dezdx + dx*f_d2ezdx )
                                + dy*( f_dezdy + dy*f_d2ezdy )
                                + dz*( f_dezdz + dz*f_d2ezdz )
                           + f_Ez0 + dx*( f_dEz0dx + dx*f_d2Ez0dx )
                                   + dy*( f_dEz0dy + dy*f_d2Ez0dy )
                                   + dz*( f_dEz0dz + dz*f_d2Ez0dz ) );
    hax += dt_2c *( f_Gx0 + dx*( f_dGx0dx + dx*f_d2Gx0dx )
                          + dy*( f_dGx0dy + dy*f_d2Gx0dy )
                          + dz*( f_dGx0dz + dz*f_d2Gx0dz ) );
    hay += dt_2c *( f_Gy0 + dx*( f_dGy0dx + dx*f_d2Gy0dx )
                          + dy*( f_dGy0dy + dy*f_d2Gy0dy )
                          + dz*( f_dGy0dz + dz*f_d2Gy0dz ) );
    haz += dt_2c *( f_Gz0 + dx*( f_dGz0dx + dx*f_d2Gz0dx )
                          + dy*( f_dGz0dy + dy*f_d2Gz0dy )
                          + dz*( f_dGz0dz + dz*f_d2Gz0dz ) );
  #else
    // Interpolate E
    float hax  = qdt_2mc*( f_ex + dx*( f_dexdx + dx*f_d2exdx )
                                + dy*( f_dexdy + dy*f_d2exdy )
                                + dz*( f_dexdz + dz*f_d2exdz ) );
    float hay  = qdt_2mc*( f_ey + dx*( f_deydx + dx*f_d2eydx )
                                + dy*( f_deydy + dy*f_d2eydy )
                                + dz*( f_deydz + dz*f_d2eydz ) );
    float haz  = qdt_2mc*( f_ez + dx*( f_dezdx + dx*f_d2ezdx )
                                + dy*( f_dezdy + dy*f_d2ezdy )
                                + dz*( f_dezdz + dz*f_d2ezdz ) );
  #endif
    // Interpolate B
    float cbx  = f_cbx + dx*( f_dcbxdx + dx*f_d2cbxdx )
                       + dy*( f_dcbxdy + dy*f_d2cbxdy )
                       + dz*( f_dcbxdz + dz*f_d2cbxdz );
    float cby  = f_cby + dx*( f_dcbydx + dx*f_d2cbydx )
                       + dy*( f_dcbydy + dy*f_d2cbydy )
                       + dz*( f_dcbydz + dz*f_d2cbydz );
    float cbz  = f_cbz + dx*( f_dcbzdx + dx*f_d2cbzdx )
                       + dy*( f_dcbzdy + dy*f_d2cbzdy )
                       + dz*( f_dcbzdz + dz*f_d2cbzdz );
#endif
#endif
    float ux   = p_ux;                             // Load momentum
    float uy   = p_uy;
    float uz   = p_uz;
    float q    = p_w;
    
    ux  += hax;                               // Half advance E
    uy  += hay;
    uz  += haz;
    v0   = qdt_2mc;///sqrtf(one + (ux*ux + (uy*uy + uz*uz)));
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

    v3   = one;///sqrtf(one + (ux*ux+ (uy*uy + uz*uz)));

    /**/                                      // Get norm displacement
    v4  = ux*cdt_dx;
    v5  = uy*cdt_dy;
    v6  = uz*cdt_dz;
    v4  *= v3;
    v5  *= v3;
    v6  *= v3;
    v0   = dx + v4;                           // Streak midpoint (inbnds)
    v1   = dy + v5;
    v2   = dz + v6;
    dx   = v0 + v4;                           // New position
    dy   = v1 + v5;
    dz   = v2 + v6;

    // printf("Pushed a particle advance_p index %d dx %e y %e z %e ux %e uy %e yz %e \n", p_index, dx, dy, dz, p_ux, p_uy, p_uz);


    
/*
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
*/


    if(  dx<=one  && dy<=one &&  dz<=one &&   // Check if inbnds
	-dx<=one && -dy<=one && -dz<=one ) {
      
      p_dx = dx;                             // Store new position
      p_dy = dy;
      p_dz = dz;

         // Common case (inbnds).  Note: accumulator values are 4 times
         // the total physical charge that passed through the appropriate
         // current quadrant in a time-step
#ifdef VARIABLE_CHARGE
      q *= qp;
#else
      q *= qsp;
#endif
         //v5 = q*ux*uy*uz*one_third;              // Compute correction

/*    
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
*/
        // TODO: That 2 needs to be 2*NGHOST eventually
        //int iii = ii;
        //int zi = iii/((nx+2)*(ny+2));
        //iii -= zi*(nx+2)*(ny+2);
        //int yi = iii/(nx+2);
        //int xi = iii - yi*(nx+2);
      
#ifdef SHAPE_NGP
      q *= rV;
      k_field_scatter_access(ii, field_var::jfx) += q*ux;
      k_field_scatter_access(ii, field_var::jfy) += q*uy;
      k_field_scatter_access(ii, field_var::jfz) += q*uz;
      k_field_scatter_access(ii, field_var::rhof) += q;
#else
#ifdef SHAPE_QS
      // stencil coefficients
      // ... OLD hybrid-VPIC with QS shape, the accumulator stores
      // ... ... p->w*qsp * two*(three - ...)
      // ... ... hyb_unload_accumulator(...) applies factor rV/12.
      // ... NEW HVPIC-K not using accumulator (yet), scatter directly to mesh,
      // ... ... so include all factors
      q *= rV12;
      w0 =  q*two*( three - v0*v0 - v1*v1 - v2*v2 );
      wx =  q*( v0 + one )*( v0 + one );
      wy =  q*( v1 + one )*( v1 + one );
      wz =  q*( v2 + one )*( v2 + one );
      wmx = q*( v0 - one )*( v0 - one );
      wmy = q*( v1 - one )*( v1 - one );
      wmz = q*( v2 - one )*( v2 - one );

      // Voxel indices
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii - yi*(nx+2);
      // Neighboring voxel 1D (flattened) indices
      int iix = VOXEL(xi+1,yi,zi,nx,ny,nz);
      int iiy = VOXEL(xi,yi+1,zi,nx,ny,nz);
      int iiz = VOXEL(xi,yi,zi+1,nx,ny,nz);
      int iimx = VOXEL(xi-1,yi,zi,nx,ny,nz);
      int iimy = VOXEL(xi,yi-1,zi,nx,ny,nz);
      int iimz = VOXEL(xi,yi,zi-1,nx,ny,nz);

      k_field_scatter_access(ii, field_var::jfx)  += w0*ux;
      k_field_scatter_access(ii, field_var::jfy)  += w0*uy;
      k_field_scatter_access(ii, field_var::jfz)  += w0*uz;
      k_field_scatter_access(ii, field_var::rhof) += w0;

      k_field_scatter_access(iix, field_var::jfx)  += wx*ux;
      k_field_scatter_access(iix, field_var::jfy)  += wx*uy;
      k_field_scatter_access(iix, field_var::jfz)  += wx*uz;
      k_field_scatter_access(iix, field_var::rhof) += wx;

      k_field_scatter_access(iiy, field_var::jfx)  += wy*ux;
      k_field_scatter_access(iiy, field_var::jfy)  += wy*uy;
      k_field_scatter_access(iiy, field_var::jfz)  += wy*uz;
      k_field_scatter_access(iiy, field_var::rhof) += wy;

      k_field_scatter_access(iiz, field_var::jfx)  += wz*ux;
      k_field_scatter_access(iiz, field_var::jfy)  += wz*uy;
      k_field_scatter_access(iiz, field_var::jfz)  += wz*uz;
      k_field_scatter_access(iiz, field_var::rhof) += wz;

      k_field_scatter_access(iimx, field_var::jfx)  += wmx*ux;
      k_field_scatter_access(iimx, field_var::jfy)  += wmx*uy;
      k_field_scatter_access(iimx, field_var::jfz)  += wmx*uz;
      k_field_scatter_access(iimx, field_var::rhof) += wmx;

      k_field_scatter_access(iimy, field_var::jfx)  += wmy*ux;
      k_field_scatter_access(iimy, field_var::jfy)  += wmy*uy;
      k_field_scatter_access(iimy, field_var::jfz)  += wmy*uz;
      k_field_scatter_access(iimy, field_var::rhof) += wmy;

      k_field_scatter_access(iimz, field_var::jfx)  += wmz*ux;
      k_field_scatter_access(iimz, field_var::jfy)  += wmz*uy;
      k_field_scatter_access(iimz, field_var::jfz)  += wmz*uz;
      k_field_scatter_access(iimz, field_var::rhof) += wmz;

#endif
#endif
    
} else {
      
      DECLARE_ALIGNED_ARRAY( particle_mover_t, 16, local_pm, 1 );
      local_pm->dispx = v4;
      local_pm->dispy = v5;
      local_pm->dispz = v6;
      local_pm->i     = p_index;
      
      //printf("Calling move_p index %d dx %e y %e z %e ux %e uy %e uz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
      if( move_p_kokkos( k_particles, k_particles_i, local_pm, // Unlikely
			 k_f_sv, g, k_neighbors, rangel, rangeh, qsp, gdx, gdy, gdz, gdt, nx, ny, nz ) )
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
#ifdef VARIABLE_CHARGE
	      k_particle_copy(nm, particle_var::qp) = p_q;
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
    //printf("Finished advance_p loop index %d dx %e y %e z %e ux %e uy %e uz %e \n", p_index, ux, uy, uz, p_ux, p_uy, p_uz);
#ifdef VPIC_ENABLE_HIERARCHICAL
    }
    });
      });
#else
    });
#endif

  Kokkos::Experimental::contribute(k_field, k_f_sv);
  
  
    // TODO: abstract this manual data copy
  //Kokkos::deep_copy(h_nm, k_nm);

  //args->seg[pipeline_rank].pm        = pm;
  //args->seg[pipeline_rank].max_nm    = max_nm;
  //args->seg[pipeline_rank].nm        = h_nm(0);
  //args->seg[pipeline_rank].n_ignored = 0; // TODO: update this
  //delete(k_local_particle_movers_p);
  //return h_nm(0);

  #undef f_ex
  #undef f_dexdx
  #undef f_dexdy
  #undef f_dexdz
  #undef f_d2exdx
  #undef f_d2exdy
  #undef f_d2exdz
  #undef f_ey
  #undef f_deydx
  #undef f_deydy
  #undef f_deydz
  #undef f_d2eydx
  #undef f_d2eydy
  #undef f_d2eydz
  #undef f_ez
  #undef f_dezdx
  #undef f_dezdy
  #undef f_dezdz
  #undef f_d2ezdx
  #undef f_d2ezdy
  #undef f_d2ezdz
  #undef f_cbx
  #undef f_dcbxdx
  #undef f_dcbxdy
  #undef f_dcbxdz
  #undef f_d2cbxdx
  #undef f_d2cbxdy
  #undef f_d2cbxdz
  #undef f_cby
  #undef f_dcbydx
  #undef f_dcbydy
  #undef f_dcbydz
  #undef f_d2cbydx
  #undef f_d2cbydy
  #undef f_d2cbydz
  #undef f_cbz
  #undef f_dcbzdx
  #undef f_dcbzdy
  #undef f_dcbzdz
  #undef f_d2cbzdx
  #undef f_d2cbzdy
  #undef f_d2cbzdz

  #undef f_Ex0
  #undef f_dEx0dx
  #undef f_dEx0dy
  #undef f_dEx0dz
  #undef f_d2Ex0dx
  #undef f_d2Ex0dy
  #undef f_d2Ex0dz
  #undef f_Ey0
  #undef f_dEy0dx
  #undef f_dEy0dy
  #undef f_dEy0dz
  #undef f_d2Ey0dx
  #undef f_d2Ey0dy
  #undef f_d2Ey0dz
  #undef f_Ez0
  #undef f_dEz0dx
  #undef f_dEz0dy
  #undef f_dEz0dz
  #undef f_d2Ez0dx
  #undef f_d2Ez0dy
  #undef f_d2Ez0dz

  #undef f_Gx0
  #undef f_dGx0dx
  #undef f_dGx0dy
  #undef f_dGx0dz
  #undef f_d2Gx0dx
  #undef f_d2Gx0dy
  #undef f_d2Gx0dz
  #undef f_Gy0
  #undef f_dGy0dx
  #undef f_dGy0dy
  #undef f_dGy0dz
  #undef f_d2Gy0dx
  #undef f_d2Gy0dy
  #undef f_d2Gy0dz
  #undef f_Gz0
  #undef f_dGz0dx
  #undef f_dGz0dy
  #undef f_dGz0dz
  #undef f_d2Gz0dx
  #undef f_d2Gz0dy
  #undef f_d2Gz0dz

      } //advance_p_kokkos_gpu

void
advance_p( /**/  species_t            * RESTRICT sp,
//           accumulator_array_t * RESTRICT aa,
           interpolator_array_t * RESTRICT ia,
           field_array_t* RESTRICT fa ) {
  //DECLARE_ALIGNED_ARRAY( advance_p_pipeline_args_t, 128, args, 1 );
  //DECLARE_ALIGNED_ARRAY( particle_mover_seg_t, 128, seg, MAX_PIPELINE+1 );
  //int rank;

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

#ifdef VARIABLE_CHARGE
  float dt_2mc = (sp->g->dt)/(2*sp->m*sp->g->cvac);
#else
  float qdt_2mc  = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
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
#ifdef VARIABLE_CHARGE
	  dt_2mc,
#else
          qdt_2mc,
#endif
          cdt_dx,
          cdt_dy,
          cdt_dz,
          sp->q,
          sp->np,
          sp->max_nm,
          sp->g->nx,
          sp->g->ny,
          sp->g->nz
  );
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


