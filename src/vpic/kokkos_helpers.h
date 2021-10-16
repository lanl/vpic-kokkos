#ifndef _kokkos_helpers_h_
#define _kokkos_helpers_h_

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <iostream>

#include "../material/material.h" // Need material_t

//#include "cnl/all.h"
//using cnl::fixed_point;
//using fixed_point_t = fixed_point<int32_t, -23>;

//#include "simd_kokkos.hpp"

#if defined(__ppc64le__)
#include "altivec.h"
#undef vector
#undef bool
typedef unsigned short fp16_t;
#endif

#if defined(__x86_64__)
#include <emmintrin.h>
typedef short fp16_t;
#endif

// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
#define PARTICLE_VAR_COUNT 7
#define PARTICLE_MOVER_VAR_COUNT 3
#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4
#define INTERPOLATOR_VAR_COUNT 18
#define MATERIAL_COEFFICIENT_VAR_COUNT 13

//#define PARTICLE_WEIGHT_CONSTANT
//#define PARTICLE_WEIGHT_SHORT
#define PARTICLE_WEIGHT_FLOAT

//#define CPU_HALF
//#define CPU_FIXED

#ifdef KOKKOS_ENABLE_CUDA
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterNonDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutLeft
#else
  #define KOKKOS_SCATTER_DUPLICATED Kokkos::Experimental::ScatterDuplicated
  #define KOKKOS_SCATTER_ATOMIC Kokkos::Experimental::ScatterNonAtomic
  #define KOKKOS_LAYOUT Kokkos::LayoutRight
#endif

typedef int16_t material_id;

//typedef float float_t;


#ifdef CPU_FIXED
typedef short Q1_14;
constexpr float Q1_14_CONST = static_cast<float>(1 << 14);
constexpr float Q1_14_INV_CONST = (1.0/static_cast<float>(1 << 14));
#endif


//#ifdef KOKKOS_ENABLE_CUDA
//typedef __half pos_t;
//typedef float mom_t;
//typedef float mixed_t;
//#else
//typedef float pos_t;
//typedef float mom_t;
//typedef float mixed_t;
//#endif

#ifdef __CUDA_ARCH__
//typedef Q1_14 pos_t;
//typedef __half pos_t;
typedef float pos_t;
typedef float mom_t;
typedef float mixed_t;
#else
//typedef Q1_14 pos_t;
//typedef __half pos_t;
//typedef __fp16 pos_t;
//typedef fp16_t pos_t;
typedef float pos_t;
typedef float mom_t;
typedef float mixed_t;
#endif

//#define GPUSpace  Kokkos::DefaultExecutionSpace::memory_space

template<typename Position, typename Momentum>
class k_particles_struct {
    public:
        Kokkos::View<Position *> dx;
        Kokkos::View<Position *> dy;
        Kokkos::View<Position *> dz;
        Kokkos::View<Momentum *> ux;
        Kokkos::View<Momentum *> uy;
        Kokkos::View<Momentum *> uz;
        Kokkos::View<int   *> i;
#if defined PARTICLE_WEIGHT_SHORT
        Kokkos::View<short *> w;
#elif defined PARTICLE_WEIGHT_FLOAT
        Kokkos::View<float *> w;
#endif

        k_particles_struct() {}

        k_particles_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
#if !defined PARTICLE_WEIGHT_CONSTANT
            i("Particle index", num_particles),
            w("Particle weight", num_particles){}
#else
            i("Particle index", num_particles){}
#endif

        KOKKOS_INLINE_FUNCTION Position get_dx(size_t j) const {
          return dx(j);
        }
        KOKKOS_INLINE_FUNCTION Position get_dy(size_t j) const {
          return dy(j);
        }
        KOKKOS_INLINE_FUNCTION Position get_dz(size_t j) const {
          return dz(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_ux(size_t j) const {
          return ux(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uy(size_t j) const {
          return uy(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uz(size_t j) const {
          return uz(j);
        }
//        KOKKOS_INLINE_FUNCTION float get_w(size_t j) const {
//          return w(j);
//        }
        KOKKOS_INLINE_FUNCTION int get_i(size_t j) const {
          return i(j);
        }

        KOKKOS_INLINE_FUNCTION void set_dx(Position val, size_t j) const {
          dx(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_dy(Position val, size_t j) const {
          dy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_dz(Position val, size_t j) const {
          dz(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_ux(Momentum val, size_t j) const {
          ux(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uy(Momentum val, size_t j) const {
          uy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uz(Momentum val, size_t j) const {
          uz(j) = val;
        }
//        KOKKOS_INLINE_FUNCTION void set_w(float val, size_t j) const {
//          w(j) = val;
//        }
        KOKKOS_INLINE_FUNCTION void set_i(int val, size_t j) const {
          i(j) = val;
        }

};

#ifdef CPU_HALF
template<typename Momentum>
class k_particles_struct<fp16_t, Momentum> {
    public:
        Kokkos::View<fp16_t *> dx;
        Kokkos::View<fp16_t *> dy;
        Kokkos::View<fp16_t *> dz;
        Kokkos::View<Momentum *> ux;
        Kokkos::View<Momentum *> uy;
        Kokkos::View<Momentum *> uz;
        Kokkos::View<int   *> i;
#if defined PARTICLE_WEIGHT_SHORT
        Kokkos::View<short *> w;
#elif defined PARTICLE_WEIGHT_FLOAT
        Kokkos::View<float *> w;
#endif

        k_particles_struct() {}

        k_particles_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
#if !defined PARTICLE_WEIGHT_CONSTANT
            i("Particle index", num_particles),
            w("Particle weight", num_particles){}
#else
            i("Particle index", num_particles){}
#endif

        KOKKOS_INLINE_FUNCTION float get_dx(int j) const {
//          int vec_idx = (j/8)*8;
//          int vec_lane = j%8;
//          __vector unsigned short half = {dx(j), 0, 0, 0, 0, 0, 0, 0};
//          __vector float single = vec_extract_fp32_from_shorth(half);
//          return single[0];
          __m128i half = _mm_set1_epi16(dx(j));
          __m128 single = _mm_cvtph_ps(half);
          return _mm_cvtss_f32(single);	
        }

        KOKKOS_INLINE_FUNCTION float get_dy(int j) const {
//          int vec_idx = (j/8)*8;
//          int vec_lane = j%8;
//          __vector unsigned short half = {dy(j), 0, 0, 0, 0, 0, 0, 0};
//          __vector float single = vec_extract_fp32_from_shorth(half);
//          return single[0];
          __m128i half = _mm_set1_epi16(dy(j));
          __m128 single = _mm_cvtph_ps(half);
          return _mm_cvtss_f32(single);	
        }

        KOKKOS_INLINE_FUNCTION float get_dz(int j) const {
//          int vec_idx = (j/8)*8;
//          int vec_lane = j%8;
//          __vector unsigned short half = {dz(j), 0, 0, 0, 0, 0, 0, 0};
//          __vector float single = vec_extract_fp32_from_shorth(half);
//          return single[0];
          __m128i half = _mm_set1_epi16(dz(j));
          __m128 single = _mm_cvtph_ps(half);
          return _mm_cvtss_f32(single);	
        }

        KOKKOS_INLINE_FUNCTION Momentum get_ux(size_t j) const {
          return ux(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uy(size_t j) const {
          return uy(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uz(size_t j) const {
          return uz(j);
        }
//        KOKKOS_INLINE_FUNCTION float get_w(size_t j) const {
//          return w(j);
//        }
        KOKKOS_INLINE_FUNCTION int get_i(size_t j) const {
          return i(j);
        }

        KOKKOS_INLINE_FUNCTION void set_dx(float val, size_t j) const {
//          __vector float single = {0.0, 0.0, 0.0, 0.0};
//          single[0] = val;
//          __vector unsigned short half = vec_pack_to_short_fp32(single, single);
//          dx(j) = half[0];
          __m128 single = _mm_set1_ps(val);
          __m128i half = _mm_cvtps_ph(single, 3);
          dx(j) = _mm_extract_epi16(half, 0);
        }

        KOKKOS_INLINE_FUNCTION void set_dy(float val, size_t j) const {
//          __vector float single = {0.0, 0.0, 0.0, 0.0};
//          single[0] = val;
//          __vector unsigned short half = vec_pack_to_short_fp32(single, single);
//          dy(j) = half[0];
          __m128 single = _mm_set1_ps(val);
          __m128i half = _mm_cvtps_ph(single, 3);
          dy(j) = _mm_extract_epi16(half, 0);
        }

        KOKKOS_INLINE_FUNCTION void set_dz(float val, size_t j) const {
//          __vector float single = {0.0, 0.0, 0.0, 0.0};
//          single[0] = val;
//          __vector unsigned short half = vec_pack_to_short_fp32(single, single);
//          dz(j) = half[0];
          __m128 single = _mm_set1_ps(val);
          __m128i half = _mm_cvtps_ph(single, 3);
          dz(j) = _mm_extract_epi16(half, 0);
        }


        KOKKOS_INLINE_FUNCTION void set_ux(Momentum val, size_t j) const {
          ux(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uy(Momentum val, size_t j) const {
          uy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uz(Momentum val, size_t j) const {
          uz(j) = val;
        }
//        KOKKOS_INLINE_FUNCTION void set_w(float val, size_t j) const {
//          w(j) = val;
//        }
        KOKKOS_INLINE_FUNCTION void set_i(int val, size_t j) const {
          i(j) = val;
        }

};
#endif

#ifdef CPU_FIXED
template<typename Momentum>
class k_particles_struct<Q1_14, Momentum> {
    public:
        Kokkos::View<Q1_14 *> dx;
        Kokkos::View<Q1_14 *> dy;
        Kokkos::View<Q1_14 *> dz;
        Kokkos::View<Momentum *> ux;
        Kokkos::View<Momentum *> uy;
        Kokkos::View<Momentum *> uz;
        Kokkos::View<int   *> i;
#if defined PARTICLE_WEIGHT_SHORT
        Kokkos::View<short *> w;
#elif defined PARTICLE_WEIGHT_FLOAT
        Kokkos::View<float *> w;
#endif

        k_particles_struct() {}

        k_particles_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
#if !defined PARTICLE_WEIGHT_CONSTANT
            i("Particle index", num_particles),
            w("Particle weight", num_particles){}
#else
            i("Particle index", num_particles){}
#endif

        KOKKOS_INLINE_FUNCTION float get_dx(size_t j) const {
          return static_cast<float>(dx(j)) * Q1_14_INV_CONST;
        }
        KOKKOS_INLINE_FUNCTION float get_dy(size_t j) const {
          return static_cast<float>(dy(j)) * Q1_14_INV_CONST;
        }
        KOKKOS_INLINE_FUNCTION float get_dz(size_t j) const {
          return static_cast<float>(dz(j)) * Q1_14_INV_CONST;
        }
        KOKKOS_INLINE_FUNCTION Momentum get_ux(size_t j) const {
          return ux(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uy(size_t j) const {
          return uy(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uz(size_t j) const {
          return uz(j);
        }
//        KOKKOS_INLINE_FUNCTION float get_w(size_t j) const {
//          return w(j);
//        }
        KOKKOS_INLINE_FUNCTION int get_i(size_t j) const {
          return i(j);
        }

        KOKKOS_INLINE_FUNCTION void set_dx(float val, size_t j) const {
//          dx(j) = static_cast<Q1_14>((val * Q1_14_CONST));
          dx(j) = static_cast<Q1_14>(roundf(val * Q1_14_CONST));
        }
        KOKKOS_INLINE_FUNCTION void set_dy(float val, size_t j) const {
//          dy(j) = static_cast<Q1_14>((val * Q1_14_CONST));
          dy(j) = static_cast<Q1_14>(roundf(val * Q1_14_CONST));
        }
        KOKKOS_INLINE_FUNCTION void set_dz(float val, size_t j) const {
//          dz(j) = static_cast<Q1_14>((val * Q1_14_CONST));
          dz(j) = static_cast<Q1_14>(roundf(val * Q1_14_CONST));
        }
        KOKKOS_INLINE_FUNCTION void set_ux(Momentum val, size_t j) const {
          ux(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uy(Momentum val, size_t j) const {
          uy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uz(Momentum val, size_t j) const {
          uz(j) = val;
        }
//        KOKKOS_INLINE_FUNCTION void set_w(float val, size_t j) const {
//          w(j) = val;
//        }
        KOKKOS_INLINE_FUNCTION void set_i(int val, size_t j) const {
          i(j) = val;
        }
};
#endif

template<typename Position, typename Momentum>
class k_particles_host_struct {
    public:
        typename Kokkos::View<Position *>::HostMirror dx;
        typename Kokkos::View<Position *>::HostMirror dy;
        typename Kokkos::View<Position *>::HostMirror dz;
        typename Kokkos::View<Momentum *>::HostMirror ux;
        typename Kokkos::View<Momentum *>::HostMirror uy;
        typename Kokkos::View<Momentum *>::HostMirror uz;
        Kokkos::View<int   *>::HostMirror i;
#if defined PARTICLE_WEIGHT_SHORT
        Kokkos::View<short *>::HostMirror w;
#elif defined PARTICLE_WEIGHT_FLOAT
        Kokkos::View<float *>::HostMirror w;
#endif

        k_particles_host_struct() {}

        k_particles_host_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
#if !defined PARTICLE_WEIGHT_CONSTANT
            i("Particle index", num_particles),
            w("Particle weight", num_particles){}
#else
            i("Particle index", num_particles){}
#endif

        k_particles_host_struct<Position,Momentum>(k_particles_struct<Position,Momentum>& particles) {
            dx = Kokkos::create_mirror_view(particles.dx);
            dy = Kokkos::create_mirror_view(particles.dy);
            dz = Kokkos::create_mirror_view(particles.dz);
            ux = Kokkos::create_mirror_view(particles.ux);
            uy = Kokkos::create_mirror_view(particles.uy);
            uz = Kokkos::create_mirror_view(particles.uz);
            i = Kokkos::create_mirror_view(particles.i);
#if !defined PARTICLE_WEIGHT_CONSTANT
            w = Kokkos::create_mirror_view(particles.w);
#endif
        }

        KOKKOS_INLINE_FUNCTION Position get_dx(size_t j) const {
          return dx(j);
        }
        KOKKOS_INLINE_FUNCTION Position get_dy(size_t j) const {
          return dy(j);
        }
        KOKKOS_INLINE_FUNCTION Position get_dz(size_t j) const {
          return dz(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_ux(size_t j) const {
          return ux(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uy(size_t j) const {
          return uy(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uz(size_t j) const {
          return uz(j);
        }
//        KOKKOS_INLINE_FUNCTION float get_w(size_t j) const {
//          return w(j);
//        }
        KOKKOS_INLINE_FUNCTION int get_i(size_t j) const {
          return i(j);
        }

        KOKKOS_INLINE_FUNCTION void set_dx(Position val, size_t j) const {
          dx(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_dy(Position val, size_t j) const {
          dy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_dz(Position val, size_t j) const {
          dz(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_ux(Momentum val, size_t j) const {
          ux(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uy(Momentum val, size_t j) const {
          uy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uz(Momentum val, size_t j) const {
          uz(j) = val;
        }
//        KOKKOS_INLINE_FUNCTION void set_w(float val, size_t j) const {
//          w(j) = val;
//        }
        KOKKOS_INLINE_FUNCTION void set_i(int val, size_t j) const {
          i(j) = val;
        }
};

#ifdef CPU_HALF
template<typename Momentum>
class k_particles_host_struct<fp16_t, Momentum> {
    public:
        typename Kokkos::View<fp16_t *>::HostMirror dx;
        typename Kokkos::View<fp16_t *>::HostMirror dy;
        typename Kokkos::View<fp16_t *>::HostMirror dz;
        typename Kokkos::View<Momentum *>::HostMirror ux;
        typename Kokkos::View<Momentum *>::HostMirror uy;
        typename Kokkos::View<Momentum *>::HostMirror uz;
        Kokkos::View<int   *>::HostMirror i;
#if defined PARTICLE_WEIGHT_SHORT
        Kokkos::View<short *>::HostMirror w;
#elif defined PARTICLE_WEIGHT_FLOAT
        Kokkos::View<float *>::HostMirror w;
#endif

        k_particles_host_struct() {}

        k_particles_host_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
#if !defined PARTICLE_WEIGHT_CONSTANT
            i("Particle index", num_particles),
            w("Particle weight", num_particles){}
#else
            i("Particle index", num_particles){}
#endif

        k_particles_host_struct<fp16_t,Momentum>(k_particles_struct<fp16_t,Momentum>& particles) {
            dx = Kokkos::create_mirror_view(particles.dx);
            dy = Kokkos::create_mirror_view(particles.dy);
            dz = Kokkos::create_mirror_view(particles.dz);
            ux = Kokkos::create_mirror_view(particles.ux);
            uy = Kokkos::create_mirror_view(particles.uy);
            uz = Kokkos::create_mirror_view(particles.uz);
            i = Kokkos::create_mirror_view(particles.i);
#if !defined PARTICLE_WEIGHT_CONSTANT
            w = Kokkos::create_mirror_view(particles.w);
#endif
        }

        KOKKOS_INLINE_FUNCTION float get_dx(int j) const {
//          int vec_idx = (j/8)*8;
//          int vec_lane = j%8;
//          __vector unsigned short half = {dx(j), 0, 0, 0, 0, 0, 0, 0};
//          __vector float single = vec_extract_fp32_from_shorth(half);
//          return single[0];
          __m128i half = _mm_set1_epi16(dx(j));
          __m128 single = _mm_cvtph_ps(half);
          return (_mm_cvtss_f32(single));	
        }

        KOKKOS_INLINE_FUNCTION float get_dy(int j) const {
//          int vec_idx = (j/8)*8;
//          int vec_lane = j%8;
//          __vector unsigned short half = {dy(j), 0, 0, 0, 0, 0, 0, 0};
//          __vector float single = vec_extract_fp32_from_shorth(half);
//          return single[0];
          __m128i half = _mm_set1_epi16(dy(j));
          __m128 single = _mm_cvtph_ps(half);
          return (_mm_cvtss_f32(single));	
        }

        KOKKOS_INLINE_FUNCTION float get_dz(int j) const {
//          int vec_idx = (j/8)*8;
//          int vec_lane = j%8;
//          __vector unsigned short half = {dz(j), 0, 0, 0, 0, 0, 0, 0};
//          __vector float single = vec_extract_fp32_from_shorth(half);
//          return single[0];
          __m128i half = _mm_set1_epi16(dz(j));
          __m128 single = _mm_cvtph_ps(half);
          return (_mm_cvtss_f32(single));	
        }

        KOKKOS_INLINE_FUNCTION Momentum get_ux(int j) const {
          return ux(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uy(int j) const {
          return uy(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uz(int j) const {
          return uz(j);
        }
//        KOKKOS_INLINE_FUNCTION float get_w(int j) const {
//          return w(j);
//        }
        KOKKOS_INLINE_FUNCTION int get_i(int j) const {
          return i(j);
        }

        KOKKOS_INLINE_FUNCTION void set_dx(float val, size_t j) const {
//          __vector float single = {0.0, 0.0, 0.0, 0.0};
//          single[0] = val;
//          __vector unsigned short half = vec_pack_to_short_fp32(single, single);
//          dx(j) = half[0];
          __m128 single = _mm_set1_ps(val);
          __m128i half = _mm_cvtps_ph(single, 3);
          dx(j) = _mm_extract_epi16(half, 0);
        }

        KOKKOS_INLINE_FUNCTION void set_dy(float val, size_t j) const {
//          __vector float single = {0.0, 0.0, 0.0, 0.0};
//          single[0] = val;
//          __vector unsigned short half = vec_pack_to_short_fp32(single, single);
//          dy(j) = half[0];
          __m128 single = _mm_set1_ps(val);
          __m128i half = _mm_cvtps_ph(single, 3);
          dy(j) = _mm_extract_epi16(half, 0);
        }

        KOKKOS_INLINE_FUNCTION void set_dz(float val, size_t j) const {
//          __vector float single = {0.0, 0.0, 0.0, 0.0};
//          single[0] = val;
//          __vector unsigned short half = vec_pack_to_short_fp32(single, single);
//          dz(j) = half[0];
          __m128 single = _mm_set1_ps(val);
          __m128i half = _mm_cvtps_ph(single, 3);
          dz(j) = _mm_extract_epi16(half, 0);
        }

        KOKKOS_INLINE_FUNCTION void set_ux(Momentum val, size_t j) const {
          ux(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uy(Momentum val, size_t j) const {
          uy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uz(Momentum val, size_t j) const {
          uz(j) = val;
        }
//        KOKKOS_INLINE_FUNCTION void set_w(float val, size_t j) const {
//          w(j) = val;
//        }
        KOKKOS_INLINE_FUNCTION void set_i(int val, size_t j) const {
          i(j) = val;
        }

};
#endif

#ifdef CPU_FIXED
template<typename Momentum>
class k_particles_host_struct<Q1_14, Momentum> {
    public:
        typename Kokkos::View<Q1_14 *>::HostMirror dx;
        typename Kokkos::View<Q1_14 *>::HostMirror dy;
        typename Kokkos::View<Q1_14 *>::HostMirror dz;
        typename Kokkos::View<Momentum *>::HostMirror ux;
        typename Kokkos::View<Momentum *>::HostMirror uy;
        typename Kokkos::View<Momentum *>::HostMirror uz;
        Kokkos::View<int   *>::HostMirror i;
#if defined PARTICLE_WEIGHT_SHORT
        Kokkos::View<short *>::HostMirror w;
#elif defined PARTICLE_WEIGHT_FLOAT
        Kokkos::View<float *>::HostMirror w;
#endif

        k_particles_host_struct() {}

        k_particles_host_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
#if !defined PARTICLE_WEIGHT_CONSTANT
            i("Particle index", num_particles),
            w("Particle weight", num_particles){}
#else
            i("Particle index", num_particles){}
#endif

        k_particles_host_struct<Q1_14,Momentum>(k_particles_struct<Q1_14,Momentum>& particles) {
            dx = Kokkos::create_mirror_view(particles.dx);
            dy = Kokkos::create_mirror_view(particles.dy);
            dz = Kokkos::create_mirror_view(particles.dz);
            ux = Kokkos::create_mirror_view(particles.ux);
            uy = Kokkos::create_mirror_view(particles.uy);
            uz = Kokkos::create_mirror_view(particles.uz);
            i = Kokkos::create_mirror_view(particles.i);
#if !defined PARTICLE_WEIGHT_CONSTANT
            w = Kokkos::create_mirror_view(particles.w);
#endif
        }

        KOKKOS_INLINE_FUNCTION float get_dx(size_t j) const {
          return static_cast<float>(dx(j)) * Q1_14_INV_CONST;
        }
        KOKKOS_INLINE_FUNCTION float get_dy(size_t j) const {
          return static_cast<float>(dy(j)) * Q1_14_INV_CONST;
        }
        KOKKOS_INLINE_FUNCTION float get_dz(size_t j) const {
          return static_cast<float>(dz(j)) * Q1_14_INV_CONST;
        }
        KOKKOS_INLINE_FUNCTION Momentum get_ux(size_t j) const {
          return ux(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uy(size_t j) const {
          return uy(j);
        }
        KOKKOS_INLINE_FUNCTION Momentum get_uz(size_t j) const {
          return uz(j);
        }
//        KOKKOS_INLINE_FUNCTION float get_w(size_t j) const {
//          return w(j);
//        }
        KOKKOS_INLINE_FUNCTION int get_i(size_t j) const {
          return i(j);
        }

        KOKKOS_INLINE_FUNCTION void set_dx(float val, size_t j) const {
//          dx(j) = static_cast<Q1_14>((val * Q1_14_CONST));
          dx(j) = static_cast<Q1_14>(roundf(val * Q1_14_CONST));
        }
        KOKKOS_INLINE_FUNCTION void set_dy(float val, size_t j) const {
//          dy(j) = static_cast<Q1_14>((val * Q1_14_CONST));
          dy(j) = static_cast<Q1_14>(roundf(val * Q1_14_CONST));
        }
        KOKKOS_INLINE_FUNCTION void set_dz(float val, size_t j) const {
//          dz(j) = static_cast<Q1_14>((val * Q1_14_CONST));
          dz(j) = static_cast<Q1_14>(roundf(val * Q1_14_CONST));
        }
        KOKKOS_INLINE_FUNCTION void set_ux(Momentum val, size_t j) const {
          ux(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uy(Momentum val, size_t j) const {
          uy(j) = val;
        }
        KOKKOS_INLINE_FUNCTION void set_uz(Momentum val, size_t j) const {
          uz(j) = val;
        }
//        KOKKOS_INLINE_FUNCTION void set_w(float val, size_t j) const {
//          w(j) = val;
//        }
        KOKKOS_INLINE_FUNCTION void set_i(int val, size_t j) const {
          i(j) = val;
        }
};
#endif

typedef struct k_particle_mover {
  int i;
  float dispx;
  float dispy;
  float dispz;
} k_particle_mover_t;

class k_particle_movers_struct {
    public:
        Kokkos::View<float *> dispx;
        Kokkos::View<float *> dispy;
        Kokkos::View<float *> dispz;
        Kokkos::View<int   *> i;

        k_particle_movers_struct() {}

        k_particle_movers_struct(int num_particles) :
            dispx("Particle dx position", num_particles),
            dispy("Particle dy position", num_particles),
            dispz("Particle dz position", num_particles),
            i("Particle index", num_particles){}
};

class k_particle_movers_host_struct {
    public:
        Kokkos::View<float *>::HostMirror dispx;
        Kokkos::View<float *>::HostMirror dispy;
        Kokkos::View<float *>::HostMirror dispz;
        Kokkos::View<int   *>::HostMirror i;

        k_particle_movers_host_struct() {}

        k_particle_movers_host_struct(int num_particles) :
            dispx("Particle dx position", num_particles),
            dispy("Particle dy position", num_particles),
            dispz("Particle dz position", num_particles),
            i("Particle index", num_particles){}

        k_particle_movers_host_struct(k_particle_movers_struct& particles) {
            dispx = Kokkos::create_mirror_view(particles.dispx);
            dispy = Kokkos::create_mirror_view(particles.dispy);
            dispz = Kokkos::create_mirror_view(particles.dispz);
            i = Kokkos::create_mirror_view(particles.i);
        }
};

//using k_particles_soa_t      = k_particles_struct<fp16_t, mom_t>;
//using k_particles_host_soa_t = k_particles_host_struct<fp16_t, mom_t>;

using k_particles_soa_t      = k_particles_struct<pos_t, mom_t>;
using k_particles_host_soa_t = k_particles_host_struct<pos_t, mom_t>;

using k_particle_movers_soa_t = k_particle_movers_struct;
using k_particle_movers_host_soa_t = k_particle_movers_host_struct;

// TODO: we dont need the [1] here
using k_iterator_t = Kokkos::View<int[1]>;

using k_field_t = Kokkos::View<float *[FIELD_VAR_COUNT]>;
using k_field_edge_t = Kokkos::View<material_id* [FIELD_EDGE_COUNT]>;

// TODO Consolidate using Cabana?
using k_particles_t = Kokkos::View<float *[PARTICLE_VAR_COUNT]>;
using k_particles_i_t = Kokkos::View<int*>;

// TODO: think about the layout here
using k_particle_copy_t = Kokkos::View<float *[PARTICLE_VAR_COUNT], Kokkos::LayoutRight>;
using k_particle_i_copy_t = Kokkos::View<int*, Kokkos::LayoutRight>;

using k_particle_movers_t = Kokkos::View<float *[PARTICLE_MOVER_VAR_COUNT]>;
using k_particle_i_movers_t = Kokkos::View<int*>;

using k_neighbor_t = Kokkos::View<int64_t*>;

using k_interpolator_t = Kokkos::View<float *[INTERPOLATOR_VAR_COUNT]>;

using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;
//using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], Kokkos::MemoryTraits<Kokkos::Atomic> >;

using k_accumulators_sa_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH]>;

using k_accumulators_sah_t = Kokkos::Experimental::ScatterView<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated, Kokkos::Experimental::ScatterNonAtomic>;

using static_sched = Kokkos::Schedule<Kokkos::Static>;
using host_execution_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, static_sched, int>;

using k_material_coefficient_t = Kokkos::View<float* [MATERIAL_COEFFICIENT_VAR_COUNT]>;

using k_field_sa_t = Kokkos::Experimental::ScatterView<float *[FIELD_VAR_COUNT]>;

#define KOKKOS_TEAM_POLICY_DEVICE  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>
#define KOKKOS_TEAM_POLICY_HOST  Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>

namespace field_var {
  enum f_v {
    ex        = 0,
    ey        = 1,
    ez        = 2,
    div_e_err = 3,
    cbx       = 4,
    cby       = 5,
    cbz       = 6,
    div_b_err = 7,
    tcax      = 8,
    tcay      = 9,
    tcaz      = 10,
    rhob      = 11,
    jfx       = 12,
    jfy       = 13,
    jfz       = 14,
    rhof      = 15
  };
};
namespace field_edge_var { \
  enum f_e_v {
    ematx = 0,
    ematy = 1,
    ematz = 2,
    nmat  = 3,
    fmatx = 4,
    fmaty = 5,
    fmatz = 6,
    cmat  = 7
  };
};

namespace interpolator_var {
  enum i_r {
    ex       = 0,
    dexdy    = 1,
    dexdz    = 2,
    d2exdydz = 3,
    ey       = 4,
    deydz    = 5,
    deydx    = 6,
    d2eydzdx = 7,
    ez       = 8,
    dezdx    = 9,
    dezdy    = 10,
    d2ezdxdy = 11,
    cbx      = 12,
    dcbxdx   = 13,
    cby      = 14,
    dcbydy   = 15,
    cbz      = 16,
    dcbzdz   = 17
  };
};

namespace particle_var {
  enum p_v {
    dx = 0,
    dy,
    dz,
    //pi = 3,
    ux,
    uy,
    uz,
    w,
  };
};

namespace particle_mover_var {
  enum p_m_v {
     dispx = 0,
     dispy = 1,
     dispz = 2,
     //pmi   = 3,
  };
};

namespace accumulator_var {
  enum a_v {
    jx = 0,
    jy = 1,
    jz = 2,
  };
};

namespace material_coeff_var {
    enum mc_v {
        decayx        = 0,
        drivex        = 1,
        decayy        = 2,
        drivey        = 3,
        decayz        = 4,
        drivez        = 5,
        rmux          = 6,
        rmuy          = 7,
        rmuz          = 8,
        nonconductive = 9,
        epsx          = 10,
        epsy          = 11,
        epsz          = 12,
    };
};

void print_particles_d(
        k_particles_t particles,
        int np
        );
void print_accumulator(k_accumulators_t fields, int n);

// The templating here is to defer the type until later in the head include chain
template <class P>
bool compareParticleMovers(P& a, P& b) {
    return a.i < b.i;
}

#endif // _kokkos_helpers_h_
