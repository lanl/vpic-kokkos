#ifndef _kokkos_helpers_h_
#define _kokkos_helpers_h_

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <iostream>

#include "../material/material.h" // Need material_t

//#include "cnl/all.h"
//using cnl::fixed_point;
//using fixed_point_t = fixed_point<int32_t, -23>;

//#ifdef KOKKOS_ENABLE_CUDA
#include "cuda_fp16.h"
//#else
#include "half.hpp"
using half_float::half;
//#endif

// This module implements kokkos macros

#define FIELD_VAR_COUNT 16
#define FIELD_EDGE_COUNT 8
#define PARTICLE_VAR_COUNT 7
#define PARTICLE_MOVER_VAR_COUNT 3
#define ACCUMULATOR_VAR_COUNT 3
#define ACCUMULATOR_ARRAY_LENGTH 4
#define INTERPOLATOR_VAR_COUNT 18
#define MATERIAL_COEFFICIENT_VAR_COUNT 13

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

typedef float float_t;

//#ifdef KOKKOS_ENABLE_CUDA
//#define half __half
//#endif
//
//#define pos_t half
//#define mom_t half
//#define mixed_t half

template <typename T> class Half {
  public:
    T _data; // Actual data

    KOKKOS_INLINE_FUNCTION Half(): _data() {}

    KOKKOS_INLINE_FUNCTION Half(T data) {
      _data = data;
    }

    KOKKOS_INLINE_FUNCTION Half(const float f) {
#ifdef __CUDA_ARCH__
      _data = __float2half(f);
#else
      _data = static_cast<half>(f);
#endif
    }

    KOKKOS_INLINE_FUNCTION float half2float() {
      return __half2float(_data);
    }

    KOKKOS_INLINE_FUNCTION operator float() const {
#ifdef __CUDA_ARCH__
      return __half2float(_data);
#else
      return static_cast<float>(_data);
#endif
    }

		KOKKOS_INLINE_FUNCTION Half& operator=(Half rhs) { 
#ifdef __CUDA_ARCH__
      _data = rhs._data;
      return *this;
#else
      _data = rhs._data;
      return *this;
#endif
    }

		KOKKOS_INLINE_FUNCTION Half& operator=(float rhs) { 
#ifdef __CUDA_ARCH__
      _data = __float2half(rhs);
      return *this;
#else
      _data = static_cast<half>(rhs);
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator+(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hadd(_data, rhs._data));
#else
      return Half<T>(_data+rhs._data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator+(const float& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hadd(_data, __float2half(rhs)));
#else
      return Half<T>(_data+static_cast<T>(_data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator-(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hsub(_data, rhs._data));
#else
      return Half<T>(_data-rhs._data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator-(const float& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hsub(_data, __float2half(rhs)));
#else
      return Half<T>(_data-static_cast<T>(_data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator*(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hmul(_data, rhs._data));
#else
      return Half<T>(_data*rhs._data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator*(const float& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hmul(_data, __float2half(rhs)));
#else
      return Half<T>(_data*static_cast<T>(_data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator/(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hdiv(_data, rhs._data));
#else
      return Half<T>(_data/rhs._data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator/(const float& rhs) {
#ifdef __CUDA_ARCH__
      return Half<T>(__hdiv(_data, __float2half(rhs)));
#else
      return Half<T>(_data/static_cast<T>(_data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator+=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hadd(_data, rhs._data);
      return *this;
#else
      _data += rhs._data;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator+=(const float& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hadd(_data, __float2half(rhs));
      return *this;
#else
      _data += rhs;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator-=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hsub(_data, rhs._data);
      return *this;
#else
      _data -= rhs._data;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator*=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hmul(_data, rhs._data);
      return *this;
#else
      _data *= rhs._data;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator/=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hdiv(_data, rhs._data);
      return *this;
#else
      _data = rhs._data;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator++() {
#ifdef __CUDA_ARCH__
      _data = __hadd(_data, Half(1.0));
      return *this;
#else
      _data++;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T>& operator--() {
#ifdef __CUDA_ARCH__
      _data = __hsub(_data, Half(1.0));
      return *this;
#else
      _data--;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator++(int a) {
#ifdef __CUDA_ARCH__
      return Half<T>( __hadd(_data, Half(1.0)));
#else
      return Half<T>(++_data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator--(int a) {
#ifdef __CUDA_ARCH__
      return Half<T>( __hsub(_data, Half(1.0)));
#else
      return Half<T>(--_data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator+() const {
#ifdef __CUDA_ARCH__
      return Half<T>(_data);
#else
      return Half<T>(_data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half<T> operator-() const {
#ifdef __CUDA_ARCH__
      return Half<T>(__hneg(_data));
#else
      return -Half<T>(_data);
#endif
    }

    KOKKOS_INLINE_FUNCTION bool operator==(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __heq(_data, rhs._data);
#else
      return _data==rhs._data;
#endif
    }

    KOKKOS_INLINE_FUNCTION bool operator!=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __hne(_data, rhs._data);
#else
      return _data!=rhs._data;
#endif
    }

    KOKKOS_INLINE_FUNCTION bool operator<=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __hle(_data, rhs._data);
#else
      return _data<=rhs._data;
#endif
    }

    KOKKOS_INLINE_FUNCTION bool operator>=(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __hge(_data, rhs._data);
#else
      return _data>=rhs._data;
#endif
    }

    KOKKOS_INLINE_FUNCTION bool operator<(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __hlt(_data, rhs._data);
#else
      return _data<rhs._data;
#endif
    }

    KOKKOS_INLINE_FUNCTION bool operator>(const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __hgt(_data, rhs._data);
#else
      return _data>rhs._data;
#endif
    }
};

template<typename T>
KOKKOS_INLINE_FUNCTION Half<T> operator*(const float lhs, const Half<T>& rhs) {
#ifdef __CUDA_ARCH__
  return Half<T>(lhs) * rhs;
#else
  return Half<T>(lhs) * rhs;
#endif
}

template<typename T> 
KOKKOS_INLINE_FUNCTION Half<T> sqrt(Half<T> h) {
#ifdef __CUDA_ARCH__
  return Half<T>(hsqrt(h._data));
#endif
}

template<typename T>
KOKKOS_INLINE_FUNCTION Half<T> fma(const Half<T>& a, const Half<T>& b, const Half<T>& c) {
#ifdef __CUDA_ARCH__
  return Half<T>(__hfma(a._data, b._data, c._data));
#else
  return Half<T>(a._data*b._data + c._data);
#endif
}

template <typename T>
class Half2 {
  public:
    T _data; // Actual data

    KOKKOS_INLINE_FUNCTION Half2<T>(): _data() {}

    KOKKOS_INLINE_FUNCTION Half2<T>(__half2 data) {
      _data = data;
    }

    KOKKOS_INLINE_FUNCTION Half2<T>(const float f1, const float f2) {
      _data = __floats2half2_rn(f1,f2);
    }

    KOKKOS_INLINE_FUNCTION Half2<T>(const float f) {
      _data = __float2half2_rn(f);
    }

    KOKKOS_INLINE_FUNCTION Half2<T>(const __half h) {
#ifdef __CUDA_ARCH__
      _data = __half2half2(h);
#endif
    }

    KOKKOS_INLINE_FUNCTION float high2float() {
      return __high2float(_data);
    }

    KOKKOS_INLINE_FUNCTION float low2float() {
      return __low2float(_data);
    }

    KOKKOS_INLINE_FUNCTION Half<__half> high2half() {
      return Half<__half>(__high2half(_data));
    }

    KOKKOS_INLINE_FUNCTION Half<__half> low2half() {
      return Half<__half>(__low2half(_data));
    }

		KOKKOS_INLINE_FUNCTION Half2<T>& operator=(Half2<T> rhs) { 
#ifdef __CUDA_ARCH__
      _data = rhs._data;
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator+(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half2<T>(__hadd2(_data, rhs._data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator+(Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half2<T>(__hadd2(_data, rhs._data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator-(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half2<T>(__hsub2(_data, rhs._data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator*(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half2<T>(__hmul2(_data, rhs._data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator/(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      return Half2<T>(__h2div(_data, rhs._data));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T>& operator+=(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hadd2(_data, rhs._data);
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T>& operator-=(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hsub2(_data, rhs._data);
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T>& operator*=(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __hmul2(_data, rhs._data);
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T>& operator/=(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      _data = __h2div(_data, rhs._data);
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T>& operator++() {
#ifdef __CUDA_ARCH__
      _data = __hadd2(_data, __floats2half2_rn(1.0,1.0));
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T>& operator--() {
#ifdef __CUDA_ARCH__
      _data = __hsub2(_data, __floats2half2_rn(1.0,1.0));
      return *this;
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator++(int a) {
#ifdef __CUDA_ARCH__
      return Half2<T>( __hadd2(_data, __floats2half2_rn(1.0,1.0)));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator--(int a) {
#ifdef __CUDA_ARCH__
      return Half2<T>( __hsub2(_data, __floats2half2_rn(1.0,1.0)));
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator+() const {
#ifdef __CUDA_ARCH__
      return Half2<T>(_data);
#endif
    }

    KOKKOS_INLINE_FUNCTION Half2<T> operator-() const {
#ifdef __CUDA_ARCH__
      return Half2<T>(__hneg2(_data));
#endif
    }

//    KOKKOS_INLINE_FUNCTION bool operator==(const Half2<T>& rhs) {
//#ifdef __CUDA_ARCH__
//      return __heq(_data, rhs._data);
//#endif
//    }
//
//    KOKKOS_INLINE_FUNCTION bool operator!=(const Half2<T>& rhs) {
//#ifdef __CUDA_ARCH__
//      return __hne(_data, rhs._data);
//#endif
//    }

    KOKKOS_INLINE_FUNCTION bool operator<=(const Half2<T>& rhs) {
#ifdef __CUDA_ARCH__
      return __hble2(_data, rhs._data);
#endif
    }

//    KOKKOS_INLINE_FUNCTION bool operator>=(const Half2<T>& rhs) {
//#ifdef __CUDA_ARCH__
//      return __hge(_data, rhs._data);
//#endif
//    }
//
//    KOKKOS_INLINE_FUNCTION bool operator<(const Half2<T>& rhs) {
//#ifdef __CUDA_ARCH__
//      return __hlt(_data, rhs._data);
//#endif
//    }
//
//    KOKKOS_INLINE_FUNCTION bool operator>(const Half2<T>& rhs) {
//#ifdef __CUDA_ARCH__
//      return __hgt(_data, rhs._data);
//#endif
//    }
};

template <typename T>
KOKKOS_INLINE_FUNCTION Half2<T> sqrt(Half2<T> h) {
#ifdef __CUDA_ARCH__
  return Half2<T>(h2sqrt(h._data));
#endif
}

//KOKKOS_INLINE_FUNCTION Half2<T> abs(Half2<T> h) {
//#ifdef __CUDA_ARCH__
//  return Half2<T>(__habs2(h._data));
//#endif
//}

template <typename T>
KOKKOS_INLINE_FUNCTION Half2<T> eq(const Half2<T>& lhs, const Half2<T> & rhs) {
#ifdef __CUDA_ARCH__
  return Half2<T>(__heq2(lhs._data, rhs._data));
#endif
}

template <typename T>
KOKKOS_INLINE_FUNCTION Half2<T> leq(const Half2<T>& lhs, const Half2<T> & rhs) {
#ifdef __CUDA_ARCH__
  return Half2<T>(__hle2(lhs._data, rhs._data));
#endif
}

template <typename T>
KOKKOS_INLINE_FUNCTION Half2<T> fma(const Half2<T>& a, const Half2<T>& b, const Half2<T>& c) {
#ifdef __CUDA_ARCH__
  return Half2<T>(__hfma2(a._data, b._data, c._data));
#endif
}

#ifdef __CUDA_ARCH__
//#define pos_t Half2<__half2>
//#define mom_t Half2<__half2>
//#define mixed_t Half2<__half2>
typedef Half<__half> pos_t;
typedef Half<__half> mom_t;
typedef Half<__half> mixed_t;
typedef Half2<__half2> packed_t;
//typedef float pos_t;
//typedef float mom_t;
//typedef float mixed_t;
#else
//#define pos_t half
//#define mom_t half
//#define mixed_t half
typedef Half<__half> pos_t;
typedef Half<__half> mom_t;
typedef Half<__half> mixed_t;
typedef Half2<__half2> packed_t;
//typedef Half<half> pos_t;
//typedef Half<half> mom_t;
//typedef Half<half> mixed_t;
//typedef float pos_t;
//typedef float mom_t;
//typedef float mixed_t;
#endif

class k_particles_struct {
    public:
        Kokkos::View<pos_t *> dx;
        Kokkos::View<pos_t *> dy;
        Kokkos::View<pos_t *> dz;
        Kokkos::View<mom_t *> ux;
        Kokkos::View<mom_t *> uy;
        Kokkos::View<mom_t *> uz;
        Kokkos::View<float *> w;
        Kokkos::View<int   *> i;

        k_particles_struct() {}

        k_particles_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
            w("Particle weight", num_particles),
            i("Particle index", num_particles){}
};

class k_particles_host_struct {
    public:
        Kokkos::View<pos_t *>::HostMirror dx;
        Kokkos::View<pos_t *>::HostMirror dy;
        Kokkos::View<pos_t *>::HostMirror dz;
        Kokkos::View<mom_t *>::HostMirror ux;
        Kokkos::View<mom_t *>::HostMirror uy;
        Kokkos::View<mom_t *>::HostMirror uz;
        Kokkos::View<float *>::HostMirror w;
        Kokkos::View<int   *>::HostMirror i;

        k_particles_host_struct() {}

        k_particles_host_struct(int num_particles) :
            dx("Particle dx position", num_particles),
            dy("Particle dy position", num_particles),
            dz("Particle dz position", num_particles),
            ux("Particle ux momentum", num_particles),
            uy("Particle uy momentum", num_particles),
            uz("Particle uz momentum", num_particles),
            w("Particle weight", num_particles),
            i("Particle index", num_particles){}

        k_particles_host_struct(k_particles_struct& particles) {
            dx = Kokkos::create_mirror_view(particles.dx);
            dy = Kokkos::create_mirror_view(particles.dy);
            dz = Kokkos::create_mirror_view(particles.dz);
            ux = Kokkos::create_mirror_view(particles.ux);
            uy = Kokkos::create_mirror_view(particles.uy);
            uz = Kokkos::create_mirror_view(particles.uz);
            w = Kokkos::create_mirror_view(particles.w);
            i = Kokkos::create_mirror_view(particles.i);
        }
};

typedef struct k_particle_mover {
  pos_t dispx;
  pos_t dispy;
  pos_t dispz;
  int i;
} k_particle_mover_t;

class k_particle_movers_struct {
    public:
        Kokkos::View<pos_t *> dispx;
        Kokkos::View<pos_t *> dispy;
        Kokkos::View<pos_t *> dispz;
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
        Kokkos::View<pos_t *>::HostMirror dispx;
        Kokkos::View<pos_t *>::HostMirror dispy;
        Kokkos::View<pos_t *>::HostMirror dispz;
        Kokkos::View<int   *>::HostMirror i;

        k_particle_movers_host_struct() {}

        k_particle_movers_host_struct(int num_particles) :
            dispx("Particle dx position", num_particles),
            dispy("Particle dy position", num_particles),
            dispz("Particle dz position", num_particles),
            i("Particle index", num_particles){}

        k_particle_movers_host_struct(k_particles_struct& particles) {
            dispx = Kokkos::create_mirror_view(particles.dx);
            dispy = Kokkos::create_mirror_view(particles.dy);
            dispz = Kokkos::create_mirror_view(particles.dz);
            i = Kokkos::create_mirror_view(particles.i);
        }
};

using k_particles_soa_t = k_particles_struct;
using k_particles_host_soa_t = k_particles_host_struct;
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

using k_accumulators_t = Kokkos::View<float *[ACCUMULATOR_VAR_COUNT][ACCUMULATOR_ARRAY_LENGTH], Kokkos::MemoryTraits<Kokkos::Atomic> >;

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
