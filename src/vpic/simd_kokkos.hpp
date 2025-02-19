#ifndef _simd_kokkos_h_
#define _simd_kokkos_h_
#include <Kokkos_Core.hpp>
//#ifdef __AVX2__
//#include <immintrin.h>
//#elif defined __VEC__
#include <altivec.h>
//#endif
// Necessary to compile decks properly due to altivec redefining bool and vector
#undef bool
#undef vector

#ifndef ALIGNED
#define ALIGNED(n)
#endif

#define __vector __attribute__((altivec(vector__)))

class vec128_half {
  __vector unsigned short v;

  public:
    KOKKOS_INLINE_FUNCTION vec128_half() {
      v = (__vector unsigned short) {0, 0, 0, 0, 0, 0, 0, 0};
    }

    KOKKOS_INLINE_FUNCTION vec128_half(__vector unsigned short data) {
      v = data;
    }

    KOKKOS_INLINE_FUNCTION vec128_half(float scalar) {
      __vector float fp32_vec = (__vector float) {scalar, scalar, scalar, scalar};
      v = vec_pack_to_short_fp32(fp32_vec, fp32_vec);
    }

    KOKKOS_INLINE_FUNCTION vec128_half(float a, float b, float c, float d, float e, float f, float g, float h) {
      __vector float fp32_v1 = (__vector float) {a, b, c, d};
      __vector float fp32_v2 = (__vector float) {e, f, g, h};
      v = vec_pack_to_short_fp32(fp32_v1, fp32_v2);;
    }

    KOKKOS_INLINE_FUNCTION float get(int i) {
      if(i < 4) {
        __vector float fp32 = vec_extract_fp32_from_shorth(v);
        return fp32[i];
      } else if(i < 8) {
        __vector float fp32 = vec_extract_fp32_from_shortl(v);
        return fp32[i-4];
      } else {
        printf("SIMD index out of range\n");
        return 0;
      }
    }

    KOKKOS_INLINE_FUNCTION void set(float val, int i) {
      if(i < 4) {
        __vector float hi = vec_extract_fp32_from_shorth(v);
        __vector float lo = vec_extract_fp32_from_shortl(v);
        hi = vec_insert(val, hi, i);
        v = vec_pack_to_short_fp32(hi, lo);
      } else if(i < 8) {
        __vector float hi = vec_extract_fp32_from_shorth(v);
        __vector float lo = vec_extract_fp32_from_shortl(v);
        lo = vec_insert(val, lo, i-4);
        v = vec_pack_to_short_fp32(hi, lo);
      } else {
        printf("SIMD index out of range\n");
      }
    }
};

//template <int N,typename T> v4 {
//  using vector = __vector T;
//  vector v;
//
//  public:
//    KOKKOS_INLINE_FUNCTION v4<N,T>(): v() {}
//
//    KOKKOS_INLINE_FUNCTION v4<N,T>(vector data) {
//      v = data;
//    }
//
//    KOKKOS_INLINE_FUNCTION v4<N,T>(T scalar) {
//      v = (vector) {scalar, scalar, scalar, scalar};
//    }
//
//    KOKKOS_INLINE_FUNCTION v4<N,T>(T a, T b, T c, T d) {
//      v = (vector) {a, b, c, d};
//    }
//
//    KOKKOS_INLINE_FUNCTION void load_4x1(const T * ALIGNED(16) array) {
//      v = vec_ld(0, array);
//    }
//
//    KOKKOS_INLINE_FUNCTION void store_4x1(const T * ALIGNED(16) array) {
//      vec_st(v, 0, array);
//    }
//
//		KOKKOS_INLINE_FUNCTION v4<N,T>& operator=(v4<N,T>& rhs) { 
//      v = rhs.v;
//      return *this;
//    }
//
//		KOKKOS_INLINE_FUNCTION v4<N,T>& operator+=(v4<N,T>& rhs) { 
//      v = vec_add(v, rhs.v);
//      return *this;
//    }
//
//		KOKKOS_INLINE_FUNCTION v4<N,T>& operator-=(v4<N,T>& rhs) { 
//      v = vec_sub(v, rhs.v);
//      return *this;
//    }
//
//		KOKKOS_INLINE_FUNCTION v4<N,T>& operator*=(v4<N,T>& rhs) { 
//      v = vec_mul(v, rhs.v);
//      return *this;
//    }
//
//		KOKKOS_INLINE_FUNCTION v4<N,T>& operator/=(v4<N,T>& rhs) { 
//      v = vec_div(v, rhs.v);
//      return *this;
//    }
//
//    KOKKOS_INLINE_FUNCTION v4<N,T> operator+(const v4<N,T>& rhs) {
//      return v4<N,T>(vec_add(v, rhs.v);
//    }
//
//    KOKKOS_INLINE_FUNCTION v4<N,T> operator-(const v4<N,T>& rhs) {
//      return v4<N,T>(vec_sub(v, rhs.v));
//    }
//    
//    KOKKOS_INLINE_FUNCTION v4<N,T> operator*(const v4<N,T>& rhs) {
//      return v4<N,T>(vec_mul(v, rhs.v));
//    }
//    
//    KOKKOS_INLINE_FUNCTION v4<N,T> operator/(const v4<N,T>& rhs) {
//      return v4<N,T>(vec_div(v, rhs.v));
//    }
//};
//
//KOKKOS_INLINE_FUNCTION v4<128,unsigned short>(v4<128,float> a, v4<128,float> b) {
//  v = v4<128,unsigned short>(vec_pack_to_short_fp32(a.v, b.v));
//}
//
//KOKKOS_INLINE_FUNCTION v4<128,float> extract_high(v4<128,unsigned short> a) {
//  return v4<128,float>(vec_extract_fp32_from_shorth(a.v));
//}
//
//KOKKOS_INLINE_FUNCTION v4<128,float> extract_low(v4<128,unsigned short> a) {
//  return v4<128,float>(vec_extract_fp32_from_shortl(a.v));
//}
//
//// a*b + c
//KOKKOS_INLINE_FUNCTION v4<128,float> fma(v4<128,float>& a, v4<128,float>& b, v4<128,float>& c) {
//  return v4<128,float>(vec_madd(a.v, b.v, c.v));
//}
//
//// a*b - c
//KOKKOS_INLINE_FUNCTION v4<128,float> fms(v4<128,float>& a, v4<128,float>& b, v4<128,float>& c) {
//  return v4<128,float>(vec_msub(a.v, b.v, c.v));
//}
//
//// Approximate reciprocal sqrt
//KOKKOS_INLINE_FUNCTION v4<128,float> rsqrt_approx(v4<128,float>& a) {
//  return v4<128,float>(vec_rsqrte(a.v));
//}
//
//// Approximate reciprocal sqrt with iterative refinement (unknown iterations)
//KOKKOS_INLINE_FUNCTION v4<128,float> rsqrt(v4<128,float>& a) {
//  return v4<128,float>(vec_rsqrt(a.v));
//}
//
//// Approximate reciprocal sqrt with iterative refinement (1 iteration)
//KOKKOS_INLINE_FUNCTION v4<128,float> rsqrt1(v4<128,float>& a) {
//  __vector float b = vec_rsqrte(a.v);
//  b = vec_madd( vec_nmsub( vec_madd( b, b, zero), a.v, one), vec_madd(b, half, zero), v);
//  return v4<128,float>(b);
//}
//
//// Approximate reciprocals
//KOKKOS_INLINE_FUNCTION v4<128,float> rcp_approx(v4<128,float>& a) {
//  return v4<128,float>(vec_re(a.v));
//}
//
//// Approximate reciprocals with iterative refinement (1 iteration)
//KOKKOS_INLINE_FUNCTION v4<128,float> rcp_approx(v4<128,float>& a) {
//  __vector float b = vec_re(a.v);
//  b = vec_madd(vec_nmsub(b, a.v, one), b, b);
//  return v4<128,float>(b);
//}

#endif // _simd_kokkos_h_
