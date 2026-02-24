#ifndef _kokkos_simd_extensions_h_
#define _kokkos_simd_extensions_h_
#include <type_traits>
#include <Kokkos_SIMD.hpp>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

namespace KokkosSIMD = Kokkos::Experimental;

using element_aligned_tag_t = KokkosSIMD::element_aligned_tag;
using vector_aligned_tag_t  = KokkosSIMD::vector_aligned_tag;

#if defined(KOKKOS_ARCH_AVX512XEON)
constexpr int native_32 = 16;
constexpr int native_64 = 8;
#elif defined(KOKKOS_ARCH_AVX2)
constexpr int native_32 = 8;
constexpr int native_64 = 4;
#elif defined(KOKKOS_ARCH_ARM_NEON)
constexpr int native_32 = 4;
constexpr int native_64 = 2;
#else
constexpr int native_32 = 1;
constexpr int native_64 = 1;
#endif

using simd_float_t          = KokkosSIMD::simd<float>;
using simd_int32_t          = KokkosSIMD::simd<int32_t>;
using simd_int64_t          = KokkosSIMD::simd<int64_t>;
using simd_float_mask_t     = KokkosSIMD::simd_mask<float>;
using simd_int32_mask_t     = KokkosSIMD::simd_mask<int32_t>;
using simd_int64_mask_t     = KokkosSIMD::simd_mask<int64_t>;
using simd_float32x4_t      = KokkosSIMD::simd<float, 4>;
using simd_float32x4_mask_t = KokkosSIMD::simd_mask<float, 4>;

//using simd_float32x4_t      = KokkosSIMD::simd<float, 4>;
//using simd_int32x4_t        = KokkosSIMD::simd<int,   4>;
//using simd_float32x4_mask_t = KokkosSIMD::simd_mask<float, 4>;
//using simd_int32x4_mask_t   = KokkosSIMD::simd_mask<int,   4>;

//using simd_float_t          = KokkosSIMD::simd<float,  KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int32_t          = KokkosSIMD::simd<int32_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int64_t          = KokkosSIMD::simd<int64_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_float_mask_t     = KokkosSIMD::simd_mask<float,  KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int32_mask_t     = KokkosSIMD::simd_mask<int32_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int64_mask_t     = KokkosSIMD::simd_mask<int64_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;

constexpr auto SIMD_LEN = simd_float_t::size();

//using SIMDFloat_t = Kokkos::Experimental::simd<float>;

template<int i0, int i1, int i2, int i3>
struct permute
{
  constexpr static int value = i0 + i1*4 + i2*16 + i3*64;
};

template<typename SIMDFloat_t>
KOKKOS_FORCEINLINE_FUNCTION
void increment( float * p, const SIMDFloat_t &v ) {
  SIMDFloat_t a;
#if KOKKOS_VERSION_MAJOR == 5
  a = KokkosSIMD::simd_unchecked_load<SIMDFloat_t>(p, vector_aligned_tag_t());
  a += v;
  KokkosSIMD::simd_unchecked_store(a, p, vector_aligned_tag_t());
#else
  a.copy_from(p, element_aligned_tag_t());
  a += v;
  a.copy_to(  p, element_aligned_tag_t());
#endif
}

KOKKOS_FORCEINLINE_FUNCTION
void swap(float& a, float& b) {
  float t = (float) a;
  a = (float) b;
  b = t;
}

#ifdef __AVX512F__

KOKKOS_INLINE_FUNCTION
KokkosSIMD::simd<float, 16> simd_cast(KokkosSIMD::simd<int32_t, 16>& a) {
  return KokkosSIMD::simd<float,16>( _mm512_castsi512_ps( (__m512i)(a) ) );
}

KOKKOS_INLINE_FUNCTION
KokkosSIMD::simd<int32_t,16> simd_cast(KokkosSIMD::simd<float,16>& a) {
  return KokkosSIMD::simd<int32_t,16>( _mm512_castps_si512( (__m512)(a) ) );
}

template<typename SIMDFloat_t, typename std::enable_if_t<SIMDFloat_t::size() == 16, bool> = true >
SIMDFloat_t rsqrt(SIMDFloat_t a)
{
  __m512 b;
  __m512 a_v = (__m512)(a), b_v;

  // b_v = _mm512_rsqrt28_ps(a_v);

  b_v = _mm512_rsqrt14_ps(a_v);

  b = _mm512_add_ps( b_v, _mm512_mul_ps( _mm512_set1_ps( 0.5f ),
				   _mm512_sub_ps( b_v,
					    _mm512_mul_ps( a_v,
							   _mm512_mul_ps( b_v,
									  _mm512_mul_ps( b_v, b_v ) ) ) ) ) );

  // Note: It is quicker to just call div_ps and sqrt_ps if more refinement
  // is desired.
  // b.v = _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( a.v ) );

  return SIMDFloat_t(b);
}

template<typename SIMDFloat_t, typename SIMDInt32_t, typename std::enable_if<SIMDFloat_t::size() == 16, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose_particles_16x8(SIMDFloat_t& dx, SIMDFloat_t& dy, SIMDFloat_t& dz, SIMDInt32_t& ii,
           		                SIMDFloat_t& ux, SIMDFloat_t& uy, SIMDFloat_t& uz, SIMDFloat_t& wt )
{
  __m512 t00, t01, t02, t03, t04, t05, t06, t07;

  __m512i idx = _mm512_set_epi32( 15, 11, 14, 10, 13, 9, 12, 8, 7, 3, 6, 2, 5, 1, 4, 0 );

  // Begin
  // dx                                                                                 //   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  // dy                                                                                 //  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  // dz                                                                                 //  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  // ii                                                                                 //  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  // ux                                                                                 //  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  // uy                                                                                 //  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  // uz                                                                                 //  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  // wt                                                                                 // 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127

  t00 = _mm512_unpacklo_ps( static_cast<__m512>(dx), static_cast<__m512>(dy) );         //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  t01 = _mm512_unpackhi_ps( static_cast<__m512>(dx), static_cast<__m512>(dy) );         //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  t02 = _mm512_unpacklo_ps( static_cast<__m512>(dz), static_cast<__m512>(ii) );         //  32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
  t03 = _mm512_unpackhi_ps( static_cast<__m512>(dz), static_cast<__m512>(ii) );         //  34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
  t04 = _mm512_unpacklo_ps( static_cast<__m512>(ux), static_cast<__m512>(uy) );         //  64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
  t05 = _mm512_unpackhi_ps( static_cast<__m512>(ux), static_cast<__m512>(uy) );         //  66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
  t06 = _mm512_unpacklo_ps( static_cast<__m512>(uz), static_cast<__m512>(wt) );         //  96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
  t07 = _mm512_unpackhi_ps( static_cast<__m512>(uz), static_cast<__m512>(wt) );         //  98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127

  dx = SIMDFloat_t( _mm512_shuffle_ps( t00, t02, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );         //   0  16  32  48   4  20  36  52   8  24  40  56  12  28  44  60
  dy = SIMDFloat_t( _mm512_shuffle_ps( t00, t02, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );         //   1  17  33  49   5  21  37  53   9  25  41  57  13  29  45  61
  dz = SIMDFloat_t( _mm512_shuffle_ps( t01, t03, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );         //   2  18  34  50   6  22  38  54  10  26  42  58  14  30  46  62
  ii = SIMDFloat_t( _mm512_shuffle_ps( t01, t03, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );         //   3  19  35  51   7  23  39  55  11  27  43  59  15  31  47  63
  ux = SIMDFloat_t( _mm512_shuffle_ps( t04, t06, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );         //  64  80  96 112  68  84 100 116  72  88 104 120  76  92 108 124
  uy = SIMDFloat_t( _mm512_shuffle_ps( t04, t06, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );         //  65  81  97 113  69  85 101 117  73  89 105 121  77  93 109 125
  uz = SIMDFloat_t( _mm512_shuffle_ps( t05, t07, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );         //  66  82  98 114  70  86 102 118  74  90 106 122  78  94 110 126
  wt = SIMDFloat_t( _mm512_shuffle_ps( t05, t07, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );         //  67  83  99 115  71  87 103 119  75  91 107 123  79  95 111 127

  t00 = _mm512_shuffle_f32x4( static_cast<__m512>(dx), static_cast<__m512>(ux), 0x88 ); //   0  16  32  48   8  24  40  56  64  80  96 112  72  88 104 120
  t01 = _mm512_shuffle_f32x4( static_cast<__m512>(dy), static_cast<__m512>(uy), 0x88 ); //   1  17  33  49   9  25  41  57  65  81  97 113  73  89 105 121
  t02 = _mm512_shuffle_f32x4( static_cast<__m512>(dz), static_cast<__m512>(uz), 0x88 ); //   2  18  34  50  10  26  42  58  66  82  98 114  74  90 106 122
  t03 = _mm512_shuffle_f32x4( static_cast<__m512>(ii), static_cast<__m512>(wt), 0x88 ); //   3  19  35  51  11  27  43  59  67  83  99 115  75  91 107 123
  t04 = _mm512_shuffle_f32x4( static_cast<__m512>(dx), static_cast<__m512>(ux), 0xdd ); //   4  20  36  52  12  28  44  60  68  84 100 116  76  92 108 124
  t05 = _mm512_shuffle_f32x4( static_cast<__m512>(dy), static_cast<__m512>(uy), 0xdd ); //   5  21  37  53  13  29  45  61  69  85 101 117  77  93 109 125
  t06 = _mm512_shuffle_f32x4( static_cast<__m512>(dz), static_cast<__m512>(uz), 0xdd ); //   6  22  38  54  14  30  46  62  70  86 102 118  78  94 110 126
  t07 = _mm512_shuffle_f32x4( static_cast<__m512>(ii), static_cast<__m512>(wt), 0xdd ); //   7  23  39  55  15  31  47  63  71  87 103 119  79  95 111 127

  dx = SIMDFloat_t( _mm512_permutexvar_ps( idx, t00 ) );                                //   0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120
  dy = SIMDFloat_t( _mm512_permutexvar_ps( idx, t01 ) );                                //   1   9  17  25  33  41  49  57  65  73  81  89  97 105 113 121
  dz = SIMDFloat_t( _mm512_permutexvar_ps( idx, t02 ) );                                //   2  10  18  26  34  42  50  58  66  74  82  90  98 106 114 122
  ii = SIMDInt32_t( _mm512_permutexvar_ps( idx, t03 ) );                                //   3  11  19  27  35  43  51  59  67  75  83  91  99 107 115 123
  ux = SIMDFloat_t( _mm512_permutexvar_ps( idx, t04 ) );                                //   4  12  20  28  36  44  52  60  68  76  84  92 100 108 116 124
  uy = SIMDFloat_t( _mm512_permutexvar_ps( idx, t05 ) );                                //   5  13  21  29  37  45  53  61  69  77  85  93 101 109 117 125
  uz = SIMDFloat_t( _mm512_permutexvar_ps( idx, t06 ) );                                //   6  14  22  30  38  46  54  62  70  78  86  94 102 110 118 126
  wt = SIMDFloat_t( _mm512_permutexvar_ps( idx, t07 ) );                                //   7  15  23  31  39  47  55  63  71  79  87  95 103 111 119 127
}

template<typename SIMDFloat_t, typename SIMDInt32_t, typename std::enable_if<SIMDFloat_t::size() == 16, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose_particles_16x8_reverse(SIMDFloat_t& dx, SIMDFloat_t& dy, SIMDFloat_t& dz, SIMDInt32_t& ii,
           		                        SIMDFloat_t& ux, SIMDFloat_t& uy, SIMDFloat_t& uz, SIMDFloat_t& wt )
{
  __m512 t00, t01, t02, t03, t04, t05, t06, t07;
  __m512 u00, u01, u02, u03, u04, u05, u06, u07;

  __m512i idx = _mm512_set_epi32( 15, 13, 11, 9, 14, 12, 10, 8, 7, 5, 3, 1, 6, 4, 2, 0 );

  __m512i idx1, idx2;

  // Start                                                     dx =   0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120
  //                                                           dy =   1   9  17  25  33  41  49  57  65  73  81  89  97 105 113 121
  //                                                           dz =   2  10  18  26  34  42  50  58  66  74  82  90  98 106 114 122
  //                                                           ii =   3  11  19  27  35  43  51  59  67  75  83  91  99 107 115 123
  //                                                           ux =   4  12  20  28  36  44  52  60  68  76  84  92 100 108 116 124
  //                                                           uy =   5  13  21  29  37  45  53  61  69  77  85  93 101 109 117 125
  //                                                           uz =   6  14  22  30  38  46  54  62  70  78  86  94 102 110 118 126
  //                                                           wt =   7  15  23  31  39  47  55  63  71  79  87  95 103 111 119 127

  t00 = _mm512_permutexvar_ps( idx, (__m512) dx );                //   0  16  32  48   8  24  40  56  64  80  96 112  72  88 104 120
  t01 = _mm512_permutexvar_ps( idx, (__m512) dy );                //   1  17  33  49   9  25  41  57  65  81  97 113  73  89 105 121
  t02 = _mm512_permutexvar_ps( idx, (__m512) dz );                //   2  18  34  50  10  26  42  58  66  82  98 114  74  90 106 122
  t03 = _mm512_permutexvar_ps( idx, static_cast<__m512>(ii));                //   3  19  35  51  11  27  43  59  67  83  99 115  75  91 107 123
  t04 = _mm512_permutexvar_ps( idx, (__m512) ux );                //   4  20  36  52  12  28  44  60  68  84 100 116  76  92 108 124
  t05 = _mm512_permutexvar_ps( idx, (__m512) uy );                //   5  21  37  53  13  29  45  61  69  85 101 117  77  93 109 125
  t06 = _mm512_permutexvar_ps( idx, (__m512) uz );                //   6  22  38  54  14  30  46  62  70  86 102 118  78  94 110 126
  t07 = _mm512_permutexvar_ps( idx, (__m512) wt );                //   7  23  39  55  15  31  47  63  71  87 103 119  79  95 111 127

  idx1 = _mm512_set_epi32(  7+16,  6+16,  5+16,  4+16,  7,  6,  5,  4,  3+16,  2+16, 1+16, 0+16,  3,  2, 1, 0 );
  idx2 = _mm512_set_epi32( 15+16, 14+16, 13+16, 12+16, 15, 14, 13, 12, 11+16, 10+16, 9+16, 8+16, 11, 10, 9, 8 );

  u00 = _mm512_permutex2var_ps( t00, idx1, t04 );                 //   0  16  32  48   4  20  36  52   8  24  40  56  12  28  44  60
  u01 = _mm512_permutex2var_ps( t01, idx1, t05 );                 //   1  17  33  49   5  21  37  53   9  25  41  57  13  29  45  61
  u02 = _mm512_permutex2var_ps( t02, idx1, t06 );                 //   2  18  34  50   6  22  38  54  10  26  42  58  14  30  46  62
  u03 = _mm512_permutex2var_ps( t03, idx1, t07 );                 //   3  19  35  51   7  23  39  55  11  27  43  59  15  31  47  63
  u04 = _mm512_permutex2var_ps( t00, idx2, t04 );                 //  64  80  96 112  68  84 100 116  72  88 104 120  76  92 108 124
  u05 = _mm512_permutex2var_ps( t01, idx2, t05 );                 //  65  81  97 113  69  85 101 117  73  89 105 121  77  93 109 125
  u06 = _mm512_permutex2var_ps( t02, idx2, t06 );                 //  66  82  98 114  70  86 102 118  74  90 106 122  78  94 110 126
  u07 = _mm512_permutex2var_ps( t03, idx2, t07 );                 //  67  83  99 115  71  87 103 119  75  91 107 123  79  95 111 127

  t00 = _mm512_shuffle_ps( u00, u01, _MM_SHUFFLE( 1, 0, 1, 0 ) ); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  t01 = _mm512_shuffle_ps( u02, u03, _MM_SHUFFLE( 1, 0, 1, 0 ) ); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  t02 = _mm512_shuffle_ps( u00, u01, _MM_SHUFFLE( 3, 2, 3, 2 ) ); //  32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
  t03 = _mm512_shuffle_ps( u02, u03, _MM_SHUFFLE( 3, 2, 3, 2 ) ); //  34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
  t04 = _mm512_shuffle_ps( u04, u05, _MM_SHUFFLE( 1, 0, 1, 0 ) ); //  64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
  t05 = _mm512_shuffle_ps( u06, u07, _MM_SHUFFLE( 1, 0, 1, 0 ) ); //  66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
  t06 = _mm512_shuffle_ps( u04, u05, _MM_SHUFFLE( 3, 2, 3, 2 ) ); //  96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
  t07 = _mm512_shuffle_ps( u06, u07, _MM_SHUFFLE( 3, 2, 3, 2 ) ); //  98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127

  dx = SIMDFloat_t(_mm512_shuffle_ps( t00, t01, _MM_SHUFFLE( 2, 0, 2, 0 ) ) ); //   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  dy = SIMDFloat_t(_mm512_shuffle_ps( t00, t01, _MM_SHUFFLE( 3, 1, 3, 1 ) ) ); //  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  dz = SIMDFloat_t(_mm512_shuffle_ps( t02, t03, _MM_SHUFFLE( 2, 0, 2, 0 ) ) ); //  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  ii = SIMDInt32_t(_mm512_shuffle_ps( t02, t03, _MM_SHUFFLE( 3, 1, 3, 1 ) ) ); //  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  ux = SIMDFloat_t(_mm512_shuffle_ps( t04, t05, _MM_SHUFFLE( 2, 0, 2, 0 ) ) ); //  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  uy = SIMDFloat_t(_mm512_shuffle_ps( t04, t05, _MM_SHUFFLE( 3, 1, 3, 1 ) ) ); //  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  uz = SIMDFloat_t(_mm512_shuffle_ps( t06, t07, _MM_SHUFFLE( 2, 0, 2, 0 ) ) ); //  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  wt = SIMDFloat_t(_mm512_shuffle_ps( t06, t07, _MM_SHUFFLE( 3, 1, 3, 1 ) ) ); // 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 16, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t& a00, SIMDFloat_t& a01, SIMDFloat_t& a02, SIMDFloat_t& a03,
           		 SIMDFloat_t& a04, SIMDFloat_t& a05, SIMDFloat_t& a06, SIMDFloat_t& a07,
           		 SIMDFloat_t& a08, SIMDFloat_t& a09, SIMDFloat_t& a10, SIMDFloat_t& a11,
           		 SIMDFloat_t& a12, SIMDFloat_t& a13, SIMDFloat_t& a14, SIMDFloat_t& a15 )
{
  __m512 t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15;

  // Start                                 a00 =   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  //                                       a01 =  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  //                                       a02 =  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  //                                       a03 =  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  //                                       a04 =  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  //                                       a05 =  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  //                                       a06 =  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  //                                       a07 = 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
  //                                       a08 = 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
  //                                       a09 = 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
  //                                       a10 = 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
  //                                       a11 = 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
  //                                       a12 = 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
  //                                       a13 = 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
  //                                       a14 = 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
  //                                       a15 = 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255

  t00 = _mm512_unpacklo_ps( static_cast<__m512>(a00), static_cast<__m512>(a01) ); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  t01 = _mm512_unpackhi_ps( static_cast<__m512>(a00), static_cast<__m512>(a01) ); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  t02 = _mm512_unpacklo_ps( static_cast<__m512>(a02), static_cast<__m512>(a03) ); //  32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
  t03 = _mm512_unpackhi_ps( static_cast<__m512>(a02), static_cast<__m512>(a03) ); //  34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
  t04 = _mm512_unpacklo_ps( static_cast<__m512>(a04), static_cast<__m512>(a05) ); //  64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
  t05 = _mm512_unpackhi_ps( static_cast<__m512>(a04), static_cast<__m512>(a05) ); //  66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
  t06 = _mm512_unpacklo_ps( static_cast<__m512>(a06), static_cast<__m512>(a07) ); //  96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
  t07 = _mm512_unpackhi_ps( static_cast<__m512>(a06), static_cast<__m512>(a07) ); //  98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127
  t08 = _mm512_unpacklo_ps( static_cast<__m512>(a08), static_cast<__m512>(a09) ); // 128 144 129 145 132 148 133 149 136 152 137 153 140 156 141 157
  t09 = _mm512_unpackhi_ps( static_cast<__m512>(a08), static_cast<__m512>(a09) ); // 130 146 131 147 134 150 135 151 138 154 139 155 142 158 143 159
  t10 = _mm512_unpacklo_ps( static_cast<__m512>(a10), static_cast<__m512>(a11) ); // 160 176 161 177 164 180 165 181 168 184 169 185 172 188 173 189
  t11 = _mm512_unpackhi_ps( static_cast<__m512>(a10), static_cast<__m512>(a11) ); // 162 178 163 179 166 182 167 183 170 186 171 187 174 190 175 191
  t12 = _mm512_unpacklo_ps( static_cast<__m512>(a12), static_cast<__m512>(a13) ); // 192 208 193 209 196 212 197 213 200 216 201 217 204 220 205 221
  t13 = _mm512_unpackhi_ps( static_cast<__m512>(a12), static_cast<__m512>(a13) ); // 194 210 195 211 198 214 199 215 202 218 203 219 206 222 207 223
  t14 = _mm512_unpacklo_ps( static_cast<__m512>(a14), static_cast<__m512>(a15) ); // 224 240 225 241 228 244 229 245 232 248 233 249 236 252 237 253
  t15 = _mm512_unpackhi_ps( static_cast<__m512>(a14), static_cast<__m512>(a15) ); // 226 242 227 243 230 246 231 247 234 250 235 251 238 254 239 255

  a00 = SIMDFloat_t( _mm512_shuffle_ps( t00, t02, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //   0  16  32  48
  a01 = SIMDFloat_t( _mm512_shuffle_ps( t00, t02, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //   1  17  33  49
  a02 = SIMDFloat_t( _mm512_shuffle_ps( t01, t03, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //   2  18  34  50 
  a03 = SIMDFloat_t( _mm512_shuffle_ps( t01, t03, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //   3  19  35  51 
  a04 = SIMDFloat_t( _mm512_shuffle_ps( t04, t06, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //  64  80  96 112 
  a05 = SIMDFloat_t( _mm512_shuffle_ps( t04, t06, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //  65  81  97 113
  a06 = SIMDFloat_t( _mm512_shuffle_ps( t05, t07, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //  66  82  98 114 
  a07 = SIMDFloat_t( _mm512_shuffle_ps( t05, t07, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //  67  83  99 115 
  a08 = SIMDFloat_t( _mm512_shuffle_ps( t08, t10, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 128 144 160 176 
  a09 = SIMDFloat_t( _mm512_shuffle_ps( t08, t10, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 129 145 161 177 
  a10 = SIMDFloat_t( _mm512_shuffle_ps( t09, t11, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 130 146 162 178 
  a11 = SIMDFloat_t( _mm512_shuffle_ps( t09, t11, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 131 147 163 179 
  a12 = SIMDFloat_t( _mm512_shuffle_ps( t12, t14, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 192 208 228 240 
  a13 = SIMDFloat_t( _mm512_shuffle_ps( t12, t14, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 193 209 229 241 
  a14 = SIMDFloat_t( _mm512_shuffle_ps( t13, t15, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 194 210 230 242 
  a15 = SIMDFloat_t( _mm512_shuffle_ps( t13, t15, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 195 211 231 243 

  t00 = _mm512_shuffle_f32x4( static_cast<__m512>(a00), static_cast<__m512>(a04), 0x88 ); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
  t01 = _mm512_shuffle_f32x4( static_cast<__m512>(a01), static_cast<__m512>(a05), 0x88 ); //   1  17  33  49 ...
  t02 = _mm512_shuffle_f32x4( static_cast<__m512>(a02), static_cast<__m512>(a06), 0x88 ); //   2  18  34  50 ...
  t03 = _mm512_shuffle_f32x4( static_cast<__m512>(a03), static_cast<__m512>(a07), 0x88 ); //   3  19  35  51 ...
  t04 = _mm512_shuffle_f32x4( static_cast<__m512>(a00), static_cast<__m512>(a04), 0xdd ); //   4  20  36  52 ...
  t05 = _mm512_shuffle_f32x4( static_cast<__m512>(a01), static_cast<__m512>(a05), 0xdd ); //   5  21  37  53 ...
  t06 = _mm512_shuffle_f32x4( static_cast<__m512>(a02), static_cast<__m512>(a06), 0xdd ); //   6  22  38  54 ...
  t07 = _mm512_shuffle_f32x4( static_cast<__m512>(a03), static_cast<__m512>(a07), 0xdd ); //   7  23  39  55 ...
  t08 = _mm512_shuffle_f32x4( static_cast<__m512>(a08), static_cast<__m512>(a12), 0x88 ); // 128 144 160 176 ...
  t09 = _mm512_shuffle_f32x4( static_cast<__m512>(a09), static_cast<__m512>(a13), 0x88 ); // 129 145 161 177 ...
  t10 = _mm512_shuffle_f32x4( static_cast<__m512>(a10), static_cast<__m512>(a14), 0x88 ); // 130 146 162 178 ...
  t11 = _mm512_shuffle_f32x4( static_cast<__m512>(a11), static_cast<__m512>(a15), 0x88 ); // 131 147 163 179 ...
  t12 = _mm512_shuffle_f32x4( static_cast<__m512>(a08), static_cast<__m512>(a12), 0xdd ); // 132 148 164 180 ...
  t13 = _mm512_shuffle_f32x4( static_cast<__m512>(a09), static_cast<__m512>(a13), 0xdd ); // 133 149 165 181 ...
  t14 = _mm512_shuffle_f32x4( static_cast<__m512>(a10), static_cast<__m512>(a14), 0xdd ); // 134 150 166 182 ...
  t15 = _mm512_shuffle_f32x4( static_cast<__m512>(a11), static_cast<__m512>(a15), 0xdd ); // 135 151 167 183 ...

  a00 = SIMDFloat_t( _mm512_shuffle_f32x4( t00, t08, 0x88 ) ); //   0  16  32  48  64  80  96 112 ... 240
  a01 = SIMDFloat_t( _mm512_shuffle_f32x4( t01, t09, 0x88 ) ); //   1  17  33  49  66  81  97 113 ... 241
  a02 = SIMDFloat_t( _mm512_shuffle_f32x4( t02, t10, 0x88 ) ); //   2  18  34  50  67  82  98 114 ... 242
  a03 = SIMDFloat_t( _mm512_shuffle_f32x4( t03, t11, 0x88 ) ); //   3  19  35  51  68  83  99 115 ... 243
  a04 = SIMDFloat_t( _mm512_shuffle_f32x4( t04, t12, 0x88 ) ); //   4 ...
  a05 = SIMDFloat_t( _mm512_shuffle_f32x4( t05, t13, 0x88 ) ); //   5 ...
  a06 = SIMDFloat_t( _mm512_shuffle_f32x4( t06, t14, 0x88 ) ); //   6 ...
  a07 = SIMDFloat_t( _mm512_shuffle_f32x4( t07, t15, 0x88 ) ); //   7 ...
  a08 = SIMDFloat_t( _mm512_shuffle_f32x4( t00, t08, 0xdd ) ); //   8 ...
  a09 = SIMDFloat_t( _mm512_shuffle_f32x4( t01, t09, 0xdd ) ); //   9 ...
  a10 = SIMDFloat_t( _mm512_shuffle_f32x4( t02, t10, 0xdd ) ); //  10 ...
  a11 = SIMDFloat_t( _mm512_shuffle_f32x4( t03, t11, 0xdd ) ); //  11 ...
  a12 = SIMDFloat_t( _mm512_shuffle_f32x4( t04, t12, 0xdd ) ); //  12 ...
  a13 = SIMDFloat_t( _mm512_shuffle_f32x4( t05, t13, 0xdd ) ); //  13 ...
  a14 = SIMDFloat_t( _mm512_shuffle_f32x4( t06, t14, 0xdd ) ); //  14 ...
  a15 = SIMDFloat_t( _mm512_shuffle_f32x4( t07, t15, 0xdd ) ); //  15  31  47  63  79  96 111 127 ... 255
}
#else
template< typename Float16 >
KOKKOS_INLINE_FUNCTION
void transpose(Float16& a00, Float16& a01, Float16& a02, Float16& a03,
               Float16& a04, Float16& a05, Float16& a06, Float16& a07,
               Float16& a08, Float16& a09, Float16& a10, Float16& a11,
               Float16& a12, Float16& a13, Float16& a14, Float16& a15) 
{
  // Start                                 a00 =   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  //                                       a01 =  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  //                                       a02 =  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  //                                       a03 =  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  //                                       a04 =  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  //                                       a05 =  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  //                                       a06 =  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  //                                       a07 = 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
  //                                       a08 = 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
  //                                       a09 = 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
  //                                       a10 = 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
  //                                       a11 = 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
  //                                       a12 = 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
  //                                       a13 = 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
  //                                       a14 = 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
  //                                       a15 = 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255

  swap(a00[1],a01[0]); swap(a00[2],a02[0]); swap(a00[3],a03[0]); swap(a00[4],a04[0]); swap(a00[5],a05[0]); swap(a00[6],a06[0]); swap(a00[7],a07[0]); swap(a00[8],a08[0]); swap(a00[9],a09[0]); swap(a00[10],a10[0]); swap(a00[11],a11[ 0]); swap(a00[12],a12[ 0]); swap(a00[13],a13[ 0]); swap(a00[14],a14[ 0]); swap(a00[15],a15[ 0]);
                       swap(a01[2],a02[1]); swap(a01[3],a03[1]); swap(a01[4],a04[1]); swap(a01[5],a05[1]); swap(a01[6],a06[1]); swap(a01[7],a07[1]); swap(a01[8],a08[1]); swap(a01[9],a09[1]); swap(a01[10],a10[1]); swap(a01[11],a11[ 1]); swap(a01[12],a12[ 1]); swap(a01[13],a13[ 1]); swap(a01[14],a14[ 1]); swap(a01[15],a15[ 1]);
                                            swap(a02[3],a03[2]); swap(a02[4],a04[2]); swap(a02[5],a05[2]); swap(a02[6],a06[2]); swap(a02[7],a07[2]); swap(a02[8],a08[2]); swap(a02[9],a09[2]); swap(a02[10],a10[2]); swap(a02[11],a11[ 2]); swap(a02[12],a12[ 2]); swap(a02[13],a13[ 2]); swap(a02[14],a14[ 2]); swap(a02[15],a15[ 2]);
                                                                 swap(a03[4],a04[3]); swap(a03[5],a05[3]); swap(a03[6],a06[3]); swap(a03[7],a07[3]); swap(a03[8],a08[3]); swap(a03[9],a09[3]); swap(a03[10],a10[3]); swap(a03[11],a11[ 3]); swap(a03[12],a12[ 3]); swap(a03[13],a13[ 3]); swap(a03[14],a14[ 3]); swap(a03[15],a15[ 3]);
                                                                                      swap(a04[5],a05[4]); swap(a04[6],a06[4]); swap(a04[7],a07[4]); swap(a04[8],a08[4]); swap(a04[9],a09[4]); swap(a04[10],a10[4]); swap(a04[11],a11[ 4]); swap(a04[12],a12[ 4]); swap(a04[13],a13[ 4]); swap(a04[14],a14[ 4]); swap(a04[15],a15[ 4]);
                                                                                                           swap(a05[6],a06[5]); swap(a05[7],a07[5]); swap(a05[8],a08[5]); swap(a05[9],a09[5]); swap(a05[10],a10[5]); swap(a05[11],a11[ 5]); swap(a05[12],a12[ 5]); swap(a05[13],a13[ 5]); swap(a05[14],a14[ 5]); swap(a05[15],a15[ 5]);
                                                                                                                                swap(a06[7],a07[6]); swap(a06[8],a08[6]); swap(a06[9],a09[6]); swap(a06[10],a10[6]); swap(a06[11],a11[ 6]); swap(a06[12],a12[ 6]); swap(a06[13],a13[ 6]); swap(a06[14],a14[ 6]); swap(a06[15],a15[ 6]);
                                                                                                                                                     swap(a07[8],a08[7]); swap(a07[9],a09[7]); swap(a07[10],a10[7]); swap(a07[11],a11[ 7]); swap(a07[12],a12[ 7]); swap(a07[13],a13[ 7]); swap(a07[14],a14[ 7]); swap(a07[15],a15[ 7]);
                                                                                                                                                                          swap(a08[9],a09[8]); swap(a08[10],a10[8]); swap(a08[11],a11[ 8]); swap(a08[12],a12[ 8]); swap(a08[13],a13[ 8]); swap(a08[14],a14[ 8]); swap(a08[15],a15[ 8]);
                                                                                                                                                                                               swap(a09[10],a10[9]); swap(a09[11],a11[ 9]); swap(a09[12],a12[ 9]); swap(a09[13],a13[ 9]); swap(a09[14],a14[ 9]); swap(a09[15],a15[ 9]);
                                                                                                                                                                                                                     swap(a10[11],a11[10]); swap(a10[12],a12[10]); swap(a10[13],a13[10]); swap(a10[14],a14[10]); swap(a10[15],a15[10]);
                                                                                                                                                                                                                                            swap(a11[12],a12[11]); swap(a11[13],a13[11]); swap(a11[14],a14[11]); swap(a11[15],a15[11]);
                                                                                                                                                                                                                                                                   swap(a12[13],a13[12]); swap(a12[14],a14[12]); swap(a12[15],a15[12]);
                                                                                                                                                                                                                                                                                          swap(a13[14],a14[13]); swap(a13[15],a15[13]);
                                                                                                                                                                                                                                                                                                                 swap(a14[15],a15[14]);
  // a00 =  0  16  32  48  64  80  96 112 ... 240
  // a01 =  1  17  33  49  66  81  97 113 ... 241
  // a02 =  2  18  34  50  67  82  98 114 ... 242
  // a03 =  3  19  35  51  68  83  99 115 ... 243
  // a04 =  4 ...
  // a05 =  5 ...
  // a06 =  6 ...
  // a07 =  7 ...
  // a08 =  8 ...
  // a09 =  9 ...
  // a10 = 10 ...
  // a11 = 11 ...
  // a12 = 12 ...
  // a13 = 13 ...
  // a14 = 14 ...
  // a15 = 15  31  47  63  79  96 111 127 ... 255
}
#endif
//
#ifdef __AVX2__

KOKKOS_INLINE_FUNCTION
KokkosSIMD::simd<float,8> simd_cast(KokkosSIMD::simd<int32_t,8>& a) {
  return KokkosSIMD::simd<float,8>( _mm256_castsi256_ps( (__m256i)(a) ) );
}

KOKKOS_INLINE_FUNCTION
KokkosSIMD::simd<int32_t,8> simd_cast(KokkosSIMD::simd<float,8>& a) {
  return KokkosSIMD::simd<int32_t,8>( _mm256_castps_si256( (__m256)(a) ) );
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 8, bool>::type=true >
KOKKOS_INLINE_FUNCTION
SIMDFloat_t rsqrt(SIMDFloat_t a)
{
  __m256 a_v = (__m256)(a), b_v, b;

  b_v = _mm256_rsqrt_ps(a_v);

  // Note: It is quicker to just call div_ps and sqrt_ps if more
  // refinement desired!
  b = _mm256_add_ps( b_v, _mm256_mul_ps( _mm256_set1_ps( 0.5f ),
			     _mm256_sub_ps( b_v,
					    _mm256_mul_ps( a_v,
							   _mm256_mul_ps( b_v,
									  _mm256_mul_ps( b_v, b_v ) ) ) ) ) );

  return SIMDFloat_t(b);
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 8, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t& a0, SIMDFloat_t& a1, SIMDFloat_t& a2, SIMDFloat_t& a3,
               SIMDFloat_t& a4, SIMDFloat_t& a5, SIMDFloat_t& a6, SIMDFloat_t& a7) {
  __m256 t0, t1, t2, t3, t4, t5, t6, t7;

  __m256 u0, u1, u2, u3, u4, u5, u6, u7;

  t0 = _mm256_unpacklo_ps( (__m256)(a0), (__m256)(a1) );
  t1 = _mm256_unpackhi_ps( (__m256)(a0), (__m256)(a1) );
  t2 = _mm256_unpacklo_ps( (__m256)(a2), (__m256)(a3) );
  t3 = _mm256_unpackhi_ps( (__m256)(a2), (__m256)(a3) );
  t4 = _mm256_unpacklo_ps( (__m256)(a4), (__m256)(a5) );
  t5 = _mm256_unpackhi_ps( (__m256)(a4), (__m256)(a5) );
  t6 = _mm256_unpacklo_ps( (__m256)(a6), (__m256)(a7) );
  t7 = _mm256_unpackhi_ps( (__m256)(a6), (__m256)(a7) );

  u0 = _mm256_shuffle_ps( t0, t2, _MM_SHUFFLE( 1, 0, 1, 0 ) );
  u1 = _mm256_shuffle_ps( t0, t2, _MM_SHUFFLE( 3, 2, 3, 2 ) );
  u2 = _mm256_shuffle_ps( t1, t3, _MM_SHUFFLE( 1, 0, 1, 0 ) );
  u3 = _mm256_shuffle_ps( t1, t3, _MM_SHUFFLE( 3, 2, 3, 2 ) );
  u4 = _mm256_shuffle_ps( t4, t6, _MM_SHUFFLE( 1, 0, 1, 0 ) );
  u5 = _mm256_shuffle_ps( t4, t6, _MM_SHUFFLE( 3, 2, 3, 2 ) );
  u6 = _mm256_shuffle_ps( t5, t7, _MM_SHUFFLE( 1, 0, 1, 0 ) );
  u7 = _mm256_shuffle_ps( t5, t7, _MM_SHUFFLE( 3, 2, 3, 2 ) );

  a0 = SIMDFloat_t( _mm256_permute2f128_ps( u0, u4, 0x20 ) );
  a1 = SIMDFloat_t( _mm256_permute2f128_ps( u1, u5, 0x20 ) );
  a2 = SIMDFloat_t( _mm256_permute2f128_ps( u2, u6, 0x20 ) );
  a3 = SIMDFloat_t( _mm256_permute2f128_ps( u3, u7, 0x20 ) );
  a4 = SIMDFloat_t( _mm256_permute2f128_ps( u0, u4, 0x31 ) );
  a5 = SIMDFloat_t( _mm256_permute2f128_ps( u1, u5, 0x31 ) );
  a6 = SIMDFloat_t( _mm256_permute2f128_ps( u2, u6, 0x31 ) );
  a7 = SIMDFloat_t( _mm256_permute2f128_ps( u3, u7, 0x31 ) );

  return;
}
#else
template< typename Float8 >
KOKKOS_INLINE_FUNCTION
void transpose(Float8& a0, Float8& a1, Float8& a2, Float8& a3,
               Float8& a4, Float8& a5, Float8& a6, Float8& a7 ) {
  swap( a0[1],a1[0] ); swap( a0[2],a2[0] ); swap( a0[3],a3[0] ); swap( a0[4],a4[0] ); swap( a0[5],a5[0] ); swap( a0[6],a6[0] ); swap( a0[7],a7[0] );
                       swap( a1[2],a2[1] ); swap( a1[3],a3[1] ); swap( a1[4],a4[1] ); swap( a1[5],a5[1] ); swap( a1[6],a6[1] ); swap( a1[7],a7[1] );
                                            swap( a2[3],a3[2] ); swap( a2[4],a4[2] ); swap( a2[5],a5[2] ); swap( a2[6],a6[2] ); swap( a2[7],a7[2] );
                                                                 swap( a3[4],a4[3] ); swap( a3[5],a5[3] ); swap( a3[6],a6[3] ); swap( a3[7],a7[3] );
                                                                                      swap( a4[5],a5[4] ); swap( a4[6],a6[4] ); swap( a4[7],a7[4] );
                                                                                                           swap( a5[6],a6[5] ); swap( a5[7],a7[5] );
                                                                                                                                swap( a6[7],a7[6] );
  return;
}
#endif

#if defined(KOKKOS_ARCH_AVX) || defined(KOKKOS_ARCH_AVX2)

KOKKOS_INLINE_FUNCTION
KokkosSIMD::simd<float,4> simd_cast(KokkosSIMD::simd<int32_t,4>& a) {
  return KokkosSIMD::simd<float,4>( _mm_castsi128_ps( (__m128i)(a) ) );
}

KOKKOS_INLINE_FUNCTION
KokkosSIMD::simd<int32_t,4> simd_cast(KokkosSIMD::simd<float,4>& a) {
  return KokkosSIMD::simd<int32_t,4>( _mm_castps_si128( (__m128)(a) ) );
}

template<typename SIMDFloat_t>
SIMDFloat_t rsqrt(SIMDFloat_t a, typename std::enable_if_t<SIMDFloat_t::size() == 4, bool> = true )
{
  __m128 b_v;

  b_v = _mm_rsqrt_ps( (__m128)(a.v) );

  SIMDFloat_t b(_mm_fmadd_ps( _mm_set1_ps( 0.5f ),
                      _mm_fnmadd_ps( (__m128)(a),
                                     _mm_mul_ps( b_v,
                                                 _mm_mul_ps( b_v, b_v ) ),
                                     b_v ),
                      b_v ));

  return b;
}

template< int i0, int i1, int i2, int i3>
KOKKOS_FORCEINLINE_FUNCTION
simd_float32x4_t shuffle(simd_float32x4_t& a) {
  return simd_float32x4_t( _mm_shuffle_ps( (__m128)(a), (__m128)(a), ( permute<i0,i1,i2,i3>::value ) ) );
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 4, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t& a, SIMDFloat_t& b, SIMDFloat_t& c, SIMDFloat_t& d) {
  // a =  0  1  2  3
  // b =  4  5  6  7
  // c =  8  9 10 11
  // d = 12 13 14 15

  __m128  r, s, t, u;

  r   = _mm_unpackhi_ps( (__m128)(a), (__m128)(b) ); // r =  2  6  3  7
  s   = _mm_unpacklo_ps( (__m128)(a), (__m128)(b) ); // s =  0  4  1  5
  t   = _mm_unpackhi_ps( (__m128)(c), (__m128)(d) ); // t = 10 14 11 15
  u   = _mm_unpacklo_ps( (__m128)(c), (__m128)(d) ); // u =  8 12  9 13

  a = SIMDFloat_t(_mm_movelh_ps( s, u )); // a = 0 4  8 12
  b = SIMDFloat_t(_mm_movehl_ps( u, s )); // b = 1 5  9 13
  c = SIMDFloat_t(_mm_movelh_ps( r, t )); // c = 2 6 10 14
  d = SIMDFloat_t(_mm_movehl_ps( t, r )); // d = 3 7 11 15

  // a = 0 4  8 12
  // b = 1 5  9 13
  // c = 2 6 10 14
  // d = 3 7 11 15
  return;
}

#elif defined KOKKOS_ARCH_ARM_SVE
#include <arm_sve.h>

KOKKOS_INLINE_FUNCTION
simd_float_t simd_cast(simd_int32_t& a) {
  vls_int32_t t = (vls_int32_t)(a);
  return simd_float_t( vls_float32_t( t ) );
}

KOKKOS_INLINE_FUNCTION
simd_int32_t simd_cast(simd_float_t& a) {
  vls_float32_t t = (vls_float32_t)(a);
  return simd_int32_t( (vls_int32_t)(t) );
}

KOKKOS_INLINE_FUNCTION
simd_float_t rsqrt(simd_float_t a)
{
  svfloat32_t b, b_v;
  svfloat32_t a_v = static_cast<vls_float32_t>(a);

  // Initial estimate
  b_v = svrsqrte_f32(a_v);

  // Two Newton steps
  b_v = svmul_f32_m(svptrue_b32(), b_v, svrsqrts_f32(svmul_f32_m(svptrue_b32(), a_v, b_v), b_v));
  b   = svmul_f32_m(svptrue_b32(), b_v, svrsqrts_f32(svmul_f32_m(svptrue_b32(), a_v, b_v), b_v));

  // Note: It is quicker to just call div_ps and sqrt_ps if more refinement
  // is desired.
  // b.v = _mm512_div_ps( _mm512_set1_ps( 1.0f ), _mm512_sqrt_ps( a.v ) );

  return simd_float_t(b);
}

template<typename SIMDFloat_t, typename SIMDInt32_t, typename std::enable_if<SIMDFloat_t::size() == 16, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose_particles_16x8(SIMDFloat_t& dx, SIMDFloat_t& dy, SIMDFloat_t& dz, SIMDInt32_t& ii,
           		                SIMDFloat_t& ux, SIMDFloat_t& uy, SIMDFloat_t& uz, SIMDFloat_t& wt )
{
  //   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  //  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  //  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  //  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  //  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  //  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  //  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  // 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127

  //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  //  32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
  //  34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
  //  64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
  //  66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
  //  96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
  //  98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127

  //   0  16  32  48   4  20  36  52   8  24  40  56  12  28  44  60
  //   1  17  33  49   5  21  37  53   9  25  41  57  13  29  45  61
  //   2  18  34  50   6  22  38  54  10  26  42  58  14  30  46  62
  //   3  19  35  51   7  23  39  55  11  27  43  59  15  31  47  63
  //  64  80  96 112  68  84 100 116  72  88 104 120  76  92 108 124
  //  65  81  97 113  69  85 101 117  73  89 105 121  77  93 109 125
  //  66  82  98 114  70  86 102 118  74  90 106 122  78  94 110 126
  //  67  83  99 115  71  87 103 119  75  91 107 123  79  95 111 127

  //   0  16  32  48   8  24  40  56  64  80  96 112  72  88 104 120
  //   1  17  33  49   9  25  41  57  65  81  97 113  73  89 105 121
  //   2  18  34  50  10  26  42  58  66  82  98 114  74  90 106 122
  //   3  19  35  51  11  27  43  59  67  83  99 115  75  91 107 123
  //   4  20  36  52  12  28  44  60  68  84 100 116  76  92 108 124
  //   5  21  37  53  13  29  45  61  69  85 101 117  77  93 109 125
  //   6  22  38  54  14  30  46  62  70  86 102 118  78  94 110 126
  //   7  23  39  55  15  31  47  63  71  87 103 119  79  95 111 127

  //   0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120
  //   1   9  17  25  33  41  49  57  65  73  81  89  97 105 113 121
  //   2  10  18  26  34  42  50  58  66  74  82  90  98 106 114 122
  //   3  11  19  27  35  43  51  59  67  75  83  91  99 107 115 123
  //   4  12  20  28  36  44  52  60  68  76  84  92 100 108 116 124
  //   5  13  21  29  37  45  53  61  69  77  85  93 101 109 117 125
  //   6  14  22  30  38  46  54  62  70  78  86  94 102 110 118 126
  //   7  15  23  31  39  47  55  63  71  79  87  95 103 111 119 127
}

template<typename SIMDFloat_t, typename SIMDInt32_t, typename std::enable_if<SIMDFloat_t::size() == 16, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose_particles_16x8_reverse(SIMDFloat_t& dx, SIMDFloat_t& dy, SIMDFloat_t& dz, SIMDInt32_t& ii,
           		                        SIMDFloat_t& ux, SIMDFloat_t& uy, SIMDFloat_t& uz, SIMDFloat_t& wt )
{
  // dx =   0   8  16  24  32  40  48  56  64  72  80  88  96 104 112 120
  // dy =   1   9  17  25  33  41  49  57  65  73  81  89  97 105 113 121
  // dz =   2  10  18  26  34  42  50  58  66  74  82  90  98 106 114 122
  // ii =   3  11  19  27  35  43  51  59  67  75  83  91  99 107 115 123
  // ux =   4  12  20  28  36  44  52  60  68  76  84  92 100 108 116 124
  // uy =   5  13  21  29  37  45  53  61  69  77  85  93 101 109 117 125
  // uz =   6  14  22  30  38  46  54  62  70  78  86  94 102 110 118 126
  // wt =   7  15  23  31  39  47  55  63  71  79  87  95 103 111 119 127

  //   0  16  32  48   8  24  40  56  64  80  96 112  72  88 104 120
  //   1  17  33  49   9  25  41  57  65  81  97 113  73  89 105 121
  //   2  18  34  50  10  26  42  58  66  82  98 114  74  90 106 122
  //   3  19  35  51  11  27  43  59  67  83  99 115  75  91 107 123
  //   4  20  36  52  12  28  44  60  68  84 100 116  76  92 108 124
  //   5  21  37  53  13  29  45  61  69  85 101 117  77  93 109 125
  //   6  22  38  54  14  30  46  62  70  86 102 118  78  94 110 126
  //   7  23  39  55  15  31  47  63  71  87 103 119  79  95 111 127

  //   0  16  32  48   4  20  36  52   8  24  40  56  12  28  44  60
  //   1  17  33  49   5  21  37  53   9  25  41  57  13  29  45  61
  //   2  18  34  50   6  22  38  54  10  26  42  58  14  30  46  62
  //   3  19  35  51   7  23  39  55  11  27  43  59  15  31  47  63
  //  64  80  96 112  68  84 100 116  72  88 104 120  76  92 108 124
  //  65  81  97 113  69  85 101 117  73  89 105 121  77  93 109 125
  //  66  82  98 114  70  86 102 118  74  90 106 122  78  94 110 126
  //  67  83  99 115  71  87 103 119  75  91 107 123  79  95 111 127

  //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  //  32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
  //  34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
  //  64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
  //  66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
  //  96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
  //  98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127

  //   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  //  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  //  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  //  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  //  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  //  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  //  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  // 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 16, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t& a00, SIMDFloat_t& a01, SIMDFloat_t& a02, SIMDFloat_t& a03,
           		 SIMDFloat_t& a04, SIMDFloat_t& a05, SIMDFloat_t& a06, SIMDFloat_t& a07,
           		 SIMDFloat_t& a08, SIMDFloat_t& a09, SIMDFloat_t& a10, SIMDFloat_t& a11,
           		 SIMDFloat_t& a12, SIMDFloat_t& a13, SIMDFloat_t& a14, SIMDFloat_t& a15)
{
  svfloat32_t t00, t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15;

  // Start                                 a00 =   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
  //                                       a01 =  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
  //                                       a02 =  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
  //                                       a03 =  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
  //                                       a04 =  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
  //                                       a05 =  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
  //                                       a06 =  96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111
  //                                       a07 = 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
  //                                       a08 = 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
  //                                       a09 = 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
  //                                       a10 = 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
  //                                       a11 = 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
  //                                       a12 = 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
  //                                       a13 = 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
  //                                       a14 = 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
  //                                       a15 = 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255

  t00 = svtrn1q_f32( (svfloat32_t)(a00), (svfloat32_t)(a08) );    // a00 =   0   1   2   3   4   5   6   7 128 129 130 131 132 133 134 135 
  t01 = svtrn1q_f32( (svfloat32_t)(a01), (svfloat32_t)(a09) );    // a01 =  16  17  18  19  20  21  22  23 144 145 146 147 148 149 150 151   
  t02 = svtrn1q_f32( (svfloat32_t)(a02), (svfloat32_t)(a10) );    // a02 =  32  33  34  35  36  37  38  39 160 161 162 163 164 165 166 167   
  t03 = svtrn1q_f32( (svfloat32_t)(a03), (svfloat32_t)(a11) );    // a03 =  48  49  50  51  52  53  54  55 176 177 178 179 180 181 182 183   
  t04 = svtrn2q_f32( (svfloat32_t)(a04), (svfloat32_t)(a12) );    // a04 =  64  65  66  67  68  69  70  71 192 193 194 195 196 197 198 199   
  t05 = svtrn2q_f32( (svfloat32_t)(a05), (svfloat32_t)(a13) );    // a05 =  80  81  82  83  84  85  86  87 208 209 210 211 212 213 214 215   
  t06 = svtrn2q_f32( (svfloat32_t)(a06), (svfloat32_t)(a14) );    // a06 =  96  97  98  99 100 101 102 103 224 225 226 227 228 229 230 231   
  t07 = svtrn2q_f32( (svfloat32_t)(a07), (svfloat32_t)(a15) );    // a07 = 112 113 114 115 116 117 118 119 240 241 242 243 244 245 246 247   
  t08 = svtrn1q_f32( (svfloat32_t)(a00), (svfloat32_t)(a08) );    // a08 =   8   9  10  11  12  13  14  15 136 137 138 139 140 141 142 143 
  t09 = svtrn1q_f32( (svfloat32_t)(a01), (svfloat32_t)(a09) );    // a09 =  24  25  26  27  28  29  30  31 152 153 154 155 156 157 158 159 
  t10 = svtrn1q_f32( (svfloat32_t)(a02), (svfloat32_t)(a10) );    // a10 =  40  41  42  43  44  45  46  47 168 169 170 171 172 173 174 175 
  t11 = svtrn1q_f32( (svfloat32_t)(a03), (svfloat32_t)(a11) );    // a11 =  56  57  58  59  60  61  62  63 184 185 186 187 188 189 190 191 
  t12 = svtrn2q_f32( (svfloat32_t)(a04), (svfloat32_t)(a12) );    // a12 =  72  73  74  75  76  77  78  79 200 201 202 203 204 205 206 207 
  t13 = svtrn2q_f32( (svfloat32_t)(a05), (svfloat32_t)(a13) );    // a13 =  88  89  90  91  92  93  94  95 216 217 218 219 220 221 222 223 
  t14 = svtrn2q_f32( (svfloat32_t)(a06), (svfloat32_t)(a14) );    // a14 = 104 105 106 107 108 109 110 111 232 233 234 235 236 237 238 239 
  t15 = svtrn2q_f32( (svfloat32_t)(a07), (svfloat32_t)(a15) );    // a15 = 120 121 122 123 124 125 126 127 248 249 250 251 252 253 254 255 

  a00 = SIMDFloat_t( svtrn1q_f32(t00, t04) );                     // a00 =   0   1   2   3  64  65  66  67 128 129 130 131 192 193 194 195 
  a01 = SIMDFloat_t( svtrn1q_f32(t01, t05) );                     // a01 =  16  17  18  19  80  81  82  83 144 145 146 147 208 209 210 211 
  a02 = SIMDFloat_t( svtrn1q_f32(t02, t06) );                     // a02 =  32  33  34  35  96  97  98  99 160 161 162 163 224 225 226 227 
  a03 = SIMDFloat_t( svtrn1q_f32(t03, t07) );                     // a03 =  48  49  50  51 112 113 114 115 176 177 178 179 240 241 242 243 
  a04 = SIMDFloat_t( svtrn2q_f32(t00, t04) );                     // a04 =   4   5   6   7  68  69  70  71 132 133 134 135 196 197 198 199
  a05 = SIMDFloat_t( svtrn2q_f32(t01, t05) );                     // a05 =  20  21  22  23  84  85  86  87 148 149 150 151 212 213 214 215
  a06 = SIMDFloat_t( svtrn2q_f32(t02, t06) );                     // a06 =  36  37  38  39 100 101 102 103 164 165 166 167 228 229 230 231
  a07 = SIMDFloat_t( svtrn2q_f32(t03, t07) );                     // a07 =  52  53  54  55 116 117 118 119 180 181 182 183 244 245 246 247
  a08 = SIMDFloat_t( svtrn1q_f32(t08, t12) );                     // a08 =   8   9  10  11  72  73  74  75 136 137 138 139 200 201 202 203
  a09 = SIMDFloat_t( svtrn1q_f32(t09, t13) );                     // a09 =  24  25  26  27  88  89  90  91 152 153 154 155 216 217 218 219
  a10 = SIMDFloat_t( svtrn1q_f32(t10, t14) );                     // a10 =  40  41  42  43 104 105 106 107 168 169 170 171 232 233 234 235
  a11 = SIMDFloat_t( svtrn1q_f32(t11, t15) );                     // a11 =  56  57  58  59 120 121 122 123 184 185 186 187 248 249 250 251
  a12 = SIMDFloat_t( svtrn2q_f32(t08, t09) );                     // a12 =  12  13  14  15  76  77  78  79 140 141 142 143 204 205 206 207
  a13 = SIMDFloat_t( svtrn2q_f32(t09, t00) );                     // a13 =  28  29  30  31  92  93  94  95 156 157 158 159 220 221 222 223
  a14 = SIMDFloat_t( svtrn2q_f32(t10, t11) );                     // a14 =  44  45  46  47 108 109 110 111 172 173 174 175 236 237 238 239
  a15 = SIMDFloat_t( svtrn2q_f32(t11, t11) );                     // a15 =  60  61  62  63 124 125 126 127 188 189 190 191 252 253 254 255

  t00 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a00), (vls_float64_t)(a02) ) );     // a00 =   0   1  32  33  64  65  96  97 128 129 160 161 192 193 224 225 
  t01 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a01), (vls_float64_t)(a03) ) );     // a01 =  16  17  48  49  80  81 112 113 144 145 176 177 208 209 240 241 
  t02 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a00), (vls_float64_t)(a02) ) );     // a02 =   2   3  34  35  66  67  98  99 130 131 162 163 194 195 226 227 
  t03 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a01), (vls_float64_t)(a03) ) );     // a03 =  18  19  50  51  82  83 114 115 146 147 178 179 210 211 242 243 
  t04 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a04), (vls_float64_t)(a06) ) );     // a04 =   4   5  36  37  68  69 100 101 132 133 164 165 196 197 228 229 
  t05 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a05), (vls_float64_t)(a07) ) );     // a05 =  20  21  52  53  84  85 116 117 148 149 180 181 212 213 244 245 
  t06 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a04), (vls_float64_t)(a06) ) );     // a06 =   6   7  38  39  70  71 102 103 134 135 166 167 198 199 230 231
  t07 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a05), (vls_float64_t)(a07) ) );     // a07 =  22  23  54  55  86  87 118 119 150 151 182 183 214 215 246 247 .
  t08 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a08), (vls_float64_t)(a10) ) );     // a08 =   8   9  40  41  72  73 104 105 136 137 168 169 200 201 232 233
  t09 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a09), (vls_float64_t)(a11) ) );     // a09 =  24  25  56  57  88  89 120 121 152 153 184 185 216 217 248 249
  t10 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a08), (vls_float64_t)(a10) ) );     // a10 =  10  11  42  43  74  75 106 107 138 139 170 171 202 203 234 235
  t11 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a09), (vls_float64_t)(a11) ) );     // a11 =  26  27  58  59  90  91 122 123 154 155 186 187 218 219 250 251
  t12 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a12), (vls_float64_t)(a14) ) );     // a12 =  12  13  44  45  76  77 108 109 140 141 172 173 204 205 236 237
  t13 = svreinterpret_f32_f64( svtrn1_f64( (vls_float64_t)(a13), (vls_float64_t)(a15) ) );     // a13 =  28  29  60  61  92  93 124 125 156 157 188 189 220 221 252 253
  t14 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a12), (vls_float64_t)(a14) ) );     // a14 =  14  15  46  47  78  79 110 111 142 143 174 175 206 207 238 239
  t15 = svreinterpret_f32_f64( svtrn2_f64( (vls_float64_t)(a13), (vls_float64_t)(a15) ) );     // a15 =  30  31  62  63  94  95 126 127 158 159 190 191 222 223 254 255

  a00 = SIMDFloat_t( svtrn1_f32(t00, t01) );                      // a00 =   0  16  32  48  64  80  96 112 128 144 160 176 192 208 224 240  
  a01 = SIMDFloat_t( svtrn2_f32(t00, t01) );                      // a00 =   1  17  33  49  65  81  97 113 129 145 161 177 193 209 225 241
  a02 = SIMDFloat_t( svtrn1_f32(t02, t03) );                      // a00 =   2  18  34 
  a03 = SIMDFloat_t( svtrn2_f32(t02, t03) );                      // a00 =   
  a04 = SIMDFloat_t( svtrn1_f32(t04, t05) );                      // a00 =   
  a05 = SIMDFloat_t( svtrn2_f32(t04, t05) );                      // a00 =   
  a06 = SIMDFloat_t( svtrn1_f32(t06, t07) );                      // a00 =   
  a07 = SIMDFloat_t( svtrn2_f32(t06, t07) );                      // a00 =   
  a08 = SIMDFloat_t( svtrn1_f32(t08, t09) );                      // a00 =   
  a09 = SIMDFloat_t( svtrn2_f32(t08, t09) );                      // a00 =   
  a10 = SIMDFloat_t( svtrn1_f32(t10, t11) );                      // a00 =   
  a11 = SIMDFloat_t( svtrn2_f32(t10, t11) );                      // a00 =   
  a12 = SIMDFloat_t( svtrn1_f32(t12, t13) );                      // a00 =   
  a13 = SIMDFloat_t( svtrn2_f32(t12, t13) );                      // a00 =   
  a14 = SIMDFloat_t( svtrn1_f32(t14, t15) );                      // a00 =   
  a15 = SIMDFloat_t( svtrn2_f32(t14, t15) );                      // a00 =   
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 8, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t& a00, SIMDFloat_t& a01, SIMDFloat_t& a02, SIMDFloat_t& a03,
           		 SIMDFloat_t& a04, SIMDFloat_t& a05, SIMDFloat_t& a06, SIMDFloat_t& a07 )
{
  svfloat32_t t00, t01, t02, t03, t04, t05, t06, t07;
  svfloat32_t u00, u01, u02, u03, u04, u05, u06, u07;
                                                           // a =  0   1   2   3   4   5   6   7
                                                           // b =  8   9  10  11  12  13  14  15
                                                           // c = 16  17  18  19  20  21  22  23
                                                           // d = 24  25  26  27  28  29  30  31
                                                           // e = 32  33  34  35  36  37  38  39
                                                           // f = 40  41  42  43  44  45  46  47
                                                           // g = 48  49  50  51  52  53  54  55
                                                           // h = 56  57  58  59  60  61  62  63

  t00 = svtrn1q_f32( (svfloat32_t)(a00), (svfloat32_t)(a04) ); // a =  0   1   2   3  32  33  34  35  svtrnq1(a,e) interleave even quadwords
  t01 = svtrn1q_f32( (svfloat32_t)(a01), (svfloat32_t)(a05) ); // b =  8   9  10  11  40  41  42  43  svtrnq1(b,f) 
  t02 = svtrn1q_f32( (svfloat32_t)(a02), (svfloat32_t)(a06) ); // c = 16  17  18  19  48  49  50  51  svtrnq1(c,g) 
  t03 = svtrn1q_f32( (svfloat32_t)(a03), (svfloat32_t)(a07) ); // d = 24  25  26  27  56  57  58  59  svtrnq1(d,h) 
  t04 = svtrn2q_f32( (svfloat32_t)(a00), (svfloat32_t)(a04) ); // e =  4   5   6   7  36  37  38  39  svtrnq2(a,e) interleave odd quadwords
  t05 = svtrn2q_f32( (svfloat32_t)(a01), (svfloat32_t)(a05) ); // f = 12  13  14  15  44  45  46  47  svtrnq2(b,f) 
  t06 = svtrn2q_f32( (svfloat32_t)(a02), (svfloat32_t)(a06) ); // g = 20  21  22  23  52  53  54  55  svtrnq2(c,g) 
  t07 = svtrn2q_f32( (svfloat32_t)(a03), (svfloat32_t)(a07) ); // h = 28  29  30  31  60  61  62  63  svtrnq2(d,h) 

  u00 = svreinterpret_f32_f64( svtrn1_f64( svreinterpret_f64_f32(t00), svreinterpret_f64_f32(t02) ) ); // a =  0   1  16  17  32  33  48  49  svtrn1_f64(a,c) interleave even 64-bits
  u01 = svreinterpret_f32_f64( svtrn1_f64( svreinterpret_f64_f32(t01), svreinterpret_f64_f32(t03) ) ); // b =  8   9  24  25  40  41  56  57  svtrn1_f64(b,d) 
  u02 = svreinterpret_f32_f64( svtrn2_f64( svreinterpret_f64_f32(t00), svreinterpret_f64_f32(t02) ) ); // c =  2   3  18  19  34  35  50  51  svtrn2_f64(a,c) interleave odd 64-bits
  u03 = svreinterpret_f32_f64( svtrn2_f64( svreinterpret_f64_f32(t01), svreinterpret_f64_f32(t03) ) ); // d = 10  11  26  27  42  43  58  59  svtrn2_f64(b,d) 
  u04 = svreinterpret_f32_f64( svtrn1_f64( svreinterpret_f64_f32(t04), svreinterpret_f64_f32(t06) ) ); // e =  4   5  20  21  36  37  52  53  svtrn1_f64(e,g)
  u05 = svreinterpret_f32_f64( svtrn1_f64( svreinterpret_f64_f32(t05), svreinterpret_f64_f32(t07) ) ); // f = 12  13  28  29  44  45  60  61  svtrn1_f64(f,h)
  u06 = svreinterpret_f32_f64( svtrn2_f64( svreinterpret_f64_f32(t04), svreinterpret_f64_f32(t06) ) ); // g =  6   7  22  23  38  39  54  55  svtrn2_f64(e,g)   
  u07 = svreinterpret_f32_f64( svtrn2_f64( svreinterpret_f64_f32(t05), svreinterpret_f64_f32(t07) ) ); // h = 14  15  30  31  46  47  62  63  svtrn2_f64(f,h)

  a00 = SIMDFloat_t( svtrn1_f32(u00, u01) );               // a =  0   8  16  24  32  40  48  56  svtrn1_f32(a,b) interleave even 32-bits
  a01 = SIMDFloat_t( svtrn2_f32(u00, u01) );               // b =  1   9  17  25  33  45  49  57  svtrn2_f32(a,b) interleave odd 32-bits
  a02 = SIMDFloat_t( svtrn1_f32(u02, u03) );               // c =  2  10  18  26  34  42  50  58  svtrn1_f32(c,d) 
  a03 = SIMDFloat_t( svtrn2_f32(u02, u03) );               // d =  3  11  19  27  35  43  51  59  svtrn2_f32(c,d) 
  a04 = SIMDFloat_t( svtrn1_f32(u04, u05) );               // e =  4  12  20  28  36  44  52  60  svtrn1_f32(e,f)
  a05 = SIMDFloat_t( svtrn2_f32(u04, u05) );               // f =  5  13  21  29  37  45  53  61  svtrn2_f32(e,f)
  a06 = SIMDFloat_t( svtrn1_f32(u06, u07) );               // g =  6  14  22  30  38  46  54  62  svtrn1_f32(g,h)   
  a07 = SIMDFloat_t( svtrn2_f32(u06, u07) );               // h =  7  15  23  31  39  47  55  63  svtrn2_f32(g,h)
}

template<typename SIMDFloat_t, typename std::enable_if<SIMDFloat_t::size() == 4, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t& a, SIMDFloat_t& b, SIMDFloat_t& c, SIMDFloat_t& d) {
  svfloat32_t t0, t1, t2, t3;
                                                           // a = a0 a1 a2 a3
                                                           // b = b0 b1 b2 b3
                                                           // c = c0 c1 c2 c3
                                                           // d = d0 d1 d2 d3

  t0 = svreinterpret_f32_f64( svtrn1_f64( svreinterpret_f64_f32( (vls_float32_t)(a) ), svreinterpret_f64_f32( (vls_float32_t)(c) ) ) ); // a = a0 a1 c0 c1 svtrn1_f64( a, c )
  t1 = svreinterpret_f32_f64( svtrn1_f64( svreinterpret_f64_f32( (vls_float32_t)(b) ), svreinterpret_f64_f32( (vls_float32_t)(d) ) ) ); // b = b0 b1 d0 d1 svtrn1_f64( b, d )
  t2 = svreinterpret_f32_f64( svtrn2_f64( svreinterpret_f64_f32( (vls_float32_t)(a) ), svreinterpret_f64_f32( (vls_float32_t)(c) ) ) ); // c = a2 a3 c2 c3 svtrn2_f64( a, c )
  t3 = svreinterpret_f32_f64( svtrn2_f64( svreinterpret_f64_f32( (vls_float32_t)(b) ), svreinterpret_f64_f32( (vls_float32_t)(d) ) ) ); // d = b2 b3 d2 d3 svtrn2_f64( b, d )

  a  = SIMDFloat_t( (vls_float32_t)( svtrn1_f32( t0, t1) ) ); // a = a0 b0 c0 d0 svtrn1_f32( a, b ) 
  b  = SIMDFloat_t( (vls_float32_t)( svtrn2_f32( t0, t1) ) ); // b = a1 b1 c1 d1 svtrn2_f32( a, b )
  c  = SIMDFloat_t( (vls_float32_t)( svtrn1_f32( t2, t3) ) ); // c = a2 b2 c2 d2 svtrn1_f32( c, d ) 
  d  = SIMDFloat_t( (vls_float32_t)( svtrn2_f32( t2, t3) ) ); // d = a3 b3 c3 d3 svtrn2_f32( c, d )
}

#elif defined KOKKOS_ARCH_ARM_NEON
#include <arm_neon.h>

KOKKOS_INLINE_FUNCTION
simd_float_t simd_cast(simd_int32_t& a) {
  return simd_float_t(vreinterpretq_f32_s32((int32x4_t)(a)));
}

KOKKOS_INLINE_FUNCTION
simd_int32_t simd_cast(simd_float_t& a) {
  return simd_int32_t(vreinterpretq_s32_f32((float32x4_t)(a)));
  //vls_float32_t t = (vls_float32_t)(a);
  //return simd_int32_t( (vls_int32_t)(t) );
}

template< int i0, int i1, int i2, int i3 >
KOKKOS_FORCEINLINE_FUNCTION
simd_float32x4_t shuffle(simd_float32x4_t& a) {
  int32x4_t mask = {i0, i1, i2, i3};
  simd_float32x4_t b = __builtin_shuffle((float32x4_t) a, mask);
  return b;
}

KOKKOS_INLINE_FUNCTION
void transpose(simd_float_t& a, simd_float_t& b, simd_float_t& c, simd_float_t& d) {
  float32x4x2_t ab, cd;

  //  0  1  2  3
  //  4  5  6  7
  //  8  9 10 11
  // 12 13 14 15

  ab = vtrnq_f32( static_cast<float32x4_t>(a), (float32x4_t)b );
  cd = vtrnq_f32( (float32x4_t)c, (float32x4_t)d );

  //  0  4  2  6
  //  1  5  3  7
  //  8 12 10 14
  //  9 13 11 15

  float64x2_t f64r0 = (float64x2_t)(ab.val[0]);
  float64x2_t f64s0 = (float64x2_t)(cd.val[0]);
  a = simd_float_t((float32x4_t)(vtrn1q_f64( (float64x2_t)(ab.val[0]), (float64x2_t)(cd.val[0]) )));
  c = simd_float_t((float32x4_t)(vtrn2q_f64( (float64x2_t)(ab.val[0]), (float64x2_t)(cd.val[0]) )));

  //  0  4  8 12
  //  1  5  3  7
  //  2  6 10 14
  //  9 13 11 15

  b = simd_float_t((float32x4_t)(vtrn1q_f64( (float64x2_t)(ab.val[1]), (float64x2_t)(cd.val[1]) )));
  d = simd_float_t((float32x4_t)(vtrn2q_f64( (float64x2_t)(ab.val[1]), (float64x2_t)(cd.val[1]) )));

  //  0  4  8 12
  //  1  5  9 13
  //  2  6 10 14
  //  3  7 11 15
  return;
}
#else
template< int i0, int i1, int i2, int i3, typename Float4 >
KOKKOS_FORCEINLINE_FUNCTION
Float4 shuffle(Float4& a) {
  Float4 b;
  b[0] = (float) a[i0];
  b[1] = (float) a[i1];
  b[2] = (float) a[i2];
  b[3] = (float) a[i3];
  return b;
}

template< typename Float4 >
KOKKOS_INLINE_FUNCTION
void transpose(Float4& a0, Float4& a1, Float4& a2, Float4& a3) {
  swap( a0[1],a1[0] ); swap( a0[2],a2[0] ); swap( a0[3],a3[0] ); 
                       swap( a1[2],a2[1] ); swap( a1[3],a3[1] ); 
                                            swap( a2[3],a3[2] ); 
  return;
}

#endif

#endif // _kokkos_simd_extensions_h_
