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
using native_abi_32 = KokkosSIMD::simd_abi::avx512_fixed_size<16>;
using native_abi_64 = KokkosSIMD::simd_abi::avx512_fixed_size<8>;
using abi_x4 = KokkosSIMD::simd_abi::avx2_fixed_size<4>;
#elif defined(KOKKOS_ARCH_AVX2)
using native_abi_32 = KokkosSIMD::simd_abi::avx2_fixed_size<8>;
using native_abi_64 = KokkosSIMD::simd_abi::avx2_fixed_size<4>;
using abi_x4 = KokkosSIMD::simd_abi::avx2_fixed_size<4>;
#elif defined(KOKKOS_ARCH_ARM_NEON)
using native_abi_32 = KokkosSIMD::simd_abi::neon_fixed_size<4>;
using native_abi_64 = KokkosSIMD::simd_abi::neon_fixed_size<2>;
using abi_x4 = KokkosSIMD::simd_abi::neon_fixed_size<4>;
#else
using native_abi_32 = KokkosSIMD::simd_abi::scalar;
using native_abi_64 = KokkosSIMD::simd_abi::scalar;
using native_x4 = KokkosSIMD::simd_abi::scalar;
#endif

using simd_float_t          = KokkosSIMD::simd<float,   native_abi_32>;
using simd_int32_t          = KokkosSIMD::simd<int32_t, native_abi_32>;
using simd_int64_t          = KokkosSIMD::simd<int64_t, native_abi_64>;
using simd_float_mask_t     = KokkosSIMD::simd_mask<float,   native_abi_32>;
using simd_int32_mask_t     = KokkosSIMD::simd_mask<int32_t, native_abi_32>;
using simd_int64_mask_t     = KokkosSIMD::simd_mask<int64_t, native_abi_64>;

using simd_float32x4_t      = KokkosSIMD::simd<float, abi_x4>;
using simd_int32x4_t        = KokkosSIMD::simd<int,   abi_x4>;
using simd_float32x4_mask_t = KokkosSIMD::simd_mask<float, abi_x4>;
using simd_int32x4_mask_t   = KokkosSIMD::simd_mask<int,   abi_x4>;

//using simd_float_t          = KokkosSIMD::simd<float,  KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int32_t          = KokkosSIMD::simd<int32_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int64_t          = KokkosSIMD::simd<int64_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_float_mask_t     = KokkosSIMD::simd_mask<float,  KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int32_mask_t     = KokkosSIMD::simd_mask<int32_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;
//using simd_int64_mask_t     = KokkosSIMD::simd_mask<int64_t,KokkosSIMD::simd_abi::auto_fixed_size<16>>;

constexpr auto SIMD_LEN = simd_float_t::size();

template<typename ABI>
using SIMDFloat_t = Kokkos::Experimental::simd<float, ABI>;

template<int i0, int i1, int i2, int i3>
struct permute
{
  constexpr static int value = i0 + i1*4 + i2*16 + i3*64;
};

template<typename ABI>
KOKKOS_FORCEINLINE_FUNCTION
void increment( float * p, const SIMDFloat_t<ABI> &v ) {
  SIMDFloat_t<ABI> a;
  a.copy_from(p, element_aligned_tag_t());
  a += v;
  a.copy_to(  p, element_aligned_tag_t());
}

KOKKOS_FORCEINLINE_FUNCTION
void swap(float& a, float& b) {
  float t = (float) a;
  a = (float) b;
  b = t;
}

#ifdef __AVX512F__
template<typename ABI, typename std::enable_if<std::is_same<ABI, Kokkos::Experimental::simd_abi::avx512_fixed_size<16>>::value, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t<ABI>& a00, SIMDFloat_t<ABI>& a01, SIMDFloat_t<ABI>& a02, SIMDFloat_t<ABI>& a03,
           		 SIMDFloat_t<ABI>& a04, SIMDFloat_t<ABI>& a05, SIMDFloat_t<ABI>& a06, SIMDFloat_t<ABI>& a07,
           		 SIMDFloat_t<ABI>& a08, SIMDFloat_t<ABI>& a09, SIMDFloat_t<ABI>& a10, SIMDFloat_t<ABI>& a11,
           		 SIMDFloat_t<ABI>& a12, SIMDFloat_t<ABI>& a13, SIMDFloat_t<ABI>& a14, SIMDFloat_t<ABI>& a15 )
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

  t00 = _mm512_unpacklo_ps( (__m512)(a00), (__m512)(a01) ); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
  t01 = _mm512_unpackhi_ps( (__m512)(a00), (__m512)(a01) ); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  t02 = _mm512_unpacklo_ps( (__m512)(a02), (__m512)(a03) ); //  32  48  33  49  36  52  37  53  40  56  41  57  44  60  45  61
  t03 = _mm512_unpackhi_ps( (__m512)(a02), (__m512)(a03) ); //  34  50  35  51  38  54  39  55  42  58  43  59  46  62  47  63
  t04 = _mm512_unpacklo_ps( (__m512)(a04), (__m512)(a05) ); //  64  80  65  81  68  84  69  85  72  88  73  89  76  92  77  93
  t05 = _mm512_unpackhi_ps( (__m512)(a04), (__m512)(a05) ); //  66  82  67  83  70  86  71  87  74  90  75  91  78  94  79  95
  t06 = _mm512_unpacklo_ps( (__m512)(a06), (__m512)(a07) ); //  96 112  97 113 100 116 101 117 104 120 105 121 108 124 109 125
  t07 = _mm512_unpackhi_ps( (__m512)(a06), (__m512)(a07) ); //  98 114  99 115 102 118 103 119 106 122 107 123 110 126 111 127
  t08 = _mm512_unpacklo_ps( (__m512)(a08), (__m512)(a09) ); // 128 144 129 145 132 148 133 149 136 152 137 153 140 156 141 157
  t09 = _mm512_unpackhi_ps( (__m512)(a08), (__m512)(a09) ); // 130 146 131 147 134 150 135 151 138 154 139 155 142 158 143 159
  t10 = _mm512_unpacklo_ps( (__m512)(a10), (__m512)(a11) ); // 160 176 161 177 164 180 165 181 168 184 169 185 172 188 173 189
  t11 = _mm512_unpackhi_ps( (__m512)(a10), (__m512)(a11) ); // 162 178 163 179 166 182 167 183 170 186 171 187 174 190 175 191
  t12 = _mm512_unpacklo_ps( (__m512)(a12), (__m512)(a13) ); // 192 208 193 209 196 212 197 213 200 216 201 217 204 220 205 221
  t13 = _mm512_unpackhi_ps( (__m512)(a12), (__m512)(a13) ); // 194 210 195 211 198 214 199 215 202 218 203 219 206 222 207 223
  t14 = _mm512_unpacklo_ps( (__m512)(a14), (__m512)(a15) ); // 224 240 225 241 228 244 229 245 232 248 233 249 236 252 237 253
  t15 = _mm512_unpackhi_ps( (__m512)(a14), (__m512)(a15) ); // 226 242 227 243 230 246 231 247 234 250 235 251 238 254 239 255

  a00 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t00, t02, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //   0  16  32  48
  a01 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t00, t02, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //   1  17  33  49
  a02 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t01, t03, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //   2  18  34  50 
  a03 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t01, t03, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //   3  19  35  51 
  a04 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t04, t06, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //  64  80  96 112 
  a05 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t04, t06, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //  65  81  97 113
  a06 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t05, t07, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  //  66  82  98 114 
  a07 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t05, t07, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  //  67  83  99 115 
  a08 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t08, t10, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 128 144 160 176 
  a09 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t08, t10, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 129 145 161 177 
  a10 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t09, t11, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 130 146 162 178 
  a11 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t09, t11, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 131 147 163 179 
  a12 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t12, t14, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 192 208 228 240 
  a13 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t12, t14, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 193 209 229 241 
  a14 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t13, t15, _MM_SHUFFLE( 1, 0, 1, 0 ) ) );  // 194 210 230 242 
  a15 = SIMDFloat_t<ABI>( _mm512_shuffle_ps( t13, t15, _MM_SHUFFLE( 3, 2, 3, 2 ) ) );  // 195 211 231 243 

  t00 = _mm512_shuffle_f32x4( (__m512)(a00), (__m512)(a04), 0x88 ); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
  t01 = _mm512_shuffle_f32x4( (__m512)(a01), (__m512)(a05), 0x88 ); //   1  17  33  49 ...
  t02 = _mm512_shuffle_f32x4( (__m512)(a02), (__m512)(a06), 0x88 ); //   2  18  34  50 ...
  t03 = _mm512_shuffle_f32x4( (__m512)(a03), (__m512)(a07), 0x88 ); //   3  19  35  51 ...
  t04 = _mm512_shuffle_f32x4( (__m512)(a00), (__m512)(a04), 0xdd ); //   4  20  36  52 ...
  t05 = _mm512_shuffle_f32x4( (__m512)(a01), (__m512)(a05), 0xdd ); //   5  21  37  53 ...
  t06 = _mm512_shuffle_f32x4( (__m512)(a02), (__m512)(a06), 0xdd ); //   6  22  38  54 ...
  t07 = _mm512_shuffle_f32x4( (__m512)(a03), (__m512)(a07), 0xdd ); //   7  23  39  55 ...
  t08 = _mm512_shuffle_f32x4( (__m512)(a08), (__m512)(a12), 0x88 ); // 128 144 160 176 ...
  t09 = _mm512_shuffle_f32x4( (__m512)(a09), (__m512)(a13), 0x88 ); // 129 145 161 177 ...
  t10 = _mm512_shuffle_f32x4( (__m512)(a10), (__m512)(a14), 0x88 ); // 130 146 162 178 ...
  t11 = _mm512_shuffle_f32x4( (__m512)(a11), (__m512)(a15), 0x88 ); // 131 147 163 179 ...
  t12 = _mm512_shuffle_f32x4( (__m512)(a08), (__m512)(a12), 0xdd ); // 132 148 164 180 ...
  t13 = _mm512_shuffle_f32x4( (__m512)(a09), (__m512)(a13), 0xdd ); // 133 149 165 181 ...
  t14 = _mm512_shuffle_f32x4( (__m512)(a10), (__m512)(a14), 0xdd ); // 134 150 166 182 ...
  t15 = _mm512_shuffle_f32x4( (__m512)(a11), (__m512)(a15), 0xdd ); // 135 151 167 183 ...

  a00 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t00, t08, 0x88 ) ); //   0  16  32  48  64  80  96 112 ... 240
  a01 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t01, t09, 0x88 ) ); //   1  17  33  49  66  81  97 113 ... 241
  a02 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t02, t10, 0x88 ) ); //   2  18  34  50  67  82  98 114 ... 242
  a03 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t03, t11, 0x88 ) ); //   3  19  35  51  68  83  99 115 ... 243
  a04 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t04, t12, 0x88 ) ); //   4 ...
  a05 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t05, t13, 0x88 ) ); //   5 ...
  a06 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t06, t14, 0x88 ) ); //   6 ...
  a07 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t07, t15, 0x88 ) ); //   7 ...
  a08 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t00, t08, 0xdd ) ); //   8 ...
  a09 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t01, t09, 0xdd ) ); //   9 ...
  a10 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t02, t10, 0xdd ) ); //  10 ...
  a11 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t03, t11, 0xdd ) ); //  11 ...
  a12 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t04, t12, 0xdd ) ); //  12 ...
  a13 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t05, t13, 0xdd ) ); //  13 ...
  a14 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t06, t14, 0xdd ) ); //  14 ...
  a15 = SIMDFloat_t<ABI>( _mm512_shuffle_f32x4( t07, t15, 0xdd ) ); //  15  31  47  63  79  96 111 127 ... 255
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

template<typename ABI, typename std::enable_if<SIMDFloat_t<ABI>::size() == 8, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t<ABI>& a0, SIMDFloat_t<ABI>& a1, SIMDFloat_t<ABI>& a2, SIMDFloat_t<ABI>& a3,
               SIMDFloat_t<ABI>& a4, SIMDFloat_t<ABI>& a5, SIMDFloat_t<ABI>& a6, SIMDFloat_t<ABI>& a7) {
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

  a0 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u0, u4, 0x20 ) );
  a1 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u1, u5, 0x20 ) );
  a2 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u2, u6, 0x20 ) );
  a3 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u3, u7, 0x20 ) );
  a4 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u0, u4, 0x31 ) );
  a5 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u1, u5, 0x31 ) );
  a6 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u2, u6, 0x31 ) );
  a7 = SIMDFloat_t<ABI>( _mm256_permute2f128_ps( u3, u7, 0x31 ) );

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

#ifdef __AVX__
template< int i0, int i1, int i2, int i3>
KOKKOS_FORCEINLINE_FUNCTION
simd_float32x4_t shuffle(simd_float32x4_t& a) {
  return simd_float32x4_t( _mm_shuffle_ps( (__m128)(a), (__m128)(a), ( permute<i0,i1,i2,i3>::value ) ) );
}

template<typename ABI, typename std::enable_if<SIMDFloat_t<ABI>::size() == 4, bool>::type=true >
KOKKOS_INLINE_FUNCTION
void transpose(SIMDFloat_t<ABI>& a, SIMDFloat_t<ABI>& b, SIMDFloat_t<ABI>& c, SIMDFloat_t<ABI>& d) {
  // a =  0  1  2  3
  // b =  4  5  6  7
  // c =  8  9 10 11
  // d = 12 13 14 15

  __m128  r, s, t, u;

  r   = _mm_unpackhi_ps( (__m128)(a), (__m128)(b) ); // r =  2  6  3  7
  s   = _mm_unpacklo_ps( (__m128)(a), (__m128)(b) ); // s =  0  4  1  5
  t   = _mm_unpackhi_ps( (__m128)(c), (__m128)(d) ); // t = 10 14 11 15
  u   = _mm_unpacklo_ps( (__m128)(c), (__m128)(d) ); // u =  8 12  9 13

  a = SIMDFloat_t<ABI>(_mm_movelh_ps( s, u )); // a = 0 4  8 12
  b = SIMDFloat_t<ABI>(_mm_movehl_ps( u, s )); // b = 1 5  9 13
  c = SIMDFloat_t<ABI>(_mm_movelh_ps( r, t )); // c = 2 6 10 14
  d = SIMDFloat_t<ABI>(_mm_movehl_ps( t, r )); // d = 3 7 11 15

  // a = 0 4  8 12
  // b = 1 5  9 13
  // c = 2 6 10 14
  // d = 3 7 11 15
  return;
}

#elif defined KOKKOS_ARCH_ARM_NEON
#include <arm_neon.h>

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

  ab = vtrnq_f32( (float32x4_t)a, (float32x4_t)b );
  cd = vtrnq_f32( (float32x4_t)c, (float32x4_t)d );

  float64x2_t f64r0 = (float64x2_t)(ab.val[0]);
  float64x2_t f64s0 = (float64x2_t)(cd.val[0]);
  a = simd_float_t((float32x4_t)(vtrn1q_f64( (float64x2_t)(ab.val[0]), (float64x2_t)(cd.val[0]) )));
  c = simd_float_t((float32x4_t)(vtrn2q_f64( (float64x2_t)(ab.val[0]), (float64x2_t)(cd.val[0]) )));

  b = simd_float_t((float32x4_t)(vtrn1q_f64( (float64x2_t)(ab.val[1]), (float64x2_t)(cd.val[1]) )));
  d = simd_float_t((float32x4_t)(vtrn2q_f64( (float64x2_t)(ab.val[1]), (float64x2_t)(cd.val[1]) )));
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
