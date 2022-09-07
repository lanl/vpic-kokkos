#define IN_sf_interface
#include "sf_interface_private.h"

//#include <iostream>
//#include <fstream>

// FIXME: N_ARRAY>1 ALWAYS BUT THIS ISN'T STRICTLY NECESSARY BECAUSE
// HOST IS THREAD FOR THE SERIAL AND THREADED DISPATCHERS.  SHOULD
// PROBABLY CHANGE N_ARRAY TO
// max({serial,thread}.n_pipeline,spu.n_pipeline+1)

void
hyb_reduce_accumulators_pipeline( accumulators_pipeline_args_t * args,
                              int pipeline_rank,
                              int n_pipeline ) {
  int i, i1, si = sizeof(accumulator_t) / sizeof(float);
  int r, nr = args->n_array-1, sr = si*args->s_array;
int j, k, l;

  DISTRIBUTE( args->n, accumulators_n_block,
              pipeline_rank, n_pipeline, i, i1 ); i1 += i;

  // a is broken into restricted rw and ro parts to allow the compiler
  // to do more aggresive optimizations

  /**/  float * RESTRICT ALIGNED(16) a = args->a->jx;
  const float * RESTRICT ALIGNED(16) b = a + sr;

# if defined(V4_ACCELERATION)

  using namespace v4;

  v4float v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;

# define LOOP(OP)                               \
  for( ; i<i1; i++ ) {                          \
    k = i*si;                                   \
    OP(k   ); OP(k+ 4); OP(k+ 8);               \
  }
# define A(k)   load_4x1(  &a[k],          v0   );
# define B(k,r) load_4x1(  &b[k+(r-1)*sr], v##r );
# define C(k,v) store_4x1( v, &a[k] )
# define O1(k)A(k  )B(k,1)                                                 \
              C(k,   v0+v1)
# define O2(k)A(k  )B(k,1)B(k,2)                                           \
              C(k,  (v0+v1)+ v2)
# define O3(k)A(k  )B(k,1)B(k,2)B(k,3)                                     \
              C(k,  (v0+v1)+(v2+v3))
# define O4(k)A(k  )B(k,1)B(k,2)B(k,3)B(k,4)                               \
              C(k, ((v0+v1)+(v2+v3))+  v4)
# define O5(k)A(k  )B(k,1)B(k,2)B(k,3)B(k,4)B(k,5)                         \
              C(k, ((v0+v1)+(v2+v3))+ (v4+v5))
# define O6(k)A(k  )B(k,1)B(k,2)B(k,3)B(k,4)B(k,5)B(k,6)                   \
              C(k, ((v0+v1)+(v2+v3))+((v4+v5)+ v6))
# define O7(k)A(k  )B(k,1)B(k,2)B(k,3)B(k,4)B(k,5)B(k,6)B(k,7)             \
              C(k, ((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)))
# define O8(k)A(k  )B(k,1)B(k,2)B(k,3)B(k,4)B(k,5)B(k,6)B(k,7)B(k,8)       \
              C(k,(((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)))+   v8)
# define O9(k)A(k  )B(k,1)B(k,2)B(k,3)B(k,4)B(k,5)B(k,6)B(k,7)B(k,8)B(k,9) \
              C(k,(((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)))+  (v8+v9))

# else

  float f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27;

# define LOOP(OP,rep)							\
  for( ; i<i1; i++ ) {							\
    k = i*si;								\
    OP(k   ,rep);OP(k+ 1,rep);OP(k+ 2,rep);OP(k+ 3,rep);		\
    OP(k+ 4,rep);OP(k+ 5,rep);OP(k+ 6,rep);OP(k+ 7,rep);		\
    OP(k+ 8,rep);OP(k+ 9,rep);OP(k+10,rep);OP(k+11,rep);		\
    OP(k+ 12,rep);OP(k+ 13,rep);OP(k+14,rep);OP(k+15,rep);		\
    OP(k+ 16,rep);OP(k+ 17,rep);OP(k+18,rep);OP(k+19,rep);		\
    OP(k+ 20,rep);OP(k+ 21,rep);OP(k+22,rep);OP(k+23,rep);		\
    OP(k+ 24,rep);OP(k+ 25,rep);OP(k+26,rep);OP(k+27,rep);		\
  }

# define OP(k,rep)		\
  for (l=0; l<rep; l++) {	\
    a[(k)] += b[ (k) + l*sr ];	\
  }

# endif
  
  switch( nr ) {        
  case 0:           break;
  case 1: LOOP(OP,1); break;
  case 2: LOOP(OP,2); break;
  case 3: LOOP(OP,3); break;
  case 4: LOOP(OP,4); break;
  case 5: LOOP(OP,5); break;
  case 6: LOOP(OP,6); break;
  case 7: LOOP(OP,7); break;
  case 8: LOOP(OP,8); break;
  case 9: LOOP(OP,9); break;
  default:
#   if defined(V4_ACCELERATION)
    for( ; i<i1; i++ ) {
      j = i*si;
      load_4x1(&a[j+0],v0);  load_4x1(&a[j+4],v1);  load_4x1(&a[j+8],v2);
      for( r=0; r<nr; r++ ) {
        k = j + r*sr;
        load_4x1(&b[k+0],v3);  load_4x1(&b[k+4],v4);  load_4x1(&b[k+8],v5);
        v0 += v3;              v1 += v4;              v2 += v5;
      }
      store_4x1(v0,&a[j+0]); store_4x1(v1,&a[j+4]); store_4x1(v2,&a[j+8]);
    }
#   else
    for( ; i<i1; i++ ) {
      j = i*si;
      f0  = a[j+ 0]; f1  = a[j+ 1]; f2  = a[j+ 2]; f3  = a[j+ 3];
      f4  = a[j+ 4]; f5  = a[j+ 5]; f6  = a[j+ 6]; f7  = a[j+ 7];
      f8  = a[j+ 8]; f9  = a[j+ 9]; f10 = a[j+10]; f11 = a[j+11];
      f12  = a[j+ 12]; f13  = a[j+ 13]; f14 = a[j+14]; f15 = a[j+15];
      f16  = a[j+ 16]; f17  = a[j+ 17]; f18 = a[j+18]; f19 = a[j+19];
      f20  = a[j+ 20]; f21  = a[j+ 21]; f22 = a[j+22]; f23 = a[j+23];
      f24  = a[j+ 24]; f25  = a[j+ 25]; f26 = a[j+26]; f27 = a[j+27];


        //std::cout << "f0=" << f0 << "\n";

      for( r=0; r<nr; r++ ) {
        k = j + r*sr;
        f0  += b[k+ 0]; f1  += b[k+ 1]; f2  += b[k+ 2]; f3  += b[k+ 3];
        f4  += b[k+ 4]; f5  += b[k+ 5]; f6  += b[k+ 6]; f7  += b[k+ 7];
        f8  += b[k+ 8]; f9  += b[k+ 9]; f10 += b[k+10]; f11 += b[k+11];
        f12  += b[k+ 12]; f13  += b[k+ 13]; f14 += b[k+14]; f15 += b[k+15];
        f16  += b[k+ 16]; f17  += b[k+ 17]; f18 += b[k+18]; f19 += b[k+19];
        f20  += b[k+ 20]; f21  += b[k+ 21]; f22 += b[k+22]; f23 += b[k+23];
        f24  += b[k+ 24]; f25  += b[k+ 25]; f26 += b[k+26]; f27 += b[k+27];

      }
      a[j+ 0] =  f0; a[j+ 1] =  f1; a[j+ 2] =  f2; a[j+ 3] =  f3;
      a[j+ 4] =  f4; a[j+ 5] =  f5; a[j+ 6] =  f6; a[j+ 7] =  f7;
      a[j+ 8] =  f8; a[j+ 9] =  f9; a[j+10] = f10; a[j+11] = f11;
      a[j+ 12] =  f12; a[j+ 13] =  f13; a[j+14] = f14; a[j+15] = f15;
      a[j+ 16] =  f16; a[j+ 17] =  f17; a[j+18] = f18; a[j+19] = f19;
      a[j+ 20] =  f20; a[j+ 21] =  f21; a[j+22] = f22; a[j+23] = f23;
      a[j+ 24] =  f24; a[j+ 25] =  f25; a[j+26] = f26; a[j+27] = f27;

    }
#   endif
    break;
  }

# undef OP
# undef C
# undef B
# undef A
# undef LOOP

}

#if defined(V4_ACCELERATION) && defined(HAS_V4_PIPELINE)

#error "The regular pipeline is already V4 accelerated!"

#endif

#define VOX(x,y,z) VOXEL(x,y,z, aa->g->nx,aa->g->ny,aa->g->nz)

void
hyb_reduce_accumulator_array( accumulator_array_t * RESTRICT aa ) {
  DECLARE_ALIGNED_ARRAY( accumulators_pipeline_args_t, 128, args, 1 );
  int i0;

  if( !aa ) ERROR(( "Bad args" ));

  i0 = (VOX(1,1,1)/2)*2; // Round i0 down to even for 128B align on Cell

  args->a       = aa->a + i0;
  args->n       = (((VOX(aa->g->nx,aa->g->ny,aa->g->nz) - i0 + 1 )+1)/2)*2;
  args->n_array = aa->n_pipeline + 1;
  args->s_array = aa->stride;

  EXEC_PIPELINES( hyb_reduce_accumulators, args, 0 );
  WAIT_PIPELINES();
}
