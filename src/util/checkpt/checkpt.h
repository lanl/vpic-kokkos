#ifndef _checkpt_h_
#define _checkpt_h_

#include "../../vpic/kokkos_helpers.h"
#include "../util_base.h"

/* A checkpt_func_t serializes an object to a checkpt.  It takes a
   pointer to the object to serialize.  Objects are checkpointed in
   the order they are registered.  A checkpoint_func_t should not
   register or unregister any objects.  checkpoint_func_t must part of
   the global symbol table (e.g. not declared as static). */

typedef void
(*checkpt_func_t)( const void * obj );

/* A restore_func_t deserializes an object from a checkpt.  It returns
   a pointer to the newly restored object.  Objects are restored in
   the order they were initially registered.  A restore_func_t should
   not register or unregister any objects.  A restore_func_t must part
   of the global symbol table (e.g. not declared as static). */

typedef void *
(*restore_func_t)( void );

/* A reanimate_func_t reanimates an object.  It takes a pointer to the
   object to reanimate.  Reanimation is done _after_ all objects in a
   checkpt have been restored.  Almost always, objects need no
   reanimation.  However, if the dependencies among checkpointed
   objects do not form a DAG (directed acyclic graph) or if some
   objects need to perform internode communications to complete their
   restore process, reanimation might be necessary for some objects.
   A reanimate_func_t should not register or unregister any objects.
   A reanimate_func_t must part of the global symbol table (e.g. not
   declared as static). */

typedef void
(*reanimate_func_t)( void * obj );

/* Boot up the checkpt service */

void
boot_checkpt( int * pargc,
              char *** pargv );

/* Halt the checkpt service */

void
halt_checkpt( void );

/* All objects to be checkpointed need to be registered when they are
   initially created.  The object registers the methods necessary to
   checkpt, restore and reanimate it.  p must be non-NULL (to avoid an
   ambiguity with object_ptr error code return below) but it is okay
   to pass NULL for the functions (it indicates that the
   checkpt/restore/reanimate process need not call that particular
   function on that object).  Objects should be registered once and
   only once. */

void
register_object( void * obj,
                 checkpt_func_t   checkpt_func,
                 restore_func_t   restore_func,
                 reanimate_func_t reanimate_func );

/* When a checkpointed object is destroyed, it should also be
   unregistered from the checkpt service. */

void
unregister_object( void * obj );

/* Given a pointer to an registered object, return an identifier for
   that object that is stable across a checkpt/restore.  Returns 0 if
   the ptr does not correspond to a registered object.  This function
   is really only needed to reanimate objects that have non-DAG
   dependencies. */

size_t
object_id( const void * ptr );

/* Given the unique identifier of a registered object, return a
   pointer to that object (valid within the current applications
   execution).  Returns NULL if the object could not be found.  This
   function is really only needed to reanimate objects that have
   non-DAG dependencies. */

void *
object_ptr( size_t id );

/* Checkpt(restore) all objects to(from) the checkpt with the given
   name (a '\0'-terminated string).  If restore_objects is called
   with any objects already registered, already objects already
   registered will be silently unregistered.  Except for objects
   registered during boot_services, this is not an issue
   practically (and the objects registered during boot_services
   know how to handle this case).  In short, restore_objects
   nominally should be called after boot_services and before
   and before any new objects are registered. */

void
checkpt_objects( const char * name );

void
restore_objects( const char * name );

/* Call the reanimate functions on all objects.  This is typically
   done after the restore process. */

void
reanimate_objects( void );

/*****************************************************************************/
/* Simple checkpt / restore / reanimate primitives */
                 
/* Checkpt(restore) n bytes from(to) data. */

void
checkpt_raw( const void * data,
             size_t n_byte );

void
restore_raw( void * data,
             size_t n_byte );


/*****************************************************************************/
/* Composite checkpt / restore / reanimate primitives */

/* Checkpt(restore) n_ele elements of data.  The elements are sz_ele
   bytes in size and are separated by str_ele bytes.  The returned
   pointer of restore_data is heap allocated as:
     MALLOC( (char *)data, max_ele*str_ele )
   if align is zero or:
     MALLOC_ALIGNED( (char *)data, max_ele*str_ele, align )
   where the first n_ele elements are initialized.  It is okay to
   checkpoint no bytes (on restore, the array will be written with
   reallocated according to max_ele and str_ele but will not be
   initialized) and okay to checkpoint a NULL pointer when the
   number of bytes to write is zero (the restored pointer may not
   be NULL if max_ele and str_ele are non-trivial in this case). */

void
checkpt_data( const void * data,
              size_t sz_ele,
              size_t str_ele,
              size_t n_ele,
              size_t max_ele,
              size_t align );

void *
restore_data( void );

/* Checkpt(restore) a '\0'-terminated string.  The returned pointer of
   restore_str heap_allocated as:
     MALLOC( (char *)string, strlen_string+1 )
   It is okay to checkpoint NULL (it will be restored as NULL). */

void
checkpt_str( const char * str );

char *
restore_str( void );

/* Checkpt, restore and reanimate a "forward" pointer to a registered
   object.  It is okay to checkpt NULL (it will be restored and
   reanimated as NULL).  Outside of NULL, checkpt_fptr will give an
   error if the pointer does not point to a registered object.  Unlike
   restore_ptr below, restore_fptr is safe use n a restore function on
   any pointers to checkpointed object.  However, this flexibility has
   a price.  A restore_fptr is not safe to use _until_ it after
   reanimate_fptr called on it in a reanimate function. */

void
checkpt_fptr( const void * ptr );

void *
restore_fptr( void );

void *
reanimate_fptr( void * );

/* Checkpt(restore) a pointer to a registered object.  It is okay
   to checkpt NULL (it will be restored as NULL).  Outside of NULL,
   checkpt_ptr will give an error if the pointer does not correspond
   to a registered object.  restore_ptr is safe to use in a restore
   function provided the checkpointed objects were registered in the
   order they were created and this pointer being restored is part of
   an object registered _after_ the object being pointed to (i.e. the
   object dependency network is a DAG ... note that you have to do
   some pretty ugly code to create object dependencies that aren't a
   DAG).  If object dependencies do not form a DAG, object pointers
   can still be restored.  In this case, forward object dependencies
   need to be restored by a reanimate function.  Like checkpt_ptr, if
   restore_ptr is used to restore a pointer to an object that has
   not already been restored, it give an error. */

void
checkpt_ptr( const void * ptr );

void *
restore_ptr( void );

/* Checkpt(restore) a symbol (e.g. a function pointer).  Generally, if
   you do not change your application binary between a checkpt and
   restore and are not checkpointing a symbol from a dynamically
   linked library, this will just work, regardless of the checkpointed
   symbol's properties, platform support issues and/or any complaining
   these functions might do.

   In more general cases, for this to work reliably, the symbol must
   be part of the application's global symbol table (e.g. a non-static
   non-inlined function or a non-static file-scope variable) or
   somewhere in the default dynamically linked library search path
   (e.g. LD_LIBRARY_PATH), your platform must support reverse symbol
   table lookups (e.g. dladdr is provided as part of libdl on
   UNIX/Linux) and your application was linked with the global symbol
   table exported (e.g. with gcc, use "-rdynamic" for the linking
   step).

   If all of these are true, the symbol will be checkpointed such that
   it should be restorable under even very adverse conditions (e.g.
   these will work after changing the application between a checkpt
   and restore, will work even if restoring symbols within dynamically
   linked libraries that were upgraded to a newer---but still binary
   compatible---version between a checkpoint and restore ...).  Note
   though, even if all these are true, there are still some cases
   where checkpt_sym / restore_sym can encounter difficulty:

   - You didn't actually checkpoint a symbol.

   - You renamed functions referred to by checkpointed symbols in your
   application, recompiled and used an old checkpt on the new
   application binary.

   - You checkpointd symbols located in dynamically linked libraries
   and hid these libraries from the application between a checkpt and
   restore (e.g. changed your LD_LIBRARY_PATH).

   - You checkpointed symbols located in dynamically linked libraries
   and you (or a malcontented sysadmin) silently replaced these
   libraries with binary incompatible versions between a checkpt and
   restore.

   If these are not true, checkpt_sym/restore_sym will do the best it
   can (i.e.  good enough to support the simple case above) and issue
   a verbose warnings to the log to help diagnosing (and repairing)
   any issues caused by checkpointing and restoring various
   edge cases above.
   
   It is okay to checkpoint a NULL symbol (it will be restored as a
   NULL). */

void
checkpt_sym( const void * sym );

void *
restore_sym( void );

/****************************************************************************/

/* This robust macros are provided for  convenience.  They provide a
   checkpt/restore helpers analogous to the MALLOC/FREE,
   MALLOC_ALIGNED/FREE_ALIGNED interface and get around some pedantic
   limitations of strict C and strict C++.

   Namely:
   - When writing checkpt/restore/reanimate functions, it is more
     convenient pass pointers appropriate for the object type.
     But then, compiler will complain, when you pass these to
     register_object will complain, that (in a pedantic sense),
     these functions have the wrong signature register_object.
     REGISTER_OBJECT does the necessary casting for you.

   - Strict C and C++ do not allow conversions (either implicit
     or explicit conversions of a function pointer to a void
     pointer.  CHECKPT_SYM does the indirect casting for you.

   - Strict C++ does not allow a void pointer to be implicit cast
     into another pointer type (which kind of defeats the purpose of
     void pointers in the first place).  So, we have a hack for
     that they works in C and C++ too. */

#define REGISTER_OBJECT(p,c,r,a) register_object( (p),                  \
                                                  (checkpt_func_t)(c),  \
                                                  (restore_func_t)(r),  \
                                                  (reanimate_func_t)(a) )

#define UNREGISTER_OBJECT(p)     unregister_object( (p) )

#define CHECKPT_ALIGNED(p,n,a) do {           \
    size_t _sz = (n)*sizeof(*(p));            \
    checkpt_data( (p), _sz, _sz, 1, 1, (a) ); \
  } while(0)

#define CHECKPT_VAL( T, val ) do {              \
    T _val;                                     \
    _val = (T)(val);                            \
    checkpt_raw( &_val, sizeof(T) );            \
  } while(0)

#define CHECKPT(p,n)       CHECKPT_ALIGNED( (p), (n), 0 )
#define CHECKPT_STR(p)     checkpt_str(  (p) )
#define CHECKPT_FPTR(p)    checkpt_fptr( (p) )
#define CHECKPT_PTR(p)     checkpt_ptr(  (p) )
#define CHECKPT_SYM(p)     checkpt_sym((const void *)(size_t)(p))
#define CHECKPT_VIEW(v)    checkpt_view( (v) )

#define RESTORE_VAL( T, val ) do {              \
    T _val;                                     \
    restore_raw( &_val, sizeof(T) );            \
    (val) = (T)(_val);                          \
  } while(0)

#define RESTORE_ALIGNED(p) CXX_ILLEGAL_PTR_COPY( (p), restore_data() )
#define RESTORE(p)         RESTORE_ALIGNED((p))
#define RESTORE_STR(p)     CXX_ILLEGAL_PTR_COPY( (p), restore_str()  )
#define RESTORE_FPTR(p)    CXX_ILLEGAL_PTR_COPY( (p), restore_fptr() )
#define RESTORE_PTR(p)     CXX_ILLEGAL_PTR_COPY( (p), restore_ptr()  )
#define RESTORE_SYM(p)     CXX_ILLEGAL_PTR_COPY( (p), restore_sym()  )
#define REANIMATE_FPTR(p)  CXX_ILLEGAL_PTR_COPY( (p), reanimate_fptr((p)) )
#define RESTORE_VIEW(v)    restore_view( (v) )

/* This macro could be done without a function call, to the effect
   of "((*((void const **)(&(lv)))) = (rv))" but I think this violates
   strict aliasing rules (I believe this is an example of
   "type-punning") and thus the compiler might optimize it
   incorrectly.  This works in both C and C++ code as both allow
   implicit conversion of any data pointer to a const void data
   pointer. */

#define CXX_ILLEGAL_PTR_COPY(lv,rv) _cxx_illegal_ptr_copy(&(lv),(rv))

void
_cxx_illegal_ptr_copy( void * lv_ref,
                       const void * rv );

/**
 * @brief Mapping for View value_type to integer
 */
enum ViewValueType {
  Char   = 0,
  UInt8  = 1,
  UInt16 = 2,
  UInt32 = 3,
  UInt64 = 4,
  Int8   = 5,
  Int16  = 6,
  Int32  = 7,
  Int64  = 8,
  Float16= 9,
  Float32= 10,
  Float64= 11
};

/**
 * @brief Checkpoint a Kokkos View. Stores dimensions, capacity, and any memory
 * traits of the View. Since View type varies, this must be a template function
 * defined in the header.
 * 
 * @param view View to checkpoint
 */
template<typename View>
void checkpt_view( const View& view ) {
  size_t rank = view.rank();
  size_t span = view.span();
  //size_t size = view.size();
  size_t dims[Kokkos::ARRAY_LAYOUT_MAX_RANK];
  for(size_t i=0; i<rank; i++) {
    dims[i] = view.extent(0);
  }

  size_t datastruct = 99999;
  if( std::is_same<View, k_field_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::Fields;
  } else if( std::is_same<View, k_field_edge_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::FieldEdges;
  } else if( std::is_same<View, k_field_accum_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::FieldAccum;
  } else if( std::is_same<View, k_jf_accum_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::CurrentAccum;
  } else if( std::is_same<View, k_interpolator_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::Interpolators;
  } else if( std::is_same<View, k_accumulators_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::Accumulators;
  } else if( std::is_same<View, k_hydro_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::Hydro;
  } else if( std::is_same<View, k_particles_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::Particles;
  } else if( std::is_same<View, k_particles_i_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::ParticleCellID;
  } else if( std::is_same<View, k_particle_movers_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::ParticleMovers;
  } else if( std::is_same<View, k_particle_i_movers_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::ParticleMoverIDs;
  } else if( std::is_same<View, k_particle_partition_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::ParticlePartition;
  } else if( std::is_same<View, k_fluid_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::Fluid;
  } else if( std::is_same<View, k_neighbor_t::HostMirror>::value ) {
    datastruct = vpic_data_struct::GridNeighbors;
  } else {
    datastruct = vpic_data_struct::Other;
  }
  using ValueType = typename View::non_const_value_type;
  size_t type_id = 32;
  if( std::is_same<ValueType, char>::value ) {
    type_id = 0;
  } else if( std::is_same<ValueType, uint8_t>::value ) {
    type_id = 1;
  } else if( std::is_same<ValueType, uint16_t>::value ) {
    type_id = 2;
  } else if( std::is_same<ValueType, uint32_t>::value ) {
    type_id = 3;
  } else if( std::is_same<ValueType, uint64_t>::value ) {
    type_id = 4;
  } else if( std::is_same<ValueType, int8_t>::value ) {
    type_id = 5;
  } else if( std::is_same<ValueType, int16_t>::value ) {
    type_id = 6;
  } else if( std::is_same<ValueType, int32_t>::value ) {
    type_id = 7;
  } else if( std::is_same<ValueType, int64_t>::value ) {
    type_id = 8;
//  } else if( std::is_same<ValueType, float16_t>::value ) {
//    type_id = 9;
  } else if( std::is_same<ValueType, float>::value ) {
    type_id = 10;
  } else if( std::is_same<ValueType, double>::value ) {
    type_id = 11;
  }
  size_t layout;
  if( std::is_same<typename View::array_layout, Kokkos::LayoutLeft>::value ) {
    layout = 0;
  } else {
    layout = 1;
  }

  /* Write the data header */

  CHECKPT_VAL( size_t, 0x51E55 ); // View header ID
  CHECKPT_VAL( size_t, datastruct ); // Standard data structures used in VPIC
  CHECKPT_VAL( size_t, type_id ); // TypeID if not one of the standard data types
  CHECKPT_VAL( size_t, layout ); // Memory layout
  CHECKPT_VAL( size_t, rank ); // Number of View dimensions
  for( size_t i=0; i<rank; i++ ) 
    CHECKPT_VAL( size_t, dims[i] );

  /* Write out the individual elements */

  checkpt_raw( view.data(), sizeof(ValueType)*span );
}

/**
 * @brief Restore a Kokkos View. Loads dimensions, capacity, and any memory
 * traits of the View. Since View type varies, this must be a template function
 * defined in the header.
 * 
 * @param view View to restore
 */
template<typename View>
void restore_view( View& view ) {
  size_t dims[Kokkos::ARRAY_LAYOUT_MAX_RANK];

  size_t n, datastruct, type_id, layout, rank;

  /* Read the data header */

  restore_raw(&n, sizeof(size_t)); // View header ID
  //RESTORE_VAL( size_t, n ); // View header ID
  if( n!=0x51E55 ) ERROR(( "malformed checkpt (expected a View header)" ));
  restore_raw( &datastruct, sizeof(size_t) ); // Standard data structures used in VPIC
  restore_raw( &type_id, sizeof(size_t) ); // TypeID if not one of the standard data types
  restore_raw( &layout, sizeof(size_t) ); // Memory layout
  restore_raw( &rank, sizeof(size_t) ); // Number of View dimensions
  for( size_t i=0; i<rank; i++ ) 
    restore_raw( &(dims[i]), sizeof(size_t) );

  // Verify checkpt header matches supplied View
  size_t view_type = 99999;
  if( std::is_same<View, k_field_t::HostMirror>::value ) {
    view_type = vpic_data_struct::Fields;
  } else if( std::is_same<View, k_field_edge_t::HostMirror>::value ) {
    view_type = vpic_data_struct::FieldEdges;
  } else if( std::is_same<View, k_field_accum_t::HostMirror>::value ) {
    view_type = vpic_data_struct::FieldAccum;
  } else if( std::is_same<View, k_jf_accum_t::HostMirror>::value ) {
    view_type = vpic_data_struct::CurrentAccum;
  } else if( std::is_same<View, k_interpolator_t::HostMirror>::value ) {
    view_type = vpic_data_struct::Interpolators;
  } else if( std::is_same<View, k_accumulators_t::HostMirror>::value ) {
    view_type = vpic_data_struct::Accumulators;
  } else if( std::is_same<View, k_hydro_t::HostMirror>::value ) {
    view_type = vpic_data_struct::Hydro;
  } else if( std::is_same<View, k_particles_t::HostMirror>::value ) {
    view_type = vpic_data_struct::Particles;
  } else if( std::is_same<View, k_particles_i_t::HostMirror>::value ) {
    view_type = vpic_data_struct::ParticleCellID;
  } else if( std::is_same<View, k_particle_movers_t::HostMirror>::value ) {
    view_type = vpic_data_struct::ParticleMovers;
  } else if( std::is_same<View, k_particle_i_movers_t::HostMirror>::value ) {
    view_type = vpic_data_struct::ParticleMoverIDs;
  } else if( std::is_same<View, k_particle_partition_t::HostMirror>::value ) {
    view_type = vpic_data_struct::ParticlePartition;
  } else if( std::is_same<View, k_fluid_t::HostMirror>::value ) {
    view_type = vpic_data_struct::Fluid;
  } else if( std::is_same<View, k_neighbor_t::HostMirror>::value ) {
    view_type = vpic_data_struct::GridNeighbors;
  } else {
    view_type = vpic_data_struct::Other;
  }
  if( view_type != datastruct ) ERROR(( "incorrect checkpt View type (expected %zu, got %zu)", view_type, datastruct));

  using ValueType = typename View::non_const_value_type;
  size_t value_type_id = 32;
  if( std::is_same<ValueType, char>::value ) {
    value_type_id = 0;
  } else if( std::is_same<ValueType, uint8_t>::value ) {
    value_type_id = 1;
  } else if( std::is_same<ValueType, uint16_t>::value ) {
    value_type_id = 2;
  } else if( std::is_same<ValueType, uint32_t>::value ) {
    value_type_id = 3;
  } else if( std::is_same<ValueType, uint64_t>::value ) {
    value_type_id = 4;
  } else if( std::is_same<ValueType, int8_t>::value ) {
    value_type_id = 5;
  } else if( std::is_same<ValueType, int16_t>::value ) {
    value_type_id = 6;
  } else if( std::is_same<ValueType, int32_t>::value ) {
    value_type_id = 7;
  } else if( std::is_same<ValueType, int64_t>::value ) {
    value_type_id = 8;
//  } else if( std::is_same<ValueType, float16_t>::value ) {
//    value_type_id = 9;
  } else if( std::is_same<ValueType, float>::value ) {
    value_type_id = 10;
  } else if( std::is_same<ValueType, double>::value ) {
    value_type_id = 11;
  }
  if( value_type_id != type_id ) ERROR(( "incorrect checkpt View value type (expected %zu, got %zu)", value_type_id, type_id));

  if( std::is_same<typename View::array_layout, Kokkos::LayoutLeft>::value ) {
    if( layout != 0 ) ERROR(( "incorrect View layout (expected LayoutLeft, got LayoutRight)" ));
  } else if( std::is_same<typename View::array_layout, Kokkos::LayoutRight>::value ) {
    if( layout != 1 ) ERROR(( "incorrect View layout (expected LayoutRight, got LayoutLeft)" ));
  } else {
    ERROR(( "non standard View Layout" ));
  }

  if( rank != view.rank() ) ERROR(( "rank of supplied View did not match checkpt (%zu vs %zu)", view.rank(), rank ));
  typename View::array_layout array_layout;
  for( size_t i=0; i<rank; i++ ) 
    array_layout.dimension[i] = dims[i];

  // Resize View to fit data
  Kokkos::resize(view, array_layout);

  /* And read in the checkpointed elements */

  restore_raw( view.data(), sizeof(ValueType)*view.span() );
}

#endif /* _checkpt_h_ */
