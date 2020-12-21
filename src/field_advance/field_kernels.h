#ifndef _field_kernels_h_
#define _field_kernels_h_

// field_advance_kernels holds all the function pointers to all the
// kernels used by a specific field_advance instance.

// FIXME: DOCUMENT THESE INTERFACES HERE AND NOT IN STANDARD FIELD
// ADVANCE PRIVATE

struct field_array;

typedef struct field_advance_kernels {

  // FIXME: DUMP.CXX SHOULD BE DECENTRALIZED AND DIAGNOSTIC DUMP
  // FOR FIELDS SHOULD BE ADDED TO THIS
  // FIXME: FOR SYSTEMS WITH MAGNETIC CURRENTS (E.G. PML LAYERS)
  // WOULD INTERFACES FOR xif,kf BE USEFUL?

  void (*delete_fa)( struct field_array * RESTRICT fa );

  // Time stepping interface

  void (*advance_b)( struct field_array * RESTRICT fa, float frac );
  void (*advance_e)( struct field_array * RESTRICT fa, float frac );

  // Diagnostic interface
  // FIXME: MAY NEED MORE CAREFUL THOUGHT FOR CURVILINEAR SYSTEMS

  void (*energy_f)( /**/  double        * RESTRICT en, // 6 elem
                    const struct field_array * RESTRICT fa );

  // Accumulator interface

  void (*clear_jf       )( struct field_array * RESTRICT fa );
  void (*synchronize_jf )( struct field_array * RESTRICT fa );
  void (*clear_rhof     )( struct field_array * RESTRICT fa );
  void (*synchronize_rho)( struct field_array * RESTRICT fa );

  // Initialization interface

  void (*compute_rhob  )( struct field_array * RESTRICT fa );
  void (*compute_curl_b)( struct field_array * RESTRICT fa );

  // Local/remote shared face cleaning

  double (*synchronize_tang_e_norm_b)( struct field_array * RESTRICT fa );

  // Electric field divergence cleaning interface

  void   (*compute_div_e_err    )( /**/  struct field_array * RESTRICT fa );
  double (*compute_rms_div_e_err)( const struct field_array * RESTRICT fa );
  void   (*clean_div_e          )( /**/  struct field_array * RESTRICT fa );

  // Magnetic field divergence cleaning interface

  void   (*compute_div_b_err    )( /**/  struct field_array * RESTRICT fa );
  double (*compute_rms_div_b_err)( const struct field_array * RESTRICT fa );
  void   (*clean_div_b          )( /**/  struct field_array * RESTRICT fa );

} field_advance_kernels_t;

#endif
