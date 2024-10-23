#ifndef _fluid_advance_h_
#define _fluid_advance_h_

#include "../grid/grid.h"
//#include "../vpic/kokkos_helpers.h"

typedef int32_t fluid_species_id; // Must be 32-bit wide

// Local fluid data
typedef struct fluid {
  float den, tmp, prs;     // density, temperature, pressure
  float ux, uy, uz;        // velocity
} fluid_t;


struct fluid_species_t; // To-do: Should this be a class like species_t?

// fluid_advance_kernels holds all the function pointers to all the
// kernels used by a specific fluid_advance instance.
//typedef struct fluid_advance_kernels {

//  void (*delete_fl)( struct fluid_species * RESTRICT fla );

  // Time stepping interface
  //  void (*advance_fl)( struct fluid_array * RESTRICT fla );
  
  // Diagnostic interface
  //  void (*energy_fl)( /**/  double        * RESTRICT en, // 6 elem
  //                    const struct fluid_array * RESTRICT fla );

//} fluid_advance_kernels_t;

typedef struct fluid_species_t { // To-do: Should this be a class like species_t?

  char * name;                        // Species name of fluid
  float q;                            // Species charge
  float m;                            // Species mass
  
  fluid_t * ALIGNED(128) fl;          // Local fluid data
  grid_t  * g;                        // Underlying grid

  fluid_species_id id;                        // Unique identifier for a fluid species
  fluid_species_t* next = NULL;         // Next species in the fluid list

  
  //  void    * params;                   // Field advance specific parameters
  //  field_advance_kernels_t kernel[1];  // Field advance kernels

  /*
  // I don't want this to be a pointer, but given it only holds Kokkos data
  // this avoids a fiasco when checkpointing...
  field_buffers_t* fb;

  k_field_t k_f_d;                   // Kokkos field data on device
  k_field_t::HostMirror k_f_h;       // Kokkos field data on host
  k_field_sa_t k_field_sa_d;
  k_field_edge_t k_fe_d;             // Kokkos field_edge data (part of field_t) on device
  k_field_edge_t::HostMirror k_fe_h; // Kokkos field_edge data on host

  k_field_accum_t k_f_rhob_accum_d;//TODO: Remove when absorbing pbc on device
  k_field_accum_t::HostMirror k_f_rhob_accum_h;

  k_jf_accum_t k_jf_accum_d;
  k_jf_accum_t::HostMirror k_jf_accum_h;

  // Step when the field was last copied to to the host.  The copy can
  // take place at any time during the step, so checking
  // last_copied==step() does not mean that the host and device
  // data are the same.  Typically, copy is called immediately after the
  // step is incremented and before or during user_diagnostics.  Checking
  // last_copied==step() in these circumstances does mean the host
  // is up to date, unless you do unusual stuff in user_diagnostics.
  //
  // This number is tracked on the host only, and may be inaccurate on
  // the device.
  int64_t last_copied = -1;

  // Constructors don't get called on restart..
  // Initialize Kokkos Field Array
  field_array(int n_fields, int xyz_sz, int yzx_sz, int zxy_sz)
  {
      init_kokkos_fields(n_fields, xyz_sz, yzx_sz, zxy_sz);
  }

  void init_kokkos_fields(int n_fields, int xyz_sz, int yzx_sz, int zxy_sz)
  {
      k_f_d = k_field_t("k_fields", n_fields);
      k_field_sa_d = Kokkos::Experimental::create_scatter_view(k_f_d);
      k_fe_d = k_field_edge_t("k_field_edges", n_fields);
      k_f_h = Kokkos::create_mirror_view(k_f_d);
      k_fe_h = Kokkos::create_mirror_view(k_fe_d);

      k_f_rhob_accum_d = k_field_accum_t("k_rhob_accum", n_fields);
      k_f_rhob_accum_h = Kokkos::create_mirror_view(k_f_rhob_accum_d);

      k_jf_accum_d = k_jf_accum_t("k_jf_accum", n_fields);
      k_jf_accum_h = Kokkos::create_mirror_view(k_jf_accum_d);

      fb = new field_buffers_t(xyz_sz, yzx_sz, zxy_sz);
  }

  ~field_array()
  {
      delete fb;
  }
  */
  
  /**
   * @brief Copies the field data to the host.
   */
  //  void copy_to_host();

  /**
   * @brief Copies the field data to the device.
   */
  //  void copy_to_device();


} fluid_species_t;


// In fluid_advance.cc

int
num_fluid_species( const fluid_species_t * fsp_list );

void
delete_fluid_species_list( fluid_species_t * fsp_list );

fluid_species_t *
find_fluid_species_id( fluid_species_id id,
		       fluid_species_t * fsp_list );

fluid_species_t *
find_fluid_species_name( const char * name,
			 fluid_species_t * fsp_list );

fluid_species_t *
append_fluid_species( fluid_species_t * fsp,
		      fluid_species_t ** fsp_list ); 

fluid_species_t *
fluid_species( const char * name,
	       float q,
	       float m,
	       grid_t * g );


void
delete_fluid_species( fluid_species_t * fsp );


#endif // _fluid_advance_h_
