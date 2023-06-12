/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version (data structures based on earlier
 *                    V4PIC versions)
 *
 */

#ifndef _species_advance_h_
#define _species_advance_h_

#include <iostream>

#include "../sf_interface/sf_interface.h"
#include "Kokkos_DualView.hpp"

typedef int32_t species_id; // Must be 32-bit wide for particle_injector_t

// FIXME: Eventually particle_t (definitely) and ther other formats
// (maybe) should be opaque and specific to a particular
// species_advance implementation

typedef struct particle {
  float dx, dy, dz; // Particle position in cell coordinates (on [-1,1])
  int32_t i;        // Voxel containing the particle.  Note that
  /**/              // particled awaiting processing by boundary_p
  /**/              // have actually set this to 8*voxel + face where
  /**/              // face is the index of the face they interacted
  /**/              // with (on 0:5).  This limits the local number of
  /**/              // voxels to 2^28 but emitter handling already
  /**/              // has a stricter limit on this (2^26).
  float ux, uy, uz; // Particle normalized momentum
  float w;          // Particle weight (number of physical particles)
} particle_t;

// WARNING: FUNCTIONS THAT USE A PARTICLE_MOVER ASSUME THAT EVERYBODY
// WHO USES THAT PARTICLE MOVER WILL HAVE ACCESS TO PARTICLE ARRAY

typedef struct particle_mover {
  float dispx, dispy, dispz; // Displacement of particle
  int32_t i;                 // Index of the particle to move
} particle_mover_t;

// NOTE: THE LAYOUT OF A PARTICLE_INJECTOR _MUST_ BE COMPATIBLE WITH
// THE CONCATENATION OF A PARTICLE_T AND A PARTICLE_MOVER!

typedef struct particle_injector {
  float dx, dy, dz;          // Particle position in cell coords (on [-1,1])
  int32_t i;                 // Index of cell containing the particle
  float ux, uy, uz;          // Particle normalized momentum
  float w;                   // Particle weight (number of physical particles)
  float dispx, dispy, dispz; // Displacement of particle
  species_id sp_id;          // Species of particle
} particle_injector_t;

// Seems like this belongs in boundary.h
class species_t;
typedef struct pb_diagnostic {

    int         enable; // Wether or not to use this diagnostic
    int         enable_user; // Will the user write additional values?
    // TODO: Do I need this circular reference?
    species_t   *sp; // Pointer to the species
    char        *fname;
    int         file_counter; // How many files has this rank written?
    size_t      store_counter; // How many floats need to be written from the buffer
    size_t      write_counter; // How many floats have been written to the current file
    size_t      bufflen; // Size of the memory buffer in sizeof(float)
    float       *buff; // The buffer that stores data to be written

    int         num_user_writes; // Number of floats the user stores per particle
    int         num_writes; // Total number of floats stored per particle

    int         write_ux;
    int         write_uy;
    int         write_uz;
    int         write_momentum_magnitude;
    int         write_posx;
    int         write_posy;
    int         write_posz;
    int         write_weight;

} pb_diagnostic_t;

typedef struct annotation_vars {
  std::vector<std::string> i32_vars;
  std::vector<std::string> i64_vars;
  std::vector<std::string> f32_vars;
  std::vector<std::string> f64_vars;
 
  annotation_vars() {
  }

  annotation_vars(const annotation_vars& a) {
    i32_vars = a.i32_vars;
    i64_vars = a.i64_vars;
    f32_vars = a.f32_vars;
    f64_vars = a.f64_vars;
  }

  template<typename AnnotationType>
  int add_annotation(std::string name) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      i32_vars.push_back(name);
      return i32_vars.size()-1;
    } else if (std::is_same<AnnotationType, int64_t>::value) {
      i64_vars.push_back(name);
      return i64_vars.size()-1;
    } else if (std::is_same<AnnotationType, float>::value) {
      f32_vars.push_back(name);
      return f32_vars.size()-1;
    } else if (std::is_same<AnnotationType, double>::value) {
      f64_vars.push_back(name);
      return f64_vars.size()-1;
    } 
    return -1;
  }

  template<typename AnnotationType>
  int get_annotation_index(const std::string& name) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      for(uint32_t i=0; i<i32_vars.size(); i++) {
        if(name.compare(i32_vars[i]) == 0) {
          return i;
        }
      }
    } else if (std::is_same<AnnotationType, int64_t>::value) {
      for(uint32_t i=0; i<i64_vars.size(); i++) {
        if(name.compare(i64_vars[i]) == 0) {
          return i;
        }
      }
    } else if (std::is_same<AnnotationType, float>::value) {
      for(uint32_t i=0; i<f32_vars.size(); i++) {
        if(name.compare(f32_vars[i]) == 0) {
          return i;
        }
      }
    } else if (std::is_same<AnnotationType, double>::value) {
      for(uint32_t i=0; i<f64_vars.size(); i++) {
        if(name.compare(f64_vars[i]) == 0) {
          return i;
        }
      }
    } 
    return -1;
  }

  void combine(annotation_vars& a) {
    for(uint32_t i=0; i<a.i32_vars.size(); i++) {
      bool found = false;
      for(uint32_t j=0; j<i32_vars.size(); j++) {
        if(a.i32_vars[i].compare(i32_vars[j])) {
          found = true;
          break;
        }
      }
      if(!found) {
        add_annotation<int>(a.i32_vars[i]);
      }
    }
    for(uint32_t i=0; i<a.i64_vars.size(); i++) {
      bool found = false;
      for(uint32_t j=0; j<i64_vars.size(); j++) {
        if(a.i64_vars[i].compare(i64_vars[j])) {
          found = true;
          break;
        }
      }
      if(!found) {
        add_annotation<int64_t>(a.i64_vars[i]);
      }
    }
    for(uint32_t i=0; i<a.f32_vars.size(); i++) {
      bool found = false;
      for(uint32_t j=0; j<f32_vars.size(); j++) {
        if(a.f32_vars[i].compare(f32_vars[j])) {
          found = true;
          break;
        }
      }
      if(!found) {
        add_annotation<float>(a.f32_vars[i]);
      }
    }
    for(uint32_t i=0; i<a.f64_vars.size(); i++) {
      bool found = false;
      for(uint32_t j=0; j<f64_vars.size(); j++) {
        if(a.f64_vars[i].compare(f64_vars[j])) {
          found = true;
          break;
        }
      }
      if(!found) {
        add_annotation<double>(a.f64_vars[i]);
      }
    }
  }

} annotation_vars_t;

template<typename ExecSpace>
class annotations_t {
public:
  using memory_space = typename ExecSpace::memory_space;
  Kokkos::View<int**, Kokkos::LayoutLeft, memory_space> i32;
  Kokkos::View<int64_t**, Kokkos::LayoutLeft, memory_space> i64;
  Kokkos::View<float**, Kokkos::LayoutLeft, memory_space> f32;
  Kokkos::View<double**, Kokkos::LayoutLeft, memory_space> f64;

  annotations_t() {}

  annotations_t(int np, annotation_vars_t sizes) {
    i32 = Kokkos::View<int**, Kokkos::LayoutLeft, memory_space>("Int annotations", np, sizes.i32_vars.size());
    i64 = Kokkos::View<int64_t**, Kokkos::LayoutLeft, memory_space>("Int64 annotations", np, sizes.i64_vars.size());
    f32 = Kokkos::View<float**, Kokkos::LayoutLeft, memory_space>("Float annotations", np, sizes.f32_vars.size());
    f64 = Kokkos::View<double**, Kokkos::LayoutLeft, memory_space>("Double annotations", np, sizes.f64_vars.size());
    Kokkos::deep_copy(i32, 0);
    Kokkos::deep_copy(i64, 0);
    Kokkos::deep_copy(f32, 0.0);
    Kokkos::deep_copy(f64, 0.0);
  }

  annotations_t(annotations_t<Kokkos::DefaultExecutionSpace>& device_annotations) {
    i32 = Kokkos::create_mirror_view(device_annotations.i32);
    i64 = Kokkos::create_mirror_view(device_annotations.i64);
    f32 = Kokkos::create_mirror_view(device_annotations.f32);
    f64 = Kokkos::create_mirror_view(device_annotations.f64);
    Kokkos::deep_copy(i32, 0);
    Kokkos::deep_copy(i64, 0);
    Kokkos::deep_copy(f32, 0.0);
    Kokkos::deep_copy(f64, 0.0);
  }

  template<typename AnnotationType>
  KOKKOS_INLINE_FUNCTION
  void set(const int particle_index, const int var, AnnotationType val) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      i32(particle_index, var) = val;
    } else if (std::is_same<AnnotationType, int64_t>::value) {
      i64(particle_index, var) = val;
    } else if (std::is_same<AnnotationType, float>::value) {
      f32(particle_index, var) = val;
    } else if (std::is_same<AnnotationType, double>::value) {
      f64(particle_index, var) = val;
    } else {
      printf( "Tried setting value for non existent annotation!\n" );
    }
  }

  template<typename AnnotationType>
  KOKKOS_INLINE_FUNCTION
  AnnotationType get(const int particle_index, const int var) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      return i32(particle_index, var);
    } else if (std::is_same<AnnotationType, int64_t>::value) {
      return i64(particle_index, var);
    } else if (std::is_same<AnnotationType, float>::value) {
      return f32(particle_index, var);
    } else if (std::is_same<AnnotationType, double>::value) {
      return f64(particle_index, var);
    }
    printf( "Tried getting value for non existent annotation!\n" );
    return 0;
  }
};

enum class TracerType { Copy, Move };

class species_t {
    public:

        char * name;                        // Species name
        float q;                            // Species particle charge
        float m;                            // Species particle rest mass

        int np = 0, max_np = 0;             // Number and max local particles
        particle_t * ALIGNED(128) p;        // Array of particles for the species

        // TODO: these could be unsigned?
        int nm = 0, max_nm = 0;             // Number and max local movers in use

        particle_mover_t * ALIGNED(128) pm; // Particle movers

        int64_t last_sorted;                // Step when the particles were last
        // sorted.
        int sort_interval;                  // How often to sort the species
        int sort_out_of_place;              // Sort method
        int * ALIGNED(128) partition;       // Static array indexed 0:
        /**/                                // (nx+2)*(ny+2)*(nz+2).  Each value
        /**/                                // corresponds to the associated particle
        /**/                                // array index of the first particle in
        /**/                                // the cell.  Array is allocated and
        /**/                                // values computed in sort_p.  Purpose is
        /**/                                // for implementing collision models
        /**/                                // This is given in terms of the
        /**/                                // underlying's grids space filling
        /**/                                // curve indexing.  Thus, immediately
        /**/                                // after a sort:
        /**/                                //   sp->p[sp->partition[g->sfc[i]  ]:
        /**/                                //         sp->partition[g->sfc[i]+1]-1]
        /**/                                // are all the particles in voxel
        /**/                                // with local index i, while:
        /**/                                //   sp->p[ sp->partition[ j   ]:
        /**/                                //          sp->partition[ j+1 ] ]
        /**/                                // are all the particles in voxel
        /**/                                // with space filling curve index j.
        /**/                                // Note: SFC NOT IN USE RIGHT NOW THUS
        /**/                                // g->sfc[i]=i ABOVE.

        grid_t * g;                         // Underlying grid
        species_id id;                      // Unique identifier for a species
        species_t* next = NULL;             // Next species in the list

        // Particle boundary diagnostic.
        pb_diagnostic_t * pb_diag = NULL;

        // Tracer type
        TracerType tracer_type;
        species_t* parent_species = NULL;


        //// END CHECKPOINTED DATA, START KOKKOS //////


        k_particles_t k_p_d;                 // kokkos particles view on device
        k_particles_i_t k_p_i_d;             // kokkos particles view on device

        k_particles_t::HostMirror k_p_h;     // kokkos particles view on host
        k_particles_i_t::HostMirror k_p_i_h; // kokkos particles view on host

        k_particle_copy_t k_pc_d;            // kokkos particles copy for movers view on device
        k_particle_i_copy_t k_pc_i_d;        // kokkos particles copy for movers view on device

        k_particle_copy_t::HostMirror k_pc_h;      // kokkos particles copy for movers view on host
        k_particle_i_copy_t::HostMirror k_pc_i_h;  // kokkos particles i copy for movers view on host

        // Only need host versions
        k_particle_copy_t::HostMirror k_pr_h;      // kokkos particles copy for received particles
        k_particle_i_copy_t::HostMirror k_pr_i_h;  // kokkos particles i copy for received particles

        k_particle_movers_t k_pm_d;         // kokkos particle movers on device
        k_particle_i_movers_t k_pm_i_d;         // kokkos particle movers on device

        k_particle_movers_t::HostMirror k_pm_h;  // kokkos particle movers on host
        k_particle_i_movers_t::HostMirror k_pm_i_h;  // kokkos particle movers on host

        // TODO: what is an iterator here??
        k_counter_t k_nm_d;               // nm iterator
        k_counter_t::HostMirror k_nm_h;

#if defined(VPIC_ENABLE_PARTICLE_ANNOTATIONS) || defined(VPIC_ENABLE_TRACER_PARTICLES)
        bool is_tracer = false;
        bool using_annotations = false;
        annotation_vars_t annotation_vars;
        annotations_t<Kokkos::DefaultExecutionSpace>     annotations_d;
        annotations_t<Kokkos::DefaultHostExecutionSpace> annotations_h;
        annotations_t<Kokkos::DefaultExecutionSpace>     annotations_copy_d;
        annotations_t<Kokkos::DefaultHostExecutionSpace> annotations_copy_h;
        annotations_t<Kokkos::DefaultHostExecutionSpace> annotations_recv_h;

        int nparticles_buffered;
        k_particles_t::HostMirror                               particle_io_buffer;
        k_particles_i_t::HostMirror                             particle_cell_io_buffer;
        std::vector<std::pair<int64_t,int64_t>>                 np_per_ts_io_buffer;
        Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror efields_io_buffer;
        Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror bfields_io_buffer;
        Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror current_dens_io_buffer;
        Kokkos::View<float*>::HostMirror                        charge_dens_io_buffer;
        Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror momentum_dens_io_buffer;
        Kokkos::View<float*>::HostMirror                        ke_dens_io_buffer;
        Kokkos::View<float*[6], Kokkos::LayoutLeft>::HostMirror stress_tensor_io_buffer;
        annotations_t<Kokkos::DefaultHostExecutionSpace>        annotations_io_buffer;
#endif


        // TODO: this should ultimatley be removeable.
        // This tracks the number of particles we need to move back to the device
        // And is basically the same as nm at certain times?
        int num_to_copy = 0;

        // Step when the species was last copied to to the host.  The copy can
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

        // Static allocations for the compressor
        Kokkos::View<int*> unsafe_index;
        Kokkos::View<int> clean_up_to_count;
        Kokkos::View<int> clean_up_from_count;
        Kokkos::View<int>::HostMirror clean_up_from_count_h;
        Kokkos::View<int*> clean_up_from;
        Kokkos::View<int*> clean_up_to;

        // Init Kokkos Particle Arrays
        species_t(int n_particles, int n_pmovers)
        {
           init_kokkos_particles(n_particles, n_pmovers);
        }

        void init_kokkos_particles()
        {
            init_kokkos_particles(max_np, max_nm);
        }

        void init_kokkos_particles(int n_particles, int n_pmovers)
        {
            k_p_d = k_particles_t("k_particles", n_particles);
            k_p_i_d = k_particles_i_t("k_particles_i", n_particles);
            k_pc_d = k_particle_copy_t("k_particle_copy_for_movers", n_pmovers);
            k_pc_i_d = k_particle_i_copy_t("k_particle_copy_for_movers_i", n_pmovers);
            k_pr_h = k_particle_copy_t::HostMirror("k_particle_send_for_movers", n_pmovers);
            k_pr_i_h = k_particle_i_copy_t::HostMirror("k_particle_send_for_movers_i", n_pmovers);
            k_pm_d = k_particle_movers_t("k_particle_movers", n_pmovers);
            k_pm_i_d = k_particle_i_movers_t("k_particle_movers_i", n_pmovers);
            k_nm_d = k_counter_t("k_nm"); // size 1 encoded in type
            unsafe_index = Kokkos::View<int*>("safe index", 2*n_pmovers);
            clean_up_to_count = Kokkos::View<int>("clean up to count");
            clean_up_from_count = Kokkos::View<int>("clean up from count");
            clean_up_from = Kokkos::View<int*>("clean up from", n_pmovers);
            clean_up_to = Kokkos::View<int*>("clean up to", n_pmovers);

            k_p_h = Kokkos::create_mirror_view(k_p_d);
            k_p_i_h = Kokkos::create_mirror_view(k_p_i_d);

            k_pc_h = Kokkos::create_mirror_view(k_pc_d);
            k_pc_i_h = Kokkos::create_mirror_view(k_pc_i_d);

            k_pm_h = Kokkos::create_mirror_view(k_pm_d);
            k_pm_i_h = Kokkos::create_mirror_view(k_pm_i_d);

            k_nm_h = Kokkos::create_mirror_view(k_nm_d);

            clean_up_from_count_h = Kokkos::create_mirror_view(clean_up_from_count);
        }

        /**
         * @brief Copies all the outbound particles and movers to the host.
         */
        void copy_outbound_to_host();

        /**
         * @brief Copies all the particles and movers from the device to the host.
         */
        void copy_to_host();

        /**
         * @brief Copies all the particles and movers from the host to the device.
         */
        void copy_to_device();

        /**
         * @brief Copies all the inbound particles from the host to the device.
         */
        void copy_inbound_to_device();

#if defined(VPIC_ENABLE_PARTICLE_ANNOTATIONS) || defined(VPIC_ENABLE_TRACER_PARTICLES)
        /**
         *  @brief Allocate memory for IO buffering tracers
         *
         *  @param N_particles       Number of particles to buffer before dumping
         *  @param over_alloc_factor Multiplier for over allocating space
         */
        void init_io_buffers(const int N_particles, const float over_alloc_factor);

        /**
         * @brief Add additional per particle annotations. 
         *
         * @param num_particles Number of particles using annotations
         * @param num_movers    Number of movers that need annotations
         * @param vars          Annotation variables
         */
        void init_annotations( int num_particles, int num_movers, annotation_vars_t& vars );

        /**
         * Create tracer particles from parent species using a predicate
         *
         * @param parent_species Species of particles to create tracers from
         * @param tracer_type    Whether to Copy or Move particles to tracers
         * @param filter         Generic function to decide whether particle is a tracer
         */
        void create_tracers_by_predicate( species_t* parent_species,
                                          const TracerType tracer_type,
                                          std::function <bool (particle_t)> filter, int rank ) {
          // Make sure parent species particles are in Kokkos Views
          auto& k_particle_h = parent_species->k_p_h;
          auto& k_particle_i_h = parent_species->k_p_i_h;
          auto& particles = parent_species->p;

          Kokkos::parallel_for("copy particles to device",
            host_execution_policy(0, parent_species->np) ,
            KOKKOS_LAMBDA (int i) {

              k_particle_h(i, particle_var::dx) = particles[i].dx;
              k_particle_h(i, particle_var::dy) = particles[i].dy;
              k_particle_h(i, particle_var::dz) = particles[i].dz;
              k_particle_h(i, particle_var::ux) = particles[i].ux;
              k_particle_h(i, particle_var::uy) = particles[i].uy;
              k_particle_h(i, particle_var::uz) = particles[i].uz;
              k_particle_h(i, particle_var::w)  = particles[i].w;
              k_particle_i_h(i) = particles[i].i;

            });
          Kokkos::fence();

          for(int i=0; i<parent_species->np; i++) {
            if(filter(parent_species->p[i])) { // Check if particle passes filter
              k_p_h(np, particle_var::dx) = parent_species->k_p_h(i, particle_var::dx); // Copy particle to tracer
              k_p_h(np, particle_var::dy) = parent_species->k_p_h(i, particle_var::dy); 
              k_p_h(np, particle_var::dz) = parent_species->k_p_h(i, particle_var::dz); 
              k_p_h(np, particle_var::ux) = parent_species->k_p_h(i, particle_var::ux); 
              k_p_h(np, particle_var::uy) = parent_species->k_p_h(i, particle_var::uy); 
              k_p_h(np, particle_var::uz) = parent_species->k_p_h(i, particle_var::uz); 
              k_p_h(np, particle_var::w)  = parent_species->k_p_h(i, particle_var::w); 
              k_p_i_h(np)                 = parent_species->k_p_i_h(i); 
              p[np] = parent_species->p[i]; // Copy legacy particle FIXME Should be unnecessary

              //TODO Test copying the rest of the annotations
              for(uint32_t j=0; j<parent_species->annotation_vars.i32_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<int>(parent_species->annotation_vars.i32_vars[j]);
                annotations_h.set<int>(np, k, parent_species->annotations_h.get<int>(i, j));
              }
              for(uint32_t j=0; j<parent_species->annotation_vars.i64_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<int64_t>(parent_species->annotation_vars.i64_vars[j]);
                annotations_h.set<int64_t>(np, k, parent_species->annotations_h.get<int64_t>(i, j));
              }
              for(uint32_t j=0; j<parent_species->annotation_vars.f32_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<float>(parent_species->annotation_vars.f32_vars[j]);
                annotations_h.set<float>(np, k, parent_species->annotations_h.get<float>(i, j));
              }
              for(uint32_t j=0; j<parent_species->annotation_vars.f64_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<double>(parent_species->annotation_vars.f64_vars[j]);
                annotations_h.set<double>(np, k, parent_species->annotations_h.get<double>(i, j));
              }

              // Create unique tracer id (32-bit rank concatenated with particle index)
              int tracer_idx = annotation_vars.get_annotation_index<int64_t>(std::string("TracerID"));
              annotations_h.set<int64_t>(np, tracer_idx, ((int64_t)(rank) << 32) | (int64_t)(np)); 
              if(tracer_type == TracerType::Copy) {
                int w_idx = annotation_vars.get_annotation_index<float>(std::string("Weight")); // Get weight annotation index
                annotations_h.set<float>(np, w_idx, k_p_h(np, particle_var::w)); // Save weight
                k_p_h(np, particle_var::w) = 0.0f; // Set tracer weight to 0 so the particle is non interactive
              } else if(tracer_type == TracerType::Move) {
                // Move last particle over to fill in gap
                parent_species->k_p_h(i, particle_var::dx) = parent_species->k_p_h(parent_species->np-1, particle_var::dx); 
                parent_species->k_p_h(i, particle_var::dy) = parent_species->k_p_h(parent_species->np-1, particle_var::dy); 
                parent_species->k_p_h(i, particle_var::dz) = parent_species->k_p_h(parent_species->np-1, particle_var::dz); 
                parent_species->k_p_h(i, particle_var::ux) = parent_species->k_p_h(parent_species->np-1, particle_var::ux); 
                parent_species->k_p_h(i, particle_var::uy) = parent_species->k_p_h(parent_species->np-1, particle_var::uy); 
                parent_species->k_p_h(i, particle_var::uz) = parent_species->k_p_h(parent_species->np-1, particle_var::uz); 
                parent_species->k_p_h(i, particle_var::w)  = parent_species->k_p_h(parent_species->np-1, particle_var::w); 
                parent_species->k_p_i_h(i)                 = parent_species->k_p_i_h(parent_species->np - 1); 
                parent_species->p[i] = parent_species->p[parent_species->np-1]; // FIXME remove legacy particles
                parent_species->np -= 1; // Decrease number of particles in parent species
                i -= 1;
              } else {
                ERROR(( "Invalid TracerType: %d", tracer_type ));
              } 
              np++; // Increase number of tracers
            }
          }
        }

        /**
         * Create tracer particles from parent species. Select every Nth particle
         *
         * @param parent_species Species of particles to create tracers from
         * @param tracer_type    Whether to Copy or Move particles to tracers
         * @param skip           Amount of particles to skip between selections
         */
        void create_tracers_by_nth( species_t* parent_species,
                                    const TracerType tracer_type,
                                    float skip, int rank) {

          // Make sure parent species particles are in Kokkos Views
          auto& k_particle_h = parent_species->k_p_h;
          auto& k_particle_i_h = parent_species->k_p_i_h;
          auto& particles = parent_species->p;

          Kokkos::parallel_for("copy particles to device",
            host_execution_policy(0, parent_species->np) ,
            KOKKOS_LAMBDA (int i) {

              k_particle_h(i, particle_var::dx) = particles[i].dx;
              k_particle_h(i, particle_var::dy) = particles[i].dy;
              k_particle_h(i, particle_var::dz) = particles[i].dz;
              k_particle_h(i, particle_var::ux) = particles[i].ux;
              k_particle_h(i, particle_var::uy) = particles[i].uy;
              k_particle_h(i, particle_var::uz) = particles[i].uz;
              k_particle_h(i, particle_var::w)  = particles[i].w;
              k_particle_i_h(i) = particles[i].i;

            });
          Kokkos::fence();

          int parent_np = parent_species->np;
          int step = 0;
          np = 0;
          for(int i=0; i<parent_np; i++) {
            if(i >= skip*np) { // Check if particle passes filter
              k_p_h(np, particle_var::dx) = parent_species->k_p_h(step, particle_var::dx); // Copy particle to tracer
              k_p_h(np, particle_var::dy) = parent_species->k_p_h(step, particle_var::dy); 
              k_p_h(np, particle_var::dz) = parent_species->k_p_h(step, particle_var::dz); 
              k_p_h(np, particle_var::ux) = parent_species->k_p_h(step, particle_var::ux); 
              k_p_h(np, particle_var::uy) = parent_species->k_p_h(step, particle_var::uy); 
              k_p_h(np, particle_var::uz) = parent_species->k_p_h(step, particle_var::uz); 
              k_p_h(np, particle_var::w)  = parent_species->k_p_h(step, particle_var::w); 
              k_p_i_h(np) = parent_species->k_p_i_h(step); 
              p[np] = parent_species->p[step]; // Copy legacy particle FIXME Should be unnecessary
              // Create unique tracer id (32-bit rank concatenated with particle index)
              int tracer_idx = annotation_vars.get_annotation_index<int64_t>(std::string("TracerID"));
              annotations_h.set<int64_t>(np, tracer_idx, ((int64_t)(rank) << 32) | (int64_t)(np)); 
              //TODO Test copying the rest of the annotations
              for(uint32_t j=0; j<parent_species->annotation_vars.i32_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<int>(parent_species->annotation_vars.i32_vars[j]);
                annotations_h.set<int>(np, k, parent_species->annotations_h.get<int>(i, j));
              }
              for(uint32_t j=0; j<parent_species->annotation_vars.i64_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<int64_t>(parent_species->annotation_vars.i64_vars[j]);
                annotations_h.set<int64_t>(np, k, parent_species->annotations_h.get<int64_t>(i, j));
              }
              for(uint32_t j=0; j<parent_species->annotation_vars.f32_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<float>(parent_species->annotation_vars.f32_vars[j]);
                annotations_h.set<float>(np, k, parent_species->annotations_h.get<float>(i, j));
              }
              for(uint32_t j=0; j<parent_species->annotation_vars.f64_vars.size(); j++) {
                int k = annotation_vars.get_annotation_index<double>(parent_species->annotation_vars.f64_vars[j]);
                annotations_h.set<double>(np, k, parent_species->annotations_h.get<double>(i, j));
              }

              if(tracer_type == TracerType::Copy) {
                int w_idx = annotation_vars.get_annotation_index<float>(std::string("Weight")); // Get weight annotation index
                annotations_h.set<float>(np, w_idx, k_p_h(np, particle_var::w)); // Save weight
                k_p_h(np, particle_var::w) = 0.0f; // Set tracer weight to 0 so the particle is non interactive
                p[np].w = 0.0f;
              } else if(tracer_type == TracerType::Move) {
                // Move last particle over to fill in gap
                parent_species->k_p_h(step, particle_var::dx) = parent_species->k_p_h(parent_species->np-1, particle_var::dx); 
                parent_species->k_p_h(step, particle_var::dy) = parent_species->k_p_h(parent_species->np-1, particle_var::dy); 
                parent_species->k_p_h(step, particle_var::dz) = parent_species->k_p_h(parent_species->np-1, particle_var::dz); 
                parent_species->k_p_h(step, particle_var::ux) = parent_species->k_p_h(parent_species->np-1, particle_var::ux); 
                parent_species->k_p_h(step, particle_var::uy) = parent_species->k_p_h(parent_species->np-1, particle_var::uy); 
                parent_species->k_p_h(step, particle_var::uz) = parent_species->k_p_h(parent_species->np-1, particle_var::uz); 
                parent_species->k_p_h(step, particle_var::w)  = parent_species->k_p_h(parent_species->np-1, particle_var::w); 
                parent_species->k_p_i_h(step)                 = parent_species->k_p_i_h(parent_species->np - 1); 
                parent_species->p[step] = parent_species->p[parent_species->np-1]; // FIXME remove legacy particles
                parent_species->np -= 1; // Decrease number of particles in parent species
                step -= 1;
              } else {
                ERROR(( "Invalid TracerType: %d", tracer_type ));
              } 
              np++; // Increase number of tracers
            }
            step++;
          }
        }
#endif
};

// In species_advance.c

int
num_species( const species_t * sp_list );

void
delete_species_list( species_t * sp_list );

species_t *
find_species_id( species_id id,
                 species_t * sp_list );

species_t *
find_species_name( const char * name,
                   species_t * sp_list );

species_t *
append_species( species_t * sp,
                species_t ** sp_list );

species_t *
species( const char * name,
         float q,
         float m,
         int max_local_np,
         int max_local_nm,
         int sort_interval,
         int sort_out_of_place,
         grid_t * g );

// FIXME: TEMPORARY HACK UNTIL THIS SPECIES_ADVANCE KERNELS
// CAN BE CONSTRUCTED ANALOGOUS TO THE FIELD_ADVANCE KERNELS
// (THESE FUNCTIONS ARE NECESSARY FOR HIGHER LEVEL CODE)

// In sort_p.c

void
sort_p( species_t * RESTRICT sp );

// In advance_p.cxx

void
advance_p( /**/  species_t            * RESTRICT sp,
                 interpolator_array_t * RESTRICT ia,
                 field_array_t* RESTRICT fa );

// In center_p.cxx

// This does a half advance field advance and a half Boris rotate on
// the particles.  As such particles with r at the time step and u
// half a step stale is moved second order accurate to have r and u on
// the time step.

void
center_p( /**/  species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia );

// In uncenter_p.cxx

// This is the inverse of center_p.  Thus, particles with r and u at
// the time step are adjusted to have r at the time step and u half a
// step stale.

void
uncenter_p( /**/  species_t            * RESTRICT sp,
            const interpolator_array_t * RESTRICT ia );

// In energy.cxx

// This computes the kinetic energy stored in the particles.  The
// calculation is done numerically robustly.  All nodes get the same
// result.

double
energy_p( const species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia );

double
energy_p_kokkos( const species_t            * RESTRICT sp,
          const interpolator_array_t * RESTRICT ia );

// In rho_p.cxx

void
accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp );

void
accumulate_rhob( field_t          * RESTRICT ALIGNED(128) f,
                 const particle_t * RESTRICT ALIGNED(32)  p,
                 const grid_t     * RESTRICT              g,
                 const float                              qsp );
void
k_accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp );

void k_accumulate_rhob(
            k_field_t& kfield,
            k_particles_t& kpart,
            k_particle_movers_t& kpart_movers,
            const grid_t* RESTRICT g,
            const float qsp,
            const int nm);

void k_accumulate_rhob_single_cpu(
            k_field_t& kfield,
            k_particles_t& kpart,
            k_particles_i_t& kpart_i,
            const int i,
            const grid_t* g,
            const float qsp
);

// In hydro_p.c

void
accumulate_hydro_p( /**/  hydro_array_t        * RESTRICT ha,
                    const species_t            * RESTRICT sp,
                    const interpolator_array_t * RESTRICT ia );

void accumulate_hydro_p_kokkos(
        k_particles_t& k_particles,
        k_particles_i_t& k_particles_i,
        k_hydro_d_t k_hydro,
        k_interpolator_t& k_interp,
        const species_t            * RESTRICT sp
);

// In move_p.cxx
int
move_p( particle_t       * ALIGNED(128) p0,
        particle_mover_t * ALIGNED(16)  pm,
        //accumulator_t    * ALIGNED(128) a0,
        k_jf_accum_t::HostMirror& k_jf_accum,
        const grid_t     *              g,
        const float                     qsp );

template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, class scatter_view_t>
int
KOKKOS_INLINE_FUNCTION
move_p_kokkos(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16)  pm,
    //accumulator_sa_t k_accumulators_sa,
    scatter_view_t scatter_view,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp,
    //field_array_t* RESTRICT fa,
    //field_view_t& k_field,
    float cx,
    float cy,
    float cz,
    const int nx,
    const int ny,
    const int nz
)
{

  #define p_dx    k_particles(pi, particle_var::dx)
  #define p_dy    k_particles(pi, particle_var::dy)
  #define p_dz    k_particles(pi, particle_var::dz)
  #define p_ux    k_particles(pi, particle_var::ux)
  #define p_uy    k_particles(pi, particle_var::uy)
  #define p_uz    k_particles(pi, particle_var::uz)
  #define p_w     k_particles(pi, particle_var::w)
  #define pii     k_particles_i(pi)

  //#define local_pm_dispx  k_local_particle_movers(0, particle_mover_var::dispx)
  //#define local_pm_dispy  k_local_particle_movers(0, particle_mover_var::dispy)
  //#define local_pm_dispz  k_local_particle_movers(0, particle_mover_var::dispz)
  //#define local_pm_i      k_local_particle_movers(0, particle_mover_var::pmi)


  //k_field_t& k_field = fa->k_f_d;
  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  //int pi = int(local_pm_i);
  int pi = pm->i;
//  auto  k_field_scatter_access = k_f_sa.access();
//  auto accum_sa = accum_sv.access();
  auto scatter_access = scatter_view.access();

  q = qsp*p_w;

    //printf("in move %d \n", pi);

  for(;;) {
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;


    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    //printf("pre axis %d x %e y %e z %e \n", axis, p_dx, p_dy, p_dz);

    //printf("disp x %e y %e z %e \n", s_dispx, s_dispy, s_dispz);

    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      v3=2,  axis=3;
    if(v0<v3) v3=v0, axis=0;
    if(v1<v3) v3=v1, axis=1;
    if(v2<v3) v3=v2, axis=2;
    v3 *= 0.5;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);

    //a = (float *)(&d_accumulators[ci]);

#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \

    //Kokkos::atomic_add(&a[0], v0);
    //Kokkos::atomic_add(&a[1], v1);
    //Kokkos::atomic_add(&a[2], v2);
    //Kokkos::atomic_add(&a[3], v3);

    if (std::is_same<scatter_view_t,k_field_sa_t>::value) {
      int iii = ii;
      int zi = iii/((nx+2)*(ny+2));
      iii -= zi*(nx+2)*(ny+2);
      int yi = iii/(nx+2);
      int xi = iii-yi*(nx+2);
      accumulate_j(x,y,z);
      scatter_access(ii, field_var::jfx) += cx*v0;
      scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx) += cx*v1;
      scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx) += cx*v2;
      scatter_access(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;

      accumulate_j(y,z,x);
      scatter_access(ii, field_var::jfy) += cy*v0;
      scatter_access(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v1;
      scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy) += cy*v2;
      scatter_access(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v3;

      accumulate_j(z,x,y);
      scatter_access(ii, field_var::jfz) += cz*v0;
      scatter_access(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz) += cz*v1;
      scatter_access(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v2;
      scatter_access(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v3;
    } else {
      accumulate_j(x,y,z);
      scatter_access(ii, 0) += cx*v0;
      scatter_access(ii, 1) += cx*v1;
      scatter_access(ii, 2) += cx*v2;
      scatter_access(ii, 3) += cx*v3;

      accumulate_j(y,z,x);
      scatter_access(ii, 4) += cy*v0;
      scatter_access(ii, 5) += cy*v1;
      scatter_access(ii, 6) += cy*v2;
      scatter_access(ii, 7) += cy*v3;

      accumulate_j(z,x,y);
      scatter_access(ii, 8) += cz*v0;
      scatter_access(ii, 9) += cz*v1;
      scatter_access(ii, 10) += cz*v2;
      scatter_access(ii, 11) += cz*v3;
    }

#   undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
    neighbor = d_neighbor( 6*ii + face );

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back


    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      k_particles(pi, particle_var::ux + axis) = -k_particles(pi, particle_var::ux + axis);

      // TODO: make this safer
      //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      //k_local_particle_movers(0, particle_mover_var::dispx + axis) = -k_local_particle_movers(0, particle_mover_var::dispx + axis);
      // TODO: replace this, it's horrible
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];


      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      pii = 8*pii + face;
      return 1; // Return "mover still in use"
      }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    pii = neighbor - rangel;
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    k_particles(pi, particle_var::dx + axis) = -v0;      // Convert coordinate system
  }
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
  #undef pii

  //#undef local_pm_dispx
  //#undef local_pm_dispy
  //#undef local_pm_dispz
  //#undef local_pm_i
  return 0; // Return "mover not in use"
}

// this has no data race protection for write into the accumulators
template<class particle_view_t, class particle_i_view_t, class neighbor_view_t, class accum_view_t>
int
move_p_kokkos_host_serial(
    const particle_view_t& k_particles,
    const particle_i_view_t& k_particles_i,
    particle_mover_t* ALIGNED(16) pm,
    accum_view_t& k_jf_accum,
    const grid_t* g,
    neighbor_view_t& d_neighbor,
    int64_t rangel,
    int64_t rangeh,
    const float qsp
)
{
  const int nx = g->nx;
  const int ny = g->ny;
  //const int nz = g->nz;
  float cx = 0.25 * g->rdy * g->rdz / g->dt;
  float cy = 0.25 * g->rdz * g->rdx / g->dt;
  float cz = 0.25 * g->rdx * g->rdy / g->dt;

  #define p_dx    k_particles(pi, particle_var::dx)
  #define p_dy    k_particles(pi, particle_var::dy)
  #define p_dz    k_particles(pi, particle_var::dz)
  #define p_ux    k_particles(pi, particle_var::ux)
  #define p_uy    k_particles(pi, particle_var::uy)
  #define p_uz    k_particles(pi, particle_var::uz)
  #define p_w     k_particles(pi, particle_var::w)
  #define pii     k_particles_i(pi)

  //#define local_pm_dispx  k_local_particle_movers(0, particle_mover_var::dispx)
  //#define local_pm_dispy  k_local_particle_movers(0, particle_mover_var::dispy)
  //#define local_pm_dispz  k_local_particle_movers(0, particle_mover_var::dispz)
  //#define local_pm_i      k_local_particle_movers(0, particle_mover_var::pmi)


  float s_midx, s_midy, s_midz;
  float s_dispx, s_dispy, s_dispz;
  float s_dir[3];
  float v0, v1, v2, v3, v4, v5, q;
  int axis, face;
  int64_t neighbor;
  //int pi = int(local_pm_i);
  int pi = pm->i;

  q = qsp*p_w;

    //printf("in move %d \n", pi);

  for(;;) {
    int ii = pii;
    s_midx = p_dx;
    s_midy = p_dy;
    s_midz = p_dz;


    s_dispx = pm->dispx;
    s_dispy = pm->dispy;
    s_dispz = pm->dispz;

    //printf("pre axis %d x %e y %e z %e \n", axis, p_dx, p_dy, p_dz);

    //printf("disp x %e y %e z %e \n", s_dispx, s_dispy, s_dispz);

    s_dir[0] = (s_dispx>0) ? 1 : -1;
    s_dir[1] = (s_dispy>0) ? 1 : -1;
    s_dir[2] = (s_dispz>0) ? 1 : -1;

    // Compute the twice the fractional distance to each potential
    // streak/cell face intersection.
    v0 = (s_dispx==0) ? 3.4e38f : (s_dir[0]-s_midx)/s_dispx;
    v1 = (s_dispy==0) ? 3.4e38f : (s_dir[1]-s_midy)/s_dispy;
    v2 = (s_dispz==0) ? 3.4e38f : (s_dir[2]-s_midz)/s_dispz;

    // Determine the fractional length and axis of current streak. The
    // streak ends on either the first face intersected by the
    // particle track or at the end of the particle track.
    //
    //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
    //   axis 3        ... streak ends at end of the particle track
    /**/      v3=2,  axis=3;
    if(v0<v3) v3=v0, axis=0;
    if(v1<v3) v3=v1, axis=1;
    if(v2<v3) v3=v2, axis=2;
    v3 *= 0.5;

    // Compute the midpoint and the normalized displacement of the streak
    s_dispx *= v3;
    s_dispy *= v3;
    s_dispz *= v3;
    s_midx += s_dispx;
    s_midy += s_dispy;
    s_midz += s_dispz;

    // Accumulate the streak.  Note: accumulator values are 4 times
    // the total physical charge that passed through the appropriate
    // current quadrant in a time-step
    v5 = q*s_dispx*s_dispy*s_dispz*(1.f/3.f);

    //a = (float *)(&d_accumulators[ci]);

#   define accumulate_j(X,Y,Z)                                        \
    v4  = q*s_disp##X;    /* v2 = q ux                            */  \
    v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
    v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
    v1 += v4;             /* v1 = q ux (1+dy)                     */  \
    v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
    v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
    v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
    v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
    v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
    v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
    v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
    v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
    v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
    v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */ 

    int iii = ii;
    int zi = iii/((nx+2)*(ny+2));
    iii -= zi*(nx+2)*(ny+2);
    int yi = iii/(nx+2);
    int xi = iii - yi*(nx+2);
    accumulate_j(x,y,z);
    k_jf_accum(ii, accumulator_var::jx) += cx*v0;
    k_jf_accum(VOXEL(xi,yi+1,zi,nx,ny,nz), accumulator_var::jx) += cx*v1;
    k_jf_accum(VOXEL(xi,yi,zi+1,nx,ny,nz), accumulator_var::jx) += cx*v2;
    k_jf_accum(VOXEL(xi,yi+1,zi+1,nx,ny,nz), accumulator_var::jx) += cx*v3;

    accumulate_j(y,z,x);
    k_jf_accum(ii, accumulator_var::jy) += cy*v0;
    k_jf_accum(VOXEL(xi,yi,zi+1,nx,ny,nz), accumulator_var::jy) += cy*v1;
    k_jf_accum(VOXEL(xi+1,yi,zi,nx,ny,nz), accumulator_var::jy) += cy*v2;
    k_jf_accum(VOXEL(xi+1,yi,zi+1,nx,ny,nz), accumulator_var::jy) += cy*v3;

    accumulate_j(z,x,y);
    k_jf_accum(ii, accumulator_var::jz) += cz*v0;
    k_jf_accum(VOXEL(xi+1,yi,zi,nx,ny,nz), accumulator_var::jz) += cz*v1;
    k_jf_accum(VOXEL(xi,yi+1,zi,nx,ny,nz), accumulator_var::jz) += cz*v2;
    k_jf_accum(VOXEL(xi+1,yi+1,zi,nx,ny,nz), accumulator_var::jz) += cz*v3;

#   undef accumulate_j

    // Compute the remaining particle displacment
    pm->dispx -= s_dispx;
    pm->dispy -= s_dispy;
    pm->dispz -= s_dispz;

    //printf("pre axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);
    // Compute the new particle offset
    p_dx += s_dispx+s_dispx;
    p_dy += s_dispy+s_dispy;
    p_dz += s_dispz+s_dispz;

    // If an end streak, return success (should be ~50% of the time)
    //printf("axis %d x %e y %e z %e disp x %e y %e z %e\n", axis, p_dx, p_dy, p_dz, s_dispx, s_dispy, s_dispz);

    if( axis==3 ) break;

    // Determine if the particle crossed into a local cell or if it
    // hit a boundary and convert the coordinate system accordingly.
    // Note: Crossing into a local cell should happen ~50% of the
    // time; hitting a boundary is usually a rare event.  Note: the
    // entry / exit coordinate for the particle is guaranteed to be
    // +/-1 _exactly_ for the particle.

    v0 = s_dir[axis];
    k_particles(pi, particle_var::dx + axis) = v0; // Avoid roundoff fiascos--put the particle
                           // _exactly_ on the boundary.
    face = axis; if( v0>0 ) face += 3;

    // TODO: clean this fixed index to an enum
    //neighbor = g->neighbor[ 6*ii + face ];
    neighbor = d_neighbor( 6*ii + face );

    // TODO: these two if statements used to be marked UNLIKELY,
    // but that intrinsic doesn't work on GPU.
    // for performance portability, maybe specialize UNLIKELY
    // for CUDA mode and put it back


    if( neighbor==reflect_particles ) {
      // Hit a reflecting boundary condition.  Reflect the particle
      // momentum and remaining displacement and keep moving the
      // particle.
      k_particles(pi, particle_var::ux + axis) = -k_particles(pi, particle_var::ux + axis);

      // TODO: make this safer
      //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis];
      //k_local_particle_movers(0, particle_mover_var::dispx + axis) = -k_local_particle_movers(0, particle_mover_var::dispx + axis);
      // TODO: replace this, it's horrible
      (&(pm->dispx))[axis] = -(&(pm->dispx))[axis];


      continue;
    }

    if( neighbor<rangel || neighbor>rangeh ) {
      // Cannot handle the boundary condition here.  Save the updated
      // particle position, face it hit and update the remaining
      // displacement in the particle mover.
      pii = 8*pii + face;
      return 1; // Return "mover still in use"
      }

    // Crossed into a normal voxel.  Update the voxel index, convert the
    // particle coordinate system and keep moving the particle.

    pii = neighbor - rangel;
    /**/                         // Note: neighbor - rangel < 2^31 / 6
    k_particles(pi, particle_var::dx + axis) = -v0;      // Convert coordinate system
  }
  #undef p_dx
  #undef p_dy
  #undef p_dz
  #undef p_ux
  #undef p_uy
  #undef p_uz
  #undef p_w
  #undef pii

  //#undef local_pm_dispx
  //#undef local_pm_dispy
  //#undef local_pm_dispz
  //#undef local_pm_i
  return 0; // Return "mover not in use"
}

// TODO: this bascially duplicates funcitonality in rho_p.cc and should be DRY'd
template<typename kf_t, typename kp_t, typename kpi_t> // k_field_t, k_particles_t, k_particles_i_t
void k_accumulate_rhob_single_cpu(
        kf_t& k_rhob_accum,
        kp_t& kpart,
        kpi_t& kpart_i,
        const int i,
        const grid_t* g,
        const float qsp
)
{
    // Extract grid vars
    const float r8V = g->r8V;
    const int nx = g->nx;
    const int ny = g->ny;
    const int nz = g->nz;
    const int sy = g->sy;
    const int sz = g->sz;

    // Kernel
    //float w0 = p->dx, w1 = p->dy, w2, w3, w4, w5, w6, w7, dz = p->dz;
    //int v = p->i, x, y, z, sy = g->sy, sz = g->sz;
    //w7 = (qsp*g->r8V)*p->w;
    float w0 = kpart(i, particle_var::dx);
    float w1 = kpart(i, particle_var::dy);
    float w7 = (qsp * r8V) * kpart(i, particle_var::w);
    float dz = kpart(i, particle_var::dz);
    int v = kpart_i(i);
    //printf("\n Vars are %g, %g, %g %g\n", w0, w1, w7, dz);

    float w6 = w7 - w0 * w7;
    w7 = w7 + w0 * w7;
    float w4 = w6 - w1 * w6;
    float w5 = w7 - w1 * w7;
    w6 = w6 + w1 * w6;
    w7 = w7 + w1 * w7;
    w0 = w4 - dz * w4;
    w1 = w5 - dz * w5;
    float w2 = w6 - dz * w6;
    float w3 = w7 - dz * w7;
    w4 = w4 + dz * w4;
    w5 = w5 + dz * w5;
    w6 = w6 + dz * w6;
    w7 = w7 + dz * w7;

    int x = v;
    int z = x/sz;
    if(z == 1) {
        w0 += w0;
        w1 += w1;
        w2 += w2;
        w3 += w3;
    }
    if(z == nz) {
        w4 += w4;
        w5 += w5;
        w6 += w6;
        w7 += w7;
    }
    x -= sz * z;
    int y = x/sy;
    if(y == 1) {
        w0 += w0;
        w1 += w1;
        w4 += w4;
        w5 += w5;
    }
    if(y == ny) {
        w2 += w2;
        w3 += w3;
        w6 += w6;
        w7 += w7;
    }
    x -= sy * y;
    if(x == 1) {
        w0 += w0;
        w2 += w2;
        w4 += w4;
        w6 += w6;
    }
    if(x == nx) {
        w1 += w1;
        w3 += w3;
        w5 += w5;
        w7 += w7;
    }
    //printf("Absorbing %d into %d for %e %e %e %e %e %e %e %e \n", i, v, w0, w1, w2, w3, w4, w5, w6, w7);
    // Save the bound charge to an accumulator array to be added to rhob on the
    // device later
    k_rhob_accum(v) += w0;
    k_rhob_accum(v+1) += w1;
    k_rhob_accum(v+sy) += w2;
    k_rhob_accum(v+sy+1) += w3;
    k_rhob_accum(v+sz) += w4;
    k_rhob_accum(v+sz+1) += w5;
    k_rhob_accum(v+sz+sy) += w6;
    k_rhob_accum(v+sz+sy+1) += w7;
}

#endif // _species_advance_h_
