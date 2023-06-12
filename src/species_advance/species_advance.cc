/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Original version
 *
 */

#include "species_advance.h"
#include "../boundary/boundary.h"

/* Private interface *********************************************************/

void
checkpt_species( const species_t * sp ) {
    //std::cout << "checkpintg " << sp->name << " with nm = " << sp->nm << std::endl;
  CHECKPT( sp, 1 );
  CHECKPT_STR( sp->name );
  checkpt_data( sp->p,
                sp->np    *sizeof(particle_t),
                sp->max_np*sizeof(particle_t), 1, 1, 128 );
  checkpt_data( sp->pm,
                sp->nm    *sizeof(particle_mover_t),
                sp->max_nm*sizeof(particle_mover_t), 1, 1, 128 );
  CHECKPT_ALIGNED( sp->partition, sp->g->nv+1, 128 );
  CHECKPT_PTR( sp->g );
  CHECKPT_PTR( sp->next );
  CHECKPT_PTR( sp->pb_diag );
}

species_t *
restore_species( void ) {
  species_t * sp;
  RESTORE( sp );
  RESTORE_STR( sp->name );
  sp->p  = (particle_t *)      restore_data();
  sp->pm = (particle_mover_t *)restore_data();
  RESTORE_ALIGNED( sp->partition );
  RESTORE_PTR( sp->g );
  RESTORE_PTR( sp->next );
  RESTORE_PTR( sp->pb_diag );
  return sp;
}

void
delete_species( species_t * sp ) {
  delete_pbd(sp->pb_diag);
  UNREGISTER_OBJECT( sp );
  FREE_ALIGNED( sp->partition );
  FREE_ALIGNED( sp->pm );
  FREE_ALIGNED( sp->p );
  FREE( sp->name );
  FREE( sp );
}

/* Public interface **********************************************************/

int
num_species( const species_t * sp_list ) {
  return sp_list ? sp_list->id+1 : 0;
}

void
delete_species_list( species_t * sp_list ) {
  species_t * sp;
  while( sp_list ) {
    sp = sp_list;
    sp_list = sp_list->next;
    delete_species( sp );
  }
}

species_t *
find_species_id( species_id id,
                 species_t * sp_list ) {
  species_t * sp;
  LIST_FIND_FIRST( sp, sp_list, sp->id==id );
  return sp;
}

species_t *
find_species_name( const char * name,
                   species_t * sp_list ) {
  species_t * sp;
  if( !name ) return NULL;
  LIST_FIND_FIRST( sp, sp_list, strcmp( sp->name, name )==0 );
  return sp;
}

species_t *
append_species( species_t * sp,
                species_t ** sp_list ) {
  if( !sp || !sp_list ) ERROR(( "Bad args" ));
  if( sp->next ) ERROR(( "Species \"%s\" already in a list", sp->name ));
  if( find_species_name( sp->name, *sp_list ) )
    ERROR(( "There is already a species in the list named \"%s\"", sp->name ));
  if( (*sp_list) && sp->g!=(*sp_list)->g )
    ERROR(( "Species \"%s\" uses a different grid from this list", sp->name ));
  sp->id   = num_species( *sp_list );
  sp->next = *sp_list;
  *sp_list = sp;
  return sp;
}

species_t *
species( const char * name,
         float q,
         float m,
         int max_local_np,
         int max_local_nm,
         int sort_interval,
         int sort_out_of_place,
         grid_t * g ) {
  species_t * sp;
  int len = name ? strlen(name) : 0;

  if( !len ) ERROR(( "Cannot create a nameless species" ));
  if( !g ) ERROR(( "NULL grid" ));
  if( g->nv == 0) ERROR(( "Allocate grid before defining species." ));
  if( max_local_np<1 ) max_local_np = 1;
  if( max_local_nm<1 ) max_local_nm = 1;

  sp = new species_t(max_local_np, max_local_nm);
  //MALLOC( sp, 1 );
  //CLEAR( sp, 1 );

  MALLOC( sp->name, len+1 );
  strcpy( sp->name, name );

  sp->q = q;
  sp->m = m;

  if(!world_rank) fprintf(stderr, "Mallocing %.4f GiB for species %s.\n",
          (double (max_local_np*sizeof(particle_t)))/pow(2,30), sp->name);
  MALLOC_ALIGNED( sp->p, max_local_np, 128 );
  sp->max_np = max_local_np;

  if(!world_rank) fprintf(stderr, "Mallocing %.4f GiB for species %s movers.\n",
          (double (max_local_nm*sizeof(particle_t)))/pow(2,30), sp->name);
  MALLOC_ALIGNED( sp->pm, max_local_nm, 128 );
  sp->max_nm = max_local_nm;

  sp->last_sorted       = INT64_MIN;
  sp->sort_interval     = sort_interval;
  sp->sort_out_of_place = sort_out_of_place;
  MALLOC_ALIGNED( sp->partition, g->nv+1, 128 );

  sp->g = g;

  sp->pb_diag = init_pb_diagnostic();
  REGISTER_OBJECT( sp->pb_diag, checkpt_pbd, restore_pbd, NULL);

  /* id, next are set by append species */

  REGISTER_OBJECT( sp, checkpt_species, restore_species, NULL );
  return sp;
}

/* Class methods **************************************************************/

void
species_t::copy_to_host()
{

  Kokkos::deep_copy(k_p_h, k_p_d);
  Kokkos::deep_copy(k_p_i_h, k_p_i_d);
  Kokkos::deep_copy(k_pm_h, k_pm_d);
  Kokkos::deep_copy(k_pm_i_h, k_pm_i_d);
  Kokkos::deep_copy(k_nm_h, k_nm_d);

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  if(using_annotations) {
    Kokkos::deep_copy(annotations_h.i32, annotations_d.i32);
    Kokkos::deep_copy(annotations_h.i64, annotations_d.i64);
    Kokkos::deep_copy(annotations_h.f32, annotations_d.f32);
    Kokkos::deep_copy(annotations_h.f64, annotations_d.f64);
  }
#endif

  nm = k_nm_h(0);

  // Avoid capturing this
  auto& k_particle_h = k_p_h;
  auto& k_particle_i_h = k_p_i_h;
  auto& particles = p;

  Kokkos::parallel_for("copy particles to host",
    host_execution_policy(0, np) ,
    KOKKOS_LAMBDA (int i) {

      particles[i].dx = k_particle_h(i, particle_var::dx);
      particles[i].dy = k_particle_h(i, particle_var::dy);
      particles[i].dz = k_particle_h(i, particle_var::dz);
      particles[i].ux = k_particle_h(i, particle_var::ux);
      particles[i].uy = k_particle_h(i, particle_var::uy);
      particles[i].uz = k_particle_h(i, particle_var::uz);
      particles[i].w  = k_particle_h(i, particle_var::w);
      particles[i].i  = k_particle_i_h(i);

    });

  // Avoid capturing this
  auto& k_particle_movers_h = k_pm_h;
  auto& k_particle_i_movers_h = k_pm_i_h;
  auto& movers = pm;

  Kokkos::parallel_for("copy movers to host",
    host_execution_policy(0, max_nm) ,
    KOKKOS_LAMBDA (int i) {

      movers[i].dispx = k_particle_movers_h(i, particle_mover_var::dispx);
      movers[i].dispy = k_particle_movers_h(i, particle_mover_var::dispy);
      movers[i].dispz = k_particle_movers_h(i, particle_mover_var::dispz);
      movers[i].i     = k_particle_i_movers_h(i);

    });

  last_copied = g->step;

}

void
species_t::copy_to_device()
{

  k_nm_h(0) = nm;

  // Avoid capturing this
  auto& k_particle_h = k_p_h;
  auto& k_particle_i_h = k_p_i_h;
  auto& particles = p;

  Kokkos::parallel_for("copy particles to device",
    host_execution_policy(0, np) ,
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

  // Avoid capturing this
  auto& k_particle_movers_h = k_pm_h;
  auto& k_particle_i_movers_h = k_pm_i_h;
  auto& movers = pm;

  Kokkos::parallel_for("copy movers to device",
    host_execution_policy(0, max_nm) ,
    KOKKOS_LAMBDA (int i) {

      k_particle_movers_h(i, particle_mover_var::dispx) = movers[i].dispx;
      k_particle_movers_h(i, particle_mover_var::dispy) = movers[i].dispy;
      k_particle_movers_h(i, particle_mover_var::dispz) = movers[i].dispz;
      k_particle_i_movers_h(i) = movers[i].i;

    });

  Kokkos::deep_copy(k_p_d, k_p_h);
  Kokkos::deep_copy(k_p_i_d, k_p_i_h);
  Kokkos::deep_copy(k_pm_d, k_pm_h);
  Kokkos::deep_copy(k_pm_i_d, k_pm_i_h);
  Kokkos::deep_copy(k_nm_d, k_nm_h);

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  if(using_annotations) {
    Kokkos::deep_copy(annotations_d.i32, annotations_h.i32);
    Kokkos::deep_copy(annotations_d.i64, annotations_h.i64);
    Kokkos::deep_copy(annotations_d.f32, annotations_h.f32);
    Kokkos::deep_copy(annotations_d.f64, annotations_h.f64);
  }
#endif
}

void
species_t::copy_outbound_to_host()
{

  Kokkos::deep_copy(k_nm_h, k_nm_d);
  nm = k_nm_h(0);

  auto pm_h_dispx = Kokkos::subview(k_pm_h, std::make_pair(0, nm), 0);
  auto pm_d_dispx = Kokkos::subview(k_pm_d, std::make_pair(0, nm), 0);
  auto pm_h_dispy = Kokkos::subview(k_pm_h, std::make_pair(0, nm), 1);
  auto pm_d_dispy = Kokkos::subview(k_pm_d, std::make_pair(0, nm), 1);
  auto pm_h_dispz = Kokkos::subview(k_pm_h, std::make_pair(0, nm), 2);
  auto pm_d_dispz = Kokkos::subview(k_pm_d, std::make_pair(0, nm), 2);

  auto pm_i_h_subview = Kokkos::subview(k_pm_i_h, std::make_pair(0, nm));
  auto pm_i_d_subview = Kokkos::subview(k_pm_i_d, std::make_pair(0, nm));

  Kokkos::deep_copy(pm_h_dispx, pm_d_dispx);
  Kokkos::deep_copy(pm_h_dispy, pm_d_dispy);
  Kokkos::deep_copy(pm_h_dispz, pm_d_dispz);
  Kokkos::deep_copy(pm_i_h_subview, pm_i_d_subview);

  // Avoid capturing this
  auto& k_particle_movers_h = k_pm_h;
  auto& k_particle_i_movers_h = k_pm_i_h;
  auto& movers = pm;

  Kokkos::parallel_for("copy movers to host",
    host_execution_policy(0, nm) ,
    KOKKOS_LAMBDA (int i) {
      movers[i].dispx = k_particle_movers_h(i, particle_mover_var::dispx);
      movers[i].dispy = k_particle_movers_h(i, particle_mover_var::dispy);
      movers[i].dispz = k_particle_movers_h(i, particle_mover_var::dispz);
      movers[i].i     = k_particle_i_movers_h(i);
    });
}

void
species_t::copy_inbound_to_device()
{

  // TODO: Why do we need particle_copy as an intermediate?
  // currently the recv particles are in particles_recv, not particle_copy
  auto pr_h_subview  = Kokkos::subview(k_pr_h,   std::make_pair(0, num_to_copy), Kokkos::ALL);
  auto pc_h_subview  = Kokkos::subview(k_pc_h,   std::make_pair(0, num_to_copy), Kokkos::ALL);
  auto pri_h_subview = Kokkos::subview(k_pr_i_h, std::make_pair(0, num_to_copy));
  auto pci_h_subview = Kokkos::subview(k_pc_i_h, std::make_pair(0, num_to_copy));
  Kokkos::deep_copy(pc_h_subview, pr_h_subview);
  Kokkos::deep_copy(pci_h_subview, pri_h_subview);

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  if(using_annotations) {
    Kokkos::deep_copy(annotations_copy_h.i32, annotations_recv_h.i32);
    Kokkos::deep_copy(annotations_copy_h.i64, annotations_recv_h.i64);
    Kokkos::deep_copy(annotations_copy_h.f32, annotations_recv_h.f32);
    Kokkos::deep_copy(annotations_copy_h.f64, annotations_recv_h.f64);
  }
#endif

  auto pc_d_subview  = Kokkos::subview(k_pc_d,   std::make_pair(0, num_to_copy), Kokkos::ALL);
  auto pci_d_subview = Kokkos::subview(k_pc_i_d, std::make_pair(0, num_to_copy));
  Kokkos::deep_copy(pc_d_subview, pc_h_subview);
  Kokkos::deep_copy(pci_d_subview, pci_h_subview);

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  if(using_annotations) {
    Kokkos::deep_copy(annotations_copy_d.i32, annotations_copy_h.i32);
    Kokkos::deep_copy(annotations_copy_d.i64, annotations_copy_h.i64);
    Kokkos::deep_copy(annotations_copy_d.f32, annotations_copy_h.f32);
    Kokkos::deep_copy(annotations_copy_d.f64, annotations_copy_h.f64);
  }
#endif

  // Append it to the particles

  // Avoid capturing this
  auto& particle_copy = k_pc_d;
  auto& particle_copy_i = k_pc_i_d;
  auto& particles = k_p_d;
  auto& particles_i = k_p_i_d;
  const int npart = np;

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
  auto& i32_annotations = annotations_d.i32;
  auto& i64_annotations = annotations_d.i64;
  auto& f32_annotations = annotations_d.f32;
  auto& f64_annotations = annotations_d.f64;
  auto& i32_annotations_copy = annotations_copy_d.i32;
  auto& i64_annotations_copy = annotations_copy_d.i64;
  auto& f32_annotations_copy = annotations_copy_d.f32;
  auto& f64_annotations_copy = annotations_copy_d.f64;
  int num_i32 = annotation_vars.i32_vars.size();
  int num_i64 = annotation_vars.i64_vars.size();
  int num_f32 = annotation_vars.f32_vars.size();
  int num_f64 = annotation_vars.f64_vars.size();
#endif

  Kokkos::parallel_for("append moved particles",
    Kokkos::RangePolicy <Kokkos::DefaultExecutionSpace> (0, num_to_copy),
    KOKKOS_LAMBDA (int i) {

      int npi = npart+i; // i goes from 0..n so no need for -1
      particles(npi, particle_var::dx) = particle_copy(i, particle_var::dx);
      particles(npi, particle_var::dy) = particle_copy(i, particle_var::dy);
      particles(npi, particle_var::dz) = particle_copy(i, particle_var::dz);
      particles(npi, particle_var::ux) = particle_copy(i, particle_var::ux);
      particles(npi, particle_var::uy) = particle_copy(i, particle_var::uy);
      particles(npi, particle_var::uz) = particle_copy(i, particle_var::uz);
      particles(npi, particle_var::w)  = particle_copy(i, particle_var::w);
      particles_i(npi) = particle_copy_i(i);

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
      // Copy int annotations
      for(int j=0; j<num_i32; j++) {
        i32_annotations(npi,j) = i32_annotations_copy(i,j);
      }
      // Copy int64_t annotations
      for(int j=0; j<num_i64; j++) {
        i64_annotations(npi,j) = i64_annotations_copy(i,j);
      }
      // Copy float annnotations
      for(int j=0; j<num_f32; j++) {
        f32_annotations(npi,j) = f32_annotations_copy(i,j);
      }
      // Copy double annnotations
      for(int j=0; j<num_f64; j++) {
        f64_annotations(npi,j) = f64_annotations_copy(i,j);
      }
#endif

    });

  // Reset this to zero now we've done the write back
  this->np += num_to_copy;
  num_to_copy = 0;
}

#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
void 
species_t::init_io_buffers(const int N_steps, const float over_alloc_factor) {
  const int nparticles = static_cast<int>(static_cast<float>(N_steps) * over_alloc_factor);
  particle_io_buffer      = k_particles_t::HostMirror("Particle io buffer", nparticles);
  particle_cell_io_buffer = k_particles_i_t::HostMirror("Particle cell io buffer", nparticles);
  efields_io_buffer       = Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror("Efield io buffer", nparticles);
  bfields_io_buffer       = Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror("Bfield io buffer", nparticles);
  current_dens_io_buffer  = Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror("Current density io buffer", nparticles);
  charge_dens_io_buffer   = Kokkos::View<float*>::HostMirror("Charge density io buffer", nparticles);;
  momentum_dens_io_buffer = Kokkos::View<float*[3], Kokkos::LayoutLeft>::HostMirror("Momentum density io buffer", nparticles);
  ke_dens_io_buffer       = Kokkos::View<float*>::HostMirror("KE density io buffer", nparticles);;
  stress_tensor_io_buffer = Kokkos::View<float*[6], Kokkos::LayoutLeft>::HostMirror("Stress tensor io buffer", nparticles);
  annotations_io_buffer   = annotations_t<Kokkos::DefaultHostExecutionSpace>(nparticles, annotation_vars);
  nparticles_buffered = 0;
}

void 
species_t::init_annotations(int num_particles, int num_movers, annotation_vars_t& vars) 
{
  using_annotations = true;
  annotation_vars = vars;
  annotations_d = annotations_t<Kokkos::DefaultExecutionSpace>(num_particles, vars);
  annotations_h = annotations_t<Kokkos::DefaultHostExecutionSpace>(annotations_d);
  annotations_copy_d = annotations_t<Kokkos::DefaultExecutionSpace>(num_movers, vars);
  annotations_copy_h = annotations_t<Kokkos::DefaultHostExecutionSpace>(annotations_copy_d);
  annotations_recv_h = annotations_t<Kokkos::DefaultHostExecutionSpace>(num_movers, vars);
}
#endif
