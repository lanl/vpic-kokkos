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

/* Private interface *********************************************************/

void
checkpt_species( species_t * sp ) {

  sp->copy_to_host();

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

  CHECKPT_VIEW( sp->k_p_d );
  CHECKPT_VIEW( sp->k_p_i_d );

  CHECKPT_VIEW( sp->k_p_h );
  CHECKPT_VIEW( sp->k_p_i_h );

  CHECKPT_VIEW( sp->k_pc_d );
  CHECKPT_VIEW( sp->k_pc_i_d );

  CHECKPT_VIEW( sp->k_pc_h );
  CHECKPT_VIEW( sp->k_pc_i_h );

  CHECKPT_VIEW( sp->k_pr_h );
  CHECKPT_VIEW( sp->k_pr_i_h );

  CHECKPT_VIEW( sp->k_pm_d );
  CHECKPT_VIEW( sp->k_pm_i_d );

  CHECKPT_VIEW( sp->k_pm_h );
  CHECKPT_VIEW( sp->k_pm_i_h );

  CHECKPT_VIEW( sp->k_nm_d );
  CHECKPT_VIEW( sp->k_nm_h );

  CHECKPT_VIEW( sp->unsafe_index );
  CHECKPT_VIEW( sp->clean_up_to_count );
  CHECKPT_VIEW( sp->clean_up_from_count );
  CHECKPT_VIEW( sp->clean_up_from_count_h );
  CHECKPT_VIEW( sp->clean_up_from );
  CHECKPT_VIEW( sp->clean_up_to );

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

  RESTORE_VIEW( &sp->k_p_d );
  RESTORE_VIEW( &sp->k_p_i_d );

  RESTORE_VIEW( &sp->k_p_h );
  RESTORE_VIEW( &sp->k_p_i_h );

  RESTORE_VIEW( &sp->k_pc_d );
  RESTORE_VIEW( &sp->k_pc_i_d );

  RESTORE_VIEW( &sp->k_pc_h );
  RESTORE_VIEW( &sp->k_pc_i_h );

  RESTORE_VIEW( &sp->k_pr_h );
  RESTORE_VIEW( &sp->k_pr_i_h );

  RESTORE_VIEW( &sp->k_pm_d );
  RESTORE_VIEW( &sp->k_pm_i_d );

  RESTORE_VIEW( &sp->k_pm_h );
  RESTORE_VIEW( &sp->k_pm_i_h );

  RESTORE_VIEW( &sp->k_nm_d );
  RESTORE_VIEW( &sp->k_nm_h );

  RESTORE_VIEW( &sp->unsafe_index );
  RESTORE_VIEW( &sp->clean_up_to_count );
  RESTORE_VIEW( &sp->clean_up_from_count );
  RESTORE_VIEW( &sp->clean_up_from_count_h );
  RESTORE_VIEW( &sp->clean_up_from );
  RESTORE_VIEW( &sp->clean_up_to );

  sp->copy_to_device();

  return sp;
}

void
delete_species( species_t * sp ) {
  UNREGISTER_OBJECT( sp );
  FREE_ALIGNED( sp->partition );
  FREE_ALIGNED( sp->pm );
  FREE_ALIGNED( sp->p );
  FREE( sp->name );
  delete(sp);
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

  sp = new species_t();

  sp->k_p_d = k_particles_t("k_particles", max_local_np);
  sp->k_p_i_d = k_particles_i_t("k_particles_i", max_local_np);
  sp->k_pc_d = k_particle_copy_t("k_particle_copy_for_movers", max_local_nm);
  sp->k_pc_i_d = k_particle_i_copy_t("k_particle_copy_for_movers_i", max_local_nm);
  sp->k_pr_h = k_particle_copy_t::HostMirror("k_particle_send_for_movers", max_local_nm);
  sp->k_pr_i_h = k_particle_i_copy_t::HostMirror("k_particle_send_for_movers_i", max_local_nm);
  sp->k_pm_d = k_particle_movers_t("k_particle_movers", max_local_nm);
  sp->k_pm_i_d = k_particle_i_movers_t("k_particle_movers_i", max_local_nm);
  sp->k_nm_d = k_counter_t("k_nm"); // size 1 encoded in type
  sp->unsafe_index = Kokkos::View<int*>("safe index", 2*max_local_nm);
  sp->clean_up_to_count = Kokkos::View<int>("clean up to count");
  sp->clean_up_from_count = Kokkos::View<int>("clean up from count");
  sp->clean_up_from = Kokkos::View<int*>("clean up from", max_local_nm);
  sp->clean_up_to = Kokkos::View<int*>("clean up to", max_local_nm);

  sp->k_p_h = Kokkos::create_mirror_view(sp->k_p_d);
  sp->k_p_i_h = Kokkos::create_mirror_view(sp->k_p_i_d);

  sp->k_pc_h = Kokkos::create_mirror_view(sp->k_pc_d);
  sp->k_pc_i_h = Kokkos::create_mirror_view(sp->k_pc_i_d);

  sp->k_pm_h = Kokkos::create_mirror_view(sp->k_pm_d);
  sp->k_pm_i_h = Kokkos::create_mirror_view(sp->k_pm_i_d);

  sp->k_nm_h = Kokkos::create_mirror_view(sp->k_nm_d);

  sp->clean_up_from_count_h = Kokkos::create_mirror_view(sp->clean_up_from_count);

  MALLOC( sp->name, len+1 );
  strcpy( sp->name, name );

  sp->q = q;
  sp->m = m;

  MALLOC_ALIGNED( sp->p, max_local_np, 128 );
  sp->max_np = max_local_np;

  MALLOC_ALIGNED( sp->pm, max_local_nm, 128 );
  sp->max_nm = max_local_nm;

  sp->last_sorted       = INT64_MIN;
  sp->last_indexed      = INT64_MIN;
  sp->sort_interval     = sort_interval;
  sp->sort_out_of_place = sort_out_of_place;
  MALLOC_ALIGNED( sp->partition, g->nv+1, 128 );

  sp->g = g;

  /* id, next are set by append species */

  REGISTER_OBJECT( sp, checkpt_species, restore_species, NULL );
  return sp;
}

/* Class methods **************************************************************/

void
species_t::copy_to_host(bool force)
{

  if( !on_device && !force )
    return;

  Kokkos::deep_copy(k_p_h, k_p_d);
  Kokkos::deep_copy(k_p_i_h, k_p_i_d);
  Kokkos::deep_copy(k_pm_h, k_pm_d);
  Kokkos::deep_copy(k_pm_i_h, k_pm_i_d);
  Kokkos::deep_copy(k_nm_h, k_nm_d);

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

  on_device = false;

}

void
species_t::copy_to_device(bool force)
{

  if( on_device && !force )
    return;

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

  on_device = true;

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


  auto pc_d_subview  = Kokkos::subview(k_pc_d,   std::make_pair(0, num_to_copy), Kokkos::ALL);
  auto pci_d_subview = Kokkos::subview(k_pc_i_d, std::make_pair(0, num_to_copy));
  Kokkos::deep_copy(pc_d_subview, pc_h_subview);
  Kokkos::deep_copy(pci_d_subview, pci_h_subview);

  // Append it to the particles

  // Avoid capturing this
  auto& particle_copy = k_pc_d;
  auto& particle_copy_i = k_pc_i_d;
  auto& particles = k_p_d;
  auto& particles_i = k_p_i_d;
  const int npart = np;

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

    });

  // Reset this to zero now we've done the write back
  this->np += num_to_copy;
  num_to_copy = 0;

}
