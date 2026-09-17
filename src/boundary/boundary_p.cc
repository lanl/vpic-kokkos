#define IN_boundary
#include "boundary_private.h"
#include <cassert>
#include <algorithm>
#include <mpi.h>
#include "Kokkos_Bitset.hpp"

// If this is defined particle and mover buffers will not resize dynamically
// (This is the common case for the users)
#define DISABLE_DYNAMIC_RESIZING

// FIXME: ARCHITECTURAL FLAW!  CUSTOM BCS AND SHARED FACES CANNOT
// COEXIST ON THE SAME FACE!  THIS MEANS THAT CUSTOM BOUNDARYS MUST
// REINJECT ALL ABSORBED PARTICLES IN THE SAME DOMAIN!

enum { MAX_PBC = 32, MAX_SP = 32 };

// Gives the local mp port associated with a local face
constexpr int f2b[6]  = { BOUNDARY(-1, 0, 0),
                          BOUNDARY( 0,-1, 0),
                          BOUNDARY( 0, 0,-1),
                          BOUNDARY( 1, 0, 0),
                          BOUNDARY( 0, 1, 0),
                          BOUNDARY( 0, 0, 1) };

// Gives the remote mp port associated with a local face
constexpr int f2rb[6] = { BOUNDARY( 1, 0, 0),
                          BOUNDARY( 0, 1, 0),
                          BOUNDARY( 0, 0, 1),
                          BOUNDARY(-1, 0, 0),
                          BOUNDARY( 0,-1, 0),
                          BOUNDARY( 0, 0,-1) };

// Gives the axis associated with a local face
constexpr int get_axis(const int idx) {
  constexpr std::array<int,6> axis = { 0, 1, 2, 0, 1, 2 };
  return axis[idx];
}

// Gives the location of sending face on the receiver
constexpr int get_dir(const int idx) {
  constexpr std::array<float,6> dir = { 1, 1, 1, -1, -1, -1 };
  return dir[idx];
}

/**
 * @brief The original boundary_p takes all moved particles, and integrates
 * them to the particle list. It requires that nm be monotonically increasing,
 * which is a bad assumption for parallel.
 *
 * Instead we try and redesign things here such that we copy only the moving
 * particles to the host, and instead of backfilling we do a compress on the
 * GPU and then add "new" particles to the end of the array.
 *
 * This may actually simplify the logic, without being significantly slower
 *
 * @param pbc_list Particle boundary condition list
 * @param sp_list Species list
 * @param fa Field array
 * @param aa Accumulator array
 */
void
boundary_p(
        particle_bc_t * RESTRICT pbc_list,
        species_t     * RESTRICT sp_list,
        field_array_t * RESTRICT fa
      )
{

  // Temporary store for local particle injectors
  // FIXME: Ugly static usage
  static particle_injector_t * RESTRICT ALIGNED(16) ci = NULL;
  static size_t max_ci = 0;

  size_t n_send[6] = {0}, n_recv[6] = {0}, n_ci;

  Kokkos::View<size_t[6]> n_send_d("Send count"), n_recv_d("Receive count");
  auto n_send_h = Kokkos::create_mirror_view(n_send_d);
  auto n_recv_h = Kokkos::create_mirror_view(n_recv_d);
  Kokkos::deep_copy(n_send_d, 0);
  Kokkos::deep_copy(n_recv_d, 0);

  Kokkos::View<particle_injector_t*> send_buff_d[6];
  Kokkos::View<particle_injector_t*> recv_buff_d[6];
  Kokkos::View<particle_injector_t*>::HostMirror send_buff_h[6];
  Kokkos::View<particle_injector_t*>::HostMirror recv_buff_h[6];

  species_t * sp;
  int face;

  // Check input args

  if( !sp_list ) return; // Nothing to do if no species
  if( !fa )
    ERROR(( "Bad args" ));

  const int num_sp = num_species( sp_list );

  // Unpack the particle boundary conditions

  //particle_bc_func_t pbc_interact[MAX_PBC];
  //void * pbc_params[MAX_PBC];
  //const int nb = num_particle_bc( pbc_list );
  //if( nb>MAX_PBC ) ERROR(( "Update this to support more particle boundary conditions" ));
  //for( particle_bc_t * pbc=pbc_list; pbc; pbc=pbc->next ) {
  //  pbc_interact[-pbc->id-3] = pbc->interact;
  //  pbc_params[  -pbc->id-3] = pbc->params;
  // }

  // Unpack fields

  //field_t * RESTRICT ALIGNED(128) f = fa->f;
  grid_t  * RESTRICT              g = fa->g;


  // Unpack the grid

Kokkos::Profiling::pushRegion("BoundaryP -> unpack grid");
  const float r8V = g->r8V;
  const int nx = g->nx, ny = g->ny, nz = g->nz;
  const float cx = 0.25 * g->rdy * g->rdz / g->dt;
  const float cy = 0.25 * g->rdz * g->rdx / g->dt;
  const float cz = 0.25 * g->rdx * g->rdy / g->dt;
  const int sy = g->sy, sz = g->sz;

  const int64_t * RESTRICT ALIGNED(128) neighbor = g->neighbor;
  mp_t* RESTRICT              mp       = g->mp;
  const int64_t rangel = g->rangel;
  const int64_t rangeh = g->rangeh;
  const int64_t rangem = g->range[world_size];

  Kokkos::View<int64_t[6]> range_d("Range array");
  auto range_h = Kokkos::create_mirror_view(range_d);

  int bc[6], shared[6];
  int64_t range[6];

  MPI_Request send_req[6], recv_req[6];
  for(int idx=0; idx<6; idx++) {
    send_req[idx] = MPI_REQUEST_NULL;
    recv_req[idx] = MPI_REQUEST_NULL;
  }
  int n_active_faces = 0;

  for( face=0; face<6; face++ ) {
    bc[face] = g->bc[f2b[face]];
    shared[face] = (bc[face]>=0) && (bc[face]<world_size) &&
                   (bc[face]!=world_rank);
    if( shared[face] ) {
      range[face] = g->range[bc[face]];
      range_h(face) = g->range[bc[face]];
    }
  }
  Kokkos::deep_copy(range_d, range_h);
Kokkos::Profiling::popRegion();

  // Begin receiving the particle counts

  for( face=0; face<6; face++ ) {
    if( shared[face] ) {
      //mp_size_recv_buffer( mp, f2b[face], sizeof(int) );
      //mp_begin_recv( mp, f2b[face], sizeof(int), bc[face], f2rb[face] );
      n_active_faces++;
      MPI_Irecv(&n_recv_h(face), 1, MPI_INT, bc[face], f2rb[face], 
                MPI_COMM_WORLD, &recv_req[face]);
    }
  }

  // Load the particle send and local injection buffers

  // Track if rhob needs to be updated on the device.
  // TODO: Do the absorbtion on the device.
  int absorbed = 0;

  do {

    particle_injector_t * RESTRICT ALIGNED(16) pi_send[6];

    // Presize the send and injection buffers
    //
    // Each buffer is large enough to hold one injector corresponding
    // to every mover in use (worst case, but plausible scenario in
    // beam simulations, is one buffer gets all the movers).
    //
    // FIXME: We could be several times more efficient in our particle
    // injector buffer sizing here.  Namely, we could create on local
    // injector buffer of nm is size.  All injection for all
    // boundaries would be done here.  The local buffer would then be
    // counted to determine the size of each send buffer.  The local
    // buffer would then move all injectors into the approate send
    // buffers (leaving only the local injectors).  This would require
    // some extra data motion though.  (But would give a more robust
    // implementation against variations in MP implementation.)
    //
    // FIXME: This presizing assumes that custom boundary conditions
    // inject at most one particle per incident particle.  Currently,
    // the invocation of pbc_interact[*] insures that assumption will
    // be satisfied (if the handlers conform that it).  We should be
    // more flexible though in the future (especially given above the
    // above overalloc).

Kokkos::Profiling::pushRegion("BoundaryP -> allocate send buffers");
    int nm = 0; 
    LIST_FOR_EACH( sp, sp_list ) {
      nm += sp->nm;
    }

    for( face=0; face<6; face++ ) {
      if( shared[face] ) {
        //mp_size_send_buffer( mp, f2b[face], 16+nm*sizeof(particle_injector_t) );
        //pi_send[face] = (particle_injector_t *)(((char *)mp_send_buffer(mp,f2b[face]))+16);
        //Kokkos::resize(send_buff[face], nm);

        send_buff_d[face] = Kokkos::View<particle_injector_t*>("Send buffer", nm);
        send_buff_h[face] = Kokkos::create_mirror_view(send_buff_d[face]);
        n_send_h(face) = 0;
      }
    }

    if( max_ci<nm ) {
      particle_injector_t * new_ci = ci;
      FREE_ALIGNED( new_ci );
      MALLOC_ALIGNED( new_ci, nm, 16 );
      ci     = new_ci;
      max_ci = nm;
    }
    n_ci = 0;
Kokkos::Profiling::popRegion();

Kokkos::Profiling::pushRegion("BoundaryP -> absorb and pack");
    // For each species, load the movers
    LIST_FOR_EACH( sp, sp_list )
    {
        //const float   sp_q  = sp->q;
        const int32_t sp_id = sp->id;

        //particle_t * RESTRICT ALIGNED(128) p0 = sp->p;
        //int np = sp->np;

        particle_mover_t * RESTRICT ALIGNED(16)  pm = sp->pm + sp->nm - 1;
        nm = sp->nm;

        particle_injector_t * RESTRICT ALIGNED(16) pi;

        // Note that particle movers for each species are processed in
        // reverse order.  This allows us to backfill holes in the
        // particle list created by boundary conditions and/or
        // communication.  This assumes particle on the mover list are
        // monotonically increasing.  That is: pm[n].i > pm[n-1].i for
        // n=1...nm-1.  advance_p and inject_particle create movers with
        // property if all aged particle injection occurs after
        // advance_p and before this

        // Here we essentially need to remove all accesses of the particle array (p0) and instead read from k_pc_h

        // Track which particles to apply particle boundary diagnostic
        Kokkos::Bitset<Kokkos::DefaultHostExecutionSpace> apply_pbd(nm);
        apply_pbd.reset();

        const auto& neighbors = g->k_neighbor_h;
        const auto& particle_send = sp->k_pc_i_h;
        const auto& particle_move = sp->k_pc_h;
        const auto& particle_move_i = sp->k_pc_i_h;
        const float qsp = sp->q;
        const bool enable_pbd = sp->pb_diag->enable;
        auto rhob_accum_sv = Kokkos::Experimental::create_scatter_view(fa->k_f_rhob_accum_h);
        Kokkos::parallel_reduce("boundary_p process particles", host_execution_policy(0,nm), 
        KOKKOS_LAMBDA (const int copy_index, int& nabsorb) {
          auto rhob_accum_sa = rhob_accum_sv.access();
          int voxel = particle_send(copy_index);
          const int face = voxel & 7;
          voxel >>= 3;
          particle_send(copy_index) = voxel;
          int64_t nn = neighbors( 6*voxel + face );
        
          if( nn==absorb_particles ) { // Absorb particle
            nabsorb++;
        
            // Send the particle to the particle boundary diagnostic
            if (enable_pbd)
              apply_pbd.set(copy_index);
              //pbd_write_to_buffer(sp, particle_move_h, particle_move_i_h, copy_index);
            
            k_accumulate_rhob_single_cpu(
                    rhob_accum_sa,
                    particle_move,
                    particle_move_i,
                    copy_index,
                    qsp,
                    r8V,
                    nx,ny,nz,
                    sy,sz
            );
          } else if( ((nn>=0) & (nn<rangel)) | ((nn>rangeh) & (nn<=rangem)) ) { // Send to neighboring node
            particle_injector_t * RESTRICT ALIGNED(16) p_injector;
            auto idx = Kokkos::atomic_fetch_inc(&(n_send_h(face)));
            p_injector = &send_buff_h[face](idx); 
        
            p_injector->dx = particle_move(copy_index, particle_var::dx);
            p_injector->dy = particle_move(copy_index, particle_var::dy);
            p_injector->dz = particle_move(copy_index, particle_var::dz);
            p_injector->i  = nn - range_h(face);
            p_injector->ux = particle_move(copy_index, particle_var::ux);
            p_injector->uy = particle_move(copy_index, particle_var::uy);
            p_injector->uz = particle_move(copy_index, particle_var::uz);
            p_injector->w  = particle_move(copy_index, particle_var::w);
        
            p_injector->dispx = pm->dispx; 
            p_injector->dispy = pm->dispy; 
            p_injector->dispz = pm->dispz;
            p_injector->sp_id = sp_id;
        
            (&p_injector->dx)[get_axis(face)] = get_dir(face);
            p_injector->i                     = nn - range_h(face);
            p_injector->sp_id                 = sp_id;
          } else {
            // User-defined handling
        
            // After a particle interacts with a boundary it is removed
            // from the local particle list.  Thus, if a boundary handler
            // does not want a particle destroyed,  it is the boundary
            // handler's job to append the destroyed particle to the list
            // of particles to inject.
            //
            // Note that these destruction and creation processes do _not_
            // adjust rhob by default.  Thus, a boundary handler is
            // responsible for insuring that the rhob is updated
            // appropriate for the incident particle it destroys and for
            // any particles it injects as a result too.
            //
            // Since most boundary handlers do local reinjection and are
            // charge neutral, this means most boundary handlers do
            // nothing to rhob.
            int64_t old_nn = nn;
            nn = -nn - 3; // Assumes reflective/absorbing are -1, -2
            /*
               if( (nn>=0) & (nn<nb) ) {
               Kokkos::abort("Custom boundary not implemented");
            //n_ci += pbc_interact[nn]( pbc_params[nn], sp, p0+i, pm,
            //ci+n_ci, 1, face );
            continue;
            }
            */
        
            // Uh-oh: We fell through
            //if( ((nn>=0) & (nn< rangel)) | ((nn>rangeh) & (nn<=rangem)) )
            Kokkos::printf("nn %ld rangel %ld rangeh %ld rangem %ld voxel %d old_nn %ld\n", nn, rangel, rangeh, rangem, face, old_nn);
        
            //WARNING(( "Unknown boundary interaction ... dropping particle "
            //            "(species=%s)", sp->name ));
          }
        }, absorbed);
        Kokkos::Experimental::contribute(fa->k_f_rhob_accum_h, rhob_accum_sv);
        Kokkos::fence();

        sp->nm = 0;
    }
Kokkos::Profiling::popRegion();

  } while(0);
  

  // Finish exchanging particle counts and start exchanging actual
  // particles.

  // Note: This is wasteful of communications.  A better protocol
  // would fuse the exchange of the counts with the exchange of the
  // messages.  in a slightly more complex protocol.  However, the MP
  // API prohibits such a model.  Unfortuantely, refining MP is not
  // much help here.  Under the hood on Roadrunner, the DaCS API also
  // prohibits such (specifically, in both, you can't do the
  // equilvanet of a MPI_Getcount to determine how much data you
  // actually received.

Kokkos::Profiling::pushRegion("BoundaryP -> comm counts");
  for( face=0; face<6; face++ ) {
    if( shared[face] ) {
      //*((int *)mp_send_buffer( mp, f2b[face] )) = n_send[face];
      //mp_begin_send( mp, f2b[face], sizeof(int), bc[face], f2b[face] );
      MPI_Isend(&n_send_h(face), 1, MPI_INT, bc[face], f2b[face], MPI_COMM_WORLD, 
                &send_req[face]);
    }
  }

//  int nhandled = 0;
//  while(nhandled < n_active_faces) {
//    MPI_Waitany(6, recv_req, &face, MPI_STATUS_IGNORE);
//    //mp_size_recv_buffer( mp, f2b[face],
//    //                     16+n_recv[face]*sizeof(particle_injector_t) );
//    //mp_begin_recv( mp, f2b[face], 16+n_recv[face]*sizeof(particle_injector_t),
//    //               bc[face], f2rb[face] );
//    //Kokkos::resize(recv_buff[face], n_recv[face]);
//    recv_buff[face] = Kokkos::View<particle_injector_t*, Kokkos::HostSpace>("recv buff", n_recv[face]);
//    MPI_Irecv(recv_buff[face].data(), n_recv[face]*sizeof(particle_injector_t), 
//              MPI_BYTE, bc[face], f2rb[face], MPI_COMM_WORLD, &recv_req[face]);
//    nhandled++;
//  }

  for( face=0; face<6; face++ ) {
    if( shared[face] )  {
      //mp_end_recv( mp, f2b[face] );
      //n_recv[face] = *((int *)mp_recv_buffer( mp, f2b[face] ));
      //mp_size_recv_buffer( mp, f2b[face],
      //                     16+n_recv[face]*sizeof(particle_injector_t) );
      //mp_begin_recv( mp, f2b[face], 16+n_recv[face]*sizeof(particle_injector_t),
      //               bc[face], f2rb[face] );
      MPI_Wait( &recv_req[face], MPI_STATUS_IGNORE );
      recv_buff_h[face] = Kokkos::View<particle_injector_t*>::HostMirror("recv buff", n_recv_h(face));
      MPI_Irecv(recv_buff_h[face].data(), n_recv_h(face)*sizeof(particle_injector_t), 
                MPI_BYTE, bc[face], f2rb[face], MPI_COMM_WORLD, &recv_req[face]);
    }
  }
Kokkos::Profiling::popRegion();

Kokkos::Profiling::pushRegion("BoundaryP -> begin send faces");
  for( face=0; face<6; face++ ) {
    if( shared[face] ) {
      //mp_end_send( mp, f2b[face] );
      // FIXME: ASSUMES MP WON'T MUCK WITH REST OF SEND BUFFER. IF WE
      // DID MORE EFFICIENT MOVER ALLOCATION ABOVE, THIS WOULD BE
      // ROBUSTED AGAINST MP IMPLEMENTATION VAGARIES
      //mp_begin_send( mp, f2b[face], 16+n_send[face]*sizeof(particle_injector_t),
      //               bc[face], f2b[face] );
      MPI_Isend(send_buff_h[face].data(), n_send_h(face)*sizeof(particle_injector_t),
                MPI_BYTE, bc[face], f2b[face], MPI_COMM_WORLD, &send_req[face]);
    }
  }
Kokkos::Profiling::popRegion();

Kokkos::Profiling::pushRegion("BoundaryP -> unpack and resend");
  do {
    // Unpack the species list for random acesss

    species_t*       sp_[ MAX_SP];
    //particle_t       * RESTRICT ALIGNED(32) sp_p[ MAX_SP];
    particle_mover_t * RESTRICT ALIGNED(32) sp_pm[MAX_SP];
    //float sp_q[MAX_SP];
    //int sp_np[MAX_SP];
    int sp_nm[MAX_SP];

    if( num_sp > MAX_SP )
    {
      ERROR(( "Update this to support more species" ));
    }

    // FIXME: I'm not sure this manual packing and storing buys us anything -- remove?
    LIST_FOR_EACH( sp, sp_list ) {
      sp_[  sp->id ] = sp;
      sp_pm[ sp->id ] = sp->pm;
      sp_nm[ sp->id ] = sp->nm;
    }

    // Inject particles.  We do custom local injection first to
    // increase message overlap opportunities.

    face = 5;
    do {
      //particle_t          * RESTRICT ALIGNED(32) p;
      particle_mover_t    * RESTRICT ALIGNED(16) pm;
      const particle_injector_t * RESTRICT ALIGNED(16) pi;
      int nm, n, id;

      face++; 
      if( face==7 ) {
        face = 0;
      }
      if( face==6 ) {
        pi = ci;
        n = n_ci;
      } else if( shared[face] ) {
        //mp_end_recv( mp, f2b[face] );
        //pi = (particle_injector_t *)
        //  (((char *)mp_recv_buffer(mp,f2b[face]))+16);
        MPI_Wait( &recv_req[face], MPI_STATUS_IGNORE );
        pi = recv_buff_h[face].data();
        n  = n_recv_h(face);
      } else {
        continue;
      }

      // WARNING: THIS TRUSTS THAT THE INJECTORS (INCLUDING THOSE
      // RECEIVED FROM OTHER NODES) HAVE VALID PARTICLE IDS.

      Kokkos::View<int[MAX_SP], Kokkos::HostSpace> sp_nm_view("nm per species");
      Kokkos::View<float[MAX_SP], Kokkos::HostSpace> sp_q_view("q for each species");
      for(int idx=0; idx<num_sp; idx++) {
        sp_nm_view(idx) = sp_nm[idx];
        sp_q_view(idx) = sp_[idx]->q;
      }
      
      k_particle_copy_t::HostMirror particle_send[MAX_SP];
      k_particle_i_copy_t::HostMirror particle_send_i[MAX_SP];
      k_particle_copy_t::HostMirror particle_recv[MAX_SP];
      k_particle_i_copy_t::HostMirror particle_recv_i[MAX_SP];
      Kokkos::View<particle_mover_t*, Kokkos::MemoryUnmanaged> sp_pm_v[MAX_SP];
      Kokkos::View<int[MAX_SP]>::HostMirror num_to_copy_h("Num to copy");
      for(int idx=0; idx<num_sp; idx++) {
        particle_send[idx]   = sp_[idx]->k_pc_h;
        particle_send_i[idx] = sp_[idx]->k_pc_i_h;
        particle_recv[idx]   = sp_[idx]->k_pr_h;
        particle_recv_i[idx] = sp_[idx]->k_pr_i_h;
        num_to_copy_h(idx)   = sp_[idx]->num_to_copy;
        sp_pm_v[idx] = Kokkos::View<particle_mover_t*, Kokkos::MemoryUnmanaged>(sp_[idx]->pm, sp_[idx]->max_nm);
      }
      auto& jf_accum = fa->k_jf_accum_h;
      const auto& neighbors = g->k_neighbor_h;
      Kokkos::View<const particle_injector_t*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> injectors(pi, n);
      Kokkos::parallel_for("boundary_p unpack/resend", host_execution_policy(0,n), KOKKOS_LAMBDA(const int idx) {
        int id = injectors(idx).sp_id;
      
        //particle_mover_t* pm = sp_pm[id];
        //int nm = sp_nm_view(id);
      
        // Extract particle data
        particle_t p = { injectors(idx).dx,
                         injectors(idx).dy,
                         injectors(idx).dz,
                         injectors(idx).i,
                         injectors(idx).ux,
                         injectors(idx).uy,
                         injectors(idx).uz,
                         injectors(idx).w };
      
        // Extract mover data
        particle_mover_t pm = { injectors(idx).dispx,
                                injectors(idx).dispy,
                                injectors(idx).dispz,
                                -1 };
        //sp_pm_v[id](nm).dispx = injectors(idx).dispx; 
        //sp_pm_v[id](nm).dispy = injectors(idx).dispy; 
        //sp_pm_v[id](nm).dispz = injectors(idx).dispz;
        //sp_pm_v[id](nm).i     = nm; //write_index; // Try tell it the index we wrote to
      
        // FIXME: this relies on serial for now -- maybe bad?
        const int ret_code = move_p_kokkos_host_serial(
                p,
                //&(pm[nm]),
                pm,
                jf_accum,
                neighbors,
                rangel,
                rangeh,
                nx, ny, nz,
                cx, cy, cz,
                sp_q_view(id)
        );
      
        if (ret_code) {
          const int keep_id = Kokkos::atomic_fetch_inc(&sp_nm_view(id));
      
          // Add particle to the "send" array for next iter
          particle_send[id](keep_id, particle_var::dx) = p.dx; 
          particle_send[id](keep_id, particle_var::dy) = p.dy; 
          particle_send[id](keep_id, particle_var::dz) = p.dz; 
          particle_send[id](keep_id, particle_var::ux) = p.ux; 
          particle_send[id](keep_id, particle_var::uy) = p.uy; 
          particle_send[id](keep_id, particle_var::uz) = p.uz; 
          particle_send[id](keep_id, particle_var::w)  = p.w;  
          particle_send_i[id](keep_id)                 = p.i;  

          sp_pm_v[id](keep_id).dispx = pm.dispx;
          sp_pm_v[id](keep_id).dispy = pm.dispy;
          sp_pm_v[id](keep_id).dispz = pm.dispz;
          sp_pm_v[id](keep_id).i = pm.i;
        } else {
          const int write_index = Kokkos::atomic_fetch_inc(&(num_to_copy_h(id)));
      
          // Write out received particle data
          particle_recv[id](write_index, particle_var::dx) = p.dx; 
          particle_recv[id](write_index, particle_var::dy) = p.dy; 
          particle_recv[id](write_index, particle_var::dz) = p.dz; 
          particle_recv[id](write_index, particle_var::ux) = p.ux; 
          particle_recv[id](write_index, particle_var::uy) = p.uy; 
          particle_recv[id](write_index, particle_var::uz) = p.uz; 
          particle_recv[id](write_index, particle_var::w)  = p.w;  
          particle_recv_i[id](write_index)                 = p.i;  
        }
      });
      Kokkos::fence();
      for(int idx=0; idx<num_sp; idx++) {
        sp_nm[idx] = sp_nm_view(idx);
        sp_[idx]->num_to_copy = num_to_copy_h(idx);
      }

    } while(face!=5);

    LIST_FOR_EACH( sp, sp_list ) {
      sp->nm=sp_nm[sp->id];
    }

  } while(0);
Kokkos::Profiling::popRegion();

  for( face=0; face<6; face++ ) {
    if( shared[face] ) {
      mp_end_send(mp,f2b[face]);
    }
  }

  // If there is additional bound charge, update rhob on device
  // Having the accumulator array saves us from copying rhob to the host every
  // step where a particle is absorbed.
  if (absorbed) {
Kokkos::Profiling::pushRegion("BoundaryP -> update rhob");
    int n_fields = fa->g->nv;
    auto& kfd = fa->k_f_d;
    auto& kfad = fa->k_f_rhob_accum_d;
    auto& kfah = fa->k_f_rhob_accum_h;
    Kokkos::deep_copy(kfad, kfah);

    Kokkos::parallel_for("Add rhob accumulation to device rhob", 
    Kokkos::RangePolicy(0, n_fields), KOKKOS_LAMBDA (const int i) {
      kfd(i, field_var::rhob) += kfad(i);
    });

    // Zero host accum array
    Kokkos::deep_copy(kfah, 0.0f);
Kokkos::Profiling::popRegion();
  }
}
