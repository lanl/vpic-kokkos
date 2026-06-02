// C++ headers
#include <cassert>
#include <iostream>
#include <filesystem>

// VPIC headers
#include "vpic.h"
#include "dumpmacros.h"
#include "../util/io/FileUtils.h"

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"
#endif

#ifdef VPIC_ENABLE_TRACER_PARTICLES
/*------------------------------------------------------------------------------
 * Text Dumps
 *---------------------------------------------------------------------------*/
void
vpic_simulation::dump_tracers_csv( const char *sp_name,
                                   uint32_t dump_vars,
                                   const char *fbase,
                                   int ftag )
{
    species_t *sp;
    constexpr int max_filename_bytes = 256;
    char fname[max_filename_bytes];
    FileIO fileIO;
    bool append = access(fname, F_OK) == 0;

    // Get species
    sp = find_species_name( sp_name, tracers_list );
    if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name ));

    if( !fbase ) ERROR(( "Invalid filename" ));

    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied) {
      sp->copy_to_host();
    }

    // Create output filename
    if( ftag ) {
        snprintf( fname, max_filename_bytes, "%s.%li.%i.csv", fbase, (long)step(), rank() );
    }
    else {
        snprintf( fname, max_filename_bytes, "%s.%i.csv", fbase, rank() );
    }

    //Add filename if missing
    std::string filename = std::string(fname);
    std::filesystem::path filepath(fbase);
    if(filepath.filename().compare("") == 0) { 
      filepath = filepath / std::filesystem::path(filename);
    }
    // Handle relative input path
    if(filepath.is_relative()) { 
      filepath = std::filesystem::current_path() / filepath;
    }
    // Make sure path exists
    if(!std::filesystem::exists(filepath.parent_path())) {
      std::filesystem::create_directories(filepath.parent_path());
    }

    if( rank()==0 )
        MESSAGE(("Dumping \"%s\" particles to \"%s\"",sp->name,filepath.c_str()));

    FileIOStatus status = fileIO.open(fname, append ? io_append : io_write);
    if( status==fail ) ERROR(( "Could not open \"%s\"", fname ));

    // Create header string
    std::string header_str = std::string("Timestep,rank,tracer_id,cell_id,dx,dy,dz,ux,uy,uz,w");
    if(dump_vars & DumpVar::GlobalPos) {
      header_str += ",posx,posy,posz";
    }
    if(dump_vars & DumpVar::Efield) {
      header_str += ",ex,ey,ez";
    }
    if(dump_vars & DumpVar::Bfield) {
      header_str += ",bx,by,bz";
    }
    if(dump_vars & DumpVar::CurrentDensity) {
      header_str += ",jx,jy,jz";
    }
    if(dump_vars & DumpVar::ChargeDensity) {
      header_str += ",rho";
    }
    if(dump_vars & DumpVar::MomentumDensity) {
      header_str += ",px,py,pz";
    }
    if(dump_vars & DumpVar::MassDensity) {
      header_str += ",mass_dens";
    }
    if(dump_vars & DumpVar::StressTensor) {
      header_str += ",txx,tyy,tzz,tyz,tzx,txy";
    }
    if(dump_vars & DumpVar::ParticleKE) {
      header_str += ",ke";
    }
    for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.i32_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.i64_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.f32_vars[j].c_str();
    }
    for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
      header_str += ",";
      header_str += sp->annotation_vars.f64_vars[j].c_str();
    }

    // Write header string
    //if( append==0 ) {
      fileIO.print( "%s\n", header_str.c_str() );
    //}

    auto& particles = sp->k_p_d;
    auto& particles_i = sp->k_p_i_d;
    auto& interpolators_k = interpolator_array->k_i_d;
    interpolator_array->copy_to_host();

    // If needed, copy weight back to particles for hydro quantities
    if(sp->tracer_type == TracerType::Copy) {
      int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_annote = Kokkos::subview(sp->annotations_h.f32, Kokkos::ALL(), w_idx);
      Kokkos::deep_copy(w_subview_h, w_annote);
      Kokkos::deep_copy(w_subview_d, w_subview_h);
    }

    // Compute hydro quantities
    if(static_cast<uint32_t>(dump_vars) >= 2*DumpVar::ParticleKE) {
      Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
      accumulate_hydro_p_kokkos(
          particles,
          particles_i,
          hydro_array->k_h_d,
          interpolators_k,
          sp
      );

      // This is slower in my tests
      synchronize_hydro_array_kokkos(hydro_array);

      hydro_array->copy_to_host();

      //synchronize_hydro_array( hydro_array );

      //hydro_array->copy_to_device();
    }

    int tracer_idx = sp->annotation_vars.get_annotation_index<int>("TracerID");
    auto& interp = interpolator_array->k_i_h;

#define _nxg (grid->nx + 2)
#define _nyg (grid->ny + 2)
#define _nzg (grid->nz + 2)
#define i0 (ii%_nxg)
#define j0 ((ii/_nxg)%_nyg)
#define k0 (ii/(_nxg*_nyg))
#define tracer_x ((i0 + (dx0-1)*0.5) * grid->dx + grid->x0)
#define tracer_y ((j0 + (dy0-1)*0.5) * grid->dy + grid->y0)
#define tracer_z ((k0 + (dz0-1)*0.5) * grid->dz + grid->z0)
    // Write tracer data
    for(size_t i=0; i<sp->np; i++) {
      float dx0 = sp->k_p_h(i, particle_var::dx);
      float dy0 = sp->k_p_h(i, particle_var::dy);
      float dz0 = sp->k_p_h(i, particle_var::dz);
      float ux0 = sp->k_p_h(i, particle_var::ux);
      float uy0 = sp->k_p_h(i, particle_var::uy);
      float uz0 = sp->k_p_h(i, particle_var::uz);
      float w0  = sp->k_p_h(i, particle_var::w);
      int   ii  = sp->k_p_i_h(i);
      fileIO.print("%ld,%d,%ld,%d,%e,%e,%e,%e,%e,%e,%e", 
        step(), rank(), sp->annotations_h.get<int>(i,tracer_idx), ii,
        dx0, dy0, dz0, ux0, uy0, uz0, w0);
      if(dump_vars & DumpVar::GlobalPos) {
        fileIO.print(",%e,%e,%e", tracer_x, tracer_y, tracer_z);
      }
      if(dump_vars & DumpVar::Efield) {
        float ex  = interp(ii,interpolator_var::ex ); 
        float ey  = interp(ii,interpolator_var::ey ); 
        float ez  = interp(ii,interpolator_var::ez ); 
        fileIO.print(",%e,%e,%e", ex, ey, ez);
      }
      if(dump_vars & DumpVar::Bfield) {
        float bx  = interp(ii,interpolator_var::cbx); 
        float by  = interp(ii,interpolator_var::cby); 
        float bz  = interp(ii,interpolator_var::cbz); 
        fileIO.print(",%e,%e,%e", bx, by, bz);
      }
      if(dump_vars & DumpVar::CurrentDensity) {
        float jx  = hydro_array->k_h_h(ii, hydro_var::jx);
        float jy  = hydro_array->k_h_h(ii, hydro_var::jy);
        float jz  = hydro_array->k_h_h(ii, hydro_var::jz);
        fileIO.print(",%e,%e,%e", jx, jy, jz);
      }
      if(dump_vars & DumpVar::ChargeDensity) {
        float rho = hydro_array->k_h_h(ii, hydro_var::rho);
        fileIO.print(",%e", rho);
      }
      if(dump_vars & DumpVar::MomentumDensity) {
        float px  = hydro_array->k_h_h(ii, hydro_var::px);
        float py  = hydro_array->k_h_h(ii, hydro_var::py);
        float pz  = hydro_array->k_h_h(ii, hydro_var::pz);
        fileIO.print(",%e,%e,%e", px, py, pz);
      }
      if(dump_vars & DumpVar::MassDensity) {
        float rho_m = hydro_array->k_h_h(ii, hydro_var::rho_m);
        fileIO.print(",%e", rho_m);
      }
      if(dump_vars & DumpVar::StressTensor) {
        float txx = hydro_array->k_h_h(ii, hydro_var::txx);
        float tyy = hydro_array->k_h_h(ii, hydro_var::tyy);
        float tzz = hydro_array->k_h_h(ii, hydro_var::tzz);
        float tyz = hydro_array->k_h_h(ii, hydro_var::tyz);
        float tzx = hydro_array->k_h_h(ii, hydro_var::tzx);
        float txy = hydro_array->k_h_h(ii, hydro_var::txy);
        fileIO.print(",%e,%e,%e,%e,%e,%e", txx, tyy, tzz, tyz, tzx, txy);
      }
      if(dump_vars & DumpVar::ParticleKE) {
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        fileIO.print(",%e", v0);
      }
      // Print annotations
      for(auto j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
        fileIO.print(",%d", sp->annotations_h.get<int>(i, j));
      }
      for(auto j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
        fileIO.print(",%ld", sp->annotations_h.get<int64_t>(i, j));
      }
      for(auto j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
        fileIO.print(",%e", sp->annotations_h.get<float>(i, j));
      }
      for(auto j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
        fileIO.print(",%e", sp->annotations_h.get<double>(i, j));
      }
      fileIO.print( "\n" );
    }
#undef nxg 
#undef nyg 
#undef nzg 
#undef i0 
#undef j0 
#undef k0 
#undef tracer_x 
#undef tracer_y 
#undef tracer_z 
#ifdef VPIC_ENABLE_PARTICLE_ANNOTATIONS
    // If needed, reset weight to 0 for copied particles
    if(sp->tracer_type == TracerType::Copy) {
      auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
      auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
      Kokkos::deep_copy(w_subview_h, 0.0);
      Kokkos::deep_copy(w_subview_d, 0.0);
    }
#endif

    if( fileIO.close() ) ERROR(("File close failed on dump particles!!!"));
}

/*------------------------------------------------------------------------------
 * HDF5 Dumps
 *---------------------------------------------------------------------------*/
#ifdef VPIC_ENABLE_HDF5
template<class ViewSlice>
herr_t write_dataset(const ViewSlice& slice, 
                     const std::string& name, 
                     const hid_t loc_id, 
                     const hid_t type_id,
                     const hid_t dataspace_id, 
                     const hid_t memspace_id, 
                     const hid_t dxpl_id,
                     const hid_t es_id = H5I_INVALID_HID) {
#ifdef VPIC_ENABLE_HDF5_ASYNC
  if(es_id != H5I_INVALID_HID) {
    hid_t dataset_id = H5Dcreate_async(loc_id, name.c_str(), type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id);
    herr_t status = H5Dwrite_async(dataset_id, type_id, memspace_id, dataspace_id, dxpl_id, slice.data(), es_id);
    status = H5Dclose_async(dataset_id, es_id);
    return status;
  } 
#endif
  hid_t dataset_id = H5Dcreate(loc_id, name.c_str(), type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status = H5Dwrite(dataset_id, type_id, memspace_id, dataspace_id, dxpl_id, slice.data());
  status = H5Dclose(dataset_id);
  return status;
}

template<class ParticleView, class CellIDView, class Slice>
herr_t write_particles(const ParticleView& particles,
                       const CellIDView& cell_ids,
                       const Slice& slice,
                       const hid_t loc_id, 
                       const hid_t dataspace_id, 
                       const hid_t memspace_id, 
                       const hid_t dxpl_id,
                       const hid_t es_id = H5I_INVALID_HID) {
  herr_t err;
  // Create subviews for data
  auto dx_subview = Kokkos::subview(particles, slice, (int)(particle_var::dx));
  auto dy_subview = Kokkos::subview(particles, slice, (int)(particle_var::dy));
  auto dz_subview = Kokkos::subview(particles, slice, (int)(particle_var::dz));
  auto ux_subview = Kokkos::subview(particles, slice, (int)(particle_var::ux));
  auto uy_subview = Kokkos::subview(particles, slice, (int)(particle_var::uy));
  auto uz_subview = Kokkos::subview(particles, slice, (int)(particle_var::uz));
  auto w_subview  = Kokkos::subview(particles, slice, (int)(particle_var::w));
#ifdef VARIABLE_CHARGE
  auto qp_subview = Kokkos::subview(particles, slice, (int)(particle_var::qp));
#endif
  auto i_subview  = Kokkos::subview(cell_ids, slice);
  
  err = write_dataset(dx_subview, "dx", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
  err = write_dataset(dy_subview, "dy", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
  err = write_dataset(dz_subview, "dz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
  err = write_dataset(ux_subview, "ux", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
  err = write_dataset(uy_subview, "uy", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
  err = write_dataset(uz_subview, "uz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
  err = write_dataset(w_subview,  "w",  loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
#ifdef VARIABLE_CHARGE
  err = write_dataset(qp_subview, "qp", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, dxpl_id, es_id);
#endif
  err = write_dataset(i_subview,  "i",  loc_id, H5T_STD_I32LE,  dataspace_id, memspace_id, dxpl_id, es_id);
  return err;
}

template<class Slice>
void 
write_tracers(species_t* sp, 
              const hydro_array_t* hydro_array,
              const k_interpolator_t::HostMirror& interp, 
              const Slice& slice,
              const bool buffered,
              const size_t num_particles,
              const uint32_t dump_vars,
              hid_t loc_id, 
              hid_t dataspace_id, 
              hid_t memspace_id,
              hid_t dxpl_id,
              hid_t es_id = H5I_INVALID_HID) {
  herr_t status;

  const grid_t* grid = sp->g;
  auto* particle_ptr = &(sp->k_p_h);
  auto* particle_i_ptr = &(sp->k_p_i_h);
  if(buffered) {
    particle_ptr = &(sp->particle_io_buffer_h);
    particle_i_ptr = &(sp->particle_cell_io_buffer_h);
  }

  status = write_particles(*particle_ptr,
                           *particle_i_ptr,
                           slice,
                           loc_id, 
                           dataspace_id, 
                           memspace_id, 
                           dxpl_id,
                           es_id);
  
  // Create subviews for data
  auto dx_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::dx));
  auto dy_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::dy));
  auto dz_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::dz));
  auto ux_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::ux));
  auto uy_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::uy));
  auto uz_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::uz));
  auto w_subview  = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::w));
#ifdef VARIABLE_CHARGE
  auto qp_subview = Kokkos::subview(*particle_ptr, slice, (int)(particle_var::qp));
#endif
  auto i_subview  = Kokkos::subview(*particle_i_ptr, slice);

  using host_memory_space = Kokkos::DefaultHostExecutionSpace::memory_space;

  auto& h_hydro = hydro_array->k_h_h;
  const int grid_nx = grid->nx, grid_ny=grid->ny, grid_nz=grid->nz;
  const float grid_dx = grid->dx, grid_dy=grid->dy, grid_dz=grid->dz;
  const float grid_x0 = grid->x0, grid_y0=grid->y0, grid_z0=grid->z0;
  auto pack_policy = Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace,size_t>(static_cast<size_t>(0), num_particles);

  // Dump Global position if specified
  if(dump_vars & DumpVar::GlobalPos) {
    auto pos_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Pos Host View", num_particles);
    Kokkos::parallel_for("Calculate global position", pack_policy, KOKKOS_LAMBDA(const size_t i) {
      // Compute global position of particle
      const int ii   = i_subview(i);
      const int nxg_ = grid_nx + 2;
      const int nyg_ = grid_ny + 2;
      const int i0   = ii % nxg_;
      const int j0   = (ii/nxg_) % nyg_;
      const int k0   = ii/(nxg_*nyg_);
      pos_view(i, 0) = (i0 + (dx_subview(i)-1)*0.5) * grid_dx + grid_x0;
      pos_view(i, 1) = (j0 + (dy_subview(i)-1)*0.5) * grid_dy + grid_y0;
      pos_view(i, 2) = (k0 + (dz_subview(i)-1)*0.5) * grid_dz + grid_z0;
    });
    auto posx_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 0);
    auto posy_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 1);
    auto posz_subview = Kokkos::subview(pos_view, Kokkos::ALL(), 2);
    status = write_dataset(posx_subview, "posx", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(posy_subview, "posy", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(posz_subview, "posz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump E field if specified
  if(dump_vars & DumpVar::Efield) {
    auto ex_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::ex);
    auto ey_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::ey);
    auto ez_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::ez);
    Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space> efield_view;
    if(!buffered) {
      efield_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("E field Host View", num_particles);
      Kokkos::parallel_for("Calculate E field", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        const int ii = i_subview(i);
        efield_view(i,0) = interp(ii,interpolator_var::ex); 
        efield_view(i,1) = interp(ii,interpolator_var::ey); 
        efield_view(i,2) = interp(ii,interpolator_var::ez); 
      });
      Kokkos::fence();
      ex_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 0);
      ey_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 1);
      ez_subview = Kokkos::subview(efield_view, Kokkos::ALL(), 2);
    }
    status = write_dataset(ex_subview, "ex", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(ey_subview, "ey", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(ez_subview, "ez", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump B field if specified
  if(dump_vars & DumpVar::Bfield) {
    auto bx_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::bx);
    auto by_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::by);
    auto bz_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::bz);
    Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space> bfield_view;
    if(!buffered) {
      bfield_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("B field Host View", num_particles);
      Kokkos::parallel_for("Calculate B field", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        const int   ii  = i_subview(i);
        bfield_view(i,0)  = interp(ii,interpolator_var::cbx); 
        bfield_view(i,1)  = interp(ii,interpolator_var::cby); 
        bfield_view(i,2)  = interp(ii,interpolator_var::cbz); 
      });
      Kokkos::fence();
      bx_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 0);
      by_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 1);
      bz_subview = Kokkos::subview(bfield_view, Kokkos::ALL(), 2);
    }
    status = write_dataset(bx_subview, "bx", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(by_subview, "by", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(bz_subview, "bz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump current density if specified
  if(dump_vars & DumpVar::CurrentDensity) {
    auto jx_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::jx);
    auto jy_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::jy);
    auto jz_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::jz);
    Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space> current_view;
    if(!buffered) {
      current_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Current density Host View", num_particles);
      Kokkos::parallel_for("Collect current density", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        const int   ii  = i_subview(i);
        current_view(i, 0) = h_hydro(ii, hydro_var::jx);
        current_view(i, 1) = h_hydro(ii, hydro_var::jy);
        current_view(i, 2) = h_hydro(ii, hydro_var::jz);
      });
      Kokkos::fence();
      jx_subview = Kokkos::subview(current_view, Kokkos::ALL(), 0);
      jy_subview = Kokkos::subview(current_view, Kokkos::ALL(), 1);
      jz_subview = Kokkos::subview(current_view, Kokkos::ALL(), 2);
    }
    status = write_dataset(jx_subview, "jx", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(jy_subview, "jy", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(jz_subview, "jz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump charge density if specified
  if(dump_vars & DumpVar::ChargeDensity) {
    auto charge_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::rho);
    Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space> charge_view;
    if(!buffered) {
      charge_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("Charge density Host View", num_particles);
      Kokkos::parallel_for("Calculate charge density", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        const int   ii  = i_subview(i);
        charge_view(i) = h_hydro(ii, hydro_var::rho);
      });
      Kokkos::fence();
      charge_subview = Kokkos::subview(charge_view, Kokkos::ALL);
    }
    status = write_dataset(charge_subview, "rho", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump momentum density if specified
  if(dump_vars & DumpVar::MomentumDensity) {
    auto px_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::px);
    auto py_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::py);
    auto pz_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::pz);
    Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space> momentum_view;
    if(!buffered) {
      momentum_view = Kokkos::View<float*[3], Kokkos::LayoutLeft, host_memory_space>("Momentum Host View", num_particles);
      Kokkos::parallel_for("Collect momentum density", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        const int   ii  = i_subview(i);
        momentum_view(i, 0) = h_hydro(ii, hydro_var::px);
        momentum_view(i, 1) = h_hydro(ii, hydro_var::py);
        momentum_view(i, 2) = h_hydro(ii, hydro_var::pz);
      });
      Kokkos::fence();
      px_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 0);
      py_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 1);
      pz_subview = Kokkos::subview(momentum_view, Kokkos::ALL(), 2);
    }
    status = write_dataset(px_subview, "px", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(py_subview, "py", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(pz_subview, "pz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump mass density if specified
  if(dump_vars & DumpVar::MassDensity) {
    auto rho_m_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::rho_m);
    Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space> rho_m_view;
    if(!buffered) {
      rho_m_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("Mass dens Host View", num_particles);
      Kokkos::parallel_for("Collect KE density", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        int   ii  = i_subview(i);
        rho_m_view(i) = h_hydro(ii, hydro_var::rho_m);
      });
      Kokkos::fence();
      rho_m_subview = Kokkos::subview(rho_m_view, Kokkos::ALL());
    }
    status = write_dataset(rho_m_subview, "mass_dens", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump stress tensor if specified
  if(dump_vars & DumpVar::StressTensor) {
    auto txx_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::txx);
    auto tyy_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::tyy);
    auto tzz_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::tzz);
    auto tyz_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::tyz);
    auto tzx_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::tzx);
    auto txy_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::txy);
    Kokkos::View<float*[6], Kokkos::LayoutLeft, host_memory_space> stress_view;
    if(!buffered) {
      auto stress_view = Kokkos::View<float*[6], Kokkos::LayoutLeft, host_memory_space>("Stress tensor Host View", num_particles);
      Kokkos::parallel_for("Collect stress tensor", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        const int   ii  = i_subview(i);
        stress_view(i, 0) = h_hydro(ii, hydro_var::txx);
        stress_view(i, 1) = h_hydro(ii, hydro_var::tyy);
        stress_view(i, 2) = h_hydro(ii, hydro_var::tzz);
        stress_view(i, 3) = h_hydro(ii, hydro_var::tyz);
        stress_view(i, 4) = h_hydro(ii, hydro_var::tzx);
        stress_view(i, 5) = h_hydro(ii, hydro_var::txy);
      });
      Kokkos::fence();
      txx_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 0);
      tyy_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 1);
      tzz_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 2);
      tyz_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 3);
      tzx_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 4);
      txy_subview = Kokkos::subview(stress_view, Kokkos::ALL(), 5);
    }
    status = write_dataset(txx_subview, "txx", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(tyy_subview, "tyy", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(tzz_subview, "tzz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(tyz_subview, "tyz", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(tzx_subview, "tzx", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    status = write_dataset(txy_subview, "txy", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }

  // Dump kinetic energy of particle if specified
  if(dump_vars & DumpVar::ParticleKE) {
    if(buffered) {
      auto ke_subview = Kokkos::subview(sp->tracer_buffer_h, slice, (int)tracer_buffer_var::ke);
      status = write_dataset(ke_subview, "ke", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    } else {
      float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
      float msp = sp->m;
      Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space> ke_view = Kokkos::View<float*, Kokkos::LayoutLeft, host_memory_space>("KE Host View", num_particles);
      Kokkos::parallel_for("Calculate KE", pack_policy, KOKKOS_LAMBDA(const size_t i) {
        float dx0 = dx_subview(i);
        float dy0 = dy_subview(i);
        float dz0 = dz_subview(i);
        int   ii  = i_subview(i);
        float ux0 = ux_subview(i);
        float uy0 = uy_subview(i);
        float uz0 = uz_subview(i);
        float w0  = w_subview(i);
        float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
        float msp = sp->m;
        float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                               dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
        float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                               dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
        float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                               dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
        v0 = v0*v0 + v1*v1 + v2*v2;
        v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
        ke_view(i) = v0;
      });
      Kokkos::fence();
      status = write_dataset(ke_view, "ke", loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
    }
  }

  // Dump int annotations
  for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
    auto* annote = buffered ? &(sp->annotations_io_buffer_h.i32) 
                            : &(sp->annotations_h.i32);
    auto i32_subview = Kokkos::subview(*annote, slice, j);
    status = write_dataset(i32_subview, sp->annotation_vars.i32_vars[j].c_str(), loc_id, H5T_STD_I32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }
  // Dump 64-bit integer annotations
  for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
    auto* annote = &(sp->annotations_h.i64);
    if(buffered)
      annote = &(sp->annotations_io_buffer_h.i64);
    auto i64_subview = Kokkos::subview(*annote, slice, j);
    status = write_dataset(i64_subview, sp->annotation_vars.i64_vars[j].c_str(), loc_id, H5T_STD_I64LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }
  // Dump 32-bit floating-point annotations
  for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
    auto* annote = &(sp->annotations_h.f32);
    if(buffered)
      annote = &(sp->annotations_io_buffer_h.f32);
    auto f32_subview = Kokkos::subview(*annote, slice, j);
    status = write_dataset(f32_subview, sp->annotation_vars.f32_vars[j].c_str(), loc_id, H5T_IEEE_F32LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }
  // Dump 64-bit floating-point annotations
  for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
    auto* annote = &(sp->annotations_h.f64);
    if(buffered)
      annote = &(sp->annotations_io_buffer_h.f64);
    auto f64_subview = Kokkos::subview(*annote, slice, j);
    status = write_dataset(f64_subview, sp->annotation_vars.f64_vars[j].c_str(), loc_id, H5T_IEEE_F64LE, dataspace_id, memspace_id, H5P_DEFAULT, es_id);
  }
}

void buffer_tracers(species_t* sp,
                    hydro_array_t* ha,
                    interpolator_array_t* ia,
                    const uint32_t dump_vars,
                    const int rank,
                    const int step) {
  uint64_t num_particles = sp->np;
  uint64_t total_particles = 0;
  MPI_Allreduce(&num_particles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if( rank==0 )
      MESSAGE(("Step %ld: Buffering %lu \"%s\" particles", step, total_particles, sp->name));

  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interp = ia->k_i_d;

  size_t nbuffered = sp->np_buffered;
  auto particle_slice = Kokkos::make_pair(static_cast<size_t>(0), sp->np);
  auto buffer_slice = Kokkos::make_pair(nbuffered, nbuffered+sp->np);

  // Copy particles into buffer
  auto particle_subview = Kokkos::subview(sp->k_p_d, particle_slice, Kokkos::ALL());
  auto particle_buffer_subview = Kokkos::subview(sp->particle_io_buffer_d, buffer_slice, Kokkos::ALL());
  auto particle_cell_subview = Kokkos::subview(sp->k_p_i_d, particle_slice);
  auto particle_cell_buffer_subview = Kokkos::subview(sp->particle_cell_io_buffer_d, buffer_slice);
  Kokkos::deep_copy(particle_buffer_subview, particle_subview); 
  Kokkos::deep_copy(particle_cell_buffer_subview, particle_cell_subview); 

  // Copy annotations into buffers
  for(uint32_t j=0; j<sp->annotation_vars.i32_vars.size(); j++) {
    auto annote_subview = Kokkos::subview(sp->annotations_d.i32, particle_slice, j);
    auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer_d.i32, buffer_slice, j);
    Kokkos::deep_copy(buffer_subview, annote_subview);
  }
  for(uint32_t j=0; j<sp->annotation_vars.i64_vars.size(); j++) {
    auto annote_subview = Kokkos::subview(sp->annotations_d.i64, particle_slice, j);
    auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer_d.i64, buffer_slice, j);
    Kokkos::deep_copy(buffer_subview, annote_subview);
  }
  for(uint32_t j=0; j<sp->annotation_vars.f32_vars.size(); j++) {
    auto annote_subview = Kokkos::subview(sp->annotations_d.f32, particle_slice, j);
    auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer_d.f32, buffer_slice, j);
    Kokkos::deep_copy(buffer_subview, annote_subview);
  }
  for(uint32_t j=0; j<sp->annotation_vars.f64_vars.size(); j++) {
    auto annote_subview = Kokkos::subview(sp->annotations_d.f64, particle_slice, j);
    auto buffer_subview = Kokkos::subview(sp->annotations_io_buffer_d.f64, buffer_slice, j);
    Kokkos::deep_copy(buffer_subview, annote_subview);
  }

  // Buffer tracer data
  sp->np_per_ts.push_back(std::make_pair(sp->np, step));
  auto e_buffer_d           = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, Kokkos::make_pair((int)tracer_buffer_var::ex, (int)tracer_buffer_var::ez+1));
  auto b_buffer_d           = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, Kokkos::make_pair((int)tracer_buffer_var::bx, (int)tracer_buffer_var::bz+1));
  auto current_buffer_d     = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, Kokkos::make_pair((int)tracer_buffer_var::jx, (int)tracer_buffer_var::jz+1));
  auto charge_buffer_d      = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, (int)tracer_buffer_var::rho);
  auto momentum_buffer_d    = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, Kokkos::make_pair((int)tracer_buffer_var::px, (int)tracer_buffer_var::pz+1));
  auto mass_dens_buffer_d   = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, (int)tracer_buffer_var::rho_m);
  auto stress_buffer_d      = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, Kokkos::make_pair((int)tracer_buffer_var::txx, (int)tracer_buffer_var::txy+1));
  auto particle_ke_buffer_d = Kokkos::subview(sp->tracer_buffer_d, Kokkos::ALL, (int)tracer_buffer_var::ke);
  auto& hydro_d = ha->k_h_d;
  float qdt_2mc = (sp->q*sp->g->dt)/(2*sp->m*sp->g->cvac);
  float msp = sp->m;

  Kokkos::parallel_for("Buffer data", Kokkos::RangePolicy<size_t>(0, sp->np), KOKKOS_LAMBDA(const size_t i) {
    float dx0 = particles(i, particle_var::dx);
    float dy0 = particles(i, particle_var::dy);
    float dz0 = particles(i, particle_var::dz);
    float ux0 = particles(i, particle_var::ux);
    float uy0 = particles(i, particle_var::uy);
    float uz0 = particles(i, particle_var::uz);
    float w0  = particles(i, particle_var::w);
    int   ii  = particles_i(i);
    if(dump_vars & DumpVar::Efield) {
      e_buffer_d(nbuffered+i, 0) = interp(ii,interpolator_var::ex);
      e_buffer_d(nbuffered+i, 1) = interp(ii,interpolator_var::ey);
      e_buffer_d(nbuffered+i, 2) = interp(ii,interpolator_var::ez);
    }
    if(dump_vars & DumpVar::Bfield) {
      b_buffer_d(nbuffered+i, 0) = interp(ii,interpolator_var::cbx);
      b_buffer_d(nbuffered+i, 1) = interp(ii,interpolator_var::cby);
      b_buffer_d(nbuffered+i, 2) = interp(ii,interpolator_var::cbz);
    }
    if(dump_vars & DumpVar::CurrentDensity) {
      current_buffer_d(nbuffered+i, 0) = hydro_d(ii, hydro_var::jx);
      current_buffer_d(nbuffered+i, 1) = hydro_d(ii, hydro_var::jy);
      current_buffer_d(nbuffered+i, 2) = hydro_d(ii, hydro_var::jz);
    }
    if(dump_vars & DumpVar::ChargeDensity) {
      charge_buffer_d(nbuffered+i) = hydro_d(ii, hydro_var::rho);
    }
    if(dump_vars & DumpVar::MomentumDensity) {
      momentum_buffer_d(nbuffered+i, 0) = hydro_d(ii, hydro_var::px);
      momentum_buffer_d(nbuffered+i, 1) = hydro_d(ii, hydro_var::py);
      momentum_buffer_d(nbuffered+i, 2) = hydro_d(ii, hydro_var::pz);
    }
    if(dump_vars & DumpVar::MassDensity) {
      mass_dens_buffer_d(nbuffered+i) = hydro_d(ii, hydro_var::rho_m);
    }
    if(dump_vars & DumpVar::StressTensor) {
      stress_buffer_d(nbuffered+i, 0) = hydro_d(ii, hydro_var::txx);
      stress_buffer_d(nbuffered+i, 1) = hydro_d(ii, hydro_var::tyy);
      stress_buffer_d(nbuffered+i, 2) = hydro_d(ii, hydro_var::tzz);
      stress_buffer_d(nbuffered+i, 3) = hydro_d(ii, hydro_var::tyz);
      stress_buffer_d(nbuffered+i, 4) = hydro_d(ii, hydro_var::tzx);
      stress_buffer_d(nbuffered+i, 5) = hydro_d(ii, hydro_var::txy);
    }
    if(dump_vars & DumpVar::ParticleKE) {
      float v0 = ux0 + qdt_2mc*( ( interp(ii, interpolator_var::ex)    + dy0*interp(ii, interpolator_var::dexdy)    ) +
                             dz0*( interp(ii, interpolator_var::dexdz) + dy0*interp(ii, interpolator_var::d2exdydz) ) );
      float v1 = uy0 + qdt_2mc*( ( interp(ii, interpolator_var::ey)    + dz0*interp(ii, interpolator_var::deydz)    ) +
                             dx0*( interp(ii, interpolator_var::deydx) + dz0*interp(ii, interpolator_var::d2eydzdx) ) );
      float v2 = uz0 + qdt_2mc*( ( interp(ii, interpolator_var::ez)    + dx0*interp(ii, interpolator_var::dezdx)    ) +
                             dy0*( interp(ii, interpolator_var::dezdy) + dx0*interp(ii, interpolator_var::d2ezdxdy) ) );
      v0 = v0*v0 + v1*v1 + v2*v2;
      v0 = (msp * w0) * (v0 / (1 + sqrtf(1 + v0)));
      particle_ke_buffer_d(nbuffered+i) = v0;
    }
  });
  sp->np_buffered += sp->np;     
}

void
vpic_simulation::dump_tracers(const char *sp_name,
                              uint32_t dump_vars,
                              const char *fbase,
                              const bool buffer_enabled, /*= true*/
                              const bool async_enabled /*= false*/)
{
  species_t *sp;
  FileIO fileIO;
  char group_name[256];

  // Get species
  sp = find_species_name( sp_name, tracers_list );
  if( !sp ) ERROR(( "Invalid tracer species name \"%s\".", sp_name ));

  if( !fbase ) ERROR(( "Invalid filename" ));

  //Add filename if missing
  std::string filename = std::string(sp_name);
  if(buffer_enabled) {
    filename += std::string("_buffered.h5");
  } else {
    filename += std::string(".h5");
  }
  std::filesystem::path filepath(fbase);
  if(filepath.filename().compare("") == 0) { 
    filepath = filepath / std::filesystem::path(filename);
  }
  // Handle relative input path
  if(filepath.is_relative()) { 
    filepath = std::filesystem::current_path() / filepath;
  }
  // Make sure path exists
  if(!std::filesystem::exists(filepath.parent_path())) {
    std::filesystem::create_directories(filepath.parent_path());
  }

  hid_t es_id = H5I_INVALID_HID;
#ifdef VPIC_ENABLE_HDF5_ASYNC
  // Create event set at initial call
  bool append = std::filesystem::exists(filepath);
  if(async_enabled && append == 0) {
    sp->es_id = H5EScreate();
  }
  if(async_enabled) {
    es_id = sp->es_id;
  }
#endif

  // Check if any buffers are filled. If any process needs to dump then all must do it together
  int buff_has_space = (sp->np_buffered+sp->np <= sp->particle_io_buffer_h.extent(0));
  if(buffer_enabled) {
    MPI_Allreduce(MPI_IN_PLACE, &buff_has_space, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
  } else {
    buff_has_space = 0;
  }

  // If needed, copy weight back to particles for hydro quantities
  if(sp->tracer_type == TracerType::Copy) {
    const int w_idx = sp->annotation_vars.get_annotation_index<float>("Weight");
    const int w_var = static_cast<int>(particle_var::w);
    auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), w_var);
    auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), w_var);
    auto w_annote = Kokkos::subview(sp->annotations_d.f32, Kokkos::ALL(), w_idx);
    Kokkos::deep_copy(w_subview_d, w_annote);
    Kokkos::deep_copy(w_subview_h, w_annote);
  }

  // Get references to necessary data structures
  auto& particles = sp->k_p_d;
  auto& particles_i = sp->k_p_i_d;
  auto& interpolators_k = interpolator_array->k_i_d;
  auto& interp = interpolator_array->k_i_h;
  interpolator_array->copy_to_host();

  // Compute hydro quantities
  if(static_cast<uint32_t>(dump_vars) >= 2*DumpVar::ParticleKE) {
    Kokkos::deep_copy(hydro_array->k_h_d, 0.0f);
    Kokkos::deep_copy(hydro_array->k_h_h, 0.0f);
    accumulate_hydro_p_kokkos(
        particles,
        particles_i,
        hydro_array->k_h_d,
        interpolator_array->k_i_d,
        sp
    );

    // This is slower in my tests
    synchronize_hydro_array_kokkos(hydro_array);

    hydro_array->copy_to_host();

    //synchronize_hydro_array( hydro_array );

    //hydro_array->copy_to_device();
  }

  // Buffer tracers
  if( buffer_enabled && buff_has_space && (step() != num_step) ) {
    buffer_tracers(sp, hydro_array, interpolator_array, dump_vars, rank(), step());
  } else { // Dump buffered tracers
    // Create file access template with parallel IO access
    herr_t status;
    // Set MPIO to collective mode
    hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    // Try to create species HDF5 file with default file creation/access property lists
    hid_t file_id;
#ifdef VPIC_ENABLE_HDF5_ASYNC
    if(async_enabled) {
      if(!std::filesystem::exists(filepath)) {
        file_id = H5Fcreate_async(filepath.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id, es_id);
      } else {
        file_id = H5Fopen_async(filepath.c_str(), H5F_ACC_RDWR, plist_id, es_id);
      }
    }
#else
    if(!std::filesystem::exists(filepath)) {
      file_id = H5Fcreate(filepath.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    } else {
      file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDWR, plist_id);
    }
#endif

    // Track total number of tracers and per rank offsets
    uint64_t total_particles, offset;
    uint64_t num_particles;

    // Update the particles on the host only if they haven't been recently
    if (step() > sp->last_copied) {
      sp->copy_to_host();
    }
    hydro_array->copy_to_host();

    // Copy buffered tracers to host
    if(buffer_enabled) {
      Kokkos::deep_copy(sp->particle_io_buffer_h, sp->particle_io_buffer_d);
      Kokkos::deep_copy(sp->particle_cell_io_buffer_h, sp->particle_cell_io_buffer_d);
      sp->annotations_io_buffer_h.copy_from(sp->annotations_io_buffer_d);
      if(dump_vars > 0) {
        Kokkos::deep_copy(sp->tracer_buffer_h, sp->tracer_buffer_d);
      }
    }
    Kokkos::fence();
    // Write buffered tracer data
    uint64_t particle_idx = 0;
    // Iterate through each timestep, write non buffered particles last
    const auto steps_buffered = buffer_enabled ? sp->np_per_ts.size() : 0;
    for(uint32_t ts_idx=0; ts_idx < steps_buffered+1; ts_idx++) {
      int64_t time_step;

      // Get # of local particles to write for this timestep
      if(!buffer_enabled || (ts_idx == steps_buffered)) {
        time_step = step();
        num_particles = sp->np;
      } else {
        time_step = sp->np_per_ts[ts_idx].second;
        num_particles = sp->np_per_ts[ts_idx].first;
      }

      // Calculate the total number of particles for this timestep
      MPI_Allreduce(&num_particles, &total_particles, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      // Calculate the offset for each rank
      MPI_Scan(&num_particles, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      offset -= num_particles;

      // Create group for time step
      sprintf(group_name, "/Timestep_%ld", time_step);
#ifdef VPIC_ENABLE_HDF5_ASYNC
      hid_t group_id;
      if(async_enabled) {
        group_id = H5Gcreate_async(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, es_id);
      } else {
        group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }
#else
      hid_t group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#endif

      if(total_particles > 0) {
        // Create dataspace describing dims for particle datasets
        hid_t dataspace_id = H5Screate_simple(1, (hsize_t*)(&total_particles), NULL);

        // Select slab of dataset for each rank
        hsize_t stride = 1;
        hsize_t block = 1;
        status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, (hsize_t*)(&offset), &stride, (hsize_t*)(&num_particles), &block);
    
        // Create memspace
        hid_t memspace_id = H5Screate_simple(1, (hsize_t*)(&num_particles), NULL);

        // Determine if writing buffered or non buffered data
        const bool write_buffered = buffer_enabled && (ts_idx < steps_buffered);
        const size_t beg = write_buffered ? particle_idx : 0LLU;
        auto slice = Kokkos::make_pair(beg, beg+num_particles);
        write_tracers(sp, hydro_array, interp, slice, write_buffered,
                      num_particles, dump_vars, group_id,  dataspace_id, 
                      memspace_id, dxpl_id);
        // Move slice beg to next time step in buffer
        if(write_buffered) {
          particle_idx += sp->np_per_ts[ts_idx].first;
        }

        status = H5Sclose(memspace_id);
        status = H5Sclose(dataspace_id);
      }
#ifdef VPIC_ENABLE_HDF5_ASYNC
      if(async_enabled) {
        status = H5Gclose_async(group_id, es_id);
      } else {
        status = H5Gclose(group_id);
      }
#else
      status = H5Gclose(group_id);
#endif

      if(buffer_enabled && (ts_idx == steps_buffered)) {
        // Clear buffers
        sp->np_buffered = 0;
        sp->np_per_ts.clear();
      }
    }

    // Close handles
#ifdef VPIC_ENABLE_HDF5_ASYNC
    if(async_enabled) {
      status = H5Fclose_async(file_id, es_id);
    } else {
      status = H5Fclose(file_id);
    }
#else
    status = H5Fclose(file_id);
#endif
    status = H5Pclose(plist_id);
    status = H5Pclose(dxpl_id);
  }
  // If needed, reset weight to 0 for copied particles
  if(sp->tracer_type == TracerType::Copy) {
    auto w_subview_h = Kokkos::subview(sp->k_p_h, Kokkos::ALL(), static_cast<int>(particle_var::w));
    auto w_subview_d = Kokkos::subview(sp->k_p_d, Kokkos::ALL(), static_cast<int>(particle_var::w));
    Kokkos::deep_copy(w_subview_h, 0.0);
    Kokkos::deep_copy(w_subview_d, 0.0);
  }
#ifdef VPIC_ENABLE_HDF5_ASYNC
  // Wait for Async ops to finish
  if(async_enabled && (step() == num_step)) {
    size_t num_in_progress;
    hbool_t op_failed;
    H5ESwait(es_id, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
    MPI_Barrier(MPI_COMM_WORLD);
    H5ESclose(sp->es_id);
  }
#endif
}

#ifdef VPIC_ENABLE_HDF5_ASYNC
void 
vpic_simulation::dump_tracers_hdf5_async(const char* sp_name, 
                                         const uint32_t dump_vars,
                                         const char* fbase) {
  dump_tracers(sp_name, dump_vars, fbase, false, true);
}

void 
vpic_simulation::dump_tracers_buffered_hdf5_async(const char* sp_name, 
                                                  const uint32_t dump_vars,
                                                  const char* fbase) {
  dump_tracers(sp_name, dump_vars, fbase, true, true);
}
#endif

void 
vpic_simulation::dump_tracers_hdf5(const char* sp_name, 
                                   const uint32_t dump_vars,
                                   const char* fbase) {
  dump_tracers(sp_name, dump_vars, fbase, false);
}

void 
vpic_simulation::dump_tracers_buffered_hdf5(const char* sp_name, 
                                            const uint32_t dump_vars,
                                            const char* fbase) {
  dump_tracers(sp_name, dump_vars, fbase, true);
}
#endif
#endif

