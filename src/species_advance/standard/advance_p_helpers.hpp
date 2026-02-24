#ifndef __ADVANCE_P_HELPERS_HPP
#define __ADVANCE_P_HELPERS_HPP

template<typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
load_particles(const k_particles_t& k_part, const k_particles_i_t& k_part_i, size_t active_lanes, size_t start_idx, SIMDFloatMask& mask,
               SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
               SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  // Load position
  dx = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::dx);});
  dy = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::dy);});
  dz = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::dz);});
  // Load cell ID
  ii = simd_int32_t([k_part_i, start_idx] (int i) {return k_part_i( start_idx+i );});
  // Load momentum
  ux = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::ux);});
  uy = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::uy);});
  uz = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::uz);});
  // Load weight
  wt = simd_float_t([k_part, start_idx] (int i) {return k_part( start_idx+i, particle_var::w );});
};

template<typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
store_particles(const k_particles_t& k_part, const k_particles_i_t& k_part_i, size_t active_lanes, size_t start_idx, SIMDFloatMask& mask,
                SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  for(size_t i=0; i<active_lanes; i++) {
    // Store position
    k_part(start_idx+i, particle_var::dx) = dx[i];
    k_part(start_idx+i, particle_var::dy) = dy[i];
    k_part(start_idx+i, particle_var::dz) = dz[i];
    // Store cell ID
    k_part_i(start_idx+i) = ii[i];
    // Store momentum
    k_part(start_idx+i, particle_var::ux) = ux[i];
    k_part(start_idx+i, particle_var::uy) = uy[i];
    k_part(start_idx+i, particle_var::uz) = uz[i];
    // Store weight
    k_part(start_idx+i, particle_var::w ) = wt[i];
  }
}

template<typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
load_particles_simd(const k_particles_t& k_part, const k_particles_i_t& k_part_i, size_t active_lanes, size_t start_idx, SIMDFloatMask& mask,
                    SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                    SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  // Load particles
  size_t np = k_part.extent(0);
  const float* mem_dx = &(k_part( start_idx, particle_var::dx));
  const float* mem_dy = &(k_part( start_idx, particle_var::dy));
  const float* mem_dz = &(k_part( start_idx, particle_var::dz));
  const float* mem_ux = &(k_part( start_idx, particle_var::ux));
  const float* mem_uy = &(k_part( start_idx, particle_var::uy));
  const float* mem_uz = &(k_part( start_idx, particle_var::uz));
  const float* mem_wt = &(k_part( start_idx, particle_var::w ));
  const int*   mem_ii = &(k_part_i(start_idx));

  if constexpr (std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
//    if(active_lanes == SIMDFloat::size()) [[likely]] {
#if KOKKOS_VERSION_MAJOR == 5
      // Load position
      dx = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_dx, vector_aligned_tag_t());
      dy = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_dy, vector_aligned_tag_t());
      dz = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_dz, vector_aligned_tag_t());
      // Load momentum
      ux = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_ux, vector_aligned_tag_t());
      uy = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_uy, vector_aligned_tag_t());
      uz = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_uz, vector_aligned_tag_t());
      // Load weight
      wt = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_wt, vector_aligned_tag_t());
      // Load cell index
      ii = KokkosSIMD::simd_unchecked_load<simd_int32_t>(mem_ii, vector_aligned_tag_t());
#else
      // Load position
      dx.copy_from(mem_dx, vector_aligned_tag_t());
      dy.copy_from(mem_dy, vector_aligned_tag_t());
      dz.copy_from(mem_dz, vector_aligned_tag_t());
      // Load momentum
      ux.copy_from(mem_ux, vector_aligned_tag_t());
      uy.copy_from(mem_uy, vector_aligned_tag_t());
      uz.copy_from(mem_uz, vector_aligned_tag_t());
      // Load weight
      wt.copy_from(mem_wt, vector_aligned_tag_t());
      // Load cell index
      ii.copy_from(mem_ii, vector_aligned_tag_t());
#endif
//    } else {
//#if KOKKOS_VERSION_MAJOR == 5
//      // Load position
//      dx = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_dx, mask, vector_aligned_tag_t());
//      dy = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_dy, mask, vector_aligned_tag_t());
//      dz = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_dz, mask, vector_aligned_tag_t());
//      // Load momentum
//      ux = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_ux, mask, vector_aligned_tag_t());
//      uy = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_uy, mask, vector_aligned_tag_t());
//      uz = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_uz, mask, vector_aligned_tag_t());
//      // Load weight
//      wt = KokkosSIMD::simd_unchecked_load<simd_float_t>(mem_wt, mask, vector_aligned_tag_t());
//      // Load cell index
//      ii = KokkosSIMD::simd_unchecked_load<simd_int32_t>(mem_ii, mask, vector_aligned_tag_t());
//#else
//      // Load position
//      dx.copy_from(mem_dx, element_aligned_tag_t());
//      dy.copy_from(mem_dy, element_aligned_tag_t());
//      dz.copy_from(mem_dz, element_aligned_tag_t());
//      // Load momentum
//      ux.copy_from(mem_ux, element_aligned_tag_t());
//      uy.copy_from(mem_uy, element_aligned_tag_t());
//      uz.copy_from(mem_uz, element_aligned_tag_t());
//      // Load weight
//      wt.copy_from(mem_wt, element_aligned_tag_t());
//      // Load cell index
//      ii.copy_from(mem_ii, element_aligned_tag_t());
//#endif
//    }
  } else {
    simd_int32_t indices([](std::size_t i) { return i*PARTICLE_VAR_COUNT; });
#if KOKKOS_VERSION_MAJOR == 5
//    SIMDFloat v0;
//    dx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx,                      element_aligned_tag_t());
//    dy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+  PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    dz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+2*PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    ux = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+3*PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    uy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+4*PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    uz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+5*PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    wt = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+6*PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    v0 = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx+7*PARTICLE_VAR_COUNT, element_aligned_tag_t());
//    transpose(dx,dy,dz,ux,uy,uz,wt,v0);
//    // Load cell index
//    ii = simd_int32_t(mem_ii, simd_int32_mask_t(mask), element_aligned_tag_t());

    simd_int32_mask_t int_mask(mask);
    dx = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_dx, mem_dx+np), int_mask, indices, element_aligned_tag_t());
    dy = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_dy, mem_dy+np), int_mask, indices, element_aligned_tag_t());
    dz = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_dz, mem_dz+np), int_mask, indices, element_aligned_tag_t());
    // Load momentum
    ux = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_ux, mem_ux+np), int_mask, indices, element_aligned_tag_t());
    uy = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_uy, mem_uy+np), int_mask, indices, element_aligned_tag_t());
    uz = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_uz, mem_uz+np), int_mask, indices, element_aligned_tag_t());
    // Load weight
    wt = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_wt, mem_wt+np), int_mask, indices, element_aligned_tag_t());
    // Load cell index
    ii = simd_int32_t(mem_ii, int_mask, element_aligned_tag_t());
#else
    // Load position
    KokkosSIMD::where(mask, dx).gather_from(mem_dx, indices);
    KokkosSIMD::where(mask, dy).gather_from(mem_dy, indices);
    KokkosSIMD::where(mask, dz).gather_from(mem_dz, indices);
    // Load momentum
    KokkosSIMD::where(mask, ux).gather_from(mem_ux, indices);
    KokkosSIMD::where(mask, uy).gather_from(mem_uy, indices);
    KokkosSIMD::where(mask, uz).gather_from(mem_uz, indices);
    // Load weight
    KokkosSIMD::where(mask, q ).gather_from(mem_w , indices);
    // Load cell index
    KokkosSIMD::where(mask_int, ii).copy_from(mem_ii, element_aligned_tag_t());
#endif
  }
};

template<typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
store_particles_simd(const k_particles_t& k_part, const k_particles_i_t& k_part_i, size_t active_lanes, size_t start_idx, SIMDFloatMask& mask,
                     SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                     SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  const size_t np = k_part.extent(0);
  float* mem_dx = &(k_part( start_idx, particle_var::dx));
  float* mem_dy = &(k_part( start_idx, particle_var::dy));
  float* mem_dz = &(k_part( start_idx, particle_var::dz));
  float* mem_ux = &(k_part( start_idx, particle_var::ux));
  float* mem_uy = &(k_part( start_idx, particle_var::uy));
  float* mem_uz = &(k_part( start_idx, particle_var::uz));
  float* mem_wt = &(k_part( start_idx, particle_var::w ));
  int*   mem_ii = &(k_part_i(start_idx));

  if constexpr (std::is_same<Kokkos::LayoutLeft, k_particles_t::array_layout>::value) {
//    if(active_lanes == SIMD_LEN) [[likely]] {
#if KOKKOS_VERSION_MAJOR == 5
      // Store position
      KokkosSIMD::simd_unchecked_store(dx, mem_dx, vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(dy, mem_dy, vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(dz, mem_dz, vector_aligned_tag_t());
      // Store cell
      KokkosSIMD::simd_unchecked_store(ii, mem_ii, vector_aligned_tag_t());
      // Store momentum
      KokkosSIMD::simd_unchecked_store(ux, mem_ux, vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(uy, mem_uy, vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(uz, mem_uz, vector_aligned_tag_t());
      // Store weight
      KokkosSIMD::simd_unchecked_store(wt, mem_wt, vector_aligned_tag_t());
#else
      // Store position
      dx.copy_to(mem_dx, element_aligned_tag_t());
      dy.copy_to(mem_dy, element_aligned_tag_t());
      dz.copy_to(mem_dz, element_aligned_tag_t());
      // Store cell
      ii.copy_to(mem_ii, element_aligned_tag_t());
      // Store momentum
      dx.copy_to(mem_ux, element_aligned_tag_t());
      dy.copy_to(mem_uy, element_aligned_tag_t());
      dz.copy_to(mem_uz, element_aligned_tag_t());
      // Store weight
      wt.copy_to(mem_wt, element_aligned_tag_t());
#endif
//    } else {
//#if KOKKOS_VERSION_MAJOR == 5
//      // Store position
//      KokkosSIMD::simd_unchecked_store(dx, mem_dx, mask, element_aligned_tag_t());
//      KokkosSIMD::simd_unchecked_store(dy, mem_dy, mask, element_aligned_tag_t());
//      KokkosSIMD::simd_unchecked_store(dz, mem_dz, mask, element_aligned_tag_t());
//      // Store cell
//      KokkosSIMD::simd_unchecked_store(ii, mem_ii, simd_int32_mask_t(mask), element_aligned_tag_t());
//      // Store momentum
//      KokkosSIMD::simd_unchecked_store(ux, mem_ux, mask, element_aligned_tag_t());
//      KokkosSIMD::simd_unchecked_store(uy, mem_uy, mask, element_aligned_tag_t());
//      KokkosSIMD::simd_unchecked_store(uz, mem_uz, mask, element_aligned_tag_t());
//      // Store weight
//      KokkosSIMD::simd_unchecked_store(wt, mem_wt, mask, element_aligned_tag_t());
//#else
//      // Store position
//      KokkosSIMD::where(mask, dx).copy_to(mem_dx, element_aligned_tag_t());
//      KokkosSIMD::where(mask, dy).copy_to(mem_dy, element_aligned_tag_t());
//      KokkosSIMD::where(mask, dz).copy_to(mem_dz, element_aligned_tag_t());
//      // Store cell
//      KokkosSIMD::where(mask, ii).copy_to(mem_ii, element_aligned_tag_t());
//      // Store momentum
//      KokkosSIMD::where(mask, ux).copy_to(mem_ux, element_aligned_tag_t());
//      KokkosSIMD::where(mask, uy).copy_to(mem_uy, element_aligned_tag_t());
//      KokkosSIMD::where(mask, uz).copy_to(mem_uz, element_aligned_tag_t());
//      // Store weight
//      KokkosSIMD::where(mask, wt).copy_to(mem_wt, element_aligned_tag_t());
//#endif
//    }
  } else {
    simd_int32_t indices([](std::size_t i) { return i*PARTICLE_VAR_COUNT; });
#if KOKKOS_VERSION_MAJOR == 5
//    for(size_t idx=0; idx<simd_int32_t::size(); idx++) {
//      if(mask[idx]) {
//        // Store position
//        mem_dx[indices[idx]] = dx[idx];
//        mem_dy[indices[idx]] = dy[idx];
//        mem_dz[indices[idx]] = dz[idx];
//        // Store cell
//        mem_ii[indices[idx]] = ii[idx];
//        // Store momentum
//        mem_ux[indices[idx]] = ux[idx];
//        mem_uy[indices[idx]] = uy[idx];
//        mem_uz[indices[idx]] = uz[idx];
//        // Store weight
//        mem_wt[indices[idx]] = wt[idx];
//      }
//    }
    
//    simd_float_mask_t write_mask([](size_t i) {return i<7;});
//    SIMDFloat t0(dx), t1(dy), t2(dz), t3(ux), t4(uy), t5(uz), t6(wt), t7(0.0f);
//    transpose(t0, t1, t2, t3, t4, t5, t6, t7);
//    if(mask[0])
//    KokkosSIMD::simd_unchecked_store(t0, mem_dx,                      write_mask, element_aligned_tag_t());
//    if(mask[1])
//    KokkosSIMD::simd_unchecked_store(t1, mem_dx+  PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    if(mask[2])
//    KokkosSIMD::simd_unchecked_store(t2, mem_dx+2*PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    if(mask[3])
//    KokkosSIMD::simd_unchecked_store(t3, mem_dx+3*PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    if(mask[4])
//    KokkosSIMD::simd_unchecked_store(t4, mem_dx+4*PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    if(mask[5])
//    KokkosSIMD::simd_unchecked_store(t5, mem_dx+5*PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    if(mask[6])
//    KokkosSIMD::simd_unchecked_store(t6, mem_dx+6*PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    if(mask[7])
//    KokkosSIMD::simd_unchecked_store(t7, mem_dx+7*PARTICLE_VAR_COUNT, write_mask, element_aligned_tag_t());
//    simd_int32_mask_t int_mask(mask);
//    KokkosSIMD::simd_unchecked_store(ii, mem_ii, int_mask, element_aligned_tag_t());
 

      simd_int32_mask_t int_mask([active_lanes] (size_t i) {return i<active_lanes;});
      // Store position
      KokkosSIMD::unchecked_scatter_to(dx, std::ranges::subrange(mem_dx, mem_dx+np), int_mask, indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(dy, std::ranges::subrange(mem_dy, mem_dx+np), int_mask, indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(dz, std::ranges::subrange(mem_dz, mem_dx+np), int_mask, indices, element_aligned_tag_t());
      // Store cell
      //KokkosSIMD::unchecked_scatter_to(ii, std::ranges::subrange(mem_ii, mem_ii+np), int_mask, indices, element_aligned_tag_t());
      // Store momentum
      KokkosSIMD::unchecked_scatter_to(ux, std::ranges::subrange(mem_ux, mem_ux+np), int_mask, indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(uy, std::ranges::subrange(mem_uy, mem_uy+np), int_mask, indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(uz, std::ranges::subrange(mem_uz, mem_uz+np), int_mask, indices, element_aligned_tag_t());
      // Store weight
      KokkosSIMD::unchecked_scatter_to(wt, std::ranges::subrange(mem_wt, mem_wt+np), int_mask, indices, element_aligned_tag_t());
#else
    // Store position
    KokkosSIMD::where(mask, dx).scatter_to(mem_dx, indices);
    KokkosSIMD::where(mask, dy).scatter_to(mem_dy, indices);
    KokkosSIMD::where(mask, dz).scatter_to(mem_dz, indices);
//    // Store cell
//    KokkosSIMD::where(mask, ii).scatter_to(mem_ii, indices);
    // Store momentum
    KokkosSIMD::where(mask, ux).scatter_to(mem_ux, indices);
    KokkosSIMD::where(mask, uy).scatter_to(mem_uy, indices);
    KokkosSIMD::where(mask, uz).scatter_to(mem_uz, indices);
    // Store weight
    KokkosSIMD::where(mask, wt).scatter_to(mem_wt, indices);
#endif
  }
}

template<typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
load_particles_union(const particles_union_t& particles, size_t active_lanes, size_t start_idx, SIMDFloatMask& mask,
                     SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                     SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  // Load particles
  if constexpr (std::is_same<Kokkos::LayoutLeft, particles_union_t::array_layout>::value) {
    const float* mem_dx = &(particles( start_idx, 0).f32);
    const float* mem_dy = &(particles( start_idx, 1).f32);
    const float* mem_dz = &(particles( start_idx, 2).f32);
    const int*   mem_ii = &(particles( start_idx, 3).i32);
    const float* mem_ux = &(particles( start_idx, 4).f32);
    const float* mem_uy = &(particles( start_idx, 5).f32);
    const float* mem_uz = &(particles( start_idx, 6).f32);
    const float* mem_wt = &(particles( start_idx, 7).f32);

    // Load position
    dx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx, vector_aligned_tag_t());
    dy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dy, vector_aligned_tag_t());
    dz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dz, vector_aligned_tag_t());
    // Load cell index
    ii = KokkosSIMD::simd_unchecked_load<SIMDInt32>(mem_ii, vector_aligned_tag_t());
    // Load momentum
    ux = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_ux, vector_aligned_tag_t());
    uy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_uy, vector_aligned_tag_t());
    uz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_uz, vector_aligned_tag_t());
    // Load weight
    wt = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_wt, vector_aligned_tag_t());
  } else {
    if constexpr( SIMDFloat::size() == 16 ) {
      const float* mem_beg = &(particles( start_idx, 0 ).f32);
      SIMDFloat cl;
      dx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg,                       vector_aligned_tag_t());
      dy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg +   SIMDFloat::size(), vector_aligned_tag_t());
      dz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 2*SIMDFloat::size(), vector_aligned_tag_t());
      cl = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 3*SIMDFloat::size(), vector_aligned_tag_t());
      ux = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 4*SIMDFloat::size(), vector_aligned_tag_t());
      uy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 5*SIMDFloat::size(), vector_aligned_tag_t());
      uz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 6*SIMDFloat::size(), vector_aligned_tag_t());
      wt = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 7*SIMDFloat::size(), vector_aligned_tag_t());
      transpose_particles_16x8(dx, dy, dz, cl, ux, uy, uz, wt);
      ii = simd_cast(cl);
    } else if constexpr( SIMDFloat::size() == 8 ) {
      simd_float_t cl;
      const float* mem_beg = &(particles( start_idx, 0 ).f32);
      dx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg,                       vector_aligned_tag_t());
      dy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg +   SIMDFloat::size(), vector_aligned_tag_t());
      dz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 2*SIMDFloat::size(), vector_aligned_tag_t());
      cl = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 3*SIMDFloat::size(), vector_aligned_tag_t());
      ux = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 4*SIMDFloat::size(), vector_aligned_tag_t());
      uy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 5*SIMDFloat::size(), vector_aligned_tag_t());
      uz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 6*SIMDFloat::size(), vector_aligned_tag_t());
      wt = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_beg + 7*SIMDFloat::size(), vector_aligned_tag_t());
      transpose(dx, dy, dz, cl, ux, uy, uz, wt);
      ii = simd_cast(cl);
    } else if constexpr( SIMDFloat::size() == 4 ) {
      SIMDFloat cl;
      dx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx,   0).f32), vector_aligned_tag_t());
      dy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx+1, 0).f32), vector_aligned_tag_t());
      dz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx+2, 0).f32), vector_aligned_tag_t());
      cl = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx+3, 0).f32), vector_aligned_tag_t());
      transpose(dx, dy, dz, cl);
      ux = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx,   4).f32), vector_aligned_tag_t());
      uy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx+1, 4).f32), vector_aligned_tag_t());
      uz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx+2, 4).f32), vector_aligned_tag_t());
      wt = KokkosSIMD::simd_unchecked_load<SIMDFloat>(&(particles(start_idx+3, 4).f32), vector_aligned_tag_t());
      transpose(ux, uy, uz, wt);
      ii = simd_cast(cl);
    } else {
      simd_int32_t indices([active_lanes](std::size_t i) { return i < active_lanes ? i*8 : 0; });
      dx = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
      dy = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
      dz = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
      ii = KokkosSIMD::unchecked_gather_from<SIMDInt32>(std::ranges::subrange(&(particles( start_idx, 0).i32), &(particles( start_idx, 0).i32) + 128), indices, element_aligned_tag_t());
      ux = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
      uy = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
      uz = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
      wt = KokkosSIMD::unchecked_gather_from<SIMDFloat>(std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32) + 128), indices, element_aligned_tag_t());
    }
  }
};

template<typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
store_particles_union(const particles_union_t& particles, size_t active_lanes, size_t start_idx, SIMDFloatMask& mask,
                      SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                      SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  if constexpr (std::is_same<Kokkos::LayoutLeft, particles_union_t::array_layout>::value) {
    float* mem_dx = &(particles( start_idx, 0).f32);
    float* mem_dy = &(particles( start_idx, 1).f32);
    float* mem_dz = &(particles( start_idx, 2).f32);
    int*   mem_ii = &(particles( start_idx, 3).i32);
    float* mem_ux = &(particles( start_idx, 4).f32);
    float* mem_uy = &(particles( start_idx, 5).f32);
    float* mem_uz = &(particles( start_idx, 6).f32);
    float* mem_wt = &(particles( start_idx, 7).f32);

    // Store position
    KokkosSIMD::simd_unchecked_store(dx, mem_dx, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(dy, mem_dy, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(dz, mem_dz, vector_aligned_tag_t());
    // Store cell
    KokkosSIMD::simd_unchecked_store(ii, mem_ii, vector_aligned_tag_t());
    // Store momentum
    KokkosSIMD::simd_unchecked_store(ux, mem_ux, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(uy, mem_uy, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(uz, mem_uz, vector_aligned_tag_t());
    // Store weight
    KokkosSIMD::simd_unchecked_store(wt, mem_wt, vector_aligned_tag_t());
  } else {
    if constexpr( SIMDFloat::size() == 16 ) {
      float* mem_beg = &(particles( start_idx, 0 ).f32);
      SIMDFloat t0(dx), t1(dy), t2(dz), t3(simd_cast(ii)), t4(ux), t5(uy), t6(uz), t7(wt);
      transpose_particles_16x8_reverse(t0, t1, t2, t3, t4, t5, t6, t7);
      KokkosSIMD::simd_unchecked_store(t0, mem_beg,                       vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t1, mem_beg +   SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t2, mem_beg + 2*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t3, mem_beg + 3*SIMDInt32::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t4, mem_beg + 4*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t5, mem_beg + 5*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t6, mem_beg + 6*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t7, mem_beg + 7*SIMDFloat::size(), vector_aligned_tag_t());
    } else if constexpr( SIMDFloat::size() == 8 ) {
      float* mem_beg = &(particles( start_idx, 0 ).f32);
      SIMDFloat t0(dx), t1(dy), t2(dz), t3(simd_cast(ii)), t4(ux), t5(uy), t6(uz), t7(wt);
      transpose(t0, t1, t2, t3, t4, t5, t6, t7);
      KokkosSIMD::simd_unchecked_store(t0, mem_beg,                       vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t1, mem_beg +   SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t2, mem_beg + 2*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t3, mem_beg + 3*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t4, mem_beg + 4*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t5, mem_beg + 5*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t6, mem_beg + 6*SIMDFloat::size(), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t7, mem_beg + 7*SIMDFloat::size(), vector_aligned_tag_t());
    } else if constexpr( SIMDFloat::size() == 4 ) {
      float* mem_beg = &(particles( start_idx, 0 ).f32);
      SIMDFloat t0(dx), t1(dy), t2(dz), t3(simd_cast(ii)), t4(ux), t5(uy), t6(uz), t7(wt);
      transpose(t0, t1, t2, t3);
      KokkosSIMD::simd_unchecked_store(t0, &(particles( start_idx,   0 ).f32), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t1, &(particles( start_idx+1, 0 ).f32), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t2, &(particles( start_idx+2, 0 ).f32), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t3, &(particles( start_idx+3, 0 ).f32), vector_aligned_tag_t());
      transpose(t4, t5, t6, t7);
      KokkosSIMD::simd_unchecked_store(t4, &(particles( start_idx,   4 ).f32), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t5, &(particles( start_idx+1, 4 ).f32), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t6, &(particles( start_idx+2, 4 ).f32), vector_aligned_tag_t());
      KokkosSIMD::simd_unchecked_store(t7, &(particles( start_idx+3, 4 ).f32), vector_aligned_tag_t());
    } else {
      simd_int32_t indices([](std::size_t i) { return i*8; });
      KokkosSIMD::unchecked_scatter_to(dx, std::ranges::subrange(&(particles( start_idx, 0).f32), &(particles( start_idx, 0).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(dy, std::ranges::subrange(&(particles( start_idx, 1).f32), &(particles( start_idx, 1).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(dz, std::ranges::subrange(&(particles( start_idx, 2).f32), &(particles( start_idx, 2).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(ii, std::ranges::subrange(&(particles( start_idx, 3).i32), &(particles( start_idx, 3).i32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(ux, std::ranges::subrange(&(particles( start_idx, 4).f32), &(particles( start_idx, 4).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(uy, std::ranges::subrange(&(particles( start_idx, 5).f32), &(particles( start_idx, 5).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(uz, std::ranges::subrange(&(particles( start_idx, 6).f32), &(particles( start_idx, 6).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
      KokkosSIMD::unchecked_scatter_to(wt, std::ranges::subrange(&(particles( start_idx, 7).f32), &(particles( start_idx, 7).f32)+SIMDFloat::size()), indices, element_aligned_tag_t());
    }
  }
}


#ifdef ENABLE_CABANA
template<typename PosSlice, typename CellSlice, typename MomSlice, typename WeightSlice,
         typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
load_particles_aosoa(PosSlice&    pos_slice,
                     CellSlice&   cel_slice,
                     MomSlice&    mom_slice,
                     WeightSlice& wgt_slice,
                     size_t active_lanes, size_t slice_id, SIMDFloatMask& mask,
                     SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                     SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  // Load particles
  const float* mem_dx = &( pos_slice.access(slice_id, 0, 0) );
  const float* mem_dy = &( pos_slice.access(slice_id, 0, 1) );
  const float* mem_dz = &( pos_slice.access(slice_id, 0, 2) );
  const int*   mem_ii = &( cel_slice.access(slice_id, 0) );
  const float* mem_ux = &( mom_slice.access(slice_id, 0, 0) );
  const float* mem_uy = &( mom_slice.access(slice_id, 0, 1) );
  const float* mem_uz = &( mom_slice.access(slice_id, 0, 2) );
  const float* mem_wt = &( wgt_slice.access(slice_id, 0) );

//  if( pos_slice.arraySize(slice_id) == SIMDFloat::size() ) [[likely]] {
    // Load position
    dx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dx, vector_aligned_tag_t());
    dy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dy, vector_aligned_tag_t());
    dz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_dz, vector_aligned_tag_t());
    // Load cell index
    ii = KokkosSIMD::simd_unchecked_load<SIMDInt32>(mem_ii, vector_aligned_tag_t());
    // Load momentum
    ux = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_ux, vector_aligned_tag_t());
    uy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_uy, vector_aligned_tag_t());
    uz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_uz, vector_aligned_tag_t());
    // Load weight
    wt = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_wt, vector_aligned_tag_t());
//  } else {
//    // Load position
//    dx = KokkosSIMD::simd_unchecked_load(mem_dx, mask, vector_aligned_tag_t());
//    dy = KokkosSIMD::simd_unchecked_load(mem_dy, mask, vector_aligned_tag_t());
//    dz = KokkosSIMD::simd_unchecked_load(mem_dz, mask, vector_aligned_tag_t());
//    // Load cell index
//    ii = KokkosSIMD::simd_unchecked_load(mem_ii, simd_int32_mask_t(mask), vector_aligned_tag_t());
//    // Load momentum
//    ux = KokkosSIMD::simd_unchecked_load(mem_ux, mask, vector_aligned_tag_t());
//    uy = KokkosSIMD::simd_unchecked_load(mem_uy, mask, vector_aligned_tag_t());
//    uz = KokkosSIMD::simd_unchecked_load(mem_uz, mask, vector_aligned_tag_t());
//    // Load weight
//    wt = KokkosSIMD::simd_unchecked_load(mem_wt, mask, vector_aligned_tag_t());
//  }
};

template<typename PosSlice, typename CellSlice, typename MomSlice, typename WeightSlice,
         typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION
store_particles_aosoa(PosSlice&    pos_slice,
                      CellSlice&   cel_slice,
                      MomSlice&    mom_slice,
                      WeightSlice& wgt_slice,
                      size_t active_lanes, size_t slice_id, SIMDFloatMask& mask,
                      SIMDFloat& dx, SIMDFloat& dy, SIMDFloat& dz, SIMDInt32& ii,
                      SIMDFloat& ux, SIMDFloat& uy, SIMDFloat& uz, SIMDFloat& wt
) {
  // Load particles
  float* mem_dx = &( pos_slice.access(slice_id, 0, 0) );
  float* mem_dy = &( pos_slice.access(slice_id, 0, 1) );
  float* mem_dz = &( pos_slice.access(slice_id, 0, 2) );
  int*   mem_ii = &( cel_slice.access(slice_id, 0) );
  float* mem_ux = &( mom_slice.access(slice_id, 0, 0) );
  float* mem_uy = &( mom_slice.access(slice_id, 0, 1) );
  float* mem_uz = &( mom_slice.access(slice_id, 0, 2) );
  float* mem_wt = &( wgt_slice.access(slice_id, 0) );

//  if( pos_slice.arraySize(slice_id) == SIMDFloat::size() ) [[likely]] {
    // Load position
    KokkosSIMD::simd_unchecked_store(dx, mem_dx, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(dy, mem_dy, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(dz, mem_dz, vector_aligned_tag_t());
    // Load cell index
    KokkosSIMD::simd_unchecked_store(ii, mem_ii, vector_aligned_tag_t());
    // Load momentum
    KokkosSIMD::simd_unchecked_store(ux, mem_ux, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(uy, mem_uy, vector_aligned_tag_t());
    KokkosSIMD::simd_unchecked_store(uz, mem_uz, vector_aligned_tag_t());
    // Load weight
    KokkosSIMD::simd_unchecked_store(wt, mem_wt, vector_aligned_tag_t());
//  } else {
//    // Load position
//    KokkosSIMD::simd_unchecked_store(dx, mem_dx, mask, vector_aligned_tag_t());
//    KokkosSIMD::simd_unchecked_store(dy, mem_dy, mask, vector_aligned_tag_t());
//    KokkosSIMD::simd_unchecked_store(dz, mem_dz, mask, vector_aligned_tag_t());
//    // Load cell index              
//    KokkosSIMD::simd_unchecked_store(ii, mem_ii, simd_int32_mask_t(mask), vector_aligned_tag_t());
//    // Load momentum                
//    KokkosSIMD::simd_unchecked_store(ux, mem_ux, mask, vector_aligned_tag_t());
//    KokkosSIMD::simd_unchecked_store(uy, mem_uy, mask, vector_aligned_tag_t());
//    KokkosSIMD::simd_unchecked_store(uz, mem_uz, mask, vector_aligned_tag_t());
//    // Load weight                  
//    KokkosSIMD::simd_unchecked_store(wt, mem_wt, mask, vector_aligned_tag_t());
//  }
}
#endif

template<typename Interpolators, typename SIMD_F32, typename SIMD_I32, typename SIMD_F32_Mask>
void KOKKOS_INLINE_FUNCTION 
load_interpolators_basic(const Interpolators& k_interp, const size_t active_lanes, 
                         const SIMD_F32_Mask& mask, const SIMD_I32 ii, 
                         const SIMD_F32& dx, const SIMD_F32& dy, const SIMD_F32& dz,
                         SIMD_F32& hax, SIMD_F32& hay, SIMD_F32& haz,
                         SIMD_F32& cbx, SIMD_F32& cby, SIMD_F32& cbz,
                         const SIMD_F32& qdt_2mc
) {
  SIMD_F32 fex, fdexdy, fdexdz, fd2exdydz;
  SIMD_F32 fey, fdeydz, fdeydx, fd2eydzdx;
  SIMD_F32 fez, fdezdx, fdezdy, fd2ezdxdy;
  SIMD_F32 fcbx , fcby, fcbz;     
  SIMD_F32 fdcbxdx, fdcbydy, fdcbzdz;

  fex       = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ex);}); 
  fdexdy    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dexdy);}); 
  fdexdz    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dexdz);}); 
  fd2exdydz = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2exdydz);}); 
  fey       = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ey);}); 
  fdeydz    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::deydz);}); 
  fdeydx    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::deydx);}); 
  fd2eydzdx = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2eydzdx);}); 
  fez       = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ez);}); 
  fdezdx    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dezdx);}); 
  fdezdy    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dezdy);}); 
  fd2ezdxdy = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2ezdxdy);}); 
  fcbx      = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cbx);}); 
  fdcbxdx   = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbxdx);}); 
  fcby      = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cby);}); 
  fdcbydy   = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbydy);}); 
  fcbz      = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cbz);}); 
  fdcbzdz   = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbzdz);}); 
  // Interpolate E
  hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
  hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
  haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
  // Interpolate B
  cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
  cby  = Kokkos::fma(dy, fdcbydy, fcby);
  cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
}

template<typename Interpolators, typename SIMD_F32, typename SIMD_I32, typename SIMD_F32_Mask>
void KOKKOS_INLINE_FUNCTION 
load_interpolators(const Interpolators& k_interp, const size_t active_lanes, 
                   const SIMD_F32_Mask& mask, const SIMD_I32 ii, 
                   const SIMD_F32& dx, const SIMD_F32& dy, const SIMD_F32& dz,
                   SIMD_F32& hax, SIMD_F32& hay, SIMD_F32& haz,
                   SIMD_F32& cbx, SIMD_F32& cby, SIMD_F32& cbz,
                   const SIMD_F32& qdt_2mc
) {
  simd_int32_mask_t int_mask(mask);
  const int nvoxels = k_interp.extent(0);
  const float* mem_fex       = &(k_interp(0, interpolator_var::ex));     
  const float* mem_fdexdy    = &(k_interp(0, interpolator_var::dexdy));  
  const float* mem_fdexdz    = &(k_interp(0, interpolator_var::dexdz));  
  const float* mem_fd2exdydz = &(k_interp(0, interpolator_var::d2exdydz));
  const float* mem_fey       = &(k_interp(0, interpolator_var::ey));     
  const float* mem_fdeydz    = &(k_interp(0, interpolator_var::deydz));  
  const float* mem_fdeydx    = &(k_interp(0, interpolator_var::deydx));  
  const float* mem_fd2eydzdx = &(k_interp(0, interpolator_var::d2eydzdx));
  const float* mem_fez       = &(k_interp(0, interpolator_var::ez));     
  const float* mem_fdezdx    = &(k_interp(0, interpolator_var::dezdx));  
  const float* mem_fdezdy    = &(k_interp(0, interpolator_var::dezdy));  
  const float* mem_fd2ezdxdy = &(k_interp(0, interpolator_var::d2ezdxdy));
  const float* mem_fcbx      = &(k_interp(0, interpolator_var::cbx));    
  const float* mem_fdcbxdx   = &(k_interp(0, interpolator_var::dcbxdx)); 
  const float* mem_fcby      = &(k_interp(0, interpolator_var::cby));    
  const float* mem_fdcbydy   = &(k_interp(0, interpolator_var::dcbydy)); 
  const float* mem_fcbz      = &(k_interp(0, interpolator_var::cbz));    
  const float* mem_fdcbzdz   = &(k_interp(0, interpolator_var::dcbzdz)); 

  SIMD_F32 fex, fdexdy, fdexdz, fd2exdydz;
  SIMD_F32 fey, fdeydz, fdeydx, fd2eydzdx;
  SIMD_F32 fez, fdezdx, fdezdy, fd2ezdxdy;
  SIMD_F32 fcbx , fcby, fcbz;     
  SIMD_F32 fdcbxdx, fdcbydy, fdcbzdz;

//  SIMD_I32 temp = ii[0];
//  auto same = temp == ii;
//  if ((SIMD_F32::size() == active_lanes) && KokkosSIMD::all_of(same)) {
//    // Load from the same interpolator
//    fex       = SIMD_F32(k_interp(ii[0], interpolator_var::ex));
//    fdexdy    = SIMD_F32(k_interp(ii[0], interpolator_var::dexdy));
//    fdexdz    = SIMD_F32(k_interp(ii[0], interpolator_var::dexdz));
//    fd2exdydz = SIMD_F32(k_interp(ii[0], interpolator_var::d2exdydz));
//    hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
//
//    fey       = SIMD_F32(k_interp(ii[0], interpolator_var::ey));
//    fdeydz    = SIMD_F32(k_interp(ii[0], interpolator_var::deydz));
//    fdeydx    = SIMD_F32(k_interp(ii[0], interpolator_var::deydx));
//    fd2eydzdx = SIMD_F32(k_interp(ii[0], interpolator_var::d2eydzdx));
//    hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
//
//    fez       = SIMD_F32(k_interp(ii[0], interpolator_var::ez));
//    fdezdx    = SIMD_F32(k_interp(ii[0], interpolator_var::dezdx));
//    fdezdy    = SIMD_F32(k_interp(ii[0], interpolator_var::dezdy));
//    fd2ezdxdy = SIMD_F32(k_interp(ii[0], interpolator_var::d2ezdxdy));
//    haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
//
//    fcbx      = SIMD_F32(k_interp(ii[0], interpolator_var::cbx));
//    fdcbxdx   = SIMD_F32(k_interp(ii[0], interpolator_var::dcbxdx));
//    cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
//
//    fcby      = SIMD_F32(k_interp(ii[0], interpolator_var::cby));
//    fdcbydy   = SIMD_F32(k_interp(ii[0], interpolator_var::dcbydy));
//    cby  = Kokkos::fma(dy, fdcbydy, fcby);
//
//    fcbz      = SIMD_F32(k_interp(ii[0], interpolator_var::cbz));
//    fdcbzdz   = SIMD_F32(k_interp(ii[0], interpolator_var::dcbzdz));
//    cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
//  } else {
    if constexpr (std::is_same<Kokkos::LayoutLeft, k_interpolator_t::array_layout>::value) {
      // Load different interpolators for each index
      fex       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fex,       mem_fex+nvoxels),       int_mask, ii, element_aligned_tag_t());
      fdexdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdy,    mem_fdexdy+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fdexdz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdz,    mem_fdexdz+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fd2exdydz = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2exdydz, mem_fd2exdydz+nvoxels), int_mask, ii, element_aligned_tag_t());
      fey       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fey,       mem_fey+nvoxels),       int_mask, ii, element_aligned_tag_t());
      fdeydz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydz,    mem_fdeydz+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fdeydx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydx,    mem_fdeydx+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fd2eydzdx = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2eydzdx, mem_fd2eydzdx+nvoxels), int_mask, ii, element_aligned_tag_t());
      fez       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fez,       mem_fez+nvoxels),       int_mask, ii, element_aligned_tag_t());
      fdezdx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdx,    mem_fdezdx+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fdezdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdy,    mem_fdezdy+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fd2ezdxdy = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2ezdxdy, mem_fd2ezdxdy+nvoxels), int_mask, ii, element_aligned_tag_t());
      fcbx      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbx,      mem_fcbx+nvoxels),      int_mask, ii, element_aligned_tag_t());
      fdcbxdx   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbxdx,   mem_fdcbxdx+nvoxels),   int_mask, ii, element_aligned_tag_t());
      fcby      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcby,      mem_fcby+nvoxels),      int_mask, ii, element_aligned_tag_t());
      fdcbydy   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbydy,   mem_fdcbydy+nvoxels),   int_mask, ii, element_aligned_tag_t());
      fcbz      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbz,      mem_fcbz+nvoxels),      int_mask, ii, element_aligned_tag_t());
      fdcbzdz   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbzdz,   mem_fdcbzdz+nvoxels),   int_mask, ii, element_aligned_tag_t());
      // Interpolate E
      hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
      hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
      haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
      // Interpolate B
      cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
      cby  = Kokkos::fma(dy, fdcbydy, fcby);
      cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
    } else if constexpr (std::is_same<Kokkos::LayoutRight, k_interpolator_t::array_layout>::value) {
      // Load interpolators stored in LayourRight order
      const int interpolator_stride = k_interp.extent(1);
//      if(active_lanes == SIMD_LEN) {
        if constexpr (SIMD_LEN == 16) {
          // Load interpolators in LayoutRight order with 16-wide SIMD
          const float* mem_00 = &(k_interp(ii[0], interpolator_var::ex));     
          const float* mem_01 = &(k_interp(ii[1], interpolator_var::ex));  
          const float* mem_02 = &(k_interp(ii[2], interpolator_var::ex));  
          const float* mem_03 = &(k_interp(ii[3], interpolator_var::ex));
          const float* mem_04 = &(k_interp(ii[4], interpolator_var::ex));     
          const float* mem_05 = &(k_interp(ii[5], interpolator_var::ex));  
          const float* mem_06 = &(k_interp(ii[6], interpolator_var::ex));  
          const float* mem_07 = &(k_interp(ii[7], interpolator_var::ex));
          const float* mem_08 = &(k_interp(ii[8], interpolator_var::ex));     
          const float* mem_09 = &(k_interp(ii[9], interpolator_var::ex));  
          const float* mem_10 = &(k_interp(ii[10], interpolator_var::ex));  
          const float* mem_11 = &(k_interp(ii[11], interpolator_var::ex));
          const float* mem_12 = &(k_interp(ii[12], interpolator_var::ex));    
          const float* mem_13 = &(k_interp(ii[13], interpolator_var::ex)); 
          const float* mem_14 = &(k_interp(ii[14], interpolator_var::ex));    
          const float* mem_15 = &(k_interp(ii[15], interpolator_var::ex)); 
//          const float* mem_00 = mask[ 0] ? &(k_interp(ii[ 0], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_01 = mask[ 1] ? &(k_interp(ii[ 1], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_02 = mask[ 2] ? &(k_interp(ii[ 2], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_03 = mask[ 3] ? &(k_interp(ii[ 3], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_04 = mask[ 4] ? &(k_interp(ii[ 4], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_05 = mask[ 5] ? &(k_interp(ii[ 5], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_06 = mask[ 6] ? &(k_interp(ii[ 6], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_07 = mask[ 7] ? &(k_interp(ii[ 7], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_08 = mask[ 8] ? &(k_interp(ii[ 8], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_09 = mask[ 9] ? &(k_interp(ii[ 9], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_10 = mask[10] ? &(k_interp(ii[10], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_11 = mask[11] ? &(k_interp(ii[11], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_12 = mask[12] ? &(k_interp(ii[12], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_13 = mask[13] ? &(k_interp(ii[13], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_14 = mask[14] ? &(k_interp(ii[14], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
//          const float* mem_15 = mask[15] ? &(k_interp(ii[15], interpolator_var::ex)) : &(k_interp(ii[0], interpolator_var::ex));
          fex       = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_00, vector_aligned_tag_t()); 
          fdexdy    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_01, vector_aligned_tag_t()); 
          fdexdz    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_02, vector_aligned_tag_t()); 
          fd2exdydz = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_03, vector_aligned_tag_t()); 
          fey       = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_04, vector_aligned_tag_t()); 
          fdeydz    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_05, vector_aligned_tag_t()); 
          fdeydx    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_06, vector_aligned_tag_t()); 
          fd2eydzdx = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_07, vector_aligned_tag_t()); 
          fez       = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_08, vector_aligned_tag_t()); 
          fdezdx    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_09, vector_aligned_tag_t()); 
          fdezdy    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_10, vector_aligned_tag_t()); 
          fd2ezdxdy = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_11, vector_aligned_tag_t()); 
          fcbx      = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_12, vector_aligned_tag_t()); 
          fdcbxdx   = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_13, vector_aligned_tag_t()); 
          fcby      = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_14, vector_aligned_tag_t()); 
          fdcbydy   = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_15, vector_aligned_tag_t()); 
          transpose(fex, fdexdy, fdexdz, fd2exdydz, 
                    fey, fdeydz, fdeydx, fd2eydzdx, 
                    fez, fdezdx, fdezdy, fd2ezdxdy, 
                    fcbx, fdcbxdx, fcby, fdcbydy);
          // Interpolate E
          hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
          hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
          haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
          // Interpolate B
          cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
          cby  = Kokkos::fma(dy, fdcbydy, fcby);
          SIMD_F32 v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15;
          fcbz    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_00+16, vector_aligned_tag_t()); 
          fdcbzdz = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_01+16, vector_aligned_tag_t()); 
          v02     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_02+16, vector_aligned_tag_t()); 
          v03     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_03+16, vector_aligned_tag_t()); 
          v04     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_04+16, vector_aligned_tag_t()); 
          v05     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_05+16, vector_aligned_tag_t()); 
          v06     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_06+16, vector_aligned_tag_t()); 
          v07     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_07+16, vector_aligned_tag_t()); 
          v08     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_08+16, vector_aligned_tag_t()); 
          v09     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_09+16, vector_aligned_tag_t()); 
          v10     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_10+16, vector_aligned_tag_t()); 
          v11     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_11+16, vector_aligned_tag_t()); 
          v12     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_12+16, vector_aligned_tag_t()); 
          v13     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_13+16, vector_aligned_tag_t()); 
          v14     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_14+16, vector_aligned_tag_t()); 
          v15     = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_15+16, vector_aligned_tag_t()); 
          transpose(fcbz, fdcbzdz, v02, v03,
                    v04, v05, v06, v07,
                    v08, v09, v10, v11,
                    v12, v13, v14, v15);
//          auto cbz_range    = std::ranges::subrange(&(k_interp(0, interpolator_var::cbz)),    &(k_interp(nvoxels, interpolator_var::cbz)));
//          auto dcbzdz_range = std::ranges::subrange(&(k_interp(0, interpolator_var::dcbzdz)), &(k_interp(nvoxels, interpolator_var::dcbzdz)));
//          fcbz      = KokkosSIMD::unchecked_gather_from<SIMD_F32>(cbz_range,    ii*interpolator_stride, vector_aligned_tag_t());
//          fdcbzdz   = KokkosSIMD::unchecked_gather_from<SIMD_F32>(dcbzdz_range, ii*interpolator_stride, vector_aligned_tag_t());
          cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
        } else if constexpr(SIMD_LEN == 8) {
          const float* mem_00 = &(k_interp(ii[0], interpolator_var::ex));     
          const float* mem_01 = &(k_interp(ii[1], interpolator_var::ex));  
          const float* mem_02 = &(k_interp(ii[2], interpolator_var::ex));  
          const float* mem_03 = &(k_interp(ii[3], interpolator_var::ex));
          const float* mem_04 = &(k_interp(ii[4], interpolator_var::ex));     
          const float* mem_05 = &(k_interp(ii[5], interpolator_var::ex));  
          const float* mem_06 = &(k_interp(ii[6], interpolator_var::ex));  
          const float* mem_07 = &(k_interp(ii[7], interpolator_var::ex));
          
          fex       = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_00, vector_aligned_tag_t()); 
          fdexdy    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_01, vector_aligned_tag_t()); 
          fdexdz    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_02, vector_aligned_tag_t()); 
          fd2exdydz = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_03, vector_aligned_tag_t()); 
          fey       = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_04, vector_aligned_tag_t()); 
          fdeydz    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_05, vector_aligned_tag_t()); 
          fdeydx    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_06, vector_aligned_tag_t()); 
          fd2eydzdx = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_07, vector_aligned_tag_t()); 
          transpose(fex, fdexdy, fdexdz, fd2exdydz, 
                    fey, fdeydz, fdeydx, fd2eydzdx);
          // Interpolate E
          hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
          hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));

          fez       = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_00+8, vector_aligned_tag_t()); 
          fdezdx    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_01+8, vector_aligned_tag_t()); 
          fdezdy    = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_02+8, vector_aligned_tag_t()); 
          fd2ezdxdy = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_03+8, vector_aligned_tag_t()); 
          fcbx      = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_04+8, vector_aligned_tag_t()); 
          fdcbxdx   = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_05+8, vector_aligned_tag_t()); 
          fcby      = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_06+8, vector_aligned_tag_t()); 
          fdcbydy   = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_07+8, vector_aligned_tag_t()); 
          transpose(fez, fdezdx, fdezdy, fd2ezdxdy, 
                    fcbx, fdcbxdx, fcby, fdcbydy);
          // Interpolate E
          haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
          // Interpolate B
          cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
          cby  = Kokkos::fma(dy, fdcbydy, fcby);

          SIMD_F32 t0, t1, t2, t3, t4, t5;
          fcbz      = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_00+16, vector_aligned_tag_t());
          fdcbzdz   = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_01+16, vector_aligned_tag_t());
          t0        = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_02+16, vector_aligned_tag_t());
          t1        = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_03+16, vector_aligned_tag_t());
          t2        = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_04+16, vector_aligned_tag_t());
          t3        = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_05+16, vector_aligned_tag_t());
          t4        = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_06+16, vector_aligned_tag_t());
          t5        = KokkosSIMD::simd_unchecked_load<SIMD_F32>(mem_07+16, vector_aligned_tag_t());
          transpose(fcbz, fdcbzdz, t0, t1, 
                    t2, t3, t4, t5);
//          auto cbz_range    = std::ranges::subrange(&(k_interp(0, interpolator_var::cbz)), &(k_interp(nvoxels, interpolator_var::cbz)));
//          auto dcbzdz_range = std::ranges::subrange(&(k_interp(0, interpolator_var::dcbzdz)), &(k_interp(nvoxels, interpolator_var::dcbzdz)));
//          fcbz      = KokkosSIMD::unchecked_gather_from<SIMD_F32>(cbz_range,    ii*interpolator_stride, vector_aligned_tag_t());
//          fdcbzdz   = KokkosSIMD::unchecked_gather_from<SIMD_F32>(dcbzdz_range, ii*interpolator_stride, vector_aligned_tag_t());
          // Interpolate B
          cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
        } else if constexpr(SIMD_LEN == 4) {
          const float* mem_00 = &(k_interp((int)ii[0], interpolator_var::ex));     
          const float* mem_01 = &(k_interp((int)ii[1], interpolator_var::ex));  
          const float* mem_02 = &(k_interp((int)ii[2], interpolator_var::ex));  
          const float* mem_03 = &(k_interp((int)ii[3], interpolator_var::ex));
          SIMD_F32 t0, t1;
          fex       = KokkosSIMD::simd_unchecked_load(mem_00, vector_aligned_tag_t()); 
          fdexdy    = KokkosSIMD::simd_unchecked_load(mem_01, vector_aligned_tag_t()); 
          fdexdz    = KokkosSIMD::simd_unchecked_load(mem_02, vector_aligned_tag_t()); 
          fd2exdydz = KokkosSIMD::simd_unchecked_load(mem_03, vector_aligned_tag_t()); 
          transpose(fex, fdexdy, fdexdz, fd2exdydz);
          // Interpolate E
          hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));

          fey       = KokkosSIMD::simd_unchecked_load(mem_00+4, vector_aligned_tag_t()); 
          fdeydz    = KokkosSIMD::simd_unchecked_load(mem_01+4, vector_aligned_tag_t()); 
          fdeydx    = KokkosSIMD::simd_unchecked_load(mem_02+4, vector_aligned_tag_t()); 
          fd2eydzdx = KokkosSIMD::simd_unchecked_load(mem_03+4, vector_aligned_tag_t()); 
          transpose(fey, fdeydz, fdeydx, fd2eydzdx);
          // Interpolate E
          hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));

          fez       = KokkosSIMD::simd_unchecked_load(mem_00+8, vector_aligned_tag_t()); 
          fdezdx    = KokkosSIMD::simd_unchecked_load(mem_01+8, vector_aligned_tag_t()); 
          fdezdy    = KokkosSIMD::simd_unchecked_load(mem_02+8, vector_aligned_tag_t()); 
          fd2ezdxdy = KokkosSIMD::simd_unchecked_load(mem_03+8, vector_aligned_tag_t()); 
          transpose(fez, fdezdx, fdezdy, fd2ezdxdy);
          // Interpolate E
          haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));

          fcbx      = KokkosSIMD::simd_unchecked_load(mem_00+12, vector_aligned_tag_t()); 
          fdcbxdx   = KokkosSIMD::simd_unchecked_load(mem_01+12, vector_aligned_tag_t()); 
          fcby      = KokkosSIMD::simd_unchecked_load(mem_02+12, vector_aligned_tag_t()); 
          fdcbydy   = KokkosSIMD::simd_unchecked_load(mem_03+12, vector_aligned_tag_t()); 
          transpose(fcbx, fdcbxdx, fcby, fdcbydy);
          // Interpolate B
          cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
          cby  = Kokkos::fma(dy, fdcbydy, fcby);

          fcbz      = KokkosSIMD::simd_unchecked_load(mem_00+16, vector_aligned_tag_t()); 
          fdcbzdz   = KokkosSIMD::simd_unchecked_load(mem_01+16, vector_aligned_tag_t()); 
          t0        = KokkosSIMD::simd_unchecked_load(mem_02+16, vector_aligned_tag_t()); 
          t1        = KokkosSIMD::simd_unchecked_load(mem_03+16, vector_aligned_tag_t()); 
          transpose(fcbz, fdcbzdz, t0,t1);
          // Interpolate B
          cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
        }
//      } else {
//        SIMD_I32 indices = ii * interpolator_stride;
//        fex       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fex,       mem_fex+nvoxels),       int_mask, indices, element_aligned_tag_t());
//        fdexdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdy,    mem_fdexdy+nvoxels),    int_mask, indices, element_aligned_tag_t());
//        fdexdz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdz,    mem_fdexdz+nvoxels),    int_mask, indices, element_aligned_tag_t());
//        fd2exdydz = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2exdydz, mem_fd2exdydz+nvoxels), int_mask, indices, element_aligned_tag_t());
//        // Interpolate E
//        hax  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2exdydz, fdexdz), dz, Kokkos::fma(dy, fdexdy, fex)));
//        fey       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fey,       mem_fey+nvoxels),       int_mask, indices, element_aligned_tag_t());
//        fdeydz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydz,    mem_fdeydz+nvoxels),    int_mask, indices, element_aligned_tag_t());
//        fdeydx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydx,    mem_fdeydx+nvoxels),    int_mask, indices, element_aligned_tag_t());
//        fd2eydzdx = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2eydzdx, mem_fd2eydzdx+nvoxels), int_mask, indices, element_aligned_tag_t());
//        // Interpolate E
//        hay  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2eydzdx, fdeydx), dx, Kokkos::fma(dz, fdeydz, fey)));
//        fez       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fez,       mem_fez+nvoxels),       int_mask, indices, element_aligned_tag_t());
//        fdezdx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdx,    mem_fdezdx+nvoxels),    int_mask, indices, element_aligned_tag_t());
//        fdezdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdy,    mem_fdezdy+nvoxels),    int_mask, indices, element_aligned_tag_t());
//        fd2ezdxdy = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2ezdxdy, mem_fd2ezdxdy+nvoxels), int_mask, indices, element_aligned_tag_t());
//        // Interpolate E
//        haz  = qdt_2mc*( Kokkos::fma( Kokkos::fma(dy, fd2ezdxdy, fdezdy), dy, Kokkos::fma(dx, fdezdx, fez)));
//        fcbx      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbx,      mem_fcbx+nvoxels),      int_mask, indices, element_aligned_tag_t());
//        fdcbxdx   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbxdx,   mem_fdcbxdx+nvoxels),   int_mask, indices, element_aligned_tag_t());
//        // Interpolate B
//        cbx  = Kokkos::fma(dx, fdcbxdx, fcbx);
//        fcby      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcby,      mem_fcby+nvoxels),      int_mask, indices, element_aligned_tag_t());
//        fdcbydy   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbydy,   mem_fdcbydy+nvoxels),   int_mask, indices, element_aligned_tag_t());
//        // Interpolate B
//        cby  = Kokkos::fma(dy, fdcbydy, fcby);
//        fcbz      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbz,      mem_fcbz+nvoxels),      int_mask, indices, element_aligned_tag_t());
//        fdcbzdz   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbzdz,   mem_fdcbzdz+nvoxels),   int_mask, indices, element_aligned_tag_t());
//        // Interpolate B
//        cbz  = Kokkos::fma(dz, fdcbzdz, fcbz);
////        fex       = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ex);}); 
////        fdexdy    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dexdy);}); 
////        fdexdz    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dexdz);}); 
////        fd2exdydz = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2exdydz);}); 
////        fey       = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ey);}); 
////        fdeydz    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::deydz);}); 
////        fdeydx    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::deydx);}); 
////        fd2eydzdx = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2eydzdx);}); 
////        fez       = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ez);}); 
////        fdezdx    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dezdx);}); 
////        fdezdy    = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dezdy);}); 
////        fd2ezdxdy = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2ezdxdy);}); 
////        fcbx      = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cbx);}); 
////        fdcbxdx   = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbxdx);}); 
////        fcby      = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cby);}); 
////        fdcbydy   = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbydy);}); 
////        fcbz      = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cbz);}); 
////        fdcbzdz   = SIMD_F32([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbzdz);}); 
//      }
    }
//  }
}

template<typename Interpolators, typename SIMDFloat, typename SIMDFloatMask, typename SIMDInt32>
void KOKKOS_INLINE_FUNCTION 
load_interpolators_old(const Interpolators& k_interp, size_t active_lanes, SIMDFloatMask& mask, 
                   SIMDInt32& ii,
                   SIMDFloat& fex,
                   SIMDFloat& fdexdy,
                   SIMDFloat& fdexdz,
                   SIMDFloat& fd2exdydz,
                   SIMDFloat& fey,
                   SIMDFloat& fdeydz,
                   SIMDFloat& fdeydx,
                   SIMDFloat& fd2eydzdx,
                   SIMDFloat& fez,
                   SIMDFloat& fdezdx,
                   SIMDFloat& fdezdy,
                   SIMDFloat& fd2ezdxdy,
                   SIMDFloat& fcbx,
                   SIMDFloat& fdcbxdx,
                   SIMDFloat& fcby,
                   SIMDFloat& fdcbydy,
                   SIMDFloat& fcbz,
                   SIMDFloat& fdcbzdz
) {
  simd_int32_mask_t int_mask(mask);
  int nvoxels = k_interp.extent(0);
  const float* mem_fex       = &(k_interp(0, interpolator_var::ex));     
  const float* mem_fdexdy    = &(k_interp(0, interpolator_var::dexdy));  
  const float* mem_fdexdz    = &(k_interp(0, interpolator_var::dexdz));  
  const float* mem_fd2exdydz = &(k_interp(0, interpolator_var::d2exdydz));
  const float* mem_fey       = &(k_interp(0, interpolator_var::ey));     
  const float* mem_fdeydz    = &(k_interp(0, interpolator_var::deydz));  
  const float* mem_fdeydx    = &(k_interp(0, interpolator_var::deydx));  
  const float* mem_fd2eydzdx = &(k_interp(0, interpolator_var::d2eydzdx));
  const float* mem_fez       = &(k_interp(0, interpolator_var::ez));     
  const float* mem_fdezdx    = &(k_interp(0, interpolator_var::dezdx));  
  const float* mem_fdezdy    = &(k_interp(0, interpolator_var::dezdy));  
  const float* mem_fd2ezdxdy = &(k_interp(0, interpolator_var::d2ezdxdy));
  const float* mem_fcbx      = &(k_interp(0, interpolator_var::cbx));    
  const float* mem_fdcbxdx   = &(k_interp(0, interpolator_var::dcbxdx)); 
  const float* mem_fcby      = &(k_interp(0, interpolator_var::cby));    
  const float* mem_fdcbydy   = &(k_interp(0, interpolator_var::dcbydy)); 
  const float* mem_fcbz      = &(k_interp(0, interpolator_var::cbz));    
  const float* mem_fdcbzdz   = &(k_interp(0, interpolator_var::dcbzdz)); 

  if constexpr (std::is_same<Kokkos::LayoutLeft, k_interpolator_t::array_layout>::value) {
    // Load interpolators LayoutLeft
    SIMDInt32 temp = ii[0];
    auto same = temp == ii;
    if((SIMDFloat::size() == active_lanes) && KokkosSIMD::all_of(same)) {
#if KOKKOS_VERSION_MAJOR == 5
      // Load from the same interpolator
      fex       = SIMDFloat(k_interp(ii[0], interpolator_var::ex));
      fdexdy    = SIMDFloat(k_interp(ii[0], interpolator_var::dexdy));
      fdexdz    = SIMDFloat(k_interp(ii[0], interpolator_var::dexdz));
      fd2exdydz = SIMDFloat(k_interp(ii[0], interpolator_var::d2exdydz));
      fey       = SIMDFloat(k_interp(ii[0], interpolator_var::ey));
      fdeydz    = SIMDFloat(k_interp(ii[0], interpolator_var::deydz));
      fdeydx    = SIMDFloat(k_interp(ii[0], interpolator_var::deydx));
      fd2eydzdx = SIMDFloat(k_interp(ii[0], interpolator_var::d2eydzdx));
      fez       = SIMDFloat(k_interp(ii[0], interpolator_var::ez));
      fdezdx    = SIMDFloat(k_interp(ii[0], interpolator_var::dezdx));
      fdezdy    = SIMDFloat(k_interp(ii[0], interpolator_var::dezdy));
      fd2ezdxdy = SIMDFloat(k_interp(ii[0], interpolator_var::d2ezdxdy));
      fcbx      = SIMDFloat(k_interp(ii[0], interpolator_var::cbx));
      fdcbxdx   = SIMDFloat(k_interp(ii[0], interpolator_var::dcbxdx));
      fcby      = SIMDFloat(k_interp(ii[0], interpolator_var::cby));
      fdcbydy   = SIMDFloat(k_interp(ii[0], interpolator_var::dcbydy));
      fcbz      = SIMDFloat(k_interp(ii[0], interpolator_var::cbz));
      fdcbzdz   = SIMDFloat(k_interp(ii[0], interpolator_var::dcbzdz));
#else
      KokkosSIMD::where(mask, fex)       = k_interp(ii[0], interpolator_var::ex);
      KokkosSIMD::where(mask, fdexdy)    = k_interp(ii[0], interpolator_var::dexdy);
      KokkosSIMD::where(mask, fdexdz)    = k_interp(ii[0], interpolator_var::dexdz);
      KokkosSIMD::where(mask, fd2exdydz) = k_interp(ii[0], interpolator_var::d2exdydz);
      KokkosSIMD::where(mask, fey)       = k_interp(ii[0], interpolator_var::ey);
      KokkosSIMD::where(mask, fdeydz)    = k_interp(ii[0], interpolator_var::deydz);
      KokkosSIMD::where(mask, fdeydx)    = k_interp(ii[0], interpolator_var::deydx);
      KokkosSIMD::where(mask, fd2eydzdx) = k_interp(ii[0], interpolator_var::d2eydzdx);
      KokkosSIMD::where(mask, fez)       = k_interp(ii[0], interpolator_var::ez);
      KokkosSIMD::where(mask, fdezdx)    = k_interp(ii[0], interpolator_var::dezdx);
      KokkosSIMD::where(mask, fdezdy)    = k_interp(ii[0], interpolator_var::dezdy);
      KokkosSIMD::where(mask, fd2ezdxdy) = k_interp(ii[0], interpolator_var::d2ezdxdy);
      KokkosSIMD::where(mask, fcbx)      = k_interp(ii[0], interpolator_var::cbx);
      KokkosSIMD::where(mask, fdcbxdx)   = k_interp(ii[0], interpolator_var::dcbxdx);
      KokkosSIMD::where(mask, fcby)      = k_interp(ii[0], interpolator_var::cby);
      KokkosSIMD::where(mask, fdcbydy)   = k_interp(ii[0], interpolator_var::dcbydy);
      KokkosSIMD::where(mask, fcbz)      = k_interp(ii[0], interpolator_var::cbz);
      KokkosSIMD::where(mask, fdcbzdz)   = k_interp(ii[0], interpolator_var::dcbzdz);
#endif
    } else {
#if KOKKOS_VERSION_MAJOR == 5
      // Load different interpolators for each index
      fex       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fex,       mem_fex+nvoxels),       int_mask, ii, element_aligned_tag_t());
      fdexdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdy,    mem_fdexdy+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fdexdz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdz,    mem_fdexdz+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fd2exdydz = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2exdydz, mem_fd2exdydz+nvoxels), int_mask, ii, element_aligned_tag_t());
      fey       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fey,       mem_fey+nvoxels),       int_mask, ii, element_aligned_tag_t());
      fdeydz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydz,    mem_fdeydz+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fdeydx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydx,    mem_fdeydx+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fd2eydzdx = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2eydzdx, mem_fd2eydzdx+nvoxels), int_mask, ii, element_aligned_tag_t());
      fez       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fez,       mem_fez+nvoxels),       int_mask, ii, element_aligned_tag_t());
      fdezdx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdx,    mem_fdezdx+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fdezdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdy,    mem_fdezdy+nvoxels),    int_mask, ii, element_aligned_tag_t());
      fd2ezdxdy = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2ezdxdy, mem_fd2ezdxdy+nvoxels), int_mask, ii, element_aligned_tag_t());
      fcbx      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbx,      mem_fcbx+nvoxels),      int_mask, ii, element_aligned_tag_t());
      fdcbxdx   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbxdx,   mem_fdcbxdx+nvoxels),   int_mask, ii, element_aligned_tag_t());
      fcby      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcby,      mem_fcby+nvoxels),      int_mask, ii, element_aligned_tag_t());
      fdcbydy   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbydy,   mem_fdcbydy+nvoxels),   int_mask, ii, element_aligned_tag_t());
      fcbz      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbz,      mem_fcbz+nvoxels),      int_mask, ii, element_aligned_tag_t());
      fdcbzdz   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbzdz,   mem_fdcbzdz+nvoxels),   int_mask, ii, element_aligned_tag_t());
//      fex       = SIMDFloat([mem_fex, ii](std::size_t idx) {return mem_fex[ii[idx]];});
//      fdexdy    = SIMDFloat([mem_fdexdy, ii](std::size_t idx) {return mem_fdexdy[ii[idx]];});
//      fdexdz    = SIMDFloat([mem_fdexdz, ii](std::size_t idx) {return mem_fdexdz[ii[idx]];});
//      fd2exdydz = SIMDFloat([mem_fd2exdydz, ii](std::size_t idx) {return mem_fd2exdydz[ii[idx]];});
//      fey       = SIMDFloat([mem_fey, ii](std::size_t idx) {return mem_fey[ii[idx]];});
//      fdeydz    = SIMDFloat([mem_fdeydz, ii](std::size_t idx) {return mem_fdeydz[ii[idx]];});
//      fdeydx    = SIMDFloat([mem_fdeydx, ii](std::size_t idx) {return mem_fdeydx[ii[idx]];});
//      fd2eydzdx = SIMDFloat([mem_fd2eydzdx, ii](std::size_t idx) {return mem_fd2eydzdx[ii[idx]];});
//      fez       = SIMDFloat([mem_fez, ii](std::size_t idx) {return mem_fez[ii[idx]];});
//      fdezdx    = SIMDFloat([mem_fdezdx, ii](std::size_t idx) {return mem_fdezdx[ii[idx]];});
//      fdezdy    = SIMDFloat([mem_fdezdy, ii](std::size_t idx) {return mem_fdezdy[ii[idx]];});
//      fd2ezdxdy = SIMDFloat([mem_fd2ezdxdy, ii](std::size_t idx) {return mem_fd2ezdxdy[ii[idx]];});
//      fcbx      = SIMDFloat([mem_fcbx, ii](std::size_t idx) {return mem_fcbx[ii[idx]];});
//      fdcbxdx   = SIMDFloat([mem_fdcbxdx, ii](std::size_t idx) {return mem_fdcbxdx[ii[idx]];});
//      fcby      = SIMDFloat([mem_fcby, ii](std::size_t idx) {return mem_fcby[ii[idx]];});
//      fdcbydy   = SIMDFloat([mem_fdcbydy, ii](std::size_t idx) {return mem_fdcbydy[ii[idx]];});
//      fcbz      = SIMDFloat([mem_fcbz, ii](std::size_t idx) {return mem_fcbz[ii[idx]];});
//      fdcbzdz   = SIMDFloat([mem_fdcbzdz, ii](std::size_t idx) {return mem_fdcbzdz[ii[idx]];});
#else
      KokkosSIMD::where(mask, fex).gather_from(mem_fex, ii);
      KokkosSIMD::where(mask, fdexdy).gather_from(mem_fdexdy, ii);
      KokkosSIMD::where(mask, fdexdz).gather_from(mem_fdexdz, ii);
      KokkosSIMD::where(mask, fd2exdydz).gather_from(mem_fd2exdydz, ii);
      KokkosSIMD::where(mask, fey).gather_from(mem_fey, ii);
      KokkosSIMD::where(mask, fdeydz).gather_from(mem_fdeydz, ii);
      KokkosSIMD::where(mask, fdeydx).gather_from(mem_fdeydx, ii);
      KokkosSIMD::where(mask, fd2eydzdx).gather_from(mem_fd2eydzdx, ii);
      KokkosSIMD::where(mask, fez).gather_from(mem_fez, ii);
      KokkosSIMD::where(mask, fdezdx).gather_from(mem_fdezdx, ii);
      KokkosSIMD::where(mask, fdezdy).gather_from(mem_fdezdy, ii);
      KokkosSIMD::where(mask, fd2ezdxdy).gather_from(mem_fd2ezdxdy, ii);
      KokkosSIMD::where(mask, fcbx).gather_from(mem_fcbx, ii);
      KokkosSIMD::where(mask, fdcbxdx).gather_from(mem_fdcbxdx, ii);
      KokkosSIMD::where(mask, fcby).gather_from(mem_fcby, ii);
      KokkosSIMD::where(mask, fdcbydy).gather_from(mem_fdcbydy, ii);
      KokkosSIMD::where(mask, fcbz).gather_from(mem_fcbz, ii);
      KokkosSIMD::where(mask, fdcbzdz).gather_from(mem_fdcbzdz, ii);
#endif
    }
  } else if constexpr (std::is_same<Kokkos::LayoutRight, k_interpolator_t::array_layout>::value) {
    // Load interpolators stored in LayourRight order
    const int interpolator_stride = k_interp.extent(1);
    SIMDInt32 temp = ii[0];
    auto same = temp == ii;
    if((SIMD_LEN == active_lanes) && KokkosSIMD::all_of(same)) {
#if KOKKOS_VERSION_MAJOR == 5
      // Load from the same interpolator
      fex       = SIMDFloat(k_interp(ii[0], interpolator_var::ex));
      fdexdy    = SIMDFloat(k_interp(ii[0], interpolator_var::dexdy));
      fdexdz    = SIMDFloat(k_interp(ii[0], interpolator_var::dexdz));
      fd2exdydz = SIMDFloat(k_interp(ii[0], interpolator_var::d2exdydz));
      fey       = SIMDFloat(k_interp(ii[0], interpolator_var::ey));
      fdeydz    = SIMDFloat(k_interp(ii[0], interpolator_var::deydz));
      fdeydx    = SIMDFloat(k_interp(ii[0], interpolator_var::deydx));
      fd2eydzdx = SIMDFloat(k_interp(ii[0], interpolator_var::d2eydzdx));
      fez       = SIMDFloat(k_interp(ii[0], interpolator_var::ez));
      fdezdx    = SIMDFloat(k_interp(ii[0], interpolator_var::dezdx));
      fdezdy    = SIMDFloat(k_interp(ii[0], interpolator_var::dezdy));
      fd2ezdxdy = SIMDFloat(k_interp(ii[0], interpolator_var::d2ezdxdy));
      fcbx      = SIMDFloat(k_interp(ii[0], interpolator_var::cbx));
      fdcbxdx   = SIMDFloat(k_interp(ii[0], interpolator_var::dcbxdx));
      fcby      = SIMDFloat(k_interp(ii[0], interpolator_var::cby));
      fdcbydy   = SIMDFloat(k_interp(ii[0], interpolator_var::dcbydy));
      fcbz      = SIMDFloat(k_interp(ii[0], interpolator_var::cbz));
      fdcbzdz   = SIMDFloat(k_interp(ii[0], interpolator_var::dcbzdz));
#else
      KokkosSIMD::where(mask, fex)       = k_interp((int)ii[0], interpolator_var::ex);
      KokkosSIMD::where(mask, fdexdy)    = k_interp((int)ii[0], interpolator_var::dexdy);
      KokkosSIMD::where(mask, fdexdz)    = k_interp((int)ii[0], interpolator_var::dexdz);
      KokkosSIMD::where(mask, fd2exdydz) = k_interp((int)ii[0], interpolator_var::d2exdydz);
      KokkosSIMD::where(mask, fey)       = k_interp((int)ii[0], interpolator_var::ey);
      KokkosSIMD::where(mask, fdeydz)    = k_interp((int)ii[0], interpolator_var::deydz);
      KokkosSIMD::where(mask, fdeydx)    = k_interp((int)ii[0], interpolator_var::deydx);
      KokkosSIMD::where(mask, fd2eydzdx) = k_interp((int)ii[0], interpolator_var::d2eydzdx);
      KokkosSIMD::where(mask, fez)       = k_interp((int)ii[0], interpolator_var::ez);
      KokkosSIMD::where(mask, fdezdx)    = k_interp((int)ii[0], interpolator_var::dezdx);
      KokkosSIMD::where(mask, fdezdy)    = k_interp((int)ii[0], interpolator_var::dezdy);
      KokkosSIMD::where(mask, fd2ezdxdy) = k_interp((int)ii[0], interpolator_var::d2ezdxdy);
      KokkosSIMD::where(mask, fcbx)      = k_interp((int)ii[0], interpolator_var::cbx);
      KokkosSIMD::where(mask, fdcbxdx)   = k_interp((int)ii[0], interpolator_var::dcbxdx);
      KokkosSIMD::where(mask, fcby)      = k_interp((int)ii[0], interpolator_var::cby);
      KokkosSIMD::where(mask, fdcbydy)   = k_interp((int)ii[0], interpolator_var::dcbydy);
      KokkosSIMD::where(mask, fcbz)      = k_interp((int)ii[0], interpolator_var::cbz);
      KokkosSIMD::where(mask, fdcbzdz)   = k_interp((int)ii[0], interpolator_var::dcbzdz);
#endif
    } else if(active_lanes == SIMD_LEN) {
      if constexpr (SIMD_LEN == 16) {
        // Load interpolators in LayoutRight order with 16-wide SIMD
        const float* mem_00 = &(k_interp(ii[0], interpolator_var::ex));     
        const float* mem_01 = &(k_interp(ii[1], interpolator_var::ex));  
        const float* mem_02 = &(k_interp(ii[2], interpolator_var::ex));  
        const float* mem_03 = &(k_interp(ii[3], interpolator_var::ex));
        const float* mem_04 = &(k_interp(ii[4], interpolator_var::ex));     
        const float* mem_05 = &(k_interp(ii[5], interpolator_var::ex));  
        const float* mem_06 = &(k_interp(ii[6], interpolator_var::ex));  
        const float* mem_07 = &(k_interp(ii[7], interpolator_var::ex));
        const float* mem_08 = &(k_interp(ii[8], interpolator_var::ex));     
        const float* mem_09 = &(k_interp(ii[9], interpolator_var::ex));  
        const float* mem_10 = &(k_interp(ii[10], interpolator_var::ex));  
        const float* mem_11 = &(k_interp(ii[11], interpolator_var::ex));
        const float* mem_12 = &(k_interp(ii[12], interpolator_var::ex));    
        const float* mem_13 = &(k_interp(ii[13], interpolator_var::ex)); 
        const float* mem_14 = &(k_interp(ii[14], interpolator_var::ex));    
        const float* mem_15 = &(k_interp(ii[15], interpolator_var::ex)); 
#if KOKKOS_VERSION_MAJOR == 5
        fex       = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_00, vector_aligned_tag_t()); 
        fdexdy    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_01, vector_aligned_tag_t()); 
        fdexdz    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_02, vector_aligned_tag_t()); 
        fd2exdydz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_03, vector_aligned_tag_t()); 
        fey       = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_04, vector_aligned_tag_t()); 
        fdeydz    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_05, vector_aligned_tag_t()); 
        fdeydx    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_06, vector_aligned_tag_t()); 
        fd2eydzdx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_07, vector_aligned_tag_t()); 
        fez       = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_08, vector_aligned_tag_t()); 
        fdezdx    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_09, vector_aligned_tag_t()); 
        fdezdy    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_10, vector_aligned_tag_t()); 
        fd2ezdxdy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_11, vector_aligned_tag_t()); 
        fcbx      = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_12, vector_aligned_tag_t()); 
        fdcbxdx   = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_13, vector_aligned_tag_t()); 
        fcby      = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_14, vector_aligned_tag_t()); 
        fdcbydy   = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_15, vector_aligned_tag_t()); 
        transpose(fex, fdexdy, fdexdz, fd2exdydz, 
                  fey, fdeydz, fdeydx, fd2eydzdx, 
                  fez, fdezdx, fdezdy, fd2ezdxdy, 
                  fcbx, fdcbxdx, fcby, fdcbydy);
        auto cbz_range    = std::ranges::subrange(&(k_interp(0, interpolator_var::cbz)), &(k_interp(nvoxels, interpolator_var::cbz)));
        auto dcbzdz_range = std::ranges::subrange(&(k_interp(0, interpolator_var::dcbzdz)), &(k_interp(nvoxels, interpolator_var::dcbzdz)));
        fcbz      = KokkosSIMD::unchecked_gather_from<SIMDFloat>(cbz_range,    ii*interpolator_stride, vector_aligned_tag_t());
        fdcbzdz   = KokkosSIMD::unchecked_gather_from<SIMDFloat>(dcbzdz_range, ii*interpolator_stride, vector_aligned_tag_t());
#else
        fex.copy_from(      mem_00, element_aligned_tag_t());
        fdexdy.copy_from(   mem_01, element_aligned_tag_t());    
        fdexdz.copy_from(   mem_02, element_aligned_tag_t());    
        fd2exdydz.copy_from(mem_03, element_aligned_tag_t()); 
        fey.copy_from(      mem_04, element_aligned_tag_t());       
        fdeydz.copy_from(   mem_05, element_aligned_tag_t());    
        fdeydx.copy_from(   mem_06, element_aligned_tag_t());    
        fd2eydzdx.copy_from(mem_07, element_aligned_tag_t()); 
        fez.copy_from(      mem_08, element_aligned_tag_t());       
        fdezdx.copy_from(   mem_09, element_aligned_tag_t());    
        fdezdy.copy_from(   mem_10, element_aligned_tag_t());    
        fd2ezdxdy.copy_from(mem_11, element_aligned_tag_t()); 
        fcbx.copy_from(     mem_12, element_aligned_tag_t());      
        fdcbxdx.copy_from(  mem_13, element_aligned_tag_t());   
        fcby.copy_from(     mem_14, element_aligned_tag_t());      
        fdcbydy.copy_from(  mem_15, element_aligned_tag_t());   
        transpose(fex, fdexdy, fdexdz, fd2exdydz, 
                  fey, fdeydz, fdeydx, fd2eydzdx, 
                  fez, fdezdx, fdezdy, fd2ezdxdy, 
                  fcbx, fdcbxdx, fcby, fdcbydy);
        KokkosSIMD::where(mask, fcbz     ).gather_from(&(k_interp(0, interpolator_var::cbz)),      ii*interpolator_stride);
        KokkosSIMD::where(mask, fdcbzdz  ).gather_from(&(k_interp(0, interpolator_var::dcbzdz)),   ii*interpolator_stride);
#endif
      } else if constexpr(SIMD_LEN == 8) {
        const float* mem_00 = &(k_interp(ii[0], interpolator_var::ex));     
        const float* mem_01 = &(k_interp(ii[1], interpolator_var::ex));  
        const float* mem_02 = &(k_interp(ii[2], interpolator_var::ex));  
        const float* mem_03 = &(k_interp(ii[3], interpolator_var::ex));
        const float* mem_04 = &(k_interp(ii[4], interpolator_var::ex));     
        const float* mem_05 = &(k_interp(ii[5], interpolator_var::ex));  
        const float* mem_06 = &(k_interp(ii[6], interpolator_var::ex));  
        const float* mem_07 = &(k_interp(ii[7], interpolator_var::ex));
        
#if KOKKOS_VERSION_MAJOR == 5
        fex       = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_00, vector_aligned_tag_t()); 
        fdexdy    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_01, vector_aligned_tag_t()); 
        fdexdz    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_02, vector_aligned_tag_t()); 
        fd2exdydz = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_03, vector_aligned_tag_t()); 
        fey       = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_04, vector_aligned_tag_t()); 
        fdeydz    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_05, vector_aligned_tag_t()); 
        fdeydx    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_06, vector_aligned_tag_t()); 
        fd2eydzdx = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_07, vector_aligned_tag_t()); 
        transpose(fex, fdexdy, fdexdz, fd2exdydz, 
                  fey, fdeydz, fdeydx, fd2eydzdx);
        fez       = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_00+8, vector_aligned_tag_t()); 
        fdezdx    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_01+8, vector_aligned_tag_t()); 
        fdezdy    = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_02+8, vector_aligned_tag_t()); 
        fd2ezdxdy = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_03+8, vector_aligned_tag_t()); 
        fcbx      = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_04+8, vector_aligned_tag_t()); 
        fdcbxdx   = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_05+8, vector_aligned_tag_t()); 
        fcby      = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_06+8, vector_aligned_tag_t()); 
        fdcbydy   = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_07+8, vector_aligned_tag_t()); 
        transpose(fez, fdezdx, fdezdy, fd2ezdxdy, 
                  fcbx, fdcbxdx, fcby, fdcbydy);
        SIMDFloat t0, t1, t2, t3, t4, t5;
        fcbz      = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_00+16, vector_aligned_tag_t());
        fdcbzdz   = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_01+16, vector_aligned_tag_t());
        t0        = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_02+16, vector_aligned_tag_t());
        t1        = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_03+16, vector_aligned_tag_t());
        t2        = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_04+16, vector_aligned_tag_t());
        t3        = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_05+16, vector_aligned_tag_t());
        t4        = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_06+16, vector_aligned_tag_t());
        t5        = KokkosSIMD::simd_unchecked_load<SIMDFloat>(mem_07+16, vector_aligned_tag_t());
        transpose(fcbz, fdcbzdz, t0, t1, 
                  t2, t3, t4, t5);
//        auto cbz_range    = std::ranges::subrange(&(k_interp(0, interpolator_var::cbz)), &(k_interp(nvoxels, interpolator_var::cbz)));
//        auto dcbzdz_range = std::ranges::subrange(&(k_interp(0, interpolator_var::dcbzdz)), &(k_interp(nvoxels, interpolator_var::dcbzdz)));
//        fcbz      = KokkosSIMD::unchecked_gather_from<SIMDFloat>(cbz_range,    ii*interpolator_stride, vector_aligned_tag_t());
//        fdcbzdz   = KokkosSIMD::unchecked_gather_from<SIMDFloat>(dcbzdz_range, ii*interpolator_stride, vector_aligned_tag_t());
#else
        fex.copy_from(mem_00, element_aligned_tag_t());
        fdexdy.copy_from(mem_01, element_aligned_tag_t());    
        fdexdz.copy_from(mem_02, element_aligned_tag_t());    
        fd2exdydz.copy_from(mem_03, element_aligned_tag_t()); 
        fey.copy_from(mem_04, element_aligned_tag_t());       
        fdeydz.copy_from(mem_05, element_aligned_tag_t());    
        fdeydx.copy_from(mem_06, element_aligned_tag_t());    
        fd2eydzdx.copy_from(mem_07, element_aligned_tag_t()); 
        transpose(fex, fdexdy, fdexdz, fd2exdydz, 
                  fey, fdeydz, fdeydx, fd2eydzdx);
        
        fez.copy_from(mem_00+8, element_aligned_tag_t());       
        fdezdx.copy_from(mem_01+8, element_aligned_tag_t());    
        fdezdy.copy_from(mem_02+8, element_aligned_tag_t());    
        fd2ezdxdy.copy_from(mem_03+8, element_aligned_tag_t()); 
        fcbx.copy_from(mem_04+8, element_aligned_tag_t());      
        fdcbxdx.copy_from(mem_05+8, element_aligned_tag_t());   
        fcby.copy_from(mem_06+8, element_aligned_tag_t());      
        fdcbydy.copy_from(mem_07+8, element_aligned_tag_t());   
        transpose(fez, fdezdx, fdezdy, fd2ezdxdy, 
                  fcbx, fdcbxdx, fcby, fdcbydy);
        KokkosSIMD::where(mask, fcbz     ).gather_from(&(k_interp(0, interpolator_var::cbz)),      ii*interpolator_stride);
        KokkosSIMD::where(mask, fdcbzdz  ).gather_from(&(k_interp(0, interpolator_var::dcbzdz)),   ii*interpolator_stride);
#endif
      } else if constexpr(SIMD_LEN == 4) {
        const float* mem_00 = &(k_interp((int)ii[0], interpolator_var::ex));     
        const float* mem_01 = &(k_interp((int)ii[1], interpolator_var::ex));  
        const float* mem_02 = &(k_interp((int)ii[2], interpolator_var::ex));  
        const float* mem_03 = &(k_interp((int)ii[3], interpolator_var::ex));
#if KOKKOS_VERSION_MAJOR == 5
        SIMDFloat t0, t1;
        fex       = KokkosSIMD::simd_unchecked_load(mem_00, element_aligned_tag_t()); 
        fdexdy    = KokkosSIMD::simd_unchecked_load(mem_01, element_aligned_tag_t()); 
        fdexdz    = KokkosSIMD::simd_unchecked_load(mem_02, element_aligned_tag_t()); 
        fd2exdydz = KokkosSIMD::simd_unchecked_load(mem_03, element_aligned_tag_t()); 
        transpose(fex, fdexdy, fdexdz, fd2exdydz);
        fey       = KokkosSIMD::simd_unchecked_load(mem_00+4, element_aligned_tag_t()); 
        fdeydz    = KokkosSIMD::simd_unchecked_load(mem_01+4, element_aligned_tag_t()); 
        fdeydx    = KokkosSIMD::simd_unchecked_load(mem_02+4, element_aligned_tag_t()); 
        fd2eydzdx = KokkosSIMD::simd_unchecked_load(mem_03+4, element_aligned_tag_t()); 
        transpose(fey, fdeydz, fdeydx, fd2eydzdx);
        fez       = KokkosSIMD::simd_unchecked_load(mem_00+8, element_aligned_tag_t()); 
        fdezdx    = KokkosSIMD::simd_unchecked_load(mem_01+8, element_aligned_tag_t()); 
        fdezdy    = KokkosSIMD::simd_unchecked_load(mem_02+8, element_aligned_tag_t()); 
        fd2ezdxdy = KokkosSIMD::simd_unchecked_load(mem_03+8, element_aligned_tag_t()); 
        transpose(fez, fdezdx, fdezdy, fd2ezdxdy);
        fcbx      = KokkosSIMD::simd_unchecked_load(mem_00+12, element_aligned_tag_t()); 
        fdcbxdx   = KokkosSIMD::simd_unchecked_load(mem_01+12, element_aligned_tag_t()); 
        fcby      = KokkosSIMD::simd_unchecked_load(mem_02+12, element_aligned_tag_t()); 
        fdcbydy   = KokkosSIMD::simd_unchecked_load(mem_03+12, element_aligned_tag_t()); 
        transpose(fcbx, fdcbxdx, fcby, fdcbydy);
        fcbz      = KokkosSIMD::simd_unchecked_load(mem_00+16, element_aligned_tag_t()); 
        fdcbzdz   = KokkosSIMD::simd_unchecked_load(mem_01+16, element_aligned_tag_t()); 
        t0        = KokkosSIMD::simd_unchecked_load(mem_02+16, element_aligned_tag_t()); 
        t1        = KokkosSIMD::simd_unchecked_load(mem_03+16, element_aligned_tag_t()); 
        transpose(fcbz, fdcbzdz, t0,t1);
#else
        fex.copy_from(mem_00, element_aligned_tag_t());
        fdexdy.copy_from(mem_01, element_aligned_tag_t());    
        fdexdz.copy_from(mem_02, element_aligned_tag_t());    
        fd2exdydz.copy_from(mem_03, element_aligned_tag_t()); 
        transpose(fex, fdexdy, fdexdz, fd2exdydz);

        fey.copy_from(mem_00+4, element_aligned_tag_t());       
        fdeydz.copy_from(mem_01+4, element_aligned_tag_t());    
        fdeydx.copy_from(mem_02+4, element_aligned_tag_t());    
        fd2eydzdx.copy_from(mem_03+4, element_aligned_tag_t()); 
        transpose(fey, fdeydz, fdeydx, fd2eydzdx);
        
        fez.copy_from(mem_00+8, element_aligned_tag_t());       
        fdezdx.copy_from(mem_01+8, element_aligned_tag_t());    
        fdezdy.copy_from(mem_02+8, element_aligned_tag_t());    
        fd2ezdxdy.copy_from(mem_03+8, element_aligned_tag_t()); 
        transpose(fez, fdezdx, fdezdy, fd2ezdxdy);

        fcbx.copy_from(mem_00+12, element_aligned_tag_t());      
        fdcbxdx.copy_from(mem_01+12, element_aligned_tag_t());   
        fcby.copy_from(mem_02+12, element_aligned_tag_t());      
        fdcbydy.copy_from(mem_03+12, element_aligned_tag_t());   
        transpose(fcbx, fdcbxdx, fcby, fdcbydy);

        SIMDFloat t0, t1;
        fcbz.copy_from(mem_00+16, element_aligned_tag_t());      
        fdcbzdz.copy_from(mem_01+16, element_aligned_tag_t());   
        t0.copy_from(mem_02+16, element_aligned_tag_t());      
        t1.copy_from(mem_03+16, element_aligned_tag_t());   
        transpose(fcbz, fdcbzdz, t0,t1);
#endif

      }
    } else {
#if KOKKOS_VERSION_MAJOR == 5
      SIMDInt32 indices = ii * interpolator_stride;
      fex       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fex,       mem_fex+nvoxels),       int_mask, indices, element_aligned_tag_t());
      fdexdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdy,    mem_fdexdy+nvoxels),    int_mask, indices, element_aligned_tag_t());
      fdexdz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdexdz,    mem_fdexdz+nvoxels),    int_mask, indices, element_aligned_tag_t());
      fd2exdydz = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2exdydz, mem_fd2exdydz+nvoxels), int_mask, indices, element_aligned_tag_t());
      fey       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fey,       mem_fey+nvoxels),       int_mask, indices, element_aligned_tag_t());
      fdeydz    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydz,    mem_fdeydz+nvoxels),    int_mask, indices, element_aligned_tag_t());
      fdeydx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdeydx,    mem_fdeydx+nvoxels),    int_mask, indices, element_aligned_tag_t());
      fd2eydzdx = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2eydzdx, mem_fd2eydzdx+nvoxels), int_mask, indices, element_aligned_tag_t());
      fez       = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fez,       mem_fez+nvoxels),       int_mask, indices, element_aligned_tag_t());
      fdezdx    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdx,    mem_fdezdx+nvoxels),    int_mask, indices, element_aligned_tag_t());
      fdezdy    = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdezdy,    mem_fdezdy+nvoxels),    int_mask, indices, element_aligned_tag_t());
      fd2ezdxdy = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fd2ezdxdy, mem_fd2ezdxdy+nvoxels), int_mask, indices, element_aligned_tag_t());
      fcbx      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbx,      mem_fcbx+nvoxels),      int_mask, indices, element_aligned_tag_t());
      fdcbxdx   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbxdx,   mem_fdcbxdx+nvoxels),   int_mask, indices, element_aligned_tag_t());
      fcby      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcby,      mem_fcby+nvoxels),      int_mask, indices, element_aligned_tag_t());
      fdcbydy   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbydy,   mem_fdcbydy+nvoxels),   int_mask, indices, element_aligned_tag_t());
      fcbz      = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fcbz,      mem_fcbz+nvoxels),      int_mask, indices, element_aligned_tag_t());
      fdcbzdz   = KokkosSIMD::unchecked_gather_from<simd_float_t>(std::ranges::subrange(mem_fdcbzdz,   mem_fdcbzdz+nvoxels),   int_mask, indices, element_aligned_tag_t());
//      fex       = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ex);}); 
//      fdexdy    = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dexdy);}); 
//      fdexdz    = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dexdz);}); 
//      fd2exdydz = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2exdydz);}); 
//      fey       = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ey);}); 
//      fdeydz    = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::deydz);}); 
//      fdeydx    = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::deydx);}); 
//      fd2eydzdx = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2eydzdx);}); 
//      fez       = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::ez);}); 
//      fdezdx    = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dezdx);}); 
//      fdezdy    = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dezdy);}); 
//      fd2ezdxdy = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::d2ezdxdy);}); 
//      fcbx      = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cbx);}); 
//      fdcbxdx   = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbxdx);}); 
//      fcby      = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cby);}); 
//      fdcbydy   = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbydy);}); 
//      fcbz      = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::cbz);}); 
//      fdcbzdz   = SIMDFloat([k_interp, ii](std::size_t idx) {return k_interp(ii[idx], interpolator_var::dcbzdz);}); 
#else
      KokkosSIMD::where(mask, fex      ).gather_from(&(k_interp(0, interpolator_var::ex)),       ii*interpolator_stride);
      KokkosSIMD::where(mask, fdexdy   ).gather_from(&(k_interp(0, interpolator_var::dexdy)),    ii*interpolator_stride);
      KokkosSIMD::where(mask, fdexdz   ).gather_from(&(k_interp(0, interpolator_var::dexdz)),    ii*interpolator_stride);
      KokkosSIMD::where(mask, fd2exdydz).gather_from(&(k_interp(0, interpolator_var::d2exdydz)), ii*interpolator_stride);
      KokkosSIMD::where(mask, fey      ).gather_from(&(k_interp(0, interpolator_var::ey)),       ii*interpolator_stride);
      KokkosSIMD::where(mask, fdeydz   ).gather_from(&(k_interp(0, interpolator_var::deydz)),    ii*interpolator_stride);
      KokkosSIMD::where(mask, fdeydx   ).gather_from(&(k_interp(0, interpolator_var::deydx)),    ii*interpolator_stride);
      KokkosSIMD::where(mask, fd2eydzdx).gather_from(&(k_interp(0, interpolator_var::d2eydzdx)), ii*interpolator_stride);
      KokkosSIMD::where(mask, fez      ).gather_from(&(k_interp(0, interpolator_var::ez)),       ii*interpolator_stride);
      KokkosSIMD::where(mask, fdezdx   ).gather_from(&(k_interp(0, interpolator_var::dezdx)),    ii*interpolator_stride);
      KokkosSIMD::where(mask, fdezdy   ).gather_from(&(k_interp(0, interpolator_var::dezdy)),    ii*interpolator_stride);
      KokkosSIMD::where(mask, fd2ezdxdy).gather_from(&(k_interp(0, interpolator_var::d2ezdxdy)), ii*interpolator_stride);
      KokkosSIMD::where(mask, fcbx     ).gather_from(&(k_interp(0, interpolator_var::cbx)),      ii*interpolator_stride);
      KokkosSIMD::where(mask, fdcbxdx  ).gather_from(&(k_interp(0, interpolator_var::dcbxdx)),   ii*interpolator_stride);
      KokkosSIMD::where(mask, fcby     ).gather_from(&(k_interp(0, interpolator_var::cby)),      ii*interpolator_stride);
      KokkosSIMD::where(mask, fdcbydy  ).gather_from(&(k_interp(0, interpolator_var::dcbydy)),   ii*interpolator_stride);
      KokkosSIMD::where(mask, fcbz     ).gather_from(&(k_interp(0, interpolator_var::cbz)),      ii*interpolator_stride);
      KokkosSIMD::where(mask, fdcbzdz  ).gather_from(&(k_interp(0, interpolator_var::dcbzdz)),   ii*interpolator_stride);
#endif
    }
  }
}

// Write current values to either an accumulator or directly to the fields
template<class CurrentScatterAccess>
void KOKKOS_INLINE_FUNCTION
accumulate_current(CurrentScatterAccess& current_sa, int ii,
                   const int nx, const int ny, const int nz, 
                   const float cx, const float cy, const float cz, 
                   const float v0, const float v1, const float v2, const float v3,
                   const float v4, const float v5, const float v6, const float v7,
                   const float v8, const float v9, const float v10, const float v11) {
#ifdef VPIC_ENABLE_ACCUMULATORS
  current_sa(ii, 0)  += cx*v0;
  current_sa(ii, 1)  += cx*v1;
  current_sa(ii, 2)  += cx*v2;
  current_sa(ii, 3)  += cx*v3;
  
  current_sa(ii, 4)  += cy*v4;
  current_sa(ii, 5)  += cy*v5;
  current_sa(ii, 6)  += cy*v6;
  current_sa(ii, 7)  += cy*v7;
  
  current_sa(ii, 8)  += cz*v8;
  current_sa(ii, 9)  += cz*v9;
  current_sa(ii, 10) += cz*v10;
  current_sa(ii, 11) += cz*v11;
#else
  int iii = ii;
  int zi = iii/((nx+2)*(ny+2));
  iii -= zi*(nx+2)*(ny+2);
  int yi = iii/(nx+2);
  int xi = iii - yi*(nx+2);
  
  current_sa(ii, field_var::jfx)                           += cx*v0;
  current_sa(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfx)   += cx*v1;
  current_sa(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfx)   += cx*v2;
  current_sa(VOXEL(xi,yi+1,zi+1,nx,ny,nz), field_var::jfx) += cx*v3;
  
  current_sa(ii, field_var::jfy)                           += cy*v4;
  current_sa(VOXEL(xi,yi,zi+1,nx,ny,nz), field_var::jfy)   += cy*v5;
  current_sa(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfy)   += cy*v6;
  current_sa(VOXEL(xi+1,yi,zi+1,nx,ny,nz), field_var::jfy) += cy*v7;
  
  current_sa(ii, field_var::jfz)                           += cz*v8;
  current_sa(VOXEL(xi+1,yi,zi,nx,ny,nz), field_var::jfz)   += cz*v9;
  current_sa(VOXEL(xi,yi+1,zi,nx,ny,nz), field_var::jfz)   += cz*v10;
  current_sa(VOXEL(xi+1,yi+1,zi,nx,ny,nz), field_var::jfz) += cz*v11;
#endif
}

// Reduce the current for all active threads/lanes to reduce the number of writes to memory
template<class TeamMember, class CurrentScatterAccess>
void KOKKOS_INLINE_FUNCTION
reduce_and_accumulate_current(TeamMember& team_member, CurrentScatterAccess& access, 
                              const int num_iters, const int ii, 
                              const int nx, const int ny, const int nz, 
                              const float cx, const float cy, const float cz, 
                              float *v0, float *v1,  float *v2,  float *v3,
                              float *v4, float *v5,  float *v6,  float *v7,
                              float *v8, float *v9,  float *v10, float *v11) {

#ifdef VPIC_ENABLE_VECTORIZATION
  alignas(16) float valx[4] = {0.0, 0.0, 0.0, 0.0};
  alignas(16) float valy[4] = {0.0, 0.0, 0.0, 0.0};
  alignas(16) float valz[4] = {0.0, 0.0, 0.0, 0.0};
  #pragma omp simd reduction(+:valx[0:4],valy[0:4],valz[0:4])
  for(int lane=0; lane<num_iters; lane++) {
    valx[0] += v0[lane];
    valx[1] += v1[lane];
    valx[2] += v2[lane];
    valx[3] += v3[lane];
    valy[0] += v4[lane];
    valy[1] += v5[lane];
    valy[2] += v6[lane];
    valy[3] += v7[lane];
    valz[0] += v8[lane];
    valz[1] += v9[lane];
    valz[2] += v10[lane];
    valz[3] += v11[lane];
  }
  accumulate_current(access, ii, nx, ny, nz, cx, cy, cz, 
                     valx[0], valx[1], valx[2], valx[3], 
                     valy[0], valy[1], valy[2], valy[3], 
                     valz[0], valz[1], valz[2], valz[3]);
#elif defined( __CUDA_ARCH__ )
  int mask = 0xffffffff;
  for(int i=16; i>0; i=i/2) {
    v0[0]  += __shfl_down_sync(mask, v0[0],  i);
    v1[0]  += __shfl_down_sync(mask, v1[0],  i);
    v2[0]  += __shfl_down_sync(mask, v2[0],  i);
    v3[0]  += __shfl_down_sync(mask, v3[0],  i);
    v4[0]  += __shfl_down_sync(mask, v4[0],  i);
    v5[0]  += __shfl_down_sync(mask, v5[0],  i);
    v6[0]  += __shfl_down_sync(mask, v6[0],  i);
    v7[0]  += __shfl_down_sync(mask, v7[0],  i);
    v8[0]  += __shfl_down_sync(mask, v8[0],  i);
    v9[0]  += __shfl_down_sync(mask, v9[0],  i);
    v10[0] += __shfl_down_sync(mask, v10[0], i);
    v11[0] += __shfl_down_sync(mask, v11[0], i);
  }
  if(team_member.team_rank()%32 == 0) {
    accumulate_current(access, ii, nx, ny, nz, cx, cy, cz, 
                       v0[0], v1[0], v2[0], v3[0], 
                       v4[0], v5[0], v6[0], v7[0], 
                       v8[0], v9[0], v10[0], v11[0]);
  }
#else
  team_member.team_reduce(Kokkos::Sum<float>(v0[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v1[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v2[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v3[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v4[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v5[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v6[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v7[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v8[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v9[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v10[0]));
  team_member.team_reduce(Kokkos::Sum<float>(v11[0]));
  if(team_member.team_rank() == 0) {
    accumulate_current(access, ii, nx, ny, nz, cx, cy, cz, 
                       v0[0], v1[0], v2[0], v3[0], 
                       v4[0], v5[0], v6[0], v7[0], 
                       v8[0], v9[0], v10[0], v11[0]);
  }
#endif
}

template<class TeamMember, class field_sa_t, class field_var>
void KOKKOS_INLINE_FUNCTION
contribute_current(TeamMember& team_member, field_sa_t& access, int i0, int i1, int i2, int i3, field_var j, float v0, float v1,  float v2, float v3) {
#ifdef __CUDA_ARCH__
  int mask = 0xffffffff;
  int team_rank = team_member.team_rank();
  for(int i=16; i>0; i=i/2) {
    v0 += __shfl_down_sync(mask, v0, i);
    v1 += __shfl_down_sync(mask, v1, i);
    v2 += __shfl_down_sync(mask, v2, i);
    v3 += __shfl_down_sync(mask, v3, i);
  }
  if(team_rank%32 == 0) {
    access(i0, j) += v0;
    access(i1, j) += v1;
    access(i2, j) += v2;
    access(i3, j) += v3;
  }
#else
  team_member.team_reduce(Kokkos::Sum<float>(v0));
  team_member.team_reduce(Kokkos::Sum<float>(v1));
  team_member.team_reduce(Kokkos::Sum<float>(v2));
  team_member.team_reduce(Kokkos::Sum<float>(v3));
  if(team_member.team_rank() == 0) {
    access(i0, j) += v0;
    access(i1, j) += v1;
    access(i2, j) += v2;
    access(i3, j) += v3;
  }
#endif
}

// Detect whether all threads/vector lanes are processing particles belonging to the same cell
template<class TeamMember, class IndexView, class BoundsView>
int KOKKOS_INLINE_FUNCTION particles_in_same_cell(TeamMember& team_member, IndexView& ii, BoundsView& inbnds, const int num_lanes) {
#ifdef USE_GPU
  int min_inbnds = inbnds[0];
  int max_inbnds = inbnds[0];
  team_member.team_reduce(Kokkos::Max<int>(min_inbnds));
  team_member.team_reduce(Kokkos::Min<int>(max_inbnds));
  int min_index = ii[0];
  int max_index = ii[0];
  team_member.team_reduce(Kokkos::Max<int>(max_index));
  team_member.team_reduce(Kokkos::Min<int>(min_index));
  return min_inbnds == max_inbnds && min_index == max_index;
#else
  for(int lane=0; lane<num_lanes; lane++) {
    if(ii[0] != ii[lane] || inbnds[0] != inbnds[lane])
      return 0;
  }
  return 1;
#endif
}

// Load the interpolator for cell ii
KOKKOS_INLINE_FUNCTION
void simd_load_interpolator_var(float* v0, const int ii, const k_interpolator_t& k_interp, int len) {
  #pragma omp simd
  for(int i=0; i<len; i++) {
    v0[i] = k_interp(ii, i);
  }
}

// Template for unrolling a loop in reverse order
// Necessary to avoid the looping/pointer overhead when loading interpolator data
template<int N>
KOKKOS_INLINE_FUNCTION
void unrolled_simd_load(float* vals, const int* ii, const k_interpolator_t& k_interp, int len) {
  unrolled_simd_load<N-1>(vals, ii, k_interp, len);
  simd_load_interpolator_var(vals+(N-1)*18, ii[N-1], k_interp, len);
}
template<>
KOKKOS_INLINE_FUNCTION
void unrolled_simd_load<0>(float* vals, const int* ii, const k_interpolator_t& k_interp, int len) {}

// Non forced unrolled version. Potentially less performance than the template version
// This will work with arbitrary number of particles rather than having to use a multiple of the number of simd lanes
void unrolled_simd_load(float* vals, const int* ii, const k_interpolator_t& k_interp, int num_var, int num_part) {
  for(int i=0; i<num_part; i++) {
    simd_load_interpolator_var(vals+i*num_var, ii[i], k_interp, num_var);
  }
}

// Load interpolators
template<int NumLanes>
KOKKOS_INLINE_FUNCTION
void load_interpolators(
                        float* fex,
                        float* fdexdy,
                        float* fdexdz,
                        float* fd2exdydz,
                        float* fey,
                        float* fdeydz,
                        float* fdeydx,
                        float* fd2eydzdx,
                        float* fez,
                        float* fdezdx,
                        float* fdezdy,
                        float* fd2ezdxdy,
                        float* fcbx,
                        float* fdcbxdx,
                        float* fcby,
                        float* fdcbydy,
                        float* fcbz,
                        float* fdcbzdz,
                        const int* ii,
                        const int num_part,
                        const k_interpolator_t& k_interp
                        ) {
#if defined(VPIC_ENABLE_VECTORIZATION) && !defined(USE_GPU)
  int same_cell = 1;
  for(int lane=0; lane<NumLanes; lane++) {
    if(ii[0] != ii[lane]) {
      same_cell = 0;
      break;
    }
  }

  // Try to reduce the number of loads if all particles are in the same cell
  if(same_cell) {
    float vals[18];

    simd_load_interpolator_var(vals, ii[0], k_interp, 18);
    #pragma omp simd
    for(int i=0; i<NumLanes; i++) {
      fex[i]       = vals[0];
      fdexdy[i]    = vals[1];
      fdexdz[i]    = vals[2];
      fd2exdydz[i] = vals[3];
      fey[i]       = vals[4];
      fdeydz[i]    = vals[5];
      fdeydx[i]    = vals[6];
      fd2eydzdx[i] = vals[7];
      fez[i]       = vals[8];
      fdezdx[i]    = vals[9];
      fdezdy[i]    = vals[10];
      fd2ezdxdy[i] = vals[11];
      fcbx[i]      = vals[12];
      fdcbxdx[i]   = vals[13];
      fcby[i]      = vals[14];
      fdcbydy[i]   = vals[15];
      fcbz[i]      = vals[16];
      fdcbzdz[i]   = vals[17];
    }
  } else {

    // Efficient vectorized load
    float vals[18*NumLanes];
    unrolled_simd_load(vals, ii, k_interp, 18, num_part);
//    unrolled_simd_load<NumLanes>(vals, ii, k_interp, 18);

    // Essentially a transpose
    #pragma omp simd
    for(int i=0; i<num_part; i++) {
      fex[i]       = vals[18*i];
      fdexdy[i]    = vals[1+18*i];
      fdexdz[i]    = vals[2+18*i];
      fd2exdydz[i] = vals[3+18*i];
      fey[i]       = vals[4+18*i];
      fdeydz[i]    = vals[5+18*i];
      fdeydx[i]    = vals[6+18*i];
      fd2eydzdx[i] = vals[7+18*i];
      fez[i]       = vals[8+18*i];
      fdezdx[i]    = vals[9+18*i];
      fdezdy[i]    = vals[10+18*i];
      fd2ezdxdy[i] = vals[11+18*i];
      fcbx[i]      = vals[12+18*i];
      fdcbxdx[i]   = vals[13+18*i];
      fcby[i]      = vals[14+18*i];
      fdcbydy[i]   = vals[15+18*i];
      fcbz[i]      = vals[16+18*i];
      fdcbzdz[i]   = vals[17+18*i];
    }
  }
#else
  for(int lane=0; lane<NumLanes; lane++) {
    // Load interpolators
    fex[LANE]       = k_interp(ii[LANE], interpolator_var::ex);     
    fdexdy[LANE]    = k_interp(ii[LANE], interpolator_var::dexdy);  
    fdexdz[LANE]    = k_interp(ii[LANE], interpolator_var::dexdz);  
    fd2exdydz[LANE] = k_interp(ii[LANE], interpolator_var::d2exdydz);
    fey[LANE]       = k_interp(ii[LANE], interpolator_var::ey);     
    fdeydz[LANE]    = k_interp(ii[LANE], interpolator_var::deydz);  
    fdeydx[LANE]    = k_interp(ii[LANE], interpolator_var::deydx);  
    fd2eydzdx[LANE] = k_interp(ii[LANE], interpolator_var::d2eydzdx);
    fez[LANE]       = k_interp(ii[LANE], interpolator_var::ez);     
    fdezdx[LANE]    = k_interp(ii[LANE], interpolator_var::dezdx);  
    fdezdy[LANE]    = k_interp(ii[LANE], interpolator_var::dezdy);  
    fd2ezdxdy[LANE] = k_interp(ii[LANE], interpolator_var::d2ezdxdy);
    fcbx[LANE]      = k_interp(ii[LANE], interpolator_var::cbx);    
    fdcbxdx[LANE]   = k_interp(ii[LANE], interpolator_var::dcbxdx); 
    fcby[LANE]      = k_interp(ii[LANE], interpolator_var::cby);    
    fdcbydy[LANE]   = k_interp(ii[LANE], interpolator_var::dcbydy); 
    fcbz[LANE]      = k_interp(ii[LANE], interpolator_var::cbz);    
    fdcbzdz[LANE]   = k_interp(ii[LANE], interpolator_var::dcbzdz); 
  }
#endif
}

#endif // __ADVANCE_P_HELPERS_HPP
