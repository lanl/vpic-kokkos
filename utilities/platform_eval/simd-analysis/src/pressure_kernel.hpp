#ifndef __PRESSURE_KERNEL__
#define __PRESSURE_KERNEL__

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "simd_helpers.hpp"

template<class Scalar, class ViewType>
void 
pressure_auto(const Scalar cls, const Scalar p_cut, const Scalar pmin, const Scalar eosvmax, 
              const ViewType compression, 
              ViewType bvc, 
              ViewType p_new, 
              const ViewType e_old, 
              const ViewType vnewc ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = p_new.size() / league_size;
  if(per_team*league_size < p_new.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("PRESSURE_BODY1: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        bvc(i) = cls * (compression(i) + 1.0);
      });
    });
  });
  Kokkos::parallel_for("PRESSURE_BODY2: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        p_new(i) = bvc(i) * e_old(i);
        if ( Kokkos::fabs(p_new(i)) < p_cut ) p_new(i) = 0.0;
        if ( vnewc(i) >= eosvmax ) p_new(i) = 0.0;
        if ( p_new(i) < pmin ) p_new(i) = pmin;
      });
    });
  });
}

template<class Scalar, class ViewType>
void 
pressure_guided(const Scalar cls, const Scalar p_cut, const Scalar pmin, const Scalar eosvmax, 
                const ViewType compression, 
                ViewType bvc, 
                ViewType p_new, 
                const ViewType e_old, 
                const ViewType vnewc ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = p_new.size() / league_size;
  if(per_team*league_size < p_new.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("PRESSURE_BODY1: guided", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for_simd(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        bvc(i) = cls * (compression(i) + 1.0);
      });
    });
  });
  Kokkos::parallel_for("PRESSURE_BODY2: guided", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for_simd(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        p_new(i) = bvc(i) * e_old(i);
        if ( Kokkos::fabs(p_new(i)) < p_cut ) p_new(i) = 0.0;
        if ( vnewc(i) >= eosvmax ) p_new(i) = 0.0;
        if ( p_new(i) < pmin ) p_new(i) = pmin;
      });
    });
  });
}

template<class Scalar, class ViewType>
void 
pressure_manual(const Scalar cls, const Scalar p_cut, const Scalar pmin, const Scalar eosvmax, 
                const ViewType compression, 
                ViewType bvc, 
                ViewType p_new, 
                const ViewType e_old, 
                const ViewType vnewc ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = p_new.size() / league_size;
  if(per_team*league_size < p_new.size())
    per_team += 1;
  using DataType = typename ViewType::non_const_value_type;
  using SIMDType = Kokkos::Experimental::simd<DataType>;
  using MaskType = Kokkos::Experimental::simd_mask<DataType>;
  constexpr std::size_t simd_len = SIMDType::size();
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("PRESSURE_BODY1: manual", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      SIMDType bvc_vec, cls_vec(cls), comp_vec, one(1.0);
      for(size_t i=beg; i<end; i+=simd_len) {
        MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
        where(mask, comp_vec).copy_from(compression.data()+i, Kokkos::Experimental::simd_flag_default);
        bvc_vec = cls_vec * (comp_vec + one);
        where(mask, bvc_vec).copy_to(bvc.data()+i, Kokkos::Experimental::simd_flag_default);
      }
    });
  });
  Kokkos::parallel_for("PRESSURE_BODY2: manual", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      SIMDType p_new_v, bvc_v, e_old_v, vnewc_v, p_cut_v(p_cut), pmin_v(pmin), eosvmax_v(eosvmax);
      for(size_t i=beg; i<end; i+=simd_len) {
        MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
        where(mask, bvc_v).copy_from(bvc.data()+i, Kokkos::Experimental::simd_flag_default);
        where(mask, e_old_v).copy_from(e_old.data()+i, Kokkos::Experimental::simd_flag_default);
        where(mask, vnewc_v).copy_from(vnewc.data()+i, Kokkos::Experimental::simd_flag_default);

        p_new_v = bvc_v * e_old_v;
        where(mask && (Kokkos::abs(p_new_v) < p_cut_v), p_new_v) = 0.0;
        where(mask && (vnewc_v >=  eosvmax_v), p_new_v) = 0.0;
        where(mask && (p_new_v <  pmin_v), p_new_v) = pmin;

        where(mask, p_new_v).copy_to(p_new.data()+i, Kokkos::Experimental::simd_flag_default);
      }
    });
  });
}

template<class Scalar, class ViewTypeX, class ViewTypeY>
void pressure_ad_hoc(const Scalar a, const ViewTypeX x, const ViewTypeY y ) {
}

#endif //__PRESSURE_KERNEL__

