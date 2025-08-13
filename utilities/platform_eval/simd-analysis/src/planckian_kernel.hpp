#ifndef __PLANCKIAN_KERNEL__
#define __PLANCKIAN_KERNEL__

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "simd_helpers.hpp"

template<class ViewType>
void 
planckian_auto(const ViewType x, ViewType y, const ViewType u, const ViewType v, ViewType w ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = y.size() / league_size;
  if(per_team*league_size < y.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("PLANCKIAN: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        y(i) = u(i) / v(i);
        w(i) = x(i) / (Kokkos::exp(y(i)) - 1.0);
      });
    });
  });
}

template<class ViewType>
void 
planckian_guided(const ViewType x, ViewType y, const ViewType u, const ViewType v, ViewType w ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = y.size() / league_size;
  if(per_team*league_size < y.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("PLANCKIAN: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for_simd(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        y(i) = u(i) / v(i);
        w(i) = x(i) / (Kokkos::exp(y(i)) - 1.0);
      });
    });
  });
}

template<class ViewType>
void 
planckian_manual(const ViewType x, ViewType y, const ViewType u, const ViewType v, ViewType w ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = y.size() / league_size;
  if(per_team*league_size < y.size())
    per_team += 1;
  using DataType = typename ViewType::non_const_value_type;
  using SIMDType = Kokkos::Experimental::simd<DataType>;
  using MaskType = Kokkos::Experimental::simd_mask<DataType>;
  constexpr std::size_t simd_len = SIMDType::size();
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("PLANCKIAN: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      SIMDType x_vec, y_vec, u_vec, v_vec, w_vec;
      for(size_t i=beg; i<end; i+=simd_len) {
        MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
        where(mask, u_vec).copy_from(u.data()+i, Kokkos::Experimental::simd_flag_default);
        where(mask, v_vec).copy_from(v.data()+i, Kokkos::Experimental::simd_flag_default);
        y_vec = u_vec / v_vec;
        where(mask, x_vec).copy_from(x.data()+i, Kokkos::Experimental::simd_flag_default);
        w_vec = x_vec / (Kokkos::exp(y_vec) - 1.0);
        where(mask, w_vec).copy_to(w.data()+i, Kokkos::Experimental::simd_flag_default);
      }
    });
  });
}

template<class ViewType>
void 
planckian_ad_hoc(const ViewType x, ViewType y, const ViewType u, const ViewType v, ViewType w ) {}
#endif //__AXPY_KERNEL__
