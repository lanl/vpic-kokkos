#ifndef __AXPY_KERNEL__
#define __AXPY_KERNEL__

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "simd_helpers.hpp"

template<class Coef, class ViewTypeX, class ViewTypeY>
void axpy_auto(const Coef a, const ViewTypeX x, const ViewTypeY y ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = y.size() / league_size;
  if(per_team*league_size < y.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("AXPY: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        y(i) += a*x(i);
      });
    });
  });
}

template<class Coef, class ViewTypeX, class ViewTypeY>
void axpy_guided(const Coef a, const ViewTypeX x, const ViewTypeY y ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = y.size() / league_size;
  if(per_team*league_size < y.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("AXPY: guided", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for_simd(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        y(i) += a*x(i);
      });
    });
  });
}

template<class Coef, class ViewTypeX, class ViewTypeY>
void axpy_manual(const Coef a, const ViewTypeX x, const ViewTypeY y ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = y.size() / league_size;
  if(per_team*league_size < y.size())
    per_team += 1;
  using DataType = typename ViewTypeY::non_const_value_type;
  using SIMDType = Kokkos::Experimental::simd<DataType>;
  using MaskType = Kokkos::Experimental::simd_mask<DataType>;
  constexpr std::size_t simd_len = SIMDType::size();
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("AXPY: manual", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      SIMDType y_vec, x_vec, a_vec(a);
      for(size_t i=beg; i<end; i+=simd_len) {
        MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
        where(mask, x_vec).copy_from(x.data()+i, Kokkos::Experimental::simd_flag_default);
        y_vec += a_vec*x_vec;
        where(mask, y_vec).copy_to(y.data()+i, Kokkos::Experimental::simd_flag_default);
      }
    });
  });
}

template<class Coef, class ViewTypeX, class ViewTypeY>
void axpy_ad_hoc(const Coef a, const ViewTypeX x, const ViewTypeY y ) {
}

#endif //__AXPY_KERNEL__
