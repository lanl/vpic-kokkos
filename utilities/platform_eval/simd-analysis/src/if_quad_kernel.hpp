#ifndef __IF_QUAD_KERNEL__
#define __IF_QUAD_KERNEL__

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "simd_helpers.hpp"

template<class ViewType>
void if_quad_auto(const ViewType a, const ViewType b, const ViewType c,
                  ViewType x1, ViewType x2 ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = a.size() / league_size;
  if(per_team*league_size < a.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("IF_QUAD: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        double s = b(i)*b(i) - 4.0*a(i)*c(i);
        if( s >= 0 ) {
          s = Kokkos::sqrt(s);
          x2(i) = (-b(i)+s)/(2.0*a(i));
          x1(i) = (-b(i)-s)/(2.0*a(i));
        } else {
          x2(i) = 0.0;
          x1(i) = 0.0;
        }
      });
    });
  });
}

template<class ViewType>
void 
if_quad_guided(const ViewType a, const ViewType b, const ViewType c,
                  ViewType x1, ViewType x2 ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = a.size() / league_size;
  if(per_team*league_size < a.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("IF_QUAD: guided", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      Kokkos::parallel_for_simd(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i) {
        double s = b(i)*b(i) - 4.0*a(i)*c(i);
        if( s >= 0 ) {
          s = Kokkos::sqrt(s);
          x2(i) = (-b(i)+s)/(2.0*a(i));
          x1(i) = (-b(i)-s)/(2.0*a(i));
        } else {
          x2(i) = 0.0;
          x1(i) = 0.0;
        }
      });
    });
  });
}

template<class ViewType>
void if_quad_manual(const ViewType a, const ViewType b, const ViewType c,
                  ViewType x1, ViewType x2 ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = a.size() / league_size;
  if(per_team*league_size < a.size())
    per_team += 1;
  using DataType = typename ViewType::non_const_value_type;
  using SIMDType = Kokkos::Experimental::simd<DataType>;
  using MaskType = Kokkos::Experimental::simd_mask<DataType>;
  constexpr std::size_t simd_len = SIMDType::size();
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("IF_QUAD: manual", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx) {
      SIMDType s_v, a_v, b_v, c_v, two_v(2.0), four_v(4.0), zero(0.0), x1_v, x2_v;
      for(size_t i=beg; i<end; i+=simd_len) {
        MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
        where(mask, a_v).copy_from(a.data()+i, Kokkos::Experimental::simd_flag_default);
        where(mask, b_v).copy_from(b.data()+i, Kokkos::Experimental::simd_flag_default);
        where(mask, c_v).copy_from(c.data()+i, Kokkos::Experimental::simd_flag_default);
        x1_v = 0.0;
        x2_v = 0.0;

        s_v = b_v*b_v - four_v*a_v*c_v;
        where(s_v >= zero, x2_v) = (-b_v+s_v)/(two_v*a_v);
        where(s_v >= zero, x1_v) = (-b_v-s_v)/(two_v*a_v);

        where(mask, x1_v).copy_to(x1.data()+i, Kokkos::Experimental::simd_flag_default);
        where(mask, x2_v).copy_to(x2.data()+i, Kokkos::Experimental::simd_flag_default);
      }
    });
  });
}

template<class ViewType>
void if_quad_ad_hoc(const ViewType a, const ViewType b, const ViewType c,
                  ViewType x1, ViewType x2 ) {
}

#endif //__IF_QUAD_KERNEL__

