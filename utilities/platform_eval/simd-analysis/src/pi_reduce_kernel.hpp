#ifndef __PI_REDUCE_KERNEL__
#define __PI_REDUCE_KERNEL__

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "simd_helpers.hpp"

template<class ScalarType>
void 
pi_reduce_auto( ScalarType& pi, const ScalarType dx, const uint64_t len ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = len / league_size;
  if(per_team*league_size < len)
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_reduce("PI_REDUCE: auto", policy, 
  KOKKOS_LAMBDA(member_type team_member, ScalarType& total_sum) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    ScalarType sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx, double& thread_sum) {
      ScalarType simd_sum = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i, double& vec_sum) {
        double x = (double(i) + 0.5) * dx;
        vec_sum += dx / (1.0 + x * x);
      }, simd_sum);
      thread_sum += simd_sum;
    }, sum);
    total_sum += sum;
  }, pi);
}

template<class ScalarType>
void 
pi_reduce_guided(ScalarType pi, const ScalarType dx, const uint64_t len) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = len / league_size;
  if(per_team*league_size < len)
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_reduce("PI_REDUCE: guided", policy, 
  KOKKOS_LAMBDA(member_type team_member, ScalarType& total_sum) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    ScalarType sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx, double& thread_sum) {
      ScalarType simd_sum = 0.0;
      Kokkos::parallel_reduce_simd_sum(Kokkos::ThreadVectorRange(team_member, beg, end), 
      [&] (const uint64_t i, double& vec_sum) {
        double x = (double(i) + 0.5) * dx;
        vec_sum += dx / (1.0 + x * x);
      }, simd_sum);
      thread_sum += simd_sum;
    }, sum);
    total_sum += sum;
  }, pi);
}

template<class ScalarType>
void 
pi_reduce_manual(ScalarType pi, const ScalarType dx, const uint64_t len) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = len / league_size;
  if(per_team*league_size < len)
    per_team += 1;
  using SIMDType = Kokkos::Experimental::simd<ScalarType>;
  using MaskType = Kokkos::Experimental::simd_mask<ScalarType>;
  constexpr std::size_t simd_len = SIMDType::size();
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_reduce("PI_REDUCE: manual", policy, 
  KOKKOS_LAMBDA(member_type team_member, ScalarType& total_sum) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    ScalarType sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, 1),
    [&] (const uint64_t idx, double& thread_sum) {
      SIMDType x, sum_v(0.0), half(0.5), one(1.0), dx_v(dx), coef(1.0), temp;
      SIMDType i_v([beg] (std::size_t lane) { return static_cast<double>(beg+lane); });
      for(size_t i=beg; i<end; i+=simd_len) {
        MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
        x = (i_v + half) * dx_v;
        temp = dx_v / (one + x * x);
        where(mask, temp) = 0.0;
        sum_v += temp;
      }
      thread_sum += Kokkos::Experimental::reduce(sum_v, MaskType(true), 0.0);
    }, sum);
    total_sum += sum;
  }, pi);
}

template<class ScalarType>
void 
pi_reduce_ad_hoc(ScalarType pi, const ScalarType dx, const uint64_t len) {}
#endif //__AXPY_KERNEL__


