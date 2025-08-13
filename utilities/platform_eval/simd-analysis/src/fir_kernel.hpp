#ifndef __FIR_KERNEL__
#define __FIR_KERNEL__

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "simd_helpers.hpp"

template<class ViewType>
void 
fir_auto(const ViewType in, ViewType out ) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = out.size() / league_size;
  if(per_team*league_size < out.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("FIR: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    double coeff_array[16] = { 3.0, -1.0, -1.0, -1.0, \
                              -1.0,  3.0, -1.0, -1.0, \
                              -1.0, -1.0,  3.0, -1.0, \
                              -1.0, -1.0, -1.0,  3.0 };
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, beg, end),
    [&] (const uint64_t i) {
      double res = 0.0;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, 16), 
      [&] (const uint64_t j, double& sum) {
        sum += coeff_array[j] * in(i+j);
      }, res);
      out(i) = res;
    });
  });
}

template<class ViewType>
void 
fir_guided(const ViewType in, ViewType out) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = out.size() / league_size;
  if(per_team*league_size < out.size())
    per_team += 1;
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("FIR: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    double coeff_array[16] = { 3.0, -1.0, -1.0, -1.0, \
                              -1.0,  3.0, -1.0, -1.0, \
                              -1.0, -1.0,  3.0, -1.0, \
                              -1.0, -1.0, -1.0,  3.0 };
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, beg, end),
    [&] (const uint64_t i) {
      double res = 0.0;
      Kokkos::parallel_reduce_simd_sum(Kokkos::ThreadVectorRange(team_member, 16), 
      [&] (const uint64_t j, double& sum) {
        sum += coeff_array[j] * in(i+j);
      }, res);
      out(i) = res;
    });
  });
}

template<class ViewType>
void 
fir_manual(const ViewType in, ViewType out) {
  const uint64_t league_size = Kokkos::num_threads();
  uint64_t per_team = out.size() / league_size;
  if(per_team*league_size < out.size())
    per_team += 1;
  using DataType = typename ViewType::non_const_value_type;
  using SIMDType = Kokkos::Experimental::simd<DataType>;
  using MaskType = Kokkos::Experimental::simd_mask<DataType>;
  constexpr std::size_t simd_len = SIMDType::size();
  
  Kokkos::TeamPolicy<> policy(league_size, 1);
  using member_type = Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for("FIR: auto", policy, KOKKOS_LAMBDA(member_type team_member) {
    const uint64_t beg = team_member.league_rank()*per_team;
    const uint64_t end = (team_member.league_rank()+1) * per_team;
    double coeff_array[16] = { 3.0, -1.0, -1.0, -1.0, \
                              -1.0,  3.0, -1.0, -1.0, \
                              -1.0, -1.0,  3.0, -1.0, \
                              -1.0, -1.0, -1.0,  3.0 };
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, beg, end),
    [&] (const uint64_t i) {
      double res = 0.0;
      SIMDType coef_vec, in_vec, sum_vec(0.0);
      MaskType mask([i, end] (std::size_t lane) { return i+int(lane) < end; });
      for(int j=0; j<16; j+=simd_len) {
        coef_vec.copy_from(coeff_array+j, Kokkos::Experimental::simd_flag_aligned);
        in_vec.copy_from(in.data()+i+j, Kokkos::Experimental::simd_flag_aligned);
        sum_vec += in_vec * coef_vec;
      }
      out(i) = Kokkos::Experimental::reduce(sum_vec, std::plus<>());
    });
  });
}

template<class ViewType>
void 
fir_ad_hoc(const ViewType in, ViewType out) {}
#endif //__AXPY_KERNEL__

