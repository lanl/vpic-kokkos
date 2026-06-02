#ifndef __ANNOTATIONS_HPP__
#define __ANNOTATIONS_HPP__

#include <vector>
#include <algorithm>
#include <string>

struct annotation_vars_t {
  std::vector<std::string> i32_vars;
  std::vector<std::string> i64_vars;
  std::vector<std::string> f32_vars;
  std::vector<std::string> f64_vars;
 
  annotation_vars_t() {
  }

  annotation_vars_t(const annotation_vars_t& a) {
    i32_vars = a.i32_vars;
    i64_vars = a.i64_vars;
    f32_vars = a.f32_vars;
    f64_vars = a.f64_vars;
  }

  template<typename AnnotationType>
  int add_annotation(std::string name) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      i32_vars.push_back(name);
      return i32_vars.size()-1;
    } else if (std::is_same<AnnotationType, int64_t>::value) {
      i64_vars.push_back(name);
      return i64_vars.size()-1;
    } else if (std::is_same<AnnotationType, float>::value) {
      f32_vars.push_back(name);
      return f32_vars.size()-1;
    } else if (std::is_same<AnnotationType, double>::value) {
      f64_vars.push_back(name);
      return f64_vars.size()-1;
    } 
    return -1;
  }

  template<typename AnnotationType>
  int get_annotation_index(const std::string& name) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      for(uint32_t i=0; i<i32_vars.size(); i++) {
        if(name.compare(i32_vars[i]) == 0) {
          return i;
        }
      }
    } else if (std::is_same<AnnotationType, int64_t>::value) {
      for(uint32_t i=0; i<i64_vars.size(); i++) {
        if(name.compare(i64_vars[i]) == 0) {
          return i;
        }
      }
    } else if (std::is_same<AnnotationType, float>::value) {
      for(uint32_t i=0; i<f32_vars.size(); i++) {
        if(name.compare(f32_vars[i]) == 0) {
          return i;
        }
      }
    } else if (std::is_same<AnnotationType, double>::value) {
      for(uint32_t i=0; i<f64_vars.size(); i++) {
        if(name.compare(f64_vars[i]) == 0) {
          return i;
        }
      }
    } 
    return -1;
  }

  void combine(annotation_vars_t& a) {
    for(uint32_t i=0; i<a.i32_vars.size(); i++) {
      auto pos = std::find(i32_vars.begin(), i32_vars.end(), a.i32_vars[i]);
      if(pos == i32_vars.end()) {
        add_annotation<int>(a.i32_vars[i]);
      }
    }
    for(uint32_t i=0; i<a.i64_vars.size(); i++) {
      auto pos = std::find(i64_vars.begin(), i64_vars.end(), a.i64_vars[i]);
      if(pos == i64_vars.end()) {
        add_annotation<int64_t>(a.i64_vars[i]);
      }
    }
    for(uint32_t i=0; i<a.f32_vars.size(); i++) {
      auto pos = std::find(f32_vars.begin(), f32_vars.end(), a.f32_vars[i]);
      if(pos == f32_vars.end()) {
        add_annotation<float>(a.f32_vars[i]);
      }
    }
    for(uint32_t i=0; i<a.f64_vars.size(); i++) {
      auto pos = std::find(f64_vars.begin(), f64_vars.end(), a.f64_vars[i]);
      if(pos == f64_vars.end()) {
        add_annotation<double>(a.f64_vars[i]);
      }
    }
  }

};

template<typename ExecSpace=Kokkos::DefaultExecutionSpace>
class annotations_t {
public:
  using memory_space = typename ExecSpace::memory_space;
  template <class T>
  using AnnotationView = Kokkos::View<T**, Kokkos::LayoutLeft, memory_space>;

  AnnotationView<int>     i32;
  AnnotationView<int64_t> i64;
  AnnotationView<float>   f32;
  AnnotationView<double>  f64;

  annotations_t() {}

  annotations_t(const size_t np, annotation_vars_t vars) {
    i32 = AnnotationView<int>("Int annotations",       np, vars.i32_vars.size());
    i64 = AnnotationView<int64_t>("Int64 annotations", np, vars.i64_vars.size());
    f32 = AnnotationView<float>("Float annotations",   np, vars.f32_vars.size());
    f64 = AnnotationView<double>("Double annotations", np, vars.f64_vars.size());
    Kokkos::deep_copy(i32, 0);
    Kokkos::deep_copy(i64, 0);
    Kokkos::deep_copy(f32, 0.0);
    Kokkos::deep_copy(f64, 0.0);
  }

  annotations_t(annotations_t<Kokkos::DefaultExecutionSpace>& device_annotations) {
    i32 = Kokkos::create_mirror_view(device_annotations.i32);
    i64 = Kokkos::create_mirror_view(device_annotations.i64);
    f32 = Kokkos::create_mirror_view(device_annotations.f32);
    f64 = Kokkos::create_mirror_view(device_annotations.f64);
    Kokkos::deep_copy(i32, 0);
    Kokkos::deep_copy(i64, 0);
    Kokkos::deep_copy(f32, 0.0);
    Kokkos::deep_copy(f64, 0.0);
  }

  template<typename FromExecSpace>
  void copy_from(annotations_t<FromExecSpace>& from) {
    Kokkos::deep_copy(i32, from.i32);
    Kokkos::deep_copy(i64, from.i64);
    Kokkos::deep_copy(f32, from.f32);
    Kokkos::deep_copy(f64, from.f64);
  }

  template<typename FromExecSpace>
  void copy_from(annotations_t<FromExecSpace>& from, const size_t nparticles) {
    auto slice = Kokkos::make_pair(0, nparticles);
    if(i32.extent(1) > 0) {
      auto from_subview = Kokkos::subview(from.i32, slice, Kokkos::ALL);
      auto to_subview   = Kokkos::subview(i32, slice, Kokkos::ALL);
      Kokkos::deep_copy(to_subview, from_subview);
    }
    if(i64.extent(1) > 0) {
      auto from_subview = Kokkos::subview(from.i64, slice, Kokkos::ALL);
      auto to_subview   = Kokkos::subview(i64, slice, Kokkos::ALL);
      Kokkos::deep_copy(to_subview, from_subview);
    }
    if(f32.extent(1) > 0) {
      auto from_subview = Kokkos::subview(from.f32, slice, Kokkos::ALL);
      auto to_subview   = Kokkos::subview(f32, slice, Kokkos::ALL);
      Kokkos::deep_copy(to_subview, from_subview);
    }
    if(f64.extent(1) > 0) {
      auto from_subview = Kokkos::subview(from.f64, slice, Kokkos::ALL);
      auto to_subview   = Kokkos::subview(f64, slice, Kokkos::ALL);
      Kokkos::deep_copy(to_subview, from_subview);
    }
  }

  template<typename AnnotationType>
  KOKKOS_INLINE_FUNCTION
  void set(const size_t particle_index, const int var, AnnotationType val) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      i32(particle_index, var) = val;
    } else if constexpr(std::is_same<AnnotationType, int64_t>::value) {
      i64(particle_index, var) = val;
    } else if constexpr(std::is_same<AnnotationType, float>::value) {
      f32(particle_index, var) = val;
    } else if constexpr(std::is_same<AnnotationType, double>::value) {
      f64(particle_index, var) = val;
    } else {
      printf( "Tried setting value for non existent annotation!\n" );
    }
  }

  template<typename AnnotationType>
  KOKKOS_INLINE_FUNCTION
  AnnotationType get(const size_t particle_index, const int var) {
    if constexpr(std::is_same<AnnotationType, int>::value) {
      return i32(particle_index, var);
    } else if constexpr(std::is_same<AnnotationType, int64_t>::value) {
      return i64(particle_index, var);
    } else if constexpr(std::is_same<AnnotationType, float>::value) {
      return f32(particle_index, var);
    } else if constexpr(std::is_same<AnnotationType, double>::value) {
      return f64(particle_index, var);
    } else {
      printf( "Tried getting value for non existent annotation!\n" );
    }
    return 0;
  }
};


#endif // __ANNOTATIONS_HPP__
