#ifndef __KERNELS_HPP__
#define __KERNELS_HPP__

// Helper functions for indexing
KOKKOS_INLINE_FUNCTION
uint64_t 
map3Dto1D(const uint64_t x,  const uint64_t y,  const uint64_t z,
          const uint64_t nx, const uint64_t ny, const uint64_t nz, const uint64_t rad) {
  return (z*(nx+(2*rad))*(ny+(2*rad))) + (y*(nx+(2*rad))) + x;
}

KOKKOS_INLINE_FUNCTION
void 
map1Dto3D(const uint64_t v, uint64_t& x,  uint64_t& y,  uint64_t& z,
          const uint64_t nx, const uint64_t ny, const uint64_t nz, const uint64_t rad) {
  uint64_t voxel = v;
  z = voxel / ((nx+(2*rad))*(ny+(2*rad)));
  voxel -= (z*(nx+(2*rad))*(ny+(2*rad)));
  y = voxel / (nx+(2*rad));
  x = voxel - y*(nx+(2*rad));
  return;
}


template<typename KeyType, typename SrcValType, typename DstValType>
void 
gather_stencil_kernel(SrcValType src, 
                      DstValType dst, 
                      KeyType keys, 
                      const int nx,
                      const int ny,
                      const int nz,
                      const int size,
                      const int rad) {
  if(nx == 1 && ny == 1 && nz == 1) { // Normal gather
    Kokkos::parallel_for("Gather stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      for(int i=0; i<size; i++) 
        dst(idx) += src(keys(idx),i);
    });
  } else if (nx != 1 && ny == 1 && nz == 1) { // Rank 1
    Kokkos::parallel_for("Gather stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      typename DstValType::non_const_value_type sum = 0;
      for(int i=0; i<size; i++) {
        for(int j=-rad; j<=rad; j++) {
          sum += src(keys(idx)+j, i);
        }
      }
      dst(idx) = sum;
    });
  } else if (nx != 1 && ny != 1 && nz == 1) { // Rank 2
    Kokkos::parallel_for("Gather stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      uint64_t x, y, z;
      map1Dto3D(keys(idx), x, y, z, nx, ny, nz, rad);
      typename DstValType::non_const_value_type sum = 0;
      for(uint64_t i=0; i<size; i++) {
        sum += src(map3Dto1D(x,y,z,nx,ny,nz,rad), i);
        for(uint64_t j=1; j<=rad; j++) {
          // Sum stencil along x-axis
          sum += src(map3Dto1D(x-j, y,   z, nx,ny,nz, rad), i); 
          sum += src(map3Dto1D(x+j, y,   z, nx,ny,nz, rad), i); 
          // Sum stencil along y-axis
          sum += src(map3Dto1D(x,   y-j, z, nx,ny,nz, rad), i);
          sum += src(map3Dto1D(x,   y+j, z, nx,ny,nz, rad), i);
        }
      }
      dst(idx) = sum;
    });
  } else if (nx != 1 && ny != 1 && nz != 1) { // Rank 3
    Kokkos::parallel_for("Gather stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      uint64_t x, y, z;
      map1Dto3D(keys(idx), x, y, z, nx, ny, nz, rad);
      typename DstValType::non_const_value_type sum = 0;
      for(uint64_t i=0; i<size; i++) {
        sum += src(map3Dto1D(x,y,z,nx,ny,nz,rad), i);
        for(uint64_t j=1; j<=rad; j++) {
          // Sum stencil along x-axis
          sum += src(map3Dto1D(x-j, y,   z,   nx,ny,nz, rad), i); 
          sum += src(map3Dto1D(x+j, y,   z,   nx,ny,nz, rad), i); 
          // Sum stencil along y-axis
          sum += src(map3Dto1D(x,   y-j, z,   nx,ny,nz, rad), i);
          sum += src(map3Dto1D(x,   y+j, z,   nx,ny,nz, rad), i);
          // Sum stencil along z-axis
          sum += src(map3Dto1D(x,   y,   z-j, nx,ny,nz, rad), i);
          sum += src(map3Dto1D(x,   y,   z+j, nx,ny,nz, rad), i);
        }
      }
      dst(idx) = sum;
    });
  }
}

template<typename KeyType, typename SrcValType, typename DstValType>
void 
scatter_stencil_kernel(SrcValType src, 
                       DstValType dst, 
                       KeyType keys, 
                       const uint64_t nx,
                       const uint64_t ny,
                       const uint64_t nz,
                       const uint64_t size,
                       const uint64_t rad) {
  if(nx == 1 && ny == 1 && nz == 1) { // Normal gather
    Kokkos::parallel_for("Scatter stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      for(int i=0; i<size; i++) 
        Kokkos::atomic_add(&dst(keys(idx), i), src(idx));
    });
  } else if (nx != 1 && ny == 1 && nz == 1) { // Rank 1
    Kokkos::parallel_for("Scatter stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      for(int i=0; i<size; i++) {
        for(int j=-rad; j<=rad; j++) {
          Kokkos::atomic_add(&dst(keys(idx)+j, i), src(idx));
        }
      }
    });
  } else if (nx != 1 && ny != 1 && nz == 1) { // Rank 2
    Kokkos::parallel_for("Scatter stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      uint64_t x, y, z;
      map1Dto3D(keys(idx), x, y, z, nx, ny, nz, rad);
      for(uint64_t i=0; i<size; i++) {
        Kokkos::atomic_add(&dst(map3Dto1D(x,y,z,nx,ny,nz,rad), i), src(idx));
        for(uint64_t j=1; j<=rad; j++) {
          // Sum stencil along x-axis
          Kokkos::atomic_add(&dst(map3Dto1D(x-j, y,   z, nx,ny,nz, rad), i), src(idx)); 
          Kokkos::atomic_add(&dst(map3Dto1D(x+j, y,   z, nx,ny,nz, rad), i), src(idx)); 
          // Sum stencil along y-axis
          Kokkos::atomic_add(&dst(map3Dto1D(x,   y-j, z, nx,ny,nz, rad), i), src(idx));
          Kokkos::atomic_add(&dst(map3Dto1D(x,   y+j, z, nx,ny,nz, rad), i), src(idx));
        }
      }
    });
  } else if (nx != 1 && ny != 1 && nz != 1) { // Rank 3
    Kokkos::parallel_for("Scatter stencil kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
      KOKKOS_LAMBDA(const uint64_t idx) {
      uint64_t x, y, z;
      map1Dto3D(keys(idx), x, y, z, nx, ny, nz, rad);
      for(uint64_t i=0; i<size; i++) {
        Kokkos::atomic_add(&dst(map3Dto1D(x,y,z,nx,ny,nz,rad), i), src(idx));
        for(uint64_t j=1; j<=rad; j++) {
          // Sum stencil along x-axis
          Kokkos::atomic_add(&dst(map3Dto1D(x-j, y,   z,   nx,ny,nz, rad), i), src(idx)); 
          Kokkos::atomic_add(&dst(map3Dto1D(x+j, y,   z,   nx,ny,nz, rad), i), src(idx)); 
          // Sum stencil along y-axis
          Kokkos::atomic_add(&dst(map3Dto1D(x,   y-j, z,   nx,ny,nz, rad), i), src(idx));
          Kokkos::atomic_add(&dst(map3Dto1D(x,   y+j, z,   nx,ny,nz, rad), i), src(idx));
          // Sum stencil along z-axis
          Kokkos::atomic_add(&dst(map3Dto1D(x,   y,   z-j, nx,ny,nz, rad), i), src(idx));
          Kokkos::atomic_add(&dst(map3Dto1D(x,   y,   z+j, nx,ny,nz, rad), i), src(idx));
        }
      }
    });
  }
}

template<typename KeyType, typename ValType>
void 
gather_kernel(ValType src, 
              ValType dst, 
              KeyType keys, 
              const int size) {
  Kokkos::parallel_for("Gather kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
    KOKKOS_LAMBDA(const uint64_t idx) {
    auto key = keys(idx);
    for(int i=0; i<size; i++) {
      dst(idx, i) = src(key, i);
    }
  });
}

template<typename KeyType, typename ValType>
void 
scatter_kernel(ValType src, 
               ValType dst, 
               KeyType keys, 
               const int size) {
  Kokkos::parallel_for("Scatter kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
    KOKKOS_LAMBDA(const uint64_t idx) {
    auto key = keys(idx);
    // size writes
    for(int i=0; i<size; i++) {
      dst(key, i) = src(idx, i);
    }
  });
}

template<typename KeyType, typename ValType>
void 
scatter_atomic_kernel(ValType src, 
                      ValType dst, 
                      KeyType keys, 
                      const int size) {
  Kokkos::parallel_for("Scatter kernel", Kokkos::RangePolicy<uint64_t>(0, keys.extent(0)), 
    KOKKOS_LAMBDA(const uint64_t idx) {
    auto key = keys(idx);
    // size writes
    for(int i=0; i<size; i++) {
      Kokkos::atomic_add(&dst(key,i), src(idx,i));
    }
  });
}

#endif
