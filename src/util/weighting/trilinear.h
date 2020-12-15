#ifndef _trilinear_h_
#define _trilinear_h_

#include "../../grid/grid.h"

class TrilinearWeighting {
public:

    TrilinearWeighting(
        const int nx, const int ny, const int nz,
        const int sx, const int sy, const int sz
    ) : nx(nx), ny(ny), nz(nz),
        sx(sx), sy(sy), sz(sz)
    {

    }

    void KOKKOS_INLINE_FUNCTION
    set_position(float dx, float dy, float dz) {

        constexpr float one_eigth = 1./8.;

        dx *= one_eigth;
        weights[0] = one_eigth * (1 - dx);
        weights[1] = one_eigth * (1 + dx);

        weights[3] = 1 + dy;
        weights[2] = weights[0]*weights[3];
        weights[3] = weights[1]*weights[3];

        dy = 1 - dy;
        weights[0] = weights[0]*dy;
        weights[1] = weights[1]*dy;

        weights[7] = 1 + dz;
        weights[4] = weights[0]*weights[7];     // (1-x)(1-y)(1+z)/8
        weights[5] = weights[1]*weights[7];     // (1-x)(1-y)(1+z)/8
        weights[6] = weights[2]*weights[7];     // (1-x)(1-y)(1+z)/8
        weights[7] = weights[3]*weights[7];     // (1-x)(1-y)(1+z)/8

        dz = 1 - dz;
        weights[0] = weights[0]*dz;             // (1-x)(1-y)(1-z)/8
        weights[1] = weights[1]*dz;             // (1+x)(1-y)(1-z)/8
        weights[2] = weights[2]*dz;             // (1-x)(1+y)(1-z)/8
        weights[3] = weights[3]*dz;             // (1-x)(1+y)(1-z)/8

    }

    void KOKKOS_INLINE_FUNCTION
    synchronize_weights(const int voxel) {

        int x, y, z;
        x  = voxel;
        z  = x/sz;
        x -= z*sz;
        y  = x/sy;
        x -= y*sy;

        if(z == 1) {
            weights[0] += weights[0];
            weights[1] += weights[1];
            weights[2] += weights[2];
            weights[3] += weights[3];
        }
        if(z == nz) {
            weights[4] += weights[4];
            weights[5] += weights[5];
            weights[6] += weights[6];
            weights[7] += weights[7];
        }
        if(y == 1) {
            weights[0] += weights[0];
            weights[1] += weights[1];
            weights[4] += weights[4];
            weights[5] += weights[5];
        }
        if(y == ny) {
            weights[2] += weights[2];
            weights[3] += weights[3];
            weights[6] += weights[6];
            weights[7] += weights[7];
        }
        if(x == 1) {
            weights[0] += weights[0];
            weights[2] += weights[2];
            weights[4] += weights[4];
            weights[6] += weights[6];
        }
        if(x == nx) {
            weights[1] += weights[1];
            weights[3] += weights[3];
            weights[5] += weights[5];
            weights[7] += weights[7];
        }

    }

    template<class view_type_t>
    void KOKKOS_INLINE_FUNCTION
    deposit(
        view_type_t& view,
        int          voxel,
        int          var,
        const float  value
    )
    {
        view(voxel,         var) += value*weights[0];
        view(voxel+1,       var) += value*weights[1];
        view(voxel+sy,      var) += value*weights[2];
        view(voxel+sy+1,    var) += value*weights[3];
        view(voxel+sz,      var) += value*weights[4];
        view(voxel+sz+1,    var) += value*weights[5];
        view(voxel+sz+sy,   var) += value*weights[6];
        view(voxel+sz+sy+1, var) += value*weights[7];
    }


    template<class view_type_t>
    void KOKKOS_INLINE_FUNCTION
    deposit(
        view_type_t& view,
        int          voxel,
        const float  value
    )
    {
        view(voxel        ) += value*weights[0];
        view(voxel+1      ) += value*weights[1];
        view(voxel+sy     ) += value*weights[2];
        view(voxel+sy+1   ) += value*weights[3];
        view(voxel+sz     ) += value*weights[4];
        view(voxel+sz+1   ) += value*weights[5];
        view(voxel+sz+sy  ) += value*weights[6];
        view(voxel+sz+sy+1) += value*weights[7];
    }

    template<class type>
    void
    deposit_aos(
        type*       array,
        int         voxel,
        int         offset,
        const float value
    )
    {
        // In principle this is not aligned.
        char * f = (char *)array + offset;

        #define view(i) *((float *)(f + (i)*sizeof(type)))

        view(voxel        ) += value*weights[0];
        view(voxel+1      ) += value*weights[1];
        view(voxel+sy     ) += value*weights[2];
        view(voxel+sy+1   ) += value*weights[3];
        view(voxel+sz     ) += value*weights[4];
        view(voxel+sz+1   ) += value*weights[5];
        view(voxel+sz+sy  ) += value*weights[6];
        view(voxel+sz+sy+1) += value*weights[7];

        #undef view

    }

private:

    const int nx, ny, nz;
    const int sx, sy, sz;
    float weights[8];

};

#endif
