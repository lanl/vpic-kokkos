#ifndef _field_buffers_h_
#define _field_buffers_h_

#include "../vpic/kokkos_helpers.h"

class field_buffers_t {
public:

    field_buffers_t(int xyz_size, int yzx_size, int zxy_size);
    ~field_buffers_t();

    Kokkos::View<float*>   xyz_sbuf_pos;
    Kokkos::View<float*>   yzx_sbuf_pos;
    Kokkos::View<float*>   zxy_sbuf_pos;
    Kokkos::View<float*>   xyz_rbuf_pos;
    Kokkos::View<float*>   yzx_rbuf_pos;
    Kokkos::View<float*>   zxy_rbuf_pos;
    Kokkos::View<float*>   xyz_sbuf_neg;
    Kokkos::View<float*>   yzx_sbuf_neg;
    Kokkos::View<float*>   zxy_sbuf_neg;
    Kokkos::View<float*>   xyz_rbuf_neg;
    Kokkos::View<float*>   yzx_rbuf_neg;
    Kokkos::View<float*>   zxy_rbuf_neg;

    Kokkos::View<float*>::HostMirror   xyz_sbuf_pos_h;
    Kokkos::View<float*>::HostMirror   yzx_sbuf_pos_h;
    Kokkos::View<float*>::HostMirror   zxy_sbuf_pos_h;
    Kokkos::View<float*>::HostMirror   xyz_rbuf_pos_h;
    Kokkos::View<float*>::HostMirror   yzx_rbuf_pos_h;
    Kokkos::View<float*>::HostMirror   zxy_rbuf_pos_h;
    Kokkos::View<float*>::HostMirror   xyz_sbuf_neg_h;
    Kokkos::View<float*>::HostMirror   yzx_sbuf_neg_h;
    Kokkos::View<float*>::HostMirror   zxy_sbuf_neg_h;
    Kokkos::View<float*>::HostMirror   xyz_rbuf_neg_h;
    Kokkos::View<float*>::HostMirror   yzx_rbuf_neg_h;
    Kokkos::View<float*>::HostMirror   zxy_rbuf_neg_h;

};

#endif
