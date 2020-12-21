#include "field_buffers.h"

void
checkpt_field_buffers( field_buffers_t * fb ) {

  CHECKPT_VIEW( fb->xyz_sbuf_pos );
  CHECKPT_VIEW( fb->yzx_sbuf_pos );
  CHECKPT_VIEW( fb->zxy_sbuf_pos );
  CHECKPT_VIEW( fb->xyz_rbuf_pos );
  CHECKPT_VIEW( fb->yzx_rbuf_pos );
  CHECKPT_VIEW( fb->zxy_rbuf_pos );
  CHECKPT_VIEW( fb->xyz_sbuf_neg );
  CHECKPT_VIEW( fb->yzx_sbuf_neg );
  CHECKPT_VIEW( fb->zxy_sbuf_neg );
  CHECKPT_VIEW( fb->xyz_rbuf_neg );
  CHECKPT_VIEW( fb->yzx_rbuf_neg );
  CHECKPT_VIEW( fb->zxy_rbuf_neg );
  CHECKPT_VIEW( fb->xyz_sbuf_pos_h );
  CHECKPT_VIEW( fb->yzx_sbuf_pos_h );
  CHECKPT_VIEW( fb->zxy_sbuf_pos_h );
  CHECKPT_VIEW( fb->xyz_rbuf_pos_h );
  CHECKPT_VIEW( fb->yzx_rbuf_pos_h );
  CHECKPT_VIEW( fb->zxy_rbuf_pos_h );
  CHECKPT_VIEW( fb->xyz_sbuf_neg_h );
  CHECKPT_VIEW( fb->yzx_sbuf_neg_h );
  CHECKPT_VIEW( fb->zxy_sbuf_neg_h );
  CHECKPT_VIEW( fb->xyz_rbuf_neg_h );
  CHECKPT_VIEW( fb->yzx_rbuf_neg_h );
  CHECKPT_VIEW( fb->zxy_rbuf_neg_h );

}

field_buffers_t *
restore_field_buffers( void ) {

  field_buffers_t * fb;
  MALLOC(fb, 1);

  RESTORE_VIEW( &fb->xyz_sbuf_pos );
  RESTORE_VIEW( &fb->yzx_sbuf_pos );
  RESTORE_VIEW( &fb->zxy_sbuf_pos );
  RESTORE_VIEW( &fb->xyz_rbuf_pos );
  RESTORE_VIEW( &fb->yzx_rbuf_pos );
  RESTORE_VIEW( &fb->zxy_rbuf_pos );
  RESTORE_VIEW( &fb->xyz_sbuf_neg );
  RESTORE_VIEW( &fb->yzx_sbuf_neg );
  RESTORE_VIEW( &fb->zxy_sbuf_neg );
  RESTORE_VIEW( &fb->xyz_rbuf_neg );
  RESTORE_VIEW( &fb->yzx_rbuf_neg );
  RESTORE_VIEW( &fb->zxy_rbuf_neg );
  RESTORE_VIEW( &fb->xyz_sbuf_pos_h );
  RESTORE_VIEW( &fb->yzx_sbuf_pos_h );
  RESTORE_VIEW( &fb->zxy_sbuf_pos_h );
  RESTORE_VIEW( &fb->xyz_rbuf_pos_h );
  RESTORE_VIEW( &fb->yzx_rbuf_pos_h );
  RESTORE_VIEW( &fb->zxy_rbuf_pos_h );
  RESTORE_VIEW( &fb->xyz_sbuf_neg_h );
  RESTORE_VIEW( &fb->yzx_sbuf_neg_h );
  RESTORE_VIEW( &fb->zxy_sbuf_neg_h );
  RESTORE_VIEW( &fb->xyz_rbuf_neg_h );
  RESTORE_VIEW( &fb->yzx_rbuf_neg_h );
  RESTORE_VIEW( &fb->zxy_rbuf_neg_h );

  return fb;

}

field_buffers_t::field_buffers_t(
  int xyz_size,
  int yzx_size,
  int zxy_size
)
{
  xyz_sbuf_pos = Kokkos::View<float*>("Send buffer for XYZ positive face", xyz_size);
  xyz_rbuf_pos = Kokkos::View<float*>("Receive buffer for XYZ positive face", xyz_size);
  yzx_sbuf_pos = Kokkos::View<float*>("Send buffer for YZX positive face", yzx_size);
  yzx_rbuf_pos = Kokkos::View<float*>("Receive buffer for YZX positive face", yzx_size);
  zxy_sbuf_pos = Kokkos::View<float*>("Send buffer for ZXY positive face", zxy_size);
  zxy_rbuf_pos = Kokkos::View<float*>("Receive buffer for ZXY positive face", zxy_size);

  xyz_sbuf_neg = Kokkos::View<float*>("Send buffer for XYZ negative face", xyz_size);
  xyz_rbuf_neg = Kokkos::View<float*>("Receive buffer for XYZ negative face", xyz_size);
  yzx_sbuf_neg = Kokkos::View<float*>("Send buffer for YZX negative face", yzx_size);
  yzx_rbuf_neg = Kokkos::View<float*>("Receive buffer for YZX negative face", yzx_size);
  zxy_sbuf_neg = Kokkos::View<float*>("Send buffer for ZXY negative face", zxy_size);
  zxy_rbuf_neg = Kokkos::View<float*>("Receive buffer for ZXY negative face", zxy_size);

  xyz_sbuf_pos_h = Kokkos::create_mirror_view(xyz_sbuf_pos);
  yzx_sbuf_pos_h = Kokkos::create_mirror_view(yzx_sbuf_pos);
  zxy_sbuf_pos_h = Kokkos::create_mirror_view(zxy_sbuf_pos);
  xyz_rbuf_pos_h = Kokkos::create_mirror_view(xyz_rbuf_pos);
  yzx_rbuf_pos_h = Kokkos::create_mirror_view(yzx_rbuf_pos);
  zxy_rbuf_pos_h = Kokkos::create_mirror_view(zxy_rbuf_pos);

  xyz_sbuf_neg_h = Kokkos::create_mirror_view(xyz_sbuf_neg);
  yzx_sbuf_neg_h = Kokkos::create_mirror_view(yzx_sbuf_neg);
  zxy_sbuf_neg_h = Kokkos::create_mirror_view(zxy_sbuf_neg);
  xyz_rbuf_neg_h = Kokkos::create_mirror_view(xyz_rbuf_neg);
  yzx_rbuf_neg_h = Kokkos::create_mirror_view(yzx_rbuf_neg);
  zxy_rbuf_neg_h = Kokkos::create_mirror_view(zxy_rbuf_neg);

  REGISTER_OBJECT(this, checkpt_field_buffers, restore_field_buffers, NULL);
}

field_buffers_t::~field_buffers_t() {
  UNREGISTER_OBJECT(this);
}
