#ifndef _borris_h_
#define _borris_h_

void
KOKKOS_INLINE_FUNCTION
borris_advance_e(
    const float qdt_2mc,
    const field_vectors_t& f,
    float& ux,
    float& uy,
    float& uz
)
{

  // Half advance E
  ux  += qdt_2mc*f.ex;
  uy  += qdt_2mc*f.ey;
  uz  += qdt_2mc*f.ez;

}

void
KOKKOS_INLINE_FUNCTION
borris_rotate_b(
    const float qdt_2mc,
    const field_vectors_t& f,
    float& ux,
    float& uy,
    float& uz
)
{

  constexpr float one            = 1.;
  constexpr float one_third      = 1./3.;
  constexpr float two_fifteenths = 2./15.;

  float v0, v1, v2, v3, v4;

  v0   = qdt_2mc/sqrtf(one + (ux*ux + (uy*uy + uz*uz)));

  // Boris - scalars
  v1   = f.cbx*f.cbx + (f.cby*f.cby + f.cbz*f.cbz);
  v2   = (v0*v0)*v1;
  v3   = v0*(one+v2*(one_third+v2*two_fifteenths));
  v4   = v3/(one+v1*(v3*v3));
  v4  += v4;

  // Boris - uprime
  v0   = ux + v3*( uy*f.cbz - uz*f.cby );
  v1   = uy + v3*( uz*f.cbx - ux*f.cbz );
  v2   = uz + v3*( ux*f.cby - uy*f.cbx );

  // Boris - rotation
  ux  += v4*( v1*f.cbz - v2*f.cby );
  uy  += v4*( v2*f.cbx - v0*f.cbz );
  uz  += v4*( v0*f.cby - v1*f.cbx );

}

#endif
