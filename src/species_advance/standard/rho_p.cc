/*
 * Written by:
 *   Kevin J. Bowers, Ph.D.
 *   Plasma Physics Group (X-1)
 *   Applied Physics Division
 *   Los Alamos National Lab
 * March/April 2004 - Revised and extended from earlier V4PIC versions
 *
 */

#define IN_spa
#include "spa_private.h"
#include "Kokkos_Core.hpp"


#include "../../vpic/kokkos_helpers.h"

// accumulate_rho_p adds the charge density associated with the
// supplied particle array to the rhof of the fields.  Trilinear
// interpolation is used.  rhof is known at the nodes at the same time
// as particle positions.  No effort is made to fix up edges of the
// computational domain; see note in synchronize_rhob about why this
// is done this way.  All particles on the list must be inbounds.

void
accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
        const species_t     * RESTRICT sp ) {
    if( !fa || !sp || fa->g!=sp->g ) ERROR(( "Bad args" ));

    /**/  field_t    * RESTRICT ALIGNED(128) f = fa->f;
    const particle_t * RESTRICT ALIGNED(128) p = sp->p;

    const float q_8V = sp->q*sp->g->r8V;
    const int np = sp->np;
    const int sy = sp->g->sy;
    const int sz = sp->g->sz;

# if 1
    float w0, w1, w2, w3, w4, w5, w6, w7, dz;
# else
    using namespace v4;
    v4float q, wl, wh, rl, rh;
# endif

    int n, v;

    // Load the grid data
    for( n=0; n<np; n++ ) {

#   if 1
        // After detailed experiments and studying of assembly dumps, it was
        // determined that if the platform does not support efficient 4-vector
        // SIMD memory gather/scatter operations, the savings from using
        // "trilinear" are slightly outweighed by the overhead of the
        // gather/scatters.

        // Load the particle data

        w0 = p[n].dx;
        w1 = p[n].dy;
        dz = p[n].dz;
        v  = p[n].i;
        w7 = p[n].w*q_8V;

        // Compute the trilinear weights
        // Though the PPE should have hardware fma/fmaf support, it was
        // measured to be more efficient _not_ to use it here.  (Maybe the
        // compiler isn't actually generating the assembly for it.

#   define FMA( x,y,z) ((z)+(x)*(y))
#   define FNMS(x,y,z) ((z)-(x)*(y))
        w6=FNMS(w0,w7,w7);                    // q(1-dx)
        w7=FMA( w0,w7,w7);                    // q(1+dx)
        w4=FNMS(w1,w6,w6); w5=FNMS(w1,w7,w7); // q(1-dx)(1-dy), q(1+dx)(1-dy)
        w6=FMA( w1,w6,w6); w7=FMA( w1,w7,w7); // q(1-dx)(1+dy), q(1+dx)(1+dy)
        w0=FNMS(dz,w4,w4); w1=FNMS(dz,w5,w5); w2=FNMS(dz,w6,w6); w3=FNMS(dz,w7,w7);
        w4=FMA( dz,w4,w4); w5=FMA( dz,w5,w5); w6=FMA( dz,w6,w6); w7=FMA( dz,w7,w7);
#   undef FNMS
#   undef FMA

        // Reduce the particle charge to rhof

        f[v      ].rhof += w0; f[v      +1].rhof += w1;
        f[v   +sy].rhof += w2; f[v   +sy+1].rhof += w3;
        f[v+sz   ].rhof += w4; f[v+sz   +1].rhof += w5;
        f[v+sz+sy].rhof += w6; f[v+sz+sy+1].rhof += w7;

#   else

        // Gather rhof for this voxel

        v = p[n].i;
        rl = v4float( f[v      ].rhof, f[v      +1].rhof,
                f[v   +sy].rhof, f[v   +sy+1].rhof);
        rh = v4float( f[v+sz   ].rhof, f[v+sz   +1].rhof,
                f[v+sz+sy].rhof, f[v+sz+sy+1].rhof);

        // Compute the trilinear weights

        load_4x1( &p[n].dx, wl );
        trilinear( wl, wh );

        // Reduce the particle charge to rhof and scatter the result

        q = v4float( p[n].w*q_8V );
        store_4x1_tr( fma(q,wl,rl), &f[v      ].rhof, &f[v      +1].rhof,
                &f[v   +sy].rhof, &f[v   +sy+1].rhof );
        store_4x1_tr( fma(q,wh,rh), &f[v+sz   ].rhof, &f[v+sz   +1].rhof,
                &f[v+sz+sy].rhof, &f[v+sz+sy+1].rhof );

#   endif

    }
}

#if 0
using namespace v4;
// Note: If part of the body of accumulate_rhob, under the hood
// there is a check for initialization that occurs everytime
// accumulate_rhob is called!
static const v4float ax[4] = { v4float(1,1,1,1), v4float(2,1,2,1),
    v4float(1,2,1,2), v4float(2,2,2,2) };
static const v4float ay[4] = { v4float(1,1,1,1), v4float(2,2,1,1),
    v4float(1,1,2,2), v4float(2,2,2,2) };
#endif

void
accumulate_rhob( field_t          * RESTRICT ALIGNED(128) f,
        const particle_t * RESTRICT ALIGNED(32)  p,
        const grid_t     * RESTRICT              g,
        const float                              qsp ) {
# if 1

    // See note in rhof for why this variant is used.
    float w0 = p->dx, w1 = p->dy, w2, w3, w4, w5, w6, w7, dz = p->dz;
    int v = p->i, x, y, z, sy = g->sy, sz = g->sz;
    w7 = (qsp*g->r8V)*p->w;

    // Compute the trilinear weights
    // See note in rhof for why FMA and FNMS are done this way.

# define FMA( x,y,z) ((z)+(x)*(y))
# define FNMS(x,y,z) ((z)-(x)*(y))
    w6=FNMS(w0,w7,w7);                    // q(1-dx)
    w7=FMA( w0,w7,w7);                    // q(1+dx)
    w4=FNMS(w1,w6,w6); w5=FNMS(w1,w7,w7); // q(1-dx)(1-dy), q(1+dx)(1-dy)
    w6=FMA( w1,w6,w6); w7=FMA( w1,w7,w7); // q(1-dx)(1+dy), q(1+dx)(1+dy)
    w0=FNMS(dz,w4,w4); w1=FNMS(dz,w5,w5); w2=FNMS(dz,w6,w6); w3=FNMS(dz,w7,w7);
    w4=FMA( dz,w4,w4); w5=FMA( dz,w5,w5); w6=FMA( dz,w6,w6); w7=FMA( dz,w7,w7);
# undef FNMS
# undef FMA

    // Adjust the weights for a corrected local accumulation of rhob.
    // See note in synchronize_rho why we must do this for rhob and not
    // for rhof.

    x  = v;    z = x/sz;
    if( z==1     ) w0 += w0, w1 += w1, w2 += w2, w3 += w3;
    if( z==g->nz ) w4 += w4, w5 += w5, w6 += w6, w7 += w7;
    x -= sz*z; y = x/sy;
    if( y==1     ) w0 += w0, w1 += w1, w4 += w4, w5 += w5;
    if( y==g->ny ) w2 += w2, w3 += w3, w6 += w6, w7 += w7;
    x -= sy*y;
    if( x==1     ) w0 += w0, w2 += w2, w4 += w4, w6 += w6;
    if( x==g->nx ) w1 += w1, w3 += w3, w5 += w5, w7 += w7;

    // Reduce the particle charge to rhob

    f[v      ].rhob += w0; f[v      +1].rhob += w1;
    f[v   +sy].rhob += w2; f[v   +sy+1].rhob += w3;
    f[v+sz   ].rhob += w4; f[v+sz   +1].rhob += w5;
    f[v+sz+sy].rhob += w6; f[v+sz+sy+1].rhob += w7;

# else

    v4float q, wl, wh, rl, rh;
    int v, sy = g->sy, sz = g->sz;
    int i, j;

    // Gather rhob for this voxel

    v = p->i;
    rl = v4float( f[v      ].rhob, f[v      +1].rhob,
            f[v   +sy].rhob, f[v   +sy+1].rhob);
    rh = v4float( f[v+sz   ].rhob, f[v+sz   +1].rhob,
            f[v+sz+sy].rhob, f[v+sz+sy+1].rhob);

    // Compute the trilinear weights

    load_4x1( &p->dx, wl );
    trilinear( wl, wh );

    // Adjust the weights for a corrected local accumulation of rhob.
    // See note in synchronize_rho why we must do this for rhob and not
    // for rhof.  Why yes, this code snippet is branchless and evil.

    i = v;
    j = i/sz; i -= sz*j;       load_4x1( &ax[(j==1    )?3:0], q ); wl *= q;
    /**/                       load_4x1( &ax[(j==g->nz)?3:0], q ); wh *= q;
    j = i/sy; i -= sy*j;
    j = (j==1) + 2*(j==g->ny); load_4x1( &ay[j], q ); wl *= q; wh *= q;
    i = (i==1) + 2*(i==g->nx); load_4x1( &ax[i], q ); wl *= q; wh *= q;

    // Reduce the particle charge to rhof and scatter the result

    q = v4float( (qsp*g->r8V)*p->w );
    store_4x1_tr( fma(q,wl,rl), &f[v      ].rhob, &f[v      +1].rhob,
            &f[v   +sy].rhob, &f[v   +sy+1].rhob );
    store_4x1_tr( fma(q,wh,rh), &f[v+sz   ].rhob, &f[v+sz   +1].rhob,
            &f[v+sz+sy].rhob, &f[v+sz+sy+1].rhob );

# endif
}

// KOKKOS VERSION
// accumulate_rho_p adds the charge density associated with the
// supplied particle array to the rhof of the fields.  Trilinear
// interpolation is used.  rhof is known at the nodes at the same time
// as particle positions.  No effort is made to fix up edges of the
// computational domain; see note in synchronize_rhob about why this
// is done this way.  All particles on the list must be inbounds.

// TODO replace with scatter add view
struct accum_rho_p {
    k_field_sa_t kfield;
    k_particles_t kparticles;
    k_particles_i_t kparticles_i;
    int sy;
    int sz;
    float q_8V;
    int np;

    KOKKOS_INLINE_FUNCTION
    accum_rho_p(k_field_sa_t k_f_sa_, k_particles_t k_p_, k_particles_i_t k_p_i_, int sy_, int sz_, float q_8V_, int np_) : kfield(k_f_sa_), kparticles(k_p_), kparticles_i(k_p_i_), sy(sy_), sz(sz_), q_8V(q_8V_), np(np_) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int n) const {
        float w0, w1, w2, w3, w4, w5, w6, w7, dz;

        w0 = kparticles(n, particle_var::dx);
        w1 = kparticles(n, particle_var::dy);
        dz = kparticles(n, particle_var::dz);
        int v = kparticles_i(n);
        w7 = kparticles(n, particle_var::w) * q_8V;

#   define FMA( x,y,z) ((z)+(x)*(y))
#   define FNMS(x,y,z) ((z)-(x)*(y))
        w6=FNMS(w0,w7,w7);                    // q(1-dx)
        w7=FMA( w0,w7,w7);                    // q(1+dx)
        w4=FNMS(w1,w6,w6); w5=FNMS(w1,w7,w7); // q(1-dx)(1-dy), q(1+dx)(1-dy)
        w6=FMA( w1,w6,w6); w7=FMA( w1,w7,w7); // q(1-dx)(1+dy), q(1+dx)(1+dy)
        w0=FNMS(dz,w4,w4); w1=FNMS(dz,w5,w5); w2=FNMS(dz,w6,w6); w3=FNMS(dz,w7,w7);
        w4=FMA( dz,w4,w4); w5=FMA( dz,w5,w5); w6=FMA( dz,w6,w6); w7=FMA( dz,w7,w7);
#   undef FNMS
#   undef FMA

        auto scatter_view_access = kfield.access();

        scatter_view_access(v,         field_var::rhof) += w0;
        scatter_view_access(v+1,       field_var::rhof) += w1;
        scatter_view_access(v+sy,      field_var::rhof) += w2;
        scatter_view_access(v+sy+1,    field_var::rhof) += w3;
        scatter_view_access(v+sz,      field_var::rhof) += w4;
        scatter_view_access(v+sz+1,    field_var::rhof) += w5;
        scatter_view_access(v+sz+sy,   field_var::rhof) += w6;
        scatter_view_access(v+sz+sy+1, field_var::rhof) += w7;

/*
        Kokkos::atomic_add(&kfield(v,         field_var::rhof), w0);
        Kokkos::atomic_add(&kfield(v+1,       field_var::rhof), w1);
        Kokkos::atomic_add(&kfield(v+sy,      field_var::rhof), w2);
        Kokkos::atomic_add(&kfield(v+sy+1,    field_var::rhof), w3);
        Kokkos::atomic_add(&kfield(v+sz,      field_var::rhof), w4);
        Kokkos::atomic_add(&kfield(v+sz+1,    field_var::rhof), w5);
        Kokkos::atomic_add(&kfield(v+sz+sy,   field_var::rhof), w6);
        Kokkos::atomic_add(&kfield(v+sz+sy+1, field_var::rhof), w7);
*/
    }
};

struct accum_rhob {
    k_field_t kfield;
    k_particles_t kpart;
    k_particles_i_t kpart_i;
    k_particle_i_movers_t kpart_movers_i;
    float qsp;
    float r8V;
    int nx;
    int ny;
    int nz;
    int sy;
    int sz;

    KOKKOS_INLINE_FUNCTION
    accum_rhob(k_field_t k_f_, k_particles_t k_p_, k_particles_i_t k_p_i_, k_particle_i_movers_t kpart_movers_i_, float qsp_, float r8V_, int nx_, int ny_, int nz_, int sy_, int sz_) :
        kfield(k_f_), kpart(k_p_), kpart_i(k_p_i_), kpart_movers_i(kpart_movers_i_), qsp(qsp_), r8V(r8V_), nx(nx_), ny(ny_), nz(nz_), sy(sy_), sz(sz_) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const int n) const {
        int part_idx = kpart_movers_i(n);
        float w0 = kpart(part_idx, particle_var::dx);
        float w1 = kpart(part_idx, particle_var::dy);
        float w2, w3, w4, w5, w6;
        float w7 = (qsp * r8V) * kpart(part_idx, particle_var::w);
        float dz = kpart(part_idx, particle_var::dz);
        int v = kpart_i(part_idx);
        int x, y, z;

        w6 = w7 - w0 * w7;
        w7 = w7 + w0 * w7;
        w4 = w6 - w1 * w6;
        w5 = w7 - w1 * w7;
        w6 = w6 + w1 * w6;
        w7 = w7 + w1 * w7;
        w0 = w4 - dz * w4;
        w1 = w5 - dz * w5;
        w2 = w6 - dz * w6;
        w3 = w7 - dz * w7;
        w4 = w4 + dz * w4;
        w5 = w5 + dz * w5;
        w6 = w6 + dz * w6;
        w7 = w7 + dz * w7;

        x = v;
        z = x/sz;
        if(z == 1) {
            w0 += w0;
            w1 += w1;
            w2 += w2;
            w3 += w3;
        }
        if(z == nz) {
            w4 += w4;
            w5 += w5;
            w6 += w6;
            w7 += w7;
        }
        x -= sz * z;
        y = x/sy;
        if(y == 1) {
            w0 += w0;
            w1 += w1;
            w4 += w4;
            w5 += w5;
        }
        if(y == ny) {
            w2 += w2;
            w3 += w3;
            w6 += w6;
            w7 += w7;
        }
        x -= sy * y;
        if(x == 1) {
            w0 += w0;
            w2 += w2;
            w4 += w4;
            w6 += w6;
        }
        if(x == nx) {
            w1 += w1;
            w3 += w3;
            w5 += w5;
            w7 += w7;
        }
        kfield(v,         field_var::rhob) += w0;
        kfield(v+1,       field_var::rhob) += w1;
        kfield(v+sy,      field_var::rhob) += w2;
        kfield(v+sy+1,    field_var::rhob) += w3;
        kfield(v+sz,      field_var::rhob) += w4;
        kfield(v+sz+1,    field_var::rhob) += w5;
        kfield(v+sz+sy,   field_var::rhob) += w6;
        kfield(v+sz+sy+1, field_var::rhob) += w7;
    }
};

void
k_accumulate_rho_p( /**/  field_array_t * RESTRICT fa,
                  const species_t     * RESTRICT sp )
{
  if( !fa || !sp || fa->g!=sp->g ) ERROR(( "Bad args" ));

    k_field_t kfield = fa->k_f_d;
    k_particles_t kparticles = sp->k_p_d;
    k_particles_i_t kparticles_i = sp->k_p_i_d;

    const float q_8V = (sp->q)*(sp->g->r8V);
    const int np = sp->np;
    const int sy = sp->g->sy;
    const int sz = sp->g->sz;
/*
    k_field_sa_t scatter_view = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, KOKKOS_SCATTER_DUPLICATED,KOKKOS_SCATTER_ATOMIC>(kfield);
    Kokkos::parallel_for("accumulate_rho_p", Kokkos::RangePolicy < Kokkos::DefaultExecutionSpace > (0, np),
        accum_rho_p(scatter_view, kparticles, kparticles_i, sy, sz, q_8V, np));
    Kokkos::Experimental::contribute(kfield, scatter_view);
*/
    Kokkos::parallel_for("accumulate_rho_p", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, np), KOKKOS_LAMBDA(const int n) {
        float w0, w1, w2, w3, w4, w5, w6, w7, dz;

        w0 = kparticles(n, particle_var::dx);
        w1 = kparticles(n, particle_var::dy);
        dz = kparticles(n, particle_var::dz);
        int v = kparticles_i(n);
        w7 = kparticles(n, particle_var::w) * q_8V;

#   define FMA( x,y,z) ((z)+(x)*(y))
#   define FNMS(x,y,z) ((z)-(x)*(y))
        w6=FNMS(w0,w7,w7);                    // q(1-dx)
        w7=FMA( w0,w7,w7);                    // q(1+dx)
        w4=FNMS(w1,w6,w6); w5=FNMS(w1,w7,w7); // q(1-dx)(1-dy), q(1+dx)(1-dy)
        w6=FMA( w1,w6,w6); w7=FMA( w1,w7,w7); // q(1-dx)(1+dy), q(1+dx)(1+dy)
        w0=FNMS(dz,w4,w4); w1=FNMS(dz,w5,w5); w2=FNMS(dz,w6,w6); w3=FNMS(dz,w7,w7);
        w4=FMA( dz,w4,w4); w5=FMA( dz,w5,w5); w6=FMA( dz,w6,w6); w7=FMA( dz,w7,w7);
#   undef FNMS
#   undef FMA

//        auto scatter_view_access = kfield.access();
//
//        scatter_view_access(v,         field_var::rhof) += w0;
//        scatter_view_access(v+1,       field_var::rhof) += w1;
//        scatter_view_access(v+sy,      field_var::rhof) += w2;
//        scatter_view_access(v+sy+1,    field_var::rhof) += w3;
//        scatter_view_access(v+sz,      field_var::rhof) += w4;
//        scatter_view_access(v+sz+1,    field_var::rhof) += w5;
//        scatter_view_access(v+sz+sy,   field_var::rhof) += w6;
//        scatter_view_access(v+sz+sy+1, field_var::rhof) += w7;

        Kokkos::atomic_add(&kfield(v,         field_var::rhof), w0);
        Kokkos::atomic_add(&kfield(v+1,       field_var::rhof), w1);
        Kokkos::atomic_add(&kfield(v+sy,      field_var::rhof), w2);
        Kokkos::atomic_add(&kfield(v+sy+1,    field_var::rhof), w3);
        Kokkos::atomic_add(&kfield(v+sz,      field_var::rhof), w4);
        Kokkos::atomic_add(&kfield(v+sz+1,    field_var::rhof), w5);
        Kokkos::atomic_add(&kfield(v+sz+sy,   field_var::rhof), w6);
        Kokkos::atomic_add(&kfield(v+sz+sy+1, field_var::rhof), w7);

    });
}

void k_accumulate_rhob(k_field_t& kfield, k_particles_t& kpart, k_particles_i_t& kpart_i, k_particle_i_movers_t& k_part_movers_i, const grid_t* RESTRICT g, const float qsp, const int nm) {
    int sy = g->sy, sz = g->sz;
    float r8V = g->r8V;
    int nx = g->nx;
    int ny = g->ny;
    int nz = g->nz;

    Kokkos::parallel_for("accumulate_rhob", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,nm),
        accum_rhob(kfield, kpart, kpart_i, k_part_movers_i, qsp, r8V, nx, ny, nz, sy, sz));
}

