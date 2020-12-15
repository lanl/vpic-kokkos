#ifndef _rhob_h_
#define _rhob_h_

#include "../../util/weighting/trilinear.h"

template<
    class kfield_view_t, // k_field_t
    class kpart_view_t,  // k_particles_t
    class kparti_view_t,  // k_particles_i_t
    class geo_type
>
void
accumulate_rhob_from_particle(
    kfield_view_t& kfield,
    kpart_view_t& kparticles,
    kparti_view_t& kparticles_i,
    const int n,
    const grid_t* g,
    const float qsp,
    geo_type geometry
)
{

    float dx = kparticles(n, particle_var::dx);
    float dy = kparticles(n, particle_var::dy);
    float dz = kparticles(n, particle_var::dz);
    int voxel = kparticles_i(n);
    float rho = qsp*kparticles(n, particle_var::w)*geometry.inverse_voxel_volume(voxel);

    auto weighter = TrilinearWeighting(g->nx, g->ny, g->nz, g->sx, g->sy, g->sz);
    weighter.set_position(dx, dy, dz);
    weighter.synchronize_weights(voxel);
    weighter.deposit(kfield, voxel, rho);

}

template<class geo_type>
void
accumulate_rhob_from_particle(
    field_t* f,
    const particle_t* p,
    const grid_t* g,
    const float qsp,
    geo_type geometry
)
{

    int voxel = p->i;
    int offset = ((char*) &(f[0].rhob) - (char*)f);

    float rho = qsp*(p->w)*geometry.inverse_voxel_volume(voxel);

    auto weighter = TrilinearWeighting(g->nx, g->ny, g->nz, g->sx, g->sy, g->sz);
    weighter.set_position(p->dx, p->dy, p->dz);
    weighter.synchronize_weights(voxel);
    weighter.deposit_aos(f, voxel, offset, rho);

}


#endif
