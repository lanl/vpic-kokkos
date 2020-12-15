#ifndef _spa_private_h_
#define _spa_private_h_

#ifndef IN_spa
#error "Do not include spa_private.h; include species_advance.h"
#endif

#include "../species_advance.h"

#include "rhob.h"
#include "move_p.h"
#include "borris.h"

// For passing particle_t structs to kokkos functions.
class ParticleViewWrapper {
public:

    ParticleViewWrapper(particle_t * p) : p(p) {}

    float& operator() (int index, int var) const {
        particle_t * pp = p + index;
        switch(var) {
            case particle_var::dx : return pp->dx ; break ;
            case particle_var::dy : return pp->dy ; break ;
            case particle_var::dz : return pp->dz ; break ;
            case particle_var::ux : return pp->ux ; break ;
            case particle_var::uy : return pp->uy ; break ;
            case particle_var::uz : return pp->uz ; break ;
            case particle_var::w  : return pp->w  ; break ;
            default :
                ERROR(("Unknown particle var"));
                break;
        }
    }

    int32_t& operator() (int index) const {
        return (p + index)->i;
    }

private:
    particle_t * p;

};

#endif // _spa_private_h_
