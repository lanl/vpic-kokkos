#ifndef TA_COLLISION_H
#define TA_COLLISION_H

#include "../../species_advance/species_advance.h"

class takizuka_abe_t {
  public:
      std::string name;
      species_t* spi;
      species_t* spj;
      //rng_pool_t * rp;
      int interval;
      double cvar0; // Base cvar0, which will later be scaled by q and mu
      takizuka_abe_t(
              std::string _name,
              species_t* _spi,
              species_t* _spj,
              int _interval,
              double _cvar0
              ) :
          name(_name), spi(_spi), spj(_spj), interval(_interval), cvar0(_cvar0)
    {
        //empty
    }
};

void add_takizuka_abe_collision(takizuka_abe_t ta);
int have_ta_collisions();
void apply_ta_collisions();

#endif
