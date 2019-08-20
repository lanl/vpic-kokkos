#ifndef RNG_POLICY
#define RNG_POLICY

#include <Kokkos_Random.hpp>
#include "../vpic/kokkos_helpers.h"

namespace _RNG {
#include <random>

// NOTE: this policy ignore the rng it's passed..
class OriginalRNG {
    public:
        inline double uniform( rng_t * rng, double low, double high ) {
            double dx = drand( rng );
            return low*(1-dx) + high*dx;
        }
        inline double normal( rng_t * rng, double mu, double sigma ) {
            return mu + sigma*drandn( rng );
        }
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, int seed, int sync ) {
            seed_rng_pool( entropy,      seed, 0 );
            seed_rng_pool( sync_entropy, seed, 1 );
        }
};

class CppRNG {
    public:
        inline double uniform( rng_t * rng, double low, double high ) {
            std::uniform_real_distribution<double> distribution(low, high);
            return distribution(uniform_generator);
        }
        inline double normal( rng_t * rng, double mu, double sigma ) {
            std::normal_distribution<double> distribution(mu, sigma);
            return distribution(normal_generator);
        }
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, int base, int sync ) {
            // Try emulate old VPIC seeding such that different ranks use
            // different seeds to avoid rng imprinting
            int seed = (sync ? world_size : world_rank) + (world_size+1)*base;
            uniform_generator = std::default_random_engine(seed);
            normal_generator = std::default_random_engine(seed);
        }

#ifndef DEFAULT_SEED
#define DEFAULT_SEED 42
#endif
        std::default_random_engine uniform_generator;
        std::default_random_engine normal_generator;
        CppRNG()
        {
            uniform_generator = std::default_random_engine(DEFAULT_SEED);
            normal_generator = std::default_random_engine(DEFAULT_SEED);
        }
};

// Theres a better way to do Kokkos rng on device using pools
class KokkosRNG { // CPU!
    //Kokkos::Random_XorShift64_Pool<> rand_pool(12313);
    // TODO: this is going to give device rng in some cases..
    public:
        // TODO: need to pass this a state..
        Kokkos::Random_XorShift1024<Kokkos::DefaultHostExecutionSpace> rp;
        //auto rand_gen = rand_pool.get_state();
        //Kokkos::Random_XorShift1024<Kokkos::DefaultExecutionSpace> rp_device;

        inline double uniform( rng_t * rng, double low, double high ) {
            std::cout << "kokrng" << std::endl;
            return rp.drand(low, high);
        }
        inline double normal( rng_t * rng, double mu, double sigma ) {
            return rp.normal(mu, sigma);
        }
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, int seed, int sync ) {
        }
};

template <typename Policy = CppRNG>
struct RandomNumberProvider : private Policy {
    using Policy::uniform;
    using Policy::normal;
    using Policy::seed;
};
}

#endif //guard
