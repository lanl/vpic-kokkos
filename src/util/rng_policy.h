#ifndef RNG_POLICY
#define RNG_POLICY

#include <Kokkos_Random.hpp>
#include "../vpic/kokkos_helpers.h"
#include <random>

namespace _RNG {

// NOTE: this policy ignore the rng it's passed..
class OriginalRNG {
    public:
        inline double uniform( rng_t* rng, const double low, const double high ) {
            double dx = drand( rng );
            return low*(1-dx) + high*dx;
        }
        inline double normal( rng_t* rng, const double mu, const double sigma ) {
            return mu + sigma*drandn( rng );
        }
        inline unsigned int uint( rng_t* rng, const unsigned int max )
        {
            return uirand(rng) % max;
        }
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, const int seed, const int sync ) {
            // seed here is a base, that gets passed into:
            // seed = (sync ? world_size : world_rank) + (world_size+1)*rp->n_rng*seed;
            seed_rng_pool( entropy,      seed, 0 );
            seed_rng_pool( sync_entropy, seed, 1 );
        }
};

class CppRNG {
    public:
        inline double uniform( rng_t* rng, const double low, const double high ) {
            std::uniform_real_distribution<double> distribution(low, high);
            return distribution(uniform_generator);
        }
        inline double normal( rng_t* rng, const double mu, const double sigma ) {
            std::normal_distribution<double> distribution(mu, sigma);
            return distribution(normal_generator);
        }
        inline unsigned int uint( rng_t* rng, const unsigned int max )
        {
            std::uniform_int_distribution<> distribution(0, max);
            return distribution(uniform_generator);
        }
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, const int base, const int sync ) {
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
// TODO: the way this keeps polling the state seems to give the same RNG, as you need to pole the same state multiple times. This seems sketchy for init...
class KokkosRNG { // CPU!
    //Kokkos::Random_XorShift64_Pool<> rand_pool(12313);
    // TODO: this is going to give device rng in some cases..
    public:
        // TODO: need to pass this a state..
        // TODO: Explicitly setting a exec space here is bad

        // TODO: use a better quality RNG, likely needs the addition of a "pool"
        //using kokkos_rng_gen_t = Kokkos::Random_XorShift1024<Kokkos::DefaultHostExecutionSpace>;
        //auto rand_gen = rand_pool.get_state();
        //Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace> rp_device;

        using kokkos_rng_pool_t = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultHostExecutionSpace>;
        kokkos_rng_pool_t rand_pool;

        inline double uniform( const double low, const double high ) {
            auto generator = rand_pool.get_state();
            return generator.drand(low, high);
        }
        inline double uniform( rng_t* rng, const double low, const double high ) {
            return uniform(low, high);
        }
        inline double normal( const double mu, const double sigma ) {
            auto generator = rand_pool.get_state();
            return generator.normal(mu, sigma);
        }
        inline double normal( rng_t* rng, const double mu, const double sigma ) {
            return normal(mu, sigma);
        }
        inline unsigned int uint( const unsigned int max )
        {
            auto generator = rand_pool.get_state();
            return generator.urand(max);
        }
        inline unsigned int uint( rng_t* rng, const unsigned int max )
        {
            return uint(max);
        }
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, const int base, const int sync )
        {
           int seed = (sync ? world_size : world_rank) + (world_size+1)*base;
           rand_pool = kokkos_rng_pool_t(seed);
        }
        KokkosRNG() {
           rand_pool = kokkos_rng_pool_t(DEFAULT_SEED);
        }
};

template <typename Policy = CppRNG>
struct RandomNumberProvider : private Policy {
    using Policy::uniform;
    using Policy::normal;
    using Policy::uint;
    using Policy::seed;
};
}

#endif //guard
