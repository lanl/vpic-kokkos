#ifndef RNG_POLICY
#define RNG_POLICY

#include <Kokkos_Random.hpp>
#include "../vpic/kokkos_helpers.h"
#include <random>

// TODO: These don't belong here, do they?
using kokkos_rng_pool_t = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
using kokkos_rng_state_t = Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace>;

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

template<typename ExecutionSpace>
class KokkosRNG
{
    public:
        // TODO: use a better quality RNG? 1024?
        // eg
        //using kokkos_rng_pool_t = Kokkos::Random_XorShift1024<Kokkos::DefaultHostExecutionSpace>;

        using kokkos_rng_pool_t = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
        using kokkos_rng_state_t = Kokkos::Random_XorShift64<ExecutionSpace>;

        kokkos_rng_pool_t rand_pool;

        /////////// Uniform

        /**
         * @brief Generate a uniform number between (low..high]
         *
         * @param low The min for the range
         * @param high The max for the range
         *
         * @return The generated uniform number
         */
        inline double uniform( const double low, const double high ) {
            auto generator = rand_pool.get_state();
            auto r = uniform(generator, low, high);
            rand_pool.free_state(generator);
            return r;
        }

        /**
         * @brief Pass-through helper to strip off the interface-only arg rng_t
         *
         * Generate a uniform number between (low..high]
         *
         * @param generator The generated state to pass in
         * @param low The min for the range
         * @param high The max for the range
         *
         * @return The generated uniform number
         */
        inline double uniform( const rng_t* rng, const double low, const double high ) {
            return uniform(low, high);
        }

        /**
         * @brief Advanced unfirom generator where the caller holds the state
         * variable for the generator.
         *
         * Generate a uniform number between (low..high]
         *
         * @param generator The generated state to pass in
         * @param low The min for the range
         * @param high The max for the range
         *
         * @return The generated uniform number
         */
        inline double uniform( kokkos_rng_state_t& generator, const double low, const double high ) {
            return generator.drand(low, high);
        }

        /////////// Normal

        /**
         * @brief Generate a normally distributed real with given mean and standard
         * deviation
         *
         * @param mu The mean of the distribution
         * @param sigma The standard deviation of the distribution
         *
         * @return The generated normal number
         */
        inline double normal( const double mu, const double sigma ) {
            auto generator = rand_pool.get_state();
            auto r = normal(generator, mu, sigma);
            rand_pool.free_state(generator);
            return r;
        }

        /**
         * @brief Pass-through helper to strip off the interface-only arg rng_t
         *
         * Generate a normally distributed real with given mean and standard
         * deviation
         *
         * @param rnt_t IGNORED
         * @param mu The mean of the distribution
         * @param sigma The standard deviation of the distribution
         *
         * @return The generated normal number
         */
        inline double normal( const rng_t* rng, const double mu, const double sigma ) {
            return normal(mu, sigma);
        }

        /**
         * @brief Advanced normal generator where the caller holds the state
         * variable for the generator.
         *
         * Generate a normally distributed real with given mean and standard
         * deviation
         *
         * @param generator The generator/state to use
         * @param mu The mean of the distribution
         * @param sigma The standard deviation of the distribution
         *
         * @return The generated normal number
         */
        inline double normal( kokkos_rng_state_t& generator, const double mu, const double sigma ) {
            return generator.normal(mu, sigma);
        }

        //////////// uint

        /**
         * @brief Generate a random int between [0..max)
         *
         * @param max The max for the range
         *
         * @return The generated random int
         */
        inline unsigned int uint( const unsigned int max )
        {
            auto generator = rand_pool.get_state();
            auto r = generator.urand(max);
            rand_pool.free_state(generator);
            return r;
        }

        /**
         * @brief Pass-through helper to strip off the interface-only arg rng_t
         *
         * Generate a random int between [0..max)
         *
         * @param rng_t IGNORED
         * @param max The max for the range
         *
         * @return The generated random int
         */
        inline unsigned int uint( const rng_t* rng, const unsigned int max )
        {
            return uint(max);
        }

        /**
         * @brief Advanced uint generator where the caller holds the state
         * variable for the generator.
         *
         * Generate a random int between [0..max)
         *
         * @param generator The generated state to pass in
         * @param max The max for the range
         *
         * @return The generated random int
         */
        inline unsigned int uint(  kokkos_rng_state_t& generator, const unsigned int max )
        {
            return generator.urand(max);
        }

        ////////////////

        /**
         * @brief Seed the kokkos RNG, based on inputs and MPI rank
         *
         * @param entropy IGNORED
         * @param sync_entropy IGNORED
         * @param base Base offset to MPI rank seed
         * @param sync Condition to determine if we use base
         */
        void seed( rng_pool_t* entropy, rng_pool_t* sync_entropy, const int base, const int sync )
        {
           int seed = (sync ? world_size : world_rank) + (world_size+1)*base;
           rand_pool = kokkos_rng_pool_t(seed);
        }

        /**
         * @brief Helper wrapper to allow the user to manage their own state
         * variables to avoid the cost of repeatedly creating/freeing it
         */
        kokkos_rng_state_t get_state()
        {
            return rand_pool.get_state();
        }

        /**
         * @brief Helper wrapper to allow the user to free the state they manually created
         * using the above get_state() wrapper
         *
         * @param generator The kokkos generator state to free
         */
        void free_state(kokkos_rng_state_t& generator)
        {
            rand_pool.free_state(generator);
        }

        /**
         * @brief Default constructor, which will intialize the kokkos pool
         * with seed=DEFAULT_SEED, which the user can set at compile time
         */
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
