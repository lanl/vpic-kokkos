#include "kokkos_helpers.h"

// Helper functions, like printing etc

void print_accumulator(k_accumulators_t& f, int n)
{

    printf("size of accumulator is %ld\n", f.size() );

    Kokkos::parallel_for("field printer", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int i)
            {

            for (int z = 0; z < n; z++)
            {
                for (int zz = 0; zz < ACCUMULATOR_VAR_COUNT; zz++)
                {
                    printf("accum %d has %d %f %f %f %f \n", i, zz,
                        f(i, zz, 0),
                        f(i, zz, 1),
                        f(i, zz, 2),
                        f(i, zz, 3)
                      );
                }
            }
    });

}
void print_particles_d(
        k_particles_t particles,
        int np
        )
{
    printf("Particle printer from 0 to %d \n", np);
    Kokkos::parallel_for("particle printer", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int i)
    {
        for (int z = 0; z < np; z++)
        {
            auto tile = z/SIMD_LEN;
            auto pidx = z - tile*SIMD_LEN;
            printf("accum part %d has %f %f %f %f %f %f \n", z,
                    particles(pidx, particle_var::dx, tile),
                    particles(pidx, particle_var::dy, tile),
                    particles(pidx, particle_var::dz, tile),
                    particles(pidx, particle_var::ux, tile),
                    particles(pidx, particle_var::uy, tile),
                    particles(pidx, particle_var::uz, tile)
                  );
        }
    });

}
