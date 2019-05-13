#include "kokkos_helpers.h"

// Helper functions, like printing etc

void print_accumulator(k_accumulators_t fields, int n)
{

    printf("size of accumulator is %d\n", fields.size() );

    Kokkos::parallel_for("field printer", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, 1), KOKKOS_LAMBDA (int i)
            {

            for (int z = 0; z < n; z++)
            {
                for (int zz = 0; zz < ACCUMULATOR_VAR_COUNT; zz++)
                {
                    printf("accum %d has %d %f %f %f %f \n", i, zz,
                        fields(i, zz, 0),
                        fields(i, zz, 1),
                        fields(i, zz, 2),
                        fields(i, zz, 3)
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
            printf("accum part %d has %f %f %f %f %f %f at %f\n", z,
                    particles(z, particle_var::dx),
                    particles(z, particle_var::dy),
                    particles(z, particle_var::dz),
                    particles(z, particle_var::ux),
                    particles(z, particle_var::uy),
                    particles(z, particle_var::uz),
                    particles(z, particle_var::pi)
                  );
        }
    });

}
