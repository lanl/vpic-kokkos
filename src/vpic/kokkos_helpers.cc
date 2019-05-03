#include "kokkos_helpers.h"

// Helper functions, like printing etc

void print_accumulator(k_accumulators_t fields, int n)
{

    printf("size of accumulator is %d\n", fields.size() );

    Kokkos::parallel_for("field printer", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, n), KOKKOS_LAMBDA (int i)
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
    });

}
void print_particles_d(
        k_particles_t particles,
        int np
        )
{
    Kokkos::parallel_for("particle printer", Kokkos::RangePolicy <
            Kokkos::DefaultExecutionSpace > (0, np), KOKKOS_LAMBDA (int i)
    {
      printf("accum part %d has %f %f %f %f %f %f\n", i,
               particles(i, particle_var::dx),
               particles(i, particle_var::dy),
               particles(i, particle_var::dz),
               particles(i, particle_var::ux),
               particles(i, particle_var::uy),
               particles(i, particle_var::uz)
       );
    });

}
