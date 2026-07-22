General Coordinate Systems
================

VPIC includes support for generalized orthagonal coordinate systems, such as cylindrical meshes and spherical meshes. It can also support arbitrarily defined meshes such as non-uniform stretched cartesian to better resolve parts of the simulation. The user can then inject particles and define fields in their chosen coordinate system.

Initialization
================

VPIC includes several pre-defined mesh systems, including cartesian, cylindrical, and spherical meshes, which can be called from the `begin_initialization` block of the deck.

The following code initializes a cylindrical mesh with periodic boundaries. First call a define grid function to specify the bounds and resolution of your grid system. Then call `init_cylindrical_grid` to tell VPIC to interpret the first argument as radius, the second argument as theta, and the third argument as z.
  .. code-block:: c++
      define_periodic_grid( 0.1, 0, -0.5, // Low corner (min_r, min_theta, min_z)
                        0.5,  2.0*M_PI,  0.5, // High corner (max_r, max_theta, max_z)
                        nr, ntheta, nz, // Resolution
                        topology_r, topology_theta, topology_z); // Topology
      init_cylindrical_grid();

After calling `init_cylindrical_grid`, VPIC will interpret the first argument of most functions as the radial coordinate, the second argument as the azimuthal angle, and the third argument as the axial coordinate. The user can then define fields and inject particles in this cylindrical system. The only exception are the velocity components in `inject_particle`, which are taken to be in cartesian coordinates.

The user can also define their own arbitrary mesh system by providing the metric factors for their system. The following example is provided for a spherical grid (which is already implemented as a helper function, but is redefined here for illustration):
  .. code-block:: c++
    void init_spherical_grid()
    {
      const int ghost_layers_per_side = 2;
      const int nx_total = nx + 2 * ghost_layers_per_side;
      const int ny_total = ny + 2 * ghost_layers_per_side;
      const int nz_total = nz + 2 * ghost_layers_per_side;
      const int nv_cm = nx_total * ny_total * nz_total;
      k_curvilinear_mesh_d = k_curvilinear_mesh_t("k_curvilinear_mesh_d", nv_cm);
      k_curvilinear_mesh_h = Kokkos::create_mirror_view(k_curvilinear_mesh_d);

      // Local per-rank cell spacing in each curvilinear direction.
      const double dr = (x1 - x0) / nx;
      const double dtheta = (y1 - y0) / ny;
      const double dphi = (z1 - z0) / nz;

      // Define metric factors for each cell - cells can have completely different metric factors if desired
      Kokkos::parallel_for(
      "Fill curvilinear mesh view",
      host_execution_policy_md({0, 0, 0}, {nx_total, ny_total, nz_total}),
      KOKKOS_CLASS_LAMBDA (const int i, const int j, const int k) {
        const int idx = i + j * nx_total + k * nx_total * ny_total;

        // Physical (r,theta,phi) from local origin and local mesh index (i-ghost).
        double r_i = x0 + (i - ghost_layers_per_side + 0.5) * dr;
        double theta_j = y1 + (j - ghost_layers_per_side + 0.5) * dtheta;
        double phi_k = z0 + (k - ghost_layers_per_side + 0.5) * dphi;

        double cos_theta = Kokkos::cos(theta_j);
        double sin_theta = Kokkos::sin(theta_j);
        double cos_phi = Kokkos::cos(phi_k);
        double sin_phi = Kokkos::sin(phi_k);

        double x = 0 + r_i * sin_theta * cos_phi;
        double y = 0 + r_i * sin_theta * sin_phi;
        double z = 0 + r_i * cos_theta;

        // Scale factors
        k_curvilinear_mesh_h(idx,curv_mesh_var::h_1) = 1.0; // h_r
        k_curvilinear_mesh_h(idx,curv_mesh_var::h_2) = r_i;  // h_theta
        k_curvilinear_mesh_h(idx,curv_mesh_var::h_3) = r_i * sin_theta; // h_phi

        // Jacobian
        k_curvilinear_mesh_h(idx,curv_mesh_var::jac) = r_i * r_i * sin_theta;

        // Basis vectors in Cartesian components
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_u) = sin_theta * cos_phi;
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_v) = sin_theta * sin_phi;
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_1_w) = cos_theta;

        k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_u) = cos_theta * cos_phi;
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_v) = cos_theta * sin_phi;
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_2_w) = -sin_theta; 

        k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_u) = -sin_phi;
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_v) = cos_phi;
        k_curvilinear_mesh_h(idx,curv_mesh_var::e_3_w) = 0.0;

        // Cartesian cell position
        k_curvilinear_mesh_h(idx,curv_mesh_var::xg) = x;
        k_curvilinear_mesh_h(idx,curv_mesh_var::yg) = y;
        k_curvilinear_mesh_h(idx,curv_mesh_var::zg) = z;
      }
      );
      Kokkos::deep_copy(k_curvilinear_mesh_d, k_curvilinear_mesh_h);
    }

From now on, VPIC will interpret x as radius, y as theta, and z as z. For example, if the user were to access `grid->x0` they would recieve the value of the minimum radius that they specified in `define_periodic_grid`.

Note:
  * VPIC assumes grids have 2 ghost layers on all sides

Particle Injection
================
After defining a grid system, the user can inject particles using that coordinate system.

For example, after calling `init_cylindrical_grid`, the user can inject particles in cylindrical coordinates and VPIC will automatically interpret the positional arguments as being in cylindrical coordinates. By default, the velocity components are still interpreted as cartesian coordinates.
  .. code-block:: c++
      inject_particle( "ion", r, theta, z, ux, uy, uz, weight, 0, 0, q );

The particle advance solver will then automatically account for your desired mesh system.

Note:
  * If you want to inject particles uniformly on a non-uniform grid, particles should be weighted by cell volume (i.e. the jacobian of the cell you are injecting in)

Field Injection
================
Field components are also interpreted as being in the user's defined coordinate system during injection:

To inject fields in cylindrical coordinates, for example:
  .. code-block:: c++
    set_region_field(everywhere, Er, Etheta, Ez, Br, Btheta, Bz);
    set_region_bext(everywhere, Br, Btheta, Bz);

Since VPIC interprets x as radius, y as theta, and z as z, if the user injects fields using an analytic equation they should use x, y, and z but understand that these will be interpreted as r, theta, and z. For example, to inject a helical magnetic field:
  .. code-block:: c++
    set_region_field(everywhere,
                0, 0, 0 // no E field
                0 // no Br
                B0*(x/r0)*sin(k*z), // Btheta varies with radius (x) and z
                B0*(1.0 - (x/r0))*cos(k*z)); // Bz is axial and varies with r and z

For convenience, VPIC also allows you to inject fields in cartesian coordinates as well, regardless of your mesh settings, using `set_region_field_cart`. In this case, if you use analytic equations, x y and z will indeed be interpreted as cartesian x y and z rather than in your coordinate system.
  .. code-block:: c++
    set_region_field(everywhere, Ex, Ey, Ez, Bx, By, Bz);
    set_region_bext(everywhere, Bx, By, Bz);

The field advance solver will then automatically account for your desired mesh system using a generalized-coordinate form of the hybrid Maxwell's equations.

One caveat:
  * Internally, VPIC stores E in covariant logical components, B in contravariant logical components, and current in contravariant logical coordinates. This means they are not physical fields but rather unscaled by the scale factors of the cell they are in. If a user tries directly accessing any of these during runtime without a helper function they will encounter unphysical quantities. To convert these to physical quantities in their coordinate systems, simply multiply or divide each component by the scale factors corresponding to the cell that they are trying to access information from. For a covariant field, divide by scale factors and for a contravariant field multiply by scale factors to convert them back to physical components. Similarly, if a user tries to directly edit fields in a deck (say, by setting `field(i, j, k).cz0`), they should perform the opposite operation to convert to logical (unphysical) components. This is what the `set_region_field` macros do internally.