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


API Reference
================

This section documents the functions, structures, and macros that were added or
modified to support generalized coordinate systems. It is intended for both
input-deck authors and developers extending the curvilinear machinery.

The curvilinear mesh stores, per cell, the scale factors :math:`h_1, h_2, h_3`,
the Jacobian, the three basis vectors (in Cartesian components), and the cell's
physical Cartesian position. These are accessed through the ``curv_mesh_var``
enumerators (``h_1``, ``h_2``, ``h_3``, ``jac``, ``e_1_u`` ... ``e_3_w``,
``xg``, ``yg``, ``zg``).


Grid Geometry Types and Indexing (``grid.h``)
------------------------------------------------

.. cpp:enum:: grid_type

   Identifies the coordinate system associated with a grid. Set automatically
   by the ``init_*_grid`` helpers and read by the particle/field solvers to
   select the correct metric handling.

   .. cpp:enumerator:: CARTESIAN

      Uniform Cartesian mesh. Scale factors are all unity.

   .. cpp:enumerator:: CYLINDRICAL

      :math:`(r, \theta, z)` mesh. :math:`h_1 = 1`, :math:`h_2 = r`,
      :math:`h_3 = 1`, Jacobian :math:`= r`.

   .. cpp:enumerator:: SPHERICAL

      :math:`(r, \theta, \phi)` mesh. :math:`h_1 = 1`, :math:`h_2 = r`,
      :math:`h_3 = r\sin\theta`, Jacobian :math:`= r^2 \sin\theta`.

   .. cpp:enumerator:: GENERAL

      Arbitrary user-defined mesh. Scale factors, basis vectors, and
      Jacobian are supplied per-cell; reciprocal basis vectors and the
      Jacobian are recovered on the fly via quadratic B-spline
      interpolation of the stored Cartesian node positions. This is the
      default ``type`` for a freshly constructed grid.

   .. cpp:enumerator:: STRETCHED_CARTESIAN

      Cartesian-aligned mesh with position-dependent (non-uniform) cell
      sizes. Basis vectors stay axis-aligned; only the scale factors vary.

.. c:macro:: UNVOXEL(v, i, j, k, nx, ny, nz)

   Inverse of :c:macro:`VOXEL`. Decomposes a linear voxel index ``v`` on the
   one-ghost-layer grid ``(nx+2)*(ny+2)*(nz+2)`` back into the ``(i,j,k)``
   integer coordinates. Written as a ``do { } while(0)`` block that assigns
   into the caller-supplied ``i``, ``j``, ``k`` lvalues.

.. c:macro:: VOXEL_TO_MESH(v, nx, ny, nz)

   Converts a linear grid voxel index (one ghost layer,
   ``(nx+2)*(ny+2)*(nz+2)``) to the corresponding linear index in the
   curvilinear mesh array (two ghost layers, ``(nx+4)*(ny+4)*(nz+4)``). The
   mesh indices are offset by ``+1`` in each dimension relative to the grid.

.. c:macro:: GRID_TO_MESH(i, j, k, nx, ny, nz)

   Converts grid cell indices ``(i,j,k)`` directly to the linear curvilinear
   mesh index, accounting for the extra ghost layer of the mesh
   (two ghosts vs. one). Used throughout the region-setter macros and the
   push kernels to fetch per-cell scale factors and geometry.


Grid Initialization Methods (``grid_t``)
-------------------------------------------

Each of these methods allocates the per-rank curvilinear mesh view
(``k_curvilinear_mesh_d`` / ``_h``), fills the scale factors, Jacobian,
basis vectors, and Cartesian cell positions, sets ``grid_t::type``, and
deep-copies the result to the device. They should be called from
``begin_initialization`` **after** a ``define_*_grid`` call has established
``nx,ny,nz`` and the domain bounds.

.. cpp:function:: void grid_t::init_kokkos_grid(int num_neighbor)

   Allocates and populates the Kokkos neighbor view (device + host mirror)
   from the existing ``neighbor`` array. Must be called only once the
   neighbor connectivity has been finalized.

.. cpp:function:: void grid_t::init_cartesian_grid()

   Fills the curvilinear mesh for a uniform Cartesian system: all scale
   factors and the Jacobian are ``1``, basis vectors are axis-aligned, and
   node positions come from the local origin plus a uniform cell size. Sets
   ``type = CARTESIAN``.

.. cpp:function:: void grid_t::init_stretched_cartesian_grid(double beta_x = 1.0, double beta_y = 1.0, double beta_z = 1.0)

   Builds a non-uniform, Cartesian-aligned mesh using a ``tanh``-based
   stretching map in each direction. The ``beta_*`` parameters control the
   stretching strength per axis (a value ``<= 1e-10`` disables stretching on
   that axis, yielding a uniform mapping). Scale factors are the local
   derivative of the stretch map (normalized so ``h == 1`` reproduces the
   uniform case), the Jacobian is their product, and basis vectors remain
   axis-aligned. Sets ``type = STRETCHED_CARTESIAN``.

   .. note::

      The stretch map and computational coordinate are defined **globally**,
      so physical positions are computed from the global bounds
      (``gx0..gz1``) and this rank's global index offset, not the local
      ``x0/x1`` bounds.

.. cpp:function:: void grid_t::init_curvilinear_grid()

   Convenience entry point for a generic curvilinear mesh. Currently
   delegates to :cpp:func:`init_cartesian_grid`; intended as the hook for
   user-provided general meshes.

.. cpp:function:: void grid_t::init_cylindrical_grid()

   Fills the mesh for a cylindrical :math:`(r,\theta,z)` system using this
   rank's local bounds and mesh indices. Sets :math:`h_2 = r`,
   Jacobian :math:`= r`, and the radial/azimuthal basis vectors. Negative
   radii are reflected (``r -> -r``, ``theta -> theta + pi``). Sets
   ``type = CYLINDRICAL``.

.. cpp:function:: void grid_t::init_spherical_grid()

   Fills the mesh for a spherical :math:`(r,\theta,\phi)` system. Sets
   :math:`h_2 = r`, :math:`h_3 = r\sin\theta`, Jacobian
   :math:`= r^2\sin\theta`, and the corresponding orthonormal basis vectors
   expressed in Cartesian components. Sets ``type = SPHERICAL``.

.. cpp:function:: grid_geom_t grid_t::geom() const

   Returns a small, value-copyable :cpp:struct:`grid_geom_t` snapshot holding
   the geometry scalars and the device curvilinear-mesh view. Safe to capture
   by value into a ``KOKKOS_LAMBDA``; this is how the push/deposit kernels
   access geometry on-device.


Device Geometry Snapshot (``grid_geom_t``)
---------------------------------------------

.. cpp:struct:: grid::grid_geom_t

   A lightweight, trivially-copyable view of the geometry needed inside
   device kernels. Holds ``type``, ``nx,ny,nz``, the local origin
   ``x0,y0,z0``, cell sizes ``dx,dy,dz``, and both the device and host
   curvilinear-mesh Views.

   .. cpp:function:: float m(int n, int var) const

      Single accessor for a curvilinear-mesh entry (node ``n``, variable
      ``var`` from ``curv_mesh_var``). Resolves at compile time to the host
      or device View depending on the execution space.

   .. cpp:function:: void local_to_global(int voxel_i, float dx_p, float dy_p, float dz_p, double& xi_out, double& eta_out, double& mu_out) const

      Maps a particle's logical in-cell offset ``(dx_p,dy_p,dz_p)`` (each on
      ``[-1,1]``) in voxel ``voxel_i`` to global **logical** coordinates
      ``(xi,eta,mu)`` using the local origin and cell sizes.

   .. cpp:function:: void local_to_global_cart(int voxel_i, float dx_p, float dy_p, float dz_p, double& x_out, double& y_out, double& z_out) const

      Maps a particle's logical in-cell offset to global **physical
      Cartesian** coordinates. Branches on ``type``:

      * ``CARTESIAN`` – delegates to :cpp:func:`local_to_global`.
      * ``STRETCHED_CARTESIAN`` – uses stored node position plus a
        half-cell offset scaled by the local scale factors.
      * ``CYLINDRICAL`` / ``SPHERICAL`` – analytic coordinate transform.
      * ``GENERAL`` (else branch) – quadratic B-spline interpolation over the
        ``3x3x3`` stencil of stored node positions.

      Used by the ``*_cart`` region-setter macros to evaluate region
      membership and field equations in physical space.


Curvilinear Kernels (free functions in ``grid.h``)
-----------------------------------------------------

.. cpp:function:: void compute_bspline_basis(float xi, float& S_m1, float& S_0, float& S_p1, float& dS_m1, float& dS_0, float& dS_p1)

   Evaluates the quadratic B-spline basis functions and their derivatives at
   logical coordinate ``xi`` (on ``[-1,1]`` within a cell) for the three
   nodes ``i-1``, ``i``, ``i+1``. Shared by the mesh interpolation in
   :cpp:func:`grid_geom_t::local_to_global_cart`,
   :cpp:func:`compute_reciprocal_basis`, and
   :c:func:`interpolate_scale_factors`.

.. cpp:function:: void compute_reciprocal_basis(const grid::grid_geom_t& geom, float dx, float dy, float dz, int ii, int nx, int ny, int nz, float gdx, float gdy, float gdz, float& grad_xi_x, float& grad_xi_y, float& grad_xi_z, float& grad_eta_x, float& grad_eta_y, float& grad_eta_z, float& grad_mu_x, float& grad_mu_y, float& grad_mu_z, float& jac)

   Computes the reciprocal (contravariant) basis vectors
   :math:`\nabla\xi, \nabla\eta, \nabla\mu` and the Jacobian ``jac`` at a
   particle's logical position ``(dx,dy,dz)`` in voxel ``ii``. This is the
   core routine that lets the pusher deposit coordinate-consistent
   contravariant current. Each geometry type has a dedicated branch:

   * ``CARTESIAN`` – constant, diagonal reciprocal basis.
   * ``CYLINDRICAL`` / ``SPHERICAL`` – analytic reciprocal basis using the
     physical position.
   * ``STRETCHED_CARTESIAN`` – axis-aligned reciprocal basis scaled by the
     stored per-cell scale factors.
   * ``GENERAL`` (else branch) – assembles the Jacobian matrix
     :math:`\partial(x,y,z)/\partial(\xi,\eta,\mu)` by summing the B-spline
     basis-function derivatives over the ``3x3x3`` node stencil, takes its
     determinant for ``jac``, and inverts it (via cofactors) to obtain the
     reciprocal basis vectors. This is the fully general path used for
     arbitrary user meshes.

   The reciprocal basis is what allows the pusher and current deposit to
   convert between Cartesian particle momenta and contravariant logical
   velocities, ensuring the deposited current is coordinate-consistent.


Deck Region Setters (``vpic.h``)
-----------------------------------

These macros are called from ``begin_initialization`` (or the injection
callbacks) to set field, external-field, and hybrid-fluid quantities over a
region. All of the field setters below were modified for, or newly added to
support, curvilinear meshes: they fetch the per-cell scale factors
:math:`h_1,h_2,h_3` via :c:macro:`GRID_TO_MESH` and scale the user-supplied
(physical) field components into VPIC's internal logical representation.

.. c:macro:: set_point_region_field(rgn, eqn_ex, eqn_ey, eqn_ez, eqn_bx, eqn_by, eqn_bz)

   Strictly evaluates the region and field equations at the Yee-mesh
   locations *inside* the region. Scales the electric field by dividing by
   the scale factors (:math:`e_x \leftarrow e_x/h_1`, etc.) and the magnetic
   field by :math:`c/h`. Sets ``ex/ey/ez`` and ``cbx/cby/cbz``.

.. c:macro:: set_region_field(rgn, eqn_ex, eqn_ey, eqn_ez, eqn_bx, eqn_by, eqn_bz)

   The workhorse field setter. Evaluates the region and field equations at
   the mesh-mapped cell locations (not strictly inside the region) using the
   logical coordinates ``x,y,z`` (which the user interprets in their chosen
   coordinate system, e.g. ``x`` = radius for a cylindrical grid). Converts
   physical components to internal storage:

   * :math:`e_x \leftarrow e_x \cdot h_1`, :math:`e_y \leftarrow e_y \cdot h_2`,
     :math:`e_z \leftarrow e_z \cdot h_3` (covariant E).
   * :math:`cb_x \leftarrow c\,b_x / h_1`, etc. (contravariant B).

.. c:macro:: set_region_field_cart(rgn, eqn_ex, eqn_ey, eqn_ez, eqn_bx, eqn_by, eqn_bz)

   Cartesian-evaluation variant of :c:macro:`set_region_field`. Region
   membership and the field equations are evaluated in **physical (x,y,z)**
   space by mapping each Yee-mesh location through
   :cpp:func:`grid_geom_t::local_to_global_cart`, so ``x,y,z`` in the
   supplied equations are true Cartesian coordinates regardless of the grid
   type. The resulting components are still scaled by :math:`h` for internal
   storage. Use this when it is more natural to specify fields in physical
   space on a curved mesh.

.. c:macro:: set_region_bext(rgn, eqn_bx, eqn_by, eqn_bz)

   Sets the external/guide magnetic field ``cbx0/cby0/cbz0``, dividing each
   component by the corresponding scale factor (and multiplying by ``c``), so
   the external field is stored in the same contravariant logical
   representation as the dynamic B field.

.. c:macro:: set_region_bext_cart(rgn, eqn_bx, eqn_by, eqn_bz)

   Cartesian-evaluation variant of :c:macro:`set_region_bext`. Region tests
   and equations are evaluated in physical space via
   :cpp:func:`grid_geom_t::local_to_global_cart`; results are scaled by
   :math:`h` and written to ``cbx0/cby0/cbz0``.

.. c:macro:: set_region_te(rgn, eqn_te)

   Hybrid model: sets the electron temperature field ``te0`` (scaled by
   ``c``) in the mesh-mapped region.

.. c:macro:: set_region_ue(rgn, eqn_uex)

   Hybrid model: sets the electron fluid velocity component ``ux`` in the
   mesh-mapped region.

.. c:macro:: set_region_ne(rgn, eqn_ne)

   Hybrid model: sets the electron/free-charge density ``rhof`` in the
   mesh-mapped region.

.. c:macro:: set_region_eta_multipliers(rgn, eqn_tcax, eqn_tcay, eqn_tcaz)

   Hybrid model: sets the per-component resistivity/hyper-resistivity
   multiplier fields ``tcax/tcay/tcaz`` (applied to eta, hypereta, and E) at
   cell centers inside the region.

.. c:macro:: set_region_fluid(rgn, name, eqn_den, eqn_tmp, eqn_prs)

   Sets the density, temperature, and pressure (``den``, ``tmp``, ``prs``) of
   the named fluid species over the region, looked up via
   ``find_fluid_species_name``.


Particle Push and Current Deposit
------------------------------------

The particle mover and current-deposit kernels were modified so that current
is deposited as a **contravariant logical current** weighted by the inverse
Jacobian, making deposition coordinate-consistent on curved meshes.

.. c:function:: void interpolate_scale_factors(const k_curvilinear_mesh_t& k_curv, float dx, float dy, float dz, int ii, int nx, int ny, int nz, float& h_xi, float& h_eta, float& h_mu)

   (In ``advance_p.cc``.) Interpolates the three scale factors
   :math:`h_1,h_2,h_3` to a particle's logical position ``(dx,dy,dz)`` in
   voxel ``ii`` using the same quadratic B-spline ``3x3x3`` stencil as
   :cpp:func:`compute_reciprocal_basis`. Reuses
   :cpp:func:`compute_bspline_basis`.

.. cpp:function:: int move_p_kokkos(const particle_view_t& k_particles, const particle_i_view_t& k_particles_i, particle_mover_t* pm, scatter_view_t scatter_view, const grid::grid_geom_t& geom, neighbor_view_t& d_neighbor, int64_t rangel, int64_t rangeh, const float qsp, float gdx, float gdy, float gdz, float gdt, const int nx, const int ny, const int nz)

   (In ``species_advance.h``.) The Kokkos boundary/streak mover, now
   curvilinear-aware. **Signature change:** it now takes a
   :cpp:struct:`grid::grid_geom_t` by reference plus the grid spacings
   ``gdx,gdy,gdz,gdt``. Internally it:

   #. Calls :cpp:func:`compute_reciprocal_basis` at the streak midpoint to
      transform the Cartesian velocity into a contravariant logical velocity
      :math:`(\dot\xi,\dot\eta,\dot\mu)`.
   #. Recomputes the reciprocal basis at the half-step position.
   #. Deposits contravariant logical current into ``jfx/jfy/jfz`` and charge
      into ``rhof`` with a ``0.125 * inv_jac`` weight (the ``1/8`` accounts
      for the logical cell volume of ``[-1,1]^3``) under ``SHAPE_NGP``.

   Boundary crossing, reflection, and range handling are unchanged from the
   Cartesian version.

.. cpp:function:: int move_p_kokkos_host_serial(const particle_view_t& k_particles, const particle_i_view_t& k_particles_i, particle_mover_t* pm, accum_view_t& k_jf_accum, const grid_t* g, neighbor_view_t& d_neighbor, int64_t rangel, int64_t rangeh, const float qsp)

   (In ``species_advance.h``.) Host-serial variant of the mover, rewritten to
   use ``g->geom()`` and :cpp:func:`compute_reciprocal_basis`. Deposits into
   an accumulator view. Supports both the ``SHAPE_NGP`` (contravariant
   logical, inverse-Jacobian weighted) and ``SHAPE_QS`` (quadratic-spline)
   deposit paths.

.. cpp:function:: void advance_p_kokkos_unified(species_t* sp, ..., const grid_t* g, ...)

   (In ``advance_p.cc``.) The portable/vectorized CPU push kernel. Now builds
   ``const grid::grid_geom_t geom = g->geom();`` up front and passes it to
   :cpp:func:`move_p_kokkos` for out-of-bounds particles, so boundary movement
   uses the correct geometry.

.. cpp:function:: void advance_p_kokkos_gpu(species_t* sp, ..., const grid_t* g, ...)

   (In ``advance_p.cc``.) The GPU push kernel, containing the primary
   curvilinear push logic. After the Boris rotation it:

   #. Computes the reciprocal basis at the current position and transforms
      the Cartesian velocity into a contravariant logical velocity.
   #. Forms a **predictor** half-step logical position, and — via the
      "geo interpolation" block — determines whether the predictor crossed
      into a neighbor cell, selecting the neighbor voxel index ``ii_pred``
      and shifting to that cell's local logical coordinates. (A warning is
      emitted if the predictor exceeds the two-ghost-layer coverage.)
   #. Recomputes the reciprocal basis and Jacobian at the predicted half-step
      position and recomputes :math:`(\dot\xi,\dot\eta,\dot\mu)`.
   #. Advances the logical position with the half-step displacements and
      deposits inverse-Jacobian-weighted contravariant logical current into
      ``ii_pred`` (``SHAPE_NGP``) or across the ``SHAPE_QS`` stencil.

   Out-of-bounds particles are handed to :cpp:func:`move_p_kokkos` with the
   ``geom`` snapshot, exactly as in the unified kernel.

   .. note::

      ``advance_p`` currently dispatches to ``advance_p_kokkos_gpu`` for
      **both** the ``USE_GPU`` and CPU builds (the ``advance_p_kokkos_unified``
      path is commented out in the dispatch macro). Both kernels are
      curvilinear-aware; the GPU kernel contains the reference predictor/
      geo-interpolation logic.


Top-Level Driver (``advance_p``)
-----------------------------------

.. cpp:function:: void advance_p(species_t* sp, interpolator_array_t* ia, field_array_t* fa)

   Unchanged public entry point. It validates arguments, computes
   ``qdt_2mc`` (or ``dt_2mc`` under ``VARIABLE_CHARGE``) and the
   ``cdt_d{x,y,z}`` factors, then invokes the selected push kernel
   (:cpp:func:`advance_p_kokkos_gpu`) with the species' Kokkos views, the
   neighbor view, the field array, and the grid. After the push it copies
   ``k_nm`` back to host and mirrors the newly-created movers (and, if
   enabled, their annotations) to host for :cpp:func:`boundary_p`. No
   curvilinear-specific arguments are added at this level — the geometry is
   pulled from ``sp->g`` inside the kernels via :cpp:func:`grid_t::geom`.


Charge/Current Deposit Helpers (``advance_p.cc``)
----------------------------------------------------

These SIMD/team helpers were retained (and made geometry-agnostic) so the
Cartesian fast paths still function. On curvilinear meshes the primary deposit
happens through the inverse-Jacobian-weighted contravariant path inside the
push kernel and :cpp:func:`move_p_kokkos`; these helpers service the
accumulator/scatter bookkeeping.

.. cpp:function:: void accumulate_current(CurrentScatterAccess& current_sa, int ii, int nx, int ny, int nz, float rV, float v0..v11)

   Writes current/charge contributions into either a 12-component accumulator
   (``VPIC_ENABLE_ACCUMULATORS``) or directly into the scatter-view field
   (``jfx/jfy/jfz/rhof``). Handles both ``SHAPE_NGP`` (single-cell) and
   ``SHAPE_QS`` (7-point quadratic-spline stencil) deposits.

.. cpp:function:: void reduce_and_accumulate_current(TeamMember&, CurrentScatterAccess&, int num_iters, int ii, int nx, int ny, int nz, float rV, float* v0..v11)

   Reduces the per-lane/per-thread current contributions before a single
   :cpp:func:`accumulate_current` write, using ``omp simd`` reduction on CPU,
   warp shuffles on CUDA, or Kokkos team reductions otherwise. Invoked when
   :cpp:func:`particles_in_same_cell` reports that an entire team/vector block
   targets the same voxel.

.. cpp:function:: void contribute_current(TeamMember&, field_sa_t&, int i0, int i1, int i2, int i3, field_var j, float v0..v3)

   Four-node variant used by the (currently disabled) team-reduction
   Cartesian deposit path.

.. cpp:function:: int particles_in_same_cell(TeamMember&, IndexView& ii, BoundsView& inbnds, int num_lanes)

   Returns non-zero when every lane/thread in a team is processing a particle
   in the same voxel and with the same in-bounds status, enabling the reduced
   write path above.

.. cpp:function:: void load_interpolators(...)

   Templated (on ``NumLanes``) loader that gathers the interpolator
   coefficients for a block of particles. Two overloads exist, selected by
   ``SHAPE_NGP`` (6 fields: ``ex,ey,ez,cbx,cby,cbz``) or ``SHAPE_QS`` (42
   fields including first/second derivatives). Uses a "same cell" fast path
   plus a vectorized transpose load when ``VPIC_ENABLE_VECTORIZATION`` is set.


Usage Notes and Conventions
==============================

The following consolidates the invariants a deck author or developer must
respect when working with the curvilinear machinery.

Ghost layers and mesh sizing
-------------------------------

* The **grid** (``field``, ``neighbor``) carries **one** ghost layer:
  ``(nx+2)(ny+2)(nz+2)``.
* The **curvilinear mesh** (``k_curvilinear_mesh_*``) carries **two** ghost
  layers per side: ``(nx+4)(ny+4)(nz+4)``. The extra layer supports the
  ``3x3x3`` B-spline stencil at domain-adjacent cells.
* Always translate between the two with :c:macro:`GRID_TO_MESH` (or
  :c:macro:`VOXEL_TO_MESH`); never index one array with the other's linear
  index.
* The mesh is **per-rank and local**: it is sized with the local ``nx,ny,nz``
  and filled using this rank's local bounds (``x0..z1``) and index offset. The
  ``STRETCHED_CARTESIAN`` map is the exception — its physical positions are
  derived from the **global** bounds (``gx0..gz1``) so the stretch is globally
  consistent across ranks.

Field storage conventions
----------------------------

* **E** is stored in **covariant** logical components → multiply by the scale
  factor to store, divide to recover the physical component.
* **B** and **current** are stored in **contravariant** logical components →
  divide by the scale factor (times ``c`` for B) to store, multiply to
  recover.
* The ``set_region_*`` macros perform these conversions for you. If you write
  ``field(i,j,k).*`` directly, you must apply the inverse conversion yourself.
* Scale factors of exactly ``0`` (e.g. on the cylindrical/spherical axis) are
  defensively reset to ``1`` inside the macros to avoid division by zero.

Sign convention caveat
-------------------------

``set_point_region_field`` **divides** E by the scale factors, while
``set_region_field`` **multiplies** E by them (compare the ``_f->ex`` lines in
each macro). These reflect two different internal interpretations; choose the
macro that matches your intended field convention and be consistent.

Particle deposit weighting
----------------------------

* Current/charge is deposited as an **inverse-Jacobian-weighted contravariant
  logical current**. Under ``SHAPE_NGP`` the weight is ``q * 0.125 * inv_jac``
  (the ``1/8`` is the logical cell volume of ``[-1,1]^3``).
* To inject particles **uniformly** on a non-uniform mesh, weight each
  particle by the cell volume — i.e. the Jacobian (``curv_mesh_var::jac``) of
  the injection cell.
* In ``inject_particle``, positions are interpreted in your chosen coordinate
  system, but **velocity components remain Cartesian**.

Predictor / ghost coverage
-----------------------------

The GPU push predicts a half-step position and, if it crosses a cell face,
shifts to the neighbor cell's local logical coordinates (the "geo
interpolation" block). If the predictor lands more than one cell away
(exceeding the two-ghost-layer coverage) a ``WARNING`` is emitted on host
builds. If you see this warning, **reduce the timestep** so that no particle
advances more than one cell per half-step.

Defining a custom mesh
------------------------

To implement a new geometry, follow the pattern of
:cpp:func:`grid_t::init_spherical_grid`:

#. Allocate ``k_curvilinear_mesh_d`` / ``_h`` sized to
   ``(nx+4)(ny+4)(nz+4)``.
#. For every cell (including both ghost layers), fill the scale factors
   (``h_1,h_2,h_3``), the Jacobian (``jac``), the three basis vectors in
   Cartesian components (``e_1_u`` ... ``e_3_w``), and the Cartesian node
   position (``xg,yg,zg``).
#. Set ``type``. If it is not one of the analytic types, leave it as
   ``GENERAL`` so :cpp:func:`compute_reciprocal_basis` and
   :cpp:func:`grid_geom_t::local_to_global_cart` fall back to the B-spline
   path, which reconstructs the reciprocal basis and Jacobian directly from
   the stored ``xg,yg,zg`` — no analytic inverse required.
#. ``deep_copy`` the host mesh to the device.

Because the ``GENERAL`` path derives everything it needs from the stored
Cartesian node positions, you only ever have to provide the *forward* map
(logical → Cartesian) plus the scale factors and Jacobian; the reciprocal
basis is computed for you.
