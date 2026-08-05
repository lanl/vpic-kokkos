===================
HDF5 and Binary I/O
===================

VPIC can write fields, hydro moments, and particles in two formats: a simple,
self-describing **binary** format (always available) and **HDF5** (optional,
enabled with ``VPIC_ENABLE_HDF5``). HDF5 writes additionally emit an ``.xmf``
(XDMF) sidecar so results open directly in ParaView / VisIt.

All writers are members of ``vpic_simulation`` and are intended to be called
from a deck's ``begin_initialization`` or ``begin_diagnostics`` block.

.. contents:: Contents
   :local:
   :depth: 2

.. _build-configuration:

Build Configuration
===================

HDF5 support is off by default. Relevant CMake options:

+--------------------------------+---------+------------------------------------------------+
| Option                         | Default | Effect                                         |
+================================+=========+================================================+
| ``VPIC_ENABLE_HDF5``           | OFF     | Compile the HDF5 writers.                      |
+--------------------------------+---------+------------------------------------------------+
| ``VPIC_HDF5_SERIAL_ONLY``      | OFF     | Force serial HDF5 (FPP-only, no collective).   |
+--------------------------------+---------+------------------------------------------------+
| ``VPIC_ENABLE_HDF5_ASYNC``     | OFF     | Experimental HDF5 async VOL support.           |
+--------------------------------+---------+------------------------------------------------+

Three compile-time states result:

1. **Parallel HDF5** (``VPIC_ENABLE_HDF5=ON``, parallel lib found):
   both M2M and M2O modes available; ``VPIC_HDF5_PARALLEL`` is defined.
2. **Serial HDF5** (``VPIC_ENABLE_HDF5=ON`` + ``VPIC_HDF5_SERIAL_ONLY=ON``,
   or only a serial lib found): FPP (file-per-process) only. Passing
   ``single_file=true`` logs a warning and is silently downgraded to M2M.
3. **HDF5 disabled** (default): the HDF5 entry points are compiled as stubs
   that ``ERROR`` if called.

.. note::

   The build interrogates ``h5pcc`` / ``h5cc`` (via ``-show``) to resolve the
   real library paths and bake an RPATH, falling back to CMake's discovered
   libraries or bare ``-lhdf5_hl -lhdf5`` names if that fails.

Write API
=========

VPIC exposes six public write entry points on ``vpic_simulation`` (callable
directly from a deck's ``begin_initialization`` or ``begin_diagnostics`` block), plus one helper for
building time-series master files. They come in matched binary and
HDF5 pairs for fields, hydro, and particles.

Binary Writers
--------------

Binary writers are always **file-per-process (M2M)** — every rank writes its
own file — and do not require any special build flags. They are available in
every VPIC build.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``write_fields_binary(DumpParameters& params, field_array_t* fa)``
     - Writes selected E/B (and optional auxiliary/material) field components.
       Copies device→host if the field array is stale.
   * - ``write_hydro_binary(DumpParameters& params, hydro_array_t* ha, const char* sp_name)``
     - Accumulates hydro moments for species ``sp_name`` from the particle
       distribution, then writes the selected moments. **Side effect:** clears
       and refills the global ``hydro_array``.
   * - ``write_particles_binary(const char* fbase, const char* species_name, bool compute_physical_position)``
     - Writes all particles of ``species_name``. See
       :ref:`particle position modes <particle-position-modes>` for the meaning
       of ``compute_physical_position``.

HDF5 Writers
------------

HDF5 writers add a ``single_file`` argument selecting the file layout:

- ``single_file = true`` → **many-to-one (M2O)**: all ranks write into one
  shared ``.h5`` file via collective MPI-IO. Requires a **parallel** HDF5 build.
- ``single_file = false`` → **many-to-many (M2M)**: each rank writes its own
  ``.h5`` file.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``write_fields_hdf5(DumpParameters& params, field_array_t* fa, bool single_file)``
     - Repacks selected field components (AoS→SoA) and writes one 3-D dataset
       per component, plus an ``.xmf`` sidecar for ParaView.
   * - ``write_hydro_hdf5(DumpParameters& params, hydro_array_t* ha, const char* sp_name, bool single_file)``
     - Accumulates moments for ``sp_name`` (same side effect as the binary
       hydro writer), repacks, and writes one dataset per moment plus an
       ``.xmf`` sidecar.
   * - ``write_particles_hdf5(const char* fbase, const char* species_name, bool single_file, bool compute_physical_position)``
     - Writes all particles of ``species_name`` as a single compound dataset
       plus an ``.xmf`` sidecar.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``write_xdmf_timeseries(const char* series_path, const std::vector<int>& steps, const char* base_pattern, const char* base_filename)``
     - **Rank 0 only.** Emits a temporal ``Collection`` master ``.xmf`` that
       ``xi:include``\ s each step's per-step ``.xmf``, giving ParaView a single
       animatable file. ``base_pattern`` is a directory pattern with a ``%d``
       step placeholder (e.g. ``"field/T.%d"``).

.. note::

   All four HDF5 entry points (including ``write_xdmf_timeseries``) exist in
   every build, but their behavior depends on how HDF5 was configured (see
   :ref:`Build Configuration <build-configuration>`):

   - **Parallel HDF5** — full M2O and M2M support.
   - **Serial HDF5** (``VPIC_HDF5_SERIAL_ONLY=ON``) — M2O is unavailable; a call
     with ``single_file=true`` logs a warning on rank 0 and is **silently
     downgraded to M2M**. ``write_xdmf_timeseries`` additionally warns that the
     per-rank files may need extra ParaView setup.
   - **HDF5 disabled** — every HDF5 entry point is a stub that calls ``ERROR``
     ("...requires HDF5 support. Rebuild with ``VPIC_ENABLE_HDF5=ON``.").

Many-to-One vs. Many-to-Many
----------------------------

The M2O/M2M distinction is the single most important operational choice for
HDF5 output:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - M2O (``single_file=true``)
     - M2M (``single_file=false``)
   * - Files produced
     - One shared file for the whole run
     - One file per MPI rank
   * - HDF5 requirement
     - Parallel HDF5 (MPI-IO)
     - Any HDF5 (parallel or serial)
   * - Global picture
     - Complete global grid in one file; ranks write hyperslabs of their
       subdomain
     - Each file holds only that rank's local subdomain
   * - Best for
     - Post-processing / visualization of the full domain, fewer inodes on the
       filesystem
     - Maximum write scalability, serial-HDF5 builds, per-rank debugging

Binary writers are always M2M and share this per-rank file model.

Common Concepts
---------------

**Variable selection.** ``write_fields_*`` and ``write_hydro_*`` choose which
components to emit from ``params.output_vars`` (a ``BitField``). The convenient
named masks (``electric``, ``magnetic``, ``current_density``,
``charge_density``, …) are described in
:ref:`Field, Hydro & Particle Write Details <write-details>`. Only the requested
components are written, so file size scales with the selection.

.. _particle-position-modes:

**Particle position modes.** Both particle writers take
``compute_physical_position``:

- ``true`` (*physical*) — converts each particle's cell-local offset to a global
  ``(x, y, z)`` coordinate and drops the voxel index. Record is 7 values
  (28 bytes); ideal for post-processing and visualization.
- ``false`` (*logical*) — writes the raw ``particle_t`` (``dx, dy, dz, i, ux,
  uy, uz, w``, 32 bytes), preserving exact simulation state for restart/debug.

**Time-centering.** Both particle writers run ``center_p_dump`` per chunk, which
time-centers velocities and removes ghost particles (adjusting the valid count)
before writing.

.. _write-details:

Field, Hydro & Particle Write Details
=====================================

The field and hydro writers (binary and HDF5 alike) select components through a
``DumpParameters`` object. The idiom used in virtually every deck is:

.. code-block:: cpp

   DumpParameters fdParams;
   sprintf(fdParams.baseDir,      "field");
   sprintf(fdParams.baseFileName, "fields");
   fdParams.output_vars = BitField(electric | magnetic);   // choose components

   write_fields_hdf5(fdParams, field_array, /*single_file=*/true);

``output_vars`` is a ``BitField`` built by OR-ing together the named masks below.
Each mask expands to one or more component bits; only the requested components
are packed and written, so the file size scales with the selection. The same
``DumpParameters`` mask drives **both** the binary and HDF5 writers — they share
an identical component table.

.. note::

   Always prefer the **named masks** (``electric``, ``magnetic``,
   ``current_density``, …) over hand-rolling raw bit indices. The masks are the
   documented, deck-facing API defined in ``vpic.h``; the bit indices are an
   implementation detail that the per-component reference table at the end of
   this section exposes only for advanced use.


Field Writer Variables
----------------------

Selected via the field masks from ``dump.h``. Bits map directly to members of
``field_t``; HDF5 names each component's dataset as shown, and the binary writer
emits the same components in ascending bit order.

.. list-table::
   :header-rows: 1
   :widths: 20 34 26 20

   * - Mask
     - Components (HDF5 dataset names)
     - Bits
     - Type
   * - ``electric``
     - ``ex``, ``ey``, ``ez``
     - 0–2
     - float
   * - ``div_e_err``
     - ``div_e_err``
     - 3
     - float
   * - ``magnetic``
     - ``cbx``, ``cby``, ``cbz``
     - 4–6
     - float
   * - ``div_b_err``
     - ``div_b_err``
     - 7
     - float
   * - ``tca``
     - ``tcax``, ``tcay``, ``tcaz``
     - 8–10
     - float
   * - ``rhob``
     - ``rhob``
     - 11
     - float
   * - ``current``
     - ``jfx``, ``jfy``, ``jfz``
     - 12–14
     - float
   * - ``rhof``
     - ``rhof``
     - 15
     - float
   * - ``emat``
     - ``ematx``, ``ematy``, ``ematz``
     - 16–18
     - material id → float
   * - ``nmat``
     - ``nmat``
     - 19
     - material id → float
   * - ``fmat``
     - ``fmatx``, ``fmaty``, ``fmatz``
     - 20–22
     - material id → float
   * - ``cmat``
     - ``cmat``
     - 23
     - material id → float

.. note::

   **Material-ID widening.** The ``emat``/``nmat``/``fmat``/``cmat`` masks select
   ``material_id`` integer members. Both writers widen these to ``float`` on
   output (HDF5 stores them as ``H5T_NATIVE_FLOAT``; binary stores a 4-byte
   float), so every field component is a uniform 4-byte float on disk regardless
   of its in-memory type. This keeps the file format homogeneous and directly
   consumable by the generated XDMF (which declares all attributes as
   ``Float``/``Precision=4``).

The common deck selection ``BitField(electric | magnetic)`` therefore writes six
datasets — ``ex, ey, ez, cbx, cby, cbz`` — as demonstrated in the 4-rank
verification deck.


Hydro Writer Variables
----------------------

Selected via the hydro masks from ``dump.h``. Bits map to members of
``hydro_t``; datasets are named as shown. **All moments are float.**

.. list-table::
   :header-rows: 1
   :widths: 24 44 20

   * - Mask
     - Components (HDF5 dataset names)
     - Bits
   * - ``current_density``
     - ``jx``, ``jy``, ``jz``
     - 0–2
   * - ``charge_density``
     - ``rho``
     - 3
   * - ``momentum_density``
     - ``px``, ``py``, ``pz``
     - 4–6
   * - ``ke_density``
     - ``ke``
     - 7
   * - ``stress_tensor``
     - ``txx``, ``tyy``, ``tzz``, ``tyz``, ``tzx``, ``txy``
     - 8–13

.. warning::

   **The hydro writers accumulate moments as a side effect.** Both
   ``write_hydro_hdf5`` and ``write_hydro_binary`` internally clear the global
   ``hydro_array``, re-accumulate the requested species' moments from the
   current particle distribution (``accumulate_hydro_p_kokkos``), and
   ``synchronize_hydro_array`` across MPI boundaries **before** writing. Two
   consequences:

   - The ``hydro_array`` you pass in is overwritten; do not rely on its prior
     contents afterward.
   - Calling a hydro writer twice in one diagnostics block (e.g. M2M then M2O)
     re-runs the full accumulation each time — this is intentional but is the
     dominant cost of a hydro dump (it shows up as the ``Setup`` phase in the
     ``[DIAGNOSTIC]`` line), unlike field dumps where the data already exists.

The common deck selection ``BitField(current_density | charge_density)`` writes
four datasets — ``jx, jy, jz, rho``.


Particle Writer Details
-----------------------

Particle writers do not use ``DumpParameters``; they take a base filename and a
species name directly. The single behavioral knob is
``compute_physical_position`` (see
:ref:`particle position modes <particle-position-modes>` in the Write API):

- **Physical mode** (``true``) — 7 floats per particle (``x, y, z, ux, uy, uz,
  w``, 28 bytes); HDF5 uses a named compound type of the same seven fields.
- **Logical mode** (``false``) — the raw ``particle_t`` (``dx, dy, dz, i, ux,
  uy, uz, w``, 32 bytes), preserving exact simulation state.

Both modes run ``center_p_dump`` per chunk, which time-centers velocities and
strips ghost particles (adjusting the written count accordingly). Only physical
mode produces a directly ParaView-visualizable point cloud; logical-mode XDMF
emits a placeholder geometry with a warning comment.


Per-Component Reference (Advanced)
----------------------------------

The masks above are unions of the individual component bits. If you need a
single component you can OR its bit directly, but this is rarely necessary. The
full field bit layout (0–23) is given in the *Field Writer Variables* table's
"Bits" column; the hydro bit layout (0–13) is given in the *Hydro Writer
Variables* table's "Bits" column. Both tables map 1:1 to the ``field_map`` /
``hydro_map`` tables in ``write.cc``, so the on-disk order for any selection is
simply ascending bit order.

Example of selecting a single component by bit:

.. code-block:: cpp

   // Write only jfx (bit 12) — advanced, rarely needed
   fdParams.output_vars = BitField(1 << 12);

.. note::

   The bit indices are stable and match the writer tables exactly, but they are
   **not** guaranteed to match the legacy VPIC ``dump_fields``/``hydro`` ASCII
   header ordering. If you are migrating post-processing scripts from the legacy
   binary dumps, verify against the dataset names emitted in the HDF5 file (or
   the ascending-bit ordering in the binary file) rather than assuming the old
   index scheme.

Binary Format
=============

Every binary file begins with a fixed 64-byte header followed by tightly
packed ``float`` data (field/hydro) or particle records.

Header layout
-------------

::

    struct BinaryHeader {
        int32_t magic;       // 0xBEEF0002
        int32_t version;     // 1
        int32_t step;        // simulation timestep
        int32_t nx, ny, nz;  // local grid dims (cells per rank)
        float   dt;          // timestep size
        float   dx, dy, dz;  // cell dimensions
        float   x0, y0, z0;  // origin (0 for field/hydro M2M; real for particles)
        float   q_m;         // charge/mass (species; 0 for fields)
        int32_t num_vars;    // number of variables (field/hydro) or 7/8 (particles)
        int32_t var_mask;    // reserved (0) for field/hydro; particle COUNT for particles
    };

.. note::

   The final ``int32_t`` is overloaded by purpose: it is a reserved
   ``var_mask`` (written as ``0``) for field and hydro writes, but carries the
   per-rank particle **count** in particle writes.

Data ordering
-------------

Grid data is written **one full component at a time**, each component packed in
column-major ``(k, j, i)`` order over the interior cells
(``1 <= i <= nx`` etc.). That is, all of ``ex`` for every cell, then all of
``ey``, and so on — SoA on disk, not interleaved.

HDF5 Layout and Metadata
========================

Grid data (fields/hydro) is written as one **3-D dataset per component**, named
after the component (``ex``, ``cbz``, ``jx``, ...), in ``[nz, ny, nx]`` order
(Z slowest, X fastest). This is an AoS→SoA transformation done in a single
cache-friendly ``MDRangePolicy`` repack kernel into a persistent Kokkos host
buffer before writing.

Two scalar attributes are attached to the file root:

+-----------+------------------+----------------------------+
| Attribute | HDF5 type        | Value                      |
+===========+==================+============================+
| ``step``  | ``NATIVE_LONG``  | ``g->step``                |
+-----------+------------------+----------------------------+
| ``time``  | ``NATIVE_DOUBLE``| ``g->t0``                  |
+-----------+------------------+----------------------------+

Particles are written as a single 1-D dataset named ``particles`` of the
compound type described above.

File naming
-----------

+------------+---------+-------------------------------------------------------+
| Data       | Mode    | Pattern                                               |
+============+=========+=======================================================+
| Fields     | M2O     | ``<baseDir>/T.<step>/<base>.<step>.h5``               |
+------------+---------+-------------------------------------------------------+
| Fields     | M2M     | ``<baseDir>/T.<step>/<base>.<step>.<rank>.h5``        |
+------------+---------+-------------------------------------------------------+
| Hydro      | M2O     | ``<baseDir>/T.<step>/<base>.<step>.h5``               |
+------------+---------+-------------------------------------------------------+
| Hydro      | M2M     | ``<baseDir>/T.<step>/<base>.<step>.<rank>.h5``        |
+------------+---------+-------------------------------------------------------+
| Particles  | M2O     | ``<fbase>.<species>.<step>.h5``                       |
+------------+---------+-------------------------------------------------------+
| Particles  | M2M     | ``<fbase>.<species>.<rank>.<step>.h5``                |
+------------+---------+-------------------------------------------------------+


HDF5 Performance Optimizations
==============================

The parallel FAPL (``create_optimized_fapl``) applies:

1. **MPI-IO driver** (``H5Pset_fapl_mpio``) for M2O.
2. **Collective metadata** ops/writes (HDF5 >= 1.10).
3. **16 MB alignment** (``H5Pset_alignment(fapl, 4096, 16*1024*1024)``) tuned
   for Lustre stripe sizes.
4. **Deferred metadata-cache flushes** (evictions disabled, incr/decr modes
   off) to batch B-tree updates until close.

M2O writes use ``H5FD_MPIO_COLLECTIVE`` on the transfer property list. The
serial-only FAPL (``create_serial_fapl``) applies the same alignment and cache
tuning but omits all MPI-IO options.

Each writer emits two log lines from rank 0::

    [METRIC],<tag>,<step>,<time>,<MB>,<MB/s>
    [DIAGNOSTIC],<tag>,<step>,Total:..,Setup:..,MetaCreate:..,BufPack:..,Compute:..,H5Dwrite:..,MetaClose:..

The ``TIME_PURE_IO_ONLY`` compile-time flag (default ``true``) makes ``[METRIC]``
report only ``H5Dwrite`` time rather than total wall time — set it ``false`` for
end-to-end timing.


XDMF / ParaView Support
=======================

Every HDF5 write also writes an ``.xmf`` sidecar (rank 0 only) so the data opens
directly in ParaView / VisIt:

- **Fields/hydro:** ``write_xdmf_structured`` emits a ``3DCoRectMesh`` with
  ``ORIGIN_DXDYDZ`` geometry and one cell-centered ``Scalar`` attribute per
  component. Grid extents come from MPI reductions in M2O, or the local grid in
  M2M. **All ranks** participate in computing the grid info (collective), but
  **only rank 0** writes the file.
- **Particles:** ``write_xdmf_particles`` emits a ``Polyvertex`` topology. In
  physical mode it maps ``x,y,z`` as ``XYZ`` geometry and exposes
  ``ux,uy,uz,w`` via hyperslabs into the compound dataset. In logical mode it
  emits only a placeholder geometry plus a warning comment (cell-local coords
  can't be visualized directly).
- **Time series:** ``write_xdmf_timeseries`` writes a temporal ``Collection``
  that ``xi:include``\ s the per-step ``.xmf`` files, giving ParaView a single
  animatable master file.

.. warning::

   In logical particle mode the XDMF geometry is a placeholder — re-run with
   ``compute_physical_position=true`` for meaningful ParaView visualization.


Usage Example
=============

From a deck's ``begin_diagnostics`` block::

    DumpParameters fdParams;
    sprintf(fdParams.baseDir, "field");
    sprintf(fdParams.baseFileName, "fields");
    fdParams.output_vars = BitField(electric | magnetic);

    // HDF5, single shared file (M2O)
    write_fields_hdf5(fdParams, field_array, true);

    // HDF5, file-per-process (M2M)
    write_fields_hdf5(fdParams, field_array, false);

    // Binary (always M2M)
    write_fields_binary(fdParams, field_array);

    // Hydro (accumulates "electron" moments internally)
    DumpParameters hedParams;
    sprintf(hedParams.baseDir, "ehydro");
    sprintf(hedParams.baseFileName, "ehydro");
    hedParams.output_vars = BitField(current_density | charge_density);

    // Hydro to HDF5 (M2O) — accumulates "electron" moments internally
    write_hydro_hdf5(hedParams, hydro_array, "electron", true);

    // Hydro to binary (always M2M)
    write_hydro_binary(hedParams, hydro_array, "electron");

    // Particles: physical coords, single shared file
    write_particles_hdf5("particle/parts", "electron", true, true);

    // Particles: logical coords, file-per-process
    write_particles_binary("particle/parts", "electron", false);

.. note::

   For a complete, runnable HDF5 verification deck (4 ranks, 2x2x1 topology,
   deterministic field/particle patterns, both M2M and M2O), see
   ``hdf5/write_correctness_4rank.cxx`` in the test suite. It exercises every
   HDF5 writer and halts after a single step.


Time-Series Master File
=======================

After Writing several steps, build a single animatable ParaView/VisIt master
with ``write_xdmf_timeseries`` (rank 0 only)::

    std::vector<int> steps = {0, 100, 200, 300};
    write_xdmf_timeseries("fields_series.xmf", steps, "field/T.%d", "fields");

This emits a temporal ``Collection`` that ``xi:include``\ s each step's
per-step ``.xmf``. Opening ``fields_series.xmf`` gives a time-navigable view.

.. warning::

   Under serial HDF5 the per-step files are per-rank, so the generated
   time-series references only rank 0's files and logs a warning. Full-domain
   time-series visualization requires the parallel (M2O) build.


Reading the Output
==================

**HDF5** — inspect and load with the standard tools::

    h5ls -r fields.100.h5           # list datasets/attributes
    h5dump -d /ex fields.100.h5     # dump one component

In Python::

    import h5py
    with h5py.File("fields.100.h5", "r") as f:
        ex   = f["ex"][:]           # shape (nz, ny, nx)
        step = f.attrs["step"]
        time = f.attrs["time"]

**Binary** — read the 64-byte header, then the packed component arrays::

    import numpy as np

    hdr_dt = np.dtype([
        ("magic","<i4"),("version","<i4"),("step","<i4"),
        ("nx","<i4"),("ny","<i4"),("nz","<i4"),
        ("dt","<f4"),("dx","<f4"),("dy","<f4"),("dz","<f4"),
        ("x0","<f4"),("y0","<f4"),("z0","<f4"),
        ("q_m","<f4"),("num_vars","<i4"),("var_mask","<i4"),
    ])

    with open("fields.100.0", "rb") as fp:
        h = np.fromfile(fp, dtype=hdr_dt, count=1)[0]
        assert h["magic"] == np.int32(0xBEEF0002)
        ncell = h["nx"] * h["ny"] * h["nz"]
        # Components follow, one full array at a time, in bit order.
        comps = [np.fromfile(fp, dtype="<f4", count=ncell)
                     .reshape(h["nz"], h["ny"], h["nx"])
                 for _ in range(h["num_vars"])]

.. note::

   The binary format is positional: to know *which* components ``comps[0]``,
   ``comps[1]``, ... correspond to, apply the same ``output_vars`` bit ordering
   used at write time (see the field/hydro variable tables above). Only set
   bits are present, in ascending bit order.


Troubleshooting
===============

+-------------------------------------------+-------------------------------------------------+
| Symptom                                   | Likely cause / fix                              |
+===========================================+=================================================+
| ``... requires HDF5 support`` ERROR       | Built without HDF5. Rebuild with                |
|                                           | ``-DVPIC_ENABLE_HDF5=ON``.                      |
+-------------------------------------------+-------------------------------------------------+
| ``single_file=true`` silently becomes M2M | Serial HDF5 build. Rebuild against a parallel   |
| (warning logged)                          | HDF5 (drop ``VPIC_HDF5_SERIAL_ONLY``).          |
+-------------------------------------------+-------------------------------------------------+
| Link errors for ``H5*`` symbols           | ``h5pcc``/``h5cc`` not on ``PATH`` at configure |
|                                           | time; point CMake at your HDF5 install.         |
+-------------------------------------------+-------------------------------------------------+
| ParaView shows particles at origin        | Logical particle mode. Re-write with            |
|                                           | ``compute_physical_position=true``.             |
+-------------------------------------------+-------------------------------------------------+
| Wrong global extents in M2O ``.xmf``      | Grid ``x1/y1/z1`` bounds inconsistent across    |
|                                           | ranks; extents come from MPI min/max reductions.|
+-------------------------------------------+-------------------------------------------------+