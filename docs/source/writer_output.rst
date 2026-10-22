===============================
Diagnostic writers and ParaView
===============================

The ``write_*`` methods on ``vpic_simulation`` export fields, hydro moments,
and particles from a simulation deck. Binary output is always available;
HDF5 is optional and disabled by default. Fields, hydro, and physical particle
HDF5 output include XDMF metadata (``.xmf`` files) for built-in ParaView readers.
Logical particle output does not generate XDMF.

This page describes the writers in `write.cc <../../src/vpic/write.cc>`_,
not the older ``dump_*`` routines or restart files. The public declarations
and selection masks are in `vpic.h <../../src/vpic/vpic.h>`_.

.. contents:: Contents
   :local:
   :depth: 2

Build configuration
===================

.. list-table:: CMake options
   :header-rows: 1

   * - Option
     - Default
     - Effect
   * - ``VPIC_ENABLE_HDF5``
     - ``OFF``
     - Enable the HDF5 writers and XDMF helper.
   * - ``VPIC_HDF5_SERIAL_ONLY``
     - ``OFF``
     - Restrict HDF5 output to one file per rank, even if parallel support
       is available in the selected library.
   * - ``VPIC_ENABLE_HDF5_ASYNC``
     - ``OFF``
     - Experimental option; the writers described here use synchronous HDF5
       calls. This option does not make these writes asynchronous.

With HDF5 enabled, CMake requires its C and high-level libraries. Unless forced
to serial mode, it prefers parallel HDF5 and enables shared-file output when
``HDF5_IS_PARALLEL`` is true. Automatically detected serial HDF5 supports
per-rank output. Serial mode rejects shared-file requests with an error before
creating output directories or files; it does not silently change the layout.

With HDF5 disabled, the HDF5 methods and ``write_xdmf_timeseries`` are not
declared. Guard those deck calls with ``#ifdef VPIC_ENABLE_HDF5`` or use the
binary methods.

Use a separate build directory for each configuration. For example, from the
checkout root, configure with the architecture script, set the HDF5 options,
build VPIC, and then compile a deck with the generated wrapper::

    mkdir build-hdf5
    cd build-hdf5
    ../arch/cpu-Release
    cmake -DVPIC_ENABLE_HDF5=ON -DVPIC_HDF5_SERIAL_ONLY=OFF .
    make -j4
    ./bin/vpic /absolute/path/to/my_deck.cxx
    mpirun -n 4 ./my_deck.Linux

Adapt the architecture script to the machine, including its Kokkos architecture
selection. The script does not forward extra command-line arguments to CMake;
the second configure command above sets the additional options. Use
``VPIC_ENABLE_HDF5=OFF`` for binary-only builds, or set
``VPIC_HDF5_SERIAL_ONLY=ON`` for forced serial mode.

Keep the HDF5 headers, libraries, and compiler wrappers consistent. The build
consults ``h5pcc`` (or ``h5cc`` in forced serial mode) when resolving link
libraries, with CMake-discovered libraries as a fallback. Review the configure
output when several HDF5 installations are present.

Calling the writers
===================

Call writers from the deck's initialization or diagnostics blocks after the
required grid, species, and arrays have been initialized. All simulation ranks
must enter the six data writers in the same order, including per-rank binary
and HDF5 output. A rank-0-only call to a data writer can hang because these
methods also perform MPI coordination.

Before creating output, ranks check agreement on the writer, output path,
step/time, species and charge/mass where applicable, layout, and particle
coordinate mode. Shared output also checks grid spacing and, for structured
data, the component selection. Per-rank structured files may select different
components. These checks cannot compensate for a rank that never enters the
writer. Unsupported modes and detected I/O failures abort the MPI job.

API reference
-------------

.. code-block:: cpp

   // Always available; one file per rank.
   void write_fields_binary(DumpParameters& params, field_array_t* fa);
   void write_hydro_binary(DumpParameters& params, hydro_array_t* ha,
                           const char* species_name);
   void write_particles_binary(const char* fbase, const char* species_name,
                               ParticleWriteOptions options = {});

   // Available when VPIC_ENABLE_HDF5 is defined.
   void write_fields_hdf5(DumpParameters& params, field_array_t* fa,
                         bool single_file = false);
   void write_hydro_hdf5(DumpParameters& params, hydro_array_t* ha,
                        const char* species_name, bool single_file = false);
   void write_particles_hdf5(const char* fbase, const char* species_name,
                             ParticleWriteOptions options = {});
   void write_xdmf_timeseries(const char* series_path,
                             const std::vector<int>& steps,
                             const char* base_pattern,
                             const char* base_filename);

For fields and hydro, ``single_file=false`` writes one file per rank
(many-to-many, M2M); ``true`` writes one shared file per snapshot
(many-to-one, M2O). Shared files require parallel HDF5. Serial HDF5 can still
write per-rank files from a multi-rank simulation.

Particle writers use named options with the same meanings in both formats:

.. code-block:: cpp

   enum class ParticleCoordinates { Logical, Physical };
   enum class OutputLayout { PerRank, Shared };
   struct ParticleWriteOptions {
       ParticleCoordinates coordinates = ParticleCoordinates::Logical;
       OutputLayout layout = OutputLayout::PerRank;
   };

Binary particle output requires ``PerRank``. The historical boolean overloads
are deprecated but retain their argument meanings: the binary third argument
is ``physical``; the HDF5 third argument is ``single_file`` and its fourth is
``physical`` (default ``false``). Prefer named options in new decks.

Deck example
------------

This example uses per-rank output, so its HDF5 calls work with either serial
or parallel libraries. Use distinct bases for different species or particle
coordinate modes.

.. code-block:: cpp

   DumpParameters fields{};
   snprintf(fields.baseDir, sizeof(fields.baseDir), "field");
   snprintf(fields.baseFileName, sizeof(fields.baseFileName), "fields");
   fields.output_vars = BitField(electric | magnetic);
   write_fields_binary(fields, field_array);

   DumpParameters moments{};
   snprintf(moments.baseDir, sizeof(moments.baseDir), "ehydro");
   snprintf(moments.baseFileName, sizeof(moments.baseFileName), "ehydro");
   moments.output_vars = BitField(current_density | charge_density);

   ParticleWriteOptions physical{
       ParticleCoordinates::Physical, OutputLayout::PerRank};
   write_particles_binary("particle/physical", "electron", physical);

   #ifdef VPIC_ENABLE_HDF5
   write_fields_hdf5(fields, field_array);
   write_hydro_hdf5(moments, hydro_array, "electron");
   write_particles_hdf5("particle/physical", "electron", physical);
   write_particles_hdf5("particle/logical", "electron"); // Default options
   #endif

For shared particle output, pass
``{ParticleCoordinates::Physical, OutputLayout::Shared}``. For shared field
or hydro output, pass ``true`` as the final argument.

Field and hydro data
====================

Export ranges and geometry
--------------------------

Every selected component exports storage indices ``1..nx, 1..ny, 1..nz``.
The local cell counts ``nx, ny, nz`` exclude ghosts; allocated storage includes
the surrounding layer and has dimensions ``nx+2, ny+2, nz+2``. Writers uniformly
exclude that layer, including physical boundary samples stored there. They do
not add an upper boundary plane for staggered field components.

Retained field values are samples on the Yee mesh, not uniformly cell-centered
quantities. Each field component gets its own XDMF sample mesh, with
``Center="Node"`` and the appropriate spatial offset. Hydro moments use nodal
samples with zero offset. Writers do not interpolate values to a common mesh.

For a component's zero-based output index ``(i,j,k)``, the physical position is
``origin + (i,j,k) * spacing``, where ``origin`` is that component's first
retained sample. Offsets from the domain's lower corner, in cell widths, are:

.. list-table:: Sample offsets (XYZ)
   :header-rows: 1

   * - Components
     - Offset
   * - ``ex, jfx, tcax, ematx``
     - ``(0.5, 0, 0)``
   * - ``ey, jfy, tcay, ematy``
     - ``(0, 0.5, 0)``
   * - ``ez, jfz, tcaz, ematz``
     - ``(0, 0, 0.5)``
   * - ``cbx, fmatx``
     - ``(0, 0.5, 0.5)``
   * - ``cby, fmaty``
     - ``(0.5, 0, 0.5)``
   * - ``cbz, fmatz``
     - ``(0.5, 0.5, 0)``
   * - ``div_b_err, cmat``
     - ``(0.5, 0.5, 0.5)``
   * - ``div_e_err, rhob, rhof, nmat`` and all hydro moments
     - ``(0, 0, 0)``

Field writers refresh the host field array when its copy is stale. Hydro
writers **clear and refill the supplied hydro array**, accumulating the chosen
species' moments and synchronizing across rank boundaries before writing.
Previous contents of that array are overwritten. Each hydro writer call repeats
the accumulation, including when writing the same species in two formats.
Supplied arrays must use the simulation grid; hydro also checks compatibility
with the species and interpolator.

Variable selection
------------------

Set ``DumpParameters.output_vars`` with the named masks below. Binary and HDF5
use the same component ordering, in ascending bit order. Components are float32
on disk; material IDs are converted to floats.

These writers use ``baseDir``, ``baseFileName``, and ``output_vars``.
They do **not** apply ``stride_x``, ``stride_y``, ``stride_z``, or ``format``:
output remains full resolution with one array per selected component. The
format is chosen by the called method. ``baseDir`` and ``baseFileName`` each
have 128-byte storage in ``DumpParameters``; constructed output paths use
dynamic strings, subject to filesystem limits.

.. list-table:: Field masks
   :header-rows: 1

   * - Mask
     - Dataset names, in bit order
     - Bits
   * - ``electric``
     - ``ex, ey, ez``
     - 0-2
   * - ``div_e_err``
     - ``div_e_err``
     - 3
   * - ``magnetic``
     - ``cbx, cby, cbz``
     - 4-6
   * - ``div_b_err``
     - ``div_b_err``
     - 7
   * - ``tca``
     - ``tcax, tcay, tcaz``
     - 8-10
   * - ``rhob``
     - ``rhob``
     - 11
   * - ``current``
     - ``jfx, jfy, jfz``
     - 12-14
   * - ``rhof``
     - ``rhof``
     - 15
   * - ``emat``
     - ``ematx, ematy, ematz``
     - 16-18
   * - ``nmat``
     - ``nmat``
     - 19
   * - ``fmat``
     - ``fmatx, fmaty, fmatz``
     - 20-22
   * - ``cmat``
     - ``cmat``
     - 23

.. list-table:: Hydro masks
   :header-rows: 1

   * - Mask
     - Dataset names, in bit order
     - Bits
   * - ``current_density``
     - ``jx, jy, jz``
     - 0-2
   * - ``charge_density``
     - ``rho``
     - 3
   * - ``momentum_density``
     - ``px, py, pz``
     - 4-6
   * - ``ke_density``
     - ``ke``
     - 7
   * - ``stress_tensor``
     - ``txx, tyy, tzz, tyz, tzx, txy``
     - 8-13

For a single component, select its bit, for example ``BitField(1u << 12)``
for ``jfx``. Do not assume these indices match the older dump formats.

Particle data
=============

Both formats write all ``sp->np`` particles of the selected species. They
refresh the host particles and interpolator, then call ``center_p_dump`` on
temporary particle copies to center output momenta. This preserves VPIC's PIC
staggering and does not change live particle momenta. ``center_p_dump`` does
not filter particles or change the output count. The structured-array ghost
exclusion policy is not a separate particle filtering operation.

Logical mode preserves cell-local offsets and voxel indices; physical mode
converts these to global XYZ positions and omits the voxel index. Both modes
contain centered diagnostic momenta, so neither is a restart image.

HDF5 particle schema
--------------------

Schema version 2 stores aligned numeric arrays. Index ``i`` belongs to the
same particle in every dataset within one snapshot. Sorting and migration can
change ordering between snapshots; array indices are not persistent particle
identities. Shared files concatenate ranks' particles in rank order.

.. list-table:: Particle datasets (N particles in the file)
   :header-rows: 1

   * - Dataset
     - Physical mode
     - Logical mode
     - Stored type
   * - ``/position``
     - ``[N,3]`` global x, y, z
     - Absent
     - Little-endian float32
   * - ``/cell_offset``
     - Absent
     - ``[N,3]`` local dx, dy, dz
     - Little-endian float32
   * - ``/cell_index``
     - Absent
     - ``[N]`` local voxel indices
     - Little-endian int32
   * - ``/momentum``
     - ``[N,3]`` ux, uy, uz
     - ``[N,3]`` ux, uy, uz
     - Little-endian float32
   * - ``/weight``
     - ``[N]`` weights
     - ``[N]`` weights
     - Little-endian float32

The particle payload occupies **28 bytes per particle in physical mode** and
**32 bytes per particle in logical mode**, excluding file and grid metadata.
Each value is stored once. There is no ``/particles`` compound payload or
duplicate visualization dataset. XDMF references the physical arrays directly.

Packing uses bounded chunks. At the current 2,097,152-particle HDF5 chunk size,
temporary staging uses about 120 MiB per calling rank in physical mode and
128 MiB in logical mode, in addition to VPIC's host mirrors. This staging
memory does not grow with total particle count and adds no on-disk payload.

Logical coordinates and grid metadata
-------------------------------------

Logical offsets use VPIC's ``[-1,1]`` normalization about a cell center.
The voxel index uses local ghost-padded dimensions ``(nx+2,ny+2,nz+2)``.
Logical HDF5 files also contain ``/particle_grid``, a small compound metadata
table with 76 bytes per writing rank:

* ``offset, count``: uint64 particle interval within this file.
* ``nx, ny, nz``: int32 local cell counts excluding ghosts.
* ``x0, y0, z0, dx, dy, dz``: float64 domain origin and cell spacing.

A per-rank file has one row with ``offset=0``. A shared file has one row per
rank, including empty ranks. This table reconstructs positions without storing
a second set of per-particle coordinates. Physical files do not need this table.

For a voxel index ``v``, recover cell indices and physical position as follows::

    ix = v % (nx + 2)
    iy = (v // (nx + 2)) % (ny + 2)
    iz = v // ((nx + 2) * (ny + 2))
    x = x0 + (ix - 1 + 0.5 * (1 + cell_offset_x)) * dx
    y = y0 + (iy - 1 + 0.5 * (1 + cell_offset_y)) * dy
    z = z0 + (iz - 1 + 0.5 * (1 + cell_offset_z)) * dz

Use the grid row whose ``[offset, offset+count)`` interval contains the particle.

HDF5 metadata and reading
=========================

Common file attributes
----------------------

.. list-table:: Root attributes
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``schema_version``
     - int32, currently ``2``.
   * - ``data_kind``
     - ``fields``, ``hydro``, or ``particles``.
   * - ``coordinate_system``
     - ``physical`` for fields/hydro; ``logical`` or ``physical`` for particles.
   * - ``file_layout``
     - ``per_rank`` or ``shared``.
   * - ``step``
     - int64 simulation step.
   * - ``time``
     - float64 snapshot time, ``t0 + dt * step``.
   * - ``dt``
     - float64 timestep size.

Hydro and particles also record ``species``, ``charge``, ``mass``, and ``q_m``.
Particles record ``momentum_centering="center_p_dump"``. Snapshot time does not
imply that every native field component has been advanced to a common time;
the writers preserve the PIC cycle's temporal staggering.

Structured geometry
-------------------

Each field/hydro component is a float32 dataset with shape ``[nz,ny,nx]``:
Z is slowest and X is fastest. Shared files cover the global mesh with ranks
writing their subdomains; per-rank files describe their local mesh.

Root attributes ``origin``, ``spacing``, and ``dimensions`` use **XYZ** order.
``geometry_axis_order="xyz"`` and ``array_axis_order="zyx"`` declare this
distinction. The root origin is the domain's lower corner, and
``ghost_policy="exclude_surrounding_layer"`` records the export policy.
Each component dataset adds:

* ``sample_offset``: XYZ offset in cell widths, from the table above.
* ``origin``: physical XYZ location of its first retained sample.
* ``centering="sample_node"``: values lie on that component's sample mesh.

The structured XDMF ``3DCoRectMesh`` dimensions, origin, and spacing are
serialized in **ZYX** order. This differs deliberately from the XYZ HDF5
geometry attributes. Particle ``/position`` columns remain XYZ.

h5py examples
-------------

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File("field/T.100/fields.100.0.h5", "r") as f:
       assert f.attrs["schema_version"] == 2
       ex = f["ex"]
       i, j, k = 0, 0, 0             # Zero-based output indices
       xyz = ex.attrs["origin"] + np.array([i, j, k]) * f.attrs["spacing"]
       value = ex[k, j, i]
       time = f.attrs["time"]

   with h5py.File("particle/physical.electron.100.0.h5", "r") as f:
       assert f.attrs["schema_version"] == 2
       position = f["position"][:1000]
       ux = f["momentum"][:1000, 0]
       weight = f["weight"][:1000]

Use bounded slices for large particle files. Logical files use the same
``momentum`` and ``weight`` access; read ``cell_offset``, ``cell_index``, and
``particle_grid`` to interpret their positions. Standard HDF5 tools also work::

    h5ls -r field/T.100/fields.100.0.h5
    h5dump -d /ex field/T.100/fields.100.0.h5

Binary format
=============

Binary files have a 64-byte header followed by component arrays or particle
records. The layout, magic, and version remain unchanged. These are native
binary values; readers must use the producing machine's byte order (normally
little-endian on the tested Linux systems).

.. code-block:: cpp

   struct BinaryHeader {
       int32_t magic;       // Bit pattern 0xBEEF0002
       int32_t version;     // 1
       int32_t step;        // Simulation step (32-bit in this format)
       int32_t nx, ny, nz;  // Local cell counts, excluding ghosts
       float dt;
       float dx, dy, dz;    // Cell spacing
       float x0, y0, z0;    // Physical subdomain origin
       float q_m;           // Species charge/mass; 0 for fields
       int32_t num_vars;    // Component count; 7/8 for physical/logical particles
       int32_t var_mask;    // Component mask; particle count for particle output
   };

For fields/hydro, the payload contains one complete float32 array per selected
component, in ascending bit order. Each array uses C-order ``[nz,ny,nx]``
(X fastest). The header's ``var_mask`` identifies the selected components.
For particles, the payload remains interleaved records:

* Physical: ``x,y,z,ux,uy,uz,w`` (seven float32 values, 28 bytes).
* Logical: ``dx,dy,dz,i,ux,uy,uz,w`` (seven float32 values and one int32 voxel
  index, 32 bytes).

The numeric-array schema change applies to HDF5; binary particle records retain
their existing layout. The header's final member stores particle count rather
than a component mask for particle output.

For example, read a field file produced on a little-endian machine:

.. code-block:: python

   import numpy as np

   header_type = np.dtype([
       ("magic", "<u4"), ("version", "<i4"), ("step", "<i4"),
       ("nx", "<i4"), ("ny", "<i4"), ("nz", "<i4"),
       ("dt", "<f4"), ("dx", "<f4"), ("dy", "<f4"), ("dz", "<f4"),
       ("x0", "<f4"), ("y0", "<f4"), ("z0", "<f4"), ("q_m", "<f4"),
       ("num_vars", "<i4"), ("var_mask", "<i4"),
   ])
   with open("field/T.100/fields.100.0", "rb") as stream:
       header = np.fromfile(stream, dtype=header_type, count=1)[0]
       assert header["magic"] == 0xBEEF0002
       shape = tuple(int(header[name]) for name in ("nz", "ny", "nx"))
       bits = [bit for bit in range(24) if int(header["var_mask"]) & (1 << bit)]
       assert len(bits) == header["num_vars"]
       components = {
           bit: np.fromfile(stream, dtype="<f4", count=int(np.prod(shape)))
                  .reshape(shape)
           for bit in bits
       }

File names and directories
==========================

Writers create required parent directories. Structured paths derive from
``DumpParameters.baseDir`` and ``baseFileName``; particle paths derive from
``fbase`` and the species name. ``STEP`` and ``RANK`` below are decimal integers.

.. list-table:: File patterns
   :header-rows: 1

   * - Output
     - Pattern
   * - Binary fields/hydro
     - ``dir/T.STEP/base.STEP.RANK``
   * - Per-rank HDF5 fields/hydro
     - ``dir/T.STEP/base.STEP.RANK.h5``
   * - Shared HDF5 fields/hydro
     - ``dir/T.STEP/base.STEP.h5``
   * - Binary particles
     - ``fbase.species.STEP.RANK``
   * - Per-rank HDF5 particles
     - ``fbase.species.STEP.RANK.h5``
   * - Shared HDF5 particles
     - ``fbase.species.STEP.h5``

XDMF uses the same stem with ``.xmf`` in place of ``.h5``. Per-rank output also
creates a master without the rank suffix that references every rank's wrapper.
Shared output has a single wrapper. Logical particles generate no wrappers.

Different steps produce distinct particle filenames. Repeating the same
base/species/step/rank replaces the existing snapshot. Coordinate mode is not
encoded in the name: use different bases for logical and physical particles.
Hydro species are not added automatically to structured filenames, so use a
distinct base or directory per species.

ParaView and time series
========================

Open the per-step master ``.xmf`` to load all writing ranks, for example
``field/T.100/fields.100.xmf`` or ``particle/physical.electron.100.xmf``.
Per-rank output supports a full-domain collection with serial HDF5 as well as
parallel HDF5. Physical particle XDMF references ``position``, ``momentum``,
and ``weight`` directly: no plugin, conversion, or duplicate particle payload
is needed. Logical coordinates require reconstruction and do not have a
direct visualization wrapper.

After all requested snapshots have been written, build a temporal collection:

.. code-block:: cpp

   #ifdef VPIC_ENABLE_HDF5
   if (rank() == 0) {
       std::vector<int> steps = {0, 100, 200}; // Existing snapshots only
       write_xdmf_timeseries("fields_series.xmf", steps,
                             "field/T.%d", "fields");
       write_xdmf_timeseries("particles_series.xmf", steps,
                             "particle", "physical.electron");
   }
   #endif

Only rank 0 performs this helper's work; other ranks return immediately if
they call it. ``base_pattern`` is a directory relative to the working directory
(or an absolute path). Literal ``%d`` occurrences are replaced with each step;
other printf formatting such as ``%06d`` is not supported. ``base_filename``
omits the step and extension. Each source wrapper must already exist.

The helper expands the writers' rank collections and rebases HDF5 references
relative to the series file. It copies XML metadata only, not particle or
grid arrays. Moving the complete output tree preserves those relative paths.
The helper accepts these writers' generated wrappers, not arbitrary external
XDMF, and reports unreadable or malformed input and cyclic includes as errors.

Reader validation and limitations
---------------------------------

Local validation used Ubuntu ParaView 5.10's built-in XDMF3 and legacy XDMF
readers to check particle values, structured sample coordinates and values,
and time series, including a relocated output tree. These checks read the
data through ParaView; they do not establish compatibility with every reader
version or visualization configuration.

In earlier tests with multiple visualization processes, that version's
default XDMF3 reader duplicated particle points across processes. Use its
built-in **XDMF Reader** (``XDMFReader`` in Python) for that configuration.
The legacy reader also failed when fetching an entirely empty collection in
parallel. These are reader limitations; the files do not contain duplicate
particle payloads. Writing ranks and visualization processes are separate:
four simulation ranks can be read by a single ParaView process.

Compatibility and troubleshooting
=================================

When moving from the earlier unversioned particle HDF5 layout, update readers
from compound ``/particles`` member access to the numeric arrays above:
``ux,uy,uz`` become columns of ``momentum``; ``w`` becomes ``weight``;
physical ``x,y,z`` become ``position``; logical ``dx,dy,dz,i`` become
``cell_offset`` and ``cell_index``. Both coordinate modes use schema version 2.
Logical coordinates remain the default.

Particle filenames now use ``STEP.RANK`` ordering, and binary particle names
include the step. Update file globs as well as dataset access. Existing files
are not renamed or converted; use a fresh output directory when migrating.
Deprecated boolean overloads preserve source compatibility, not old schemas
or filenames.

.. list-table:: Common issues
   :header-rows: 1

   * - Symptom
     - Check
   * - HDF5 writer missing at deck compile time
     - Enable ``VPIC_ENABLE_HDF5`` and rebuild, or guard HDF5 calls and use binary.
   * - Shared output rejected
     - Use per-rank output, or rebuild with parallel HDF5 and serial-only mode off.
   * - Output contract differs across ranks
     - Check matching paths, step/time, species, layout, coordinates, and shared
       component selections; all ranks must enter writers in the same order.
   * - No particle XDMF file
     - Logical mode intentionally omits it. Select physical coordinates for
       direct ParaView visualization.
   * - Missing HDF5 file when opening a series
     - Keep referenced HDF5 files with the output tree. Regenerate older series
       wrappers with the current helper if their relative references are wrong.
   * - Stride or interleaving option has no effect
     - These writers ignore those ``DumpParameters`` members.

Runnable writer checks are in the `integrated write tests
<../../test/integrated/write/>`_. They include binary and HDF5 verification
with one and four ranks. Run ``make test`` in the configured build directory;
the HDF5 tests require HDF5 support and Python with h5py and NumPy.
