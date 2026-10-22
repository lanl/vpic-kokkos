#include "vpic.h"
#include <sys/stat.h> // for mkdir
#include <vector>
#include <filesystem>

#ifdef VPIC_ENABLE_HDF5
#include "hdf5.h"
#endif // VPIC_ENABLE_HDF5


#include <cstdarg>
#include <cstdlib>
#include <climits>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <regex>
#include <set>

#define WRITE_ERROR(args) do { log_printf args; log_printf("\n"); \
    mp_abort(1); std::abort(); } while (0)

static FILE* checked_fopen(const char* path, const char* mode) {
    FILE* file = fopen(path, mode);
    if (!file) WRITE_ERROR(("Cannot open output file: %s", path));
    return file;
}
static void checked_fwrite(const void* data, size_t size, size_t count, FILE* file) {
    if (fwrite(data, size, count, file) != count)
        WRITE_ERROR(("Short write in VPIC binary output"));
}
static void checked_fprintf(FILE* file, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = vfprintf(file, format, args);
    va_end(args);
    if (result < 0) WRITE_ERROR(("Failed to write XDMF output"));
}
static void checked_fclose(FILE* file) {
    if (fclose(file) != 0) WRITE_ERROR(("Failed to close VPIC output"));
}

static std::string xml_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        switch (c) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        default: out += c;
        }
    }
    return out;
}

// =============================================================================
// CONFIGURATION
// =============================================================================

// Binary identifier and header version are independent; preserve the existing format.
constexpr int32_t VPIC_BINARY_MAGIC = 0xBEEF0002;
constexpr int32_t VPIC_BINARY_VERSION = 1;

// Bounded processing buffers; binary and HDF5 have different I/O granularity.
constexpr int BINARY_PARTICLE_BUFFER_SIZE = 32768;
constexpr int HDF5_PARTICLE_BUFFER_SIZE = 2097152;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Ensures a directory exists, creating it if necessary
 * @param path Directory path to create
 */
static void ensure_directory(const char* path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) WRITE_ERROR(("Cannot create output directory %s: %s", path, ec.message().c_str()));
}

// All ranks enter a writer in the same order. Compare the complete contract,
// not hashes, using the portable MP interface (also available without HDF5).
static void require_matching_output(const std::string& contract) {
    if (world_size == 1) return;
    double length = double(contract.size()), min_length, max_length;
    mp_allmin_d(&length, &min_length, 1);
    mp_allmax_d(&length, &max_length, 1);
    if (min_length != max_length)
        WRITE_ERROR(("Output contract differs across ranks (writer/path/species/step/selection/mode)"));
    for (size_t start=0; start<contract.size(); start+=1024) {
        int count = int(std::min(size_t(1024), contract.size()-start));
        double local[1024], lo[1024], hi[1024];
        for (int i=0; i<count; ++i) local[i] = static_cast<unsigned char>(contract[start+i]);
        mp_allmin_d(local, lo, count);
        mp_allmax_d(local, hi, count);
        for (int i=0; i<count; ++i)
            if (lo[i] != hi[i])
                WRITE_ERROR(("Output contract differs across ranks (writer/path/species/step/selection/mode)"));
    }
}

static uint32_t output_mask(const DumpParameters& params, int count) {
    uint32_t mask = 0;
    for (int bit=0; bit<count; ++bit)
        if (params.output_vars.bitset(bit)) mask |= uint32_t(1) << bit;
    return mask;
}

static std::string structured_stem(const DumpParameters& params, const grid_t* g) {
    return (std::filesystem::path(params.baseDir) / ("T."+std::to_string(g->step)) /
            (std::string(params.baseFileName)+"."+std::to_string(g->step))).string();
}

static std::string particle_stem(const char* base, const species_t* sp) {
    return std::string(base)+"."+sp->name+"."+std::to_string(sp->g->step);
}

static std::string output_stem(const std::string& stem, bool shared, int rank=world_rank) {
    return shared ? stem : stem+"."+std::to_string(rank);
}

static void prepare_output_parent(const std::string& stem) {
    auto parent = std::filesystem::path(stem).parent_path();
    if (world_rank == 0 && !parent.empty()) ensure_directory(parent.string().c_str());
    mp_barrier();
}

static void validate_output(const char* kind, const std::string& stem, const grid_t* g,
                            const species_t* sp, bool shared, bool physical=false,
                            uint32_t selection=0) {
    std::ostringstream contract;
    contract << std::setprecision(17) << kind << '\n' << stem.size() << ':' << stem
             << '\n' << g->step << ' ' << g->t0 << ' ' << g->dt
             << ' ' << shared << ' ' << physical;
    if (sp) contract << '\n' << strlen(sp->name) << ':' << sp->name << ' ' << sp->q << ' ' << sp->m;
    // Per-rank files may select different components, but shared datasets may not.
    if (shared) contract << '\n' << selection << ' ' << g->dx << ' ' << g->dy << ' ' << g->dz;
    require_matching_output(contract.str());
}

// =============================================================================
// BINARY FORMAT STRUCTURES
// =============================================================================

/**
 * @brief Binary file header structure (64 bytes)
 * 
 * This header precedes all data in VPIC binary dump files and contains
 * metadata necessary for interpreting the file contents.
 */

struct BinaryHeader {
    int32_t magic;      ///< File format identifier (0xBEEF0002)
    int32_t version;    ///< Format version number (currently 1)
    int32_t step;       ///< Simulation timestep when dump was created
    int32_t nx, ny, nz; ///< Grid dimensions (cells per rank)
    float   dt;         ///< Timestep size
    float   dx, dy, dz; ///< Cell dimensions
    float   x0, y0, z0; ///< Physical origin of this rank
    float   q_m;        ///< Charge-to-mass ratio (species-specific, 0.0 for fields)
    int32_t num_vars;   ///< Number of variables per cell
    int32_t var_mask;   ///< Variable mask for fields/hydro; particle count for particle output
};

enum VarType { VT_FLOAT, VT_MATID };
struct FieldComponent { int bit; const char* name; size_t offset; VarType type; };
struct HydroComponent { int bit; const char* name; size_t offset; };
static const std::vector<FieldComponent> field_map = {
            {0, "ex",         offsetof(field_t, ex), VT_FLOAT},
            {1, "ey",         offsetof(field_t, ey), VT_FLOAT},
            {2, "ez",         offsetof(field_t, ez), VT_FLOAT},
            {3, "div_e_err",  offsetof(field_t, div_e_err), VT_FLOAT},
            {4, "cbx",        offsetof(field_t, cbx), VT_FLOAT},
            {5, "cby",        offsetof(field_t, cby), VT_FLOAT},
            {6, "cbz",        offsetof(field_t, cbz), VT_FLOAT},
            {7, "div_b_err",  offsetof(field_t, div_b_err), VT_FLOAT},
            {8,  "tcax",      offsetof(field_t, tcax), VT_FLOAT},
            {9,  "tcay",      offsetof(field_t, tcay), VT_FLOAT},
            {10, "tcaz",      offsetof(field_t, tcaz), VT_FLOAT},
            {11, "rhob",      offsetof(field_t, rhob), VT_FLOAT},
            {12, "jfx",       offsetof(field_t, jfx), VT_FLOAT},
            {13, "jfy",       offsetof(field_t, jfy), VT_FLOAT},
            {14, "jfz",       offsetof(field_t, jfz), VT_FLOAT},
            {15, "rhof",      offsetof(field_t, rhof), VT_FLOAT},
            {16, "ematx",     offsetof(field_t, ematx), VT_MATID},
            {17, "ematy",     offsetof(field_t, ematy), VT_MATID},
            {18, "ematz",     offsetof(field_t, ematz), VT_MATID},
            {19, "nmat",      offsetof(field_t, nmat),  VT_MATID},
            {20, "fmatx",     offsetof(field_t, fmatx), VT_MATID},
            {21, "fmaty",     offsetof(field_t, fmaty), VT_MATID},
            {22, "fmatz",     offsetof(field_t, fmatz), VT_MATID},
            {23, "cmat",      offsetof(field_t, cmat),  VT_MATID}
        };
static const std::vector<HydroComponent> hydro_map = {
            // Current density
            {0,  "jx",  offsetof(hydro_t, jx)},
            {1,  "jy",  offsetof(hydro_t, jy)},
            {2,  "jz",  offsetof(hydro_t, jz)},
            // Charge & momentum
            {3,  "rho", offsetof(hydro_t, rho)},
            {4,  "px",  offsetof(hydro_t, px)},
            {5,  "py",  offsetof(hydro_t, py)},
            {6,  "pz",  offsetof(hydro_t, pz)},
            // Energy & pressure tensor
            {7,  "ke",  offsetof(hydro_t, ke)},
            {8,  "txx", offsetof(hydro_t, txx)},
            {9,  "tyy", offsetof(hydro_t, tyy)},
            {10, "tzz", offsetof(hydro_t, tzz)},
            {11, "tyz", offsetof(hydro_t, tyz)},
            {12, "tzx", offsetof(hydro_t, tzx)},
            {13, "txy", offsetof(hydro_t, txy)}
        };

// =============================================================================
// BINARY FIELD WRITER
// =============================================================================

/**
 * @brief Writes field data to binary format using table-driven variable selection
 * 
 * @param params Dump configuration (output directory, filename, variable mask)
 * @param fa Field array containing electromagnetic field data
 * 
 * This function uses a mapping table to dynamically select which field components
 * to write based on the bitmask in params.output_vars. Data is written in
 * rank-per-file (M2M) format.
 */
void vpic_simulation::write_fields_binary(DumpParameters& params, field_array_t* fa) {
    TIC {
        if (!fa) WRITE_ERROR(("NULL field array"));
        if (step() > fa->last_copied) fa->copy_to_host();
        if (fa->g != grid) WRITE_ERROR(("Field output requires the simulation grid"));
        grid_t* g = fa->g;
        validate_output("fields_binary", structured_stem(params, g), g, nullptr,
                        false, false, output_mask(params, 24));

        const std::string stem = structured_stem(params, g);
        prepare_output_parent(stem);

        // Open output file for this rank
        const std::string filename = output_stem(stem, false);
        const char* fname = filename.c_str();
        FILE* fp = checked_fopen(fname, "wb");

        // Count requested variables
        int32_t num_vars = 0;
        int32_t var_mask = 0;
        for (const auto& comp : field_map) {
            if (params.output_vars.bitset(comp.bit)) { ++num_vars; var_mask |= int32_t(1u << comp.bit); }
        }

        double io_start = wallclock();

        // Write header
        BinaryHeader header = {
            VPIC_BINARY_MAGIC,
            VPIC_BINARY_VERSION,
            (int32_t)step(),
            g->nx, g->ny, g->nz,
            g->dt,
            g->dx, g->dy, g->dz,
            g->x0, g->y0, g->z0, // Physical subdomain origin
            0.0f,              // q_m (not applicable for fields)
            num_vars,
            var_mask
        };
        checked_fwrite(&header, sizeof(BinaryHeader), 1, fp);

        // Write field data
        size_t num_cells = size_t(g->nx) * g->ny * g->nz;
        std::vector<float> buffer(num_cells);
        auto* f_base = fa->f;

        for (const auto& comp : field_map) {
            if (params.output_vars.bitset(comp.bit)) {
                size_t idx = 0;
                for (int k = 1; k <= g->nz; k++) {
                    for (int j = 1; j <= g->ny; j++) {
                        for (int i = 1; i <= g->nx; i++) {
                            int voxel_idx = voxel(i, j, k);
                            const char* p = (char*)&f_base[voxel_idx] + comp.offset;
                            float val = (comp.type == VT_MATID)
                              ? (float)(*(const material_id*)p)
                              : *(const float*)p;
                            buffer[idx++] = val;
                        }
                    }
                }
                checked_fwrite(buffer.data(), sizeof(float), num_cells, fp);
            }
        }

        checked_fclose(fp);

        // Performance metrics
        double io_time = wallclock() - io_start;
        double max_time = 0.0;
        mp_allmax_d(&io_time, &max_time, 1);

        if (rank() == 0 && num_vars > 0) {
            double mb_total = ((double)(num_cells * num_vars * sizeof(float)) / (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / max_time;
            MESSAGE(("[METRIC],write_fields_bin,%ld,%.4f,%.4f,%.4f",
                     (long)step(), max_time, mb_total, throughput));
        }
    } TOC(write_fields_bin, 1);
}

// =============================================================================
// BINARY HYDRO WRITER
// =============================================================================

/**
 * @brief Writes hydro array data to binary format using table-driven variable selection
 * 
 * @param params Dump configuration (output directory, filename, variable mask)
 * @param ha Hydro array containing moment data
 * @param sp_name Species name for physics accumulation
 * 
 * This function performs physics accumulation from particle data before writing,
 * computing current density (J), charge density (ρ), momentum (P), kinetic energy (KE),
 * and stress tensor components (T). Data is written in rank-per-file (M2M) format.
 * 
 * @note Physics accumulation is performed inside this function by design - the hydro
 *       array must be computed from particle data before dumping.
 */
void vpic_simulation::write_hydro_binary(DumpParameters& params, 
                                         hydro_array_t* ha, 
                                         const char* sp_name) {
    TIC {
        if (!ha) WRITE_ERROR(("NULL hydro array"));
        
        // =====================================================================
        // PHYSICS ACCUMULATION
        // =====================================================================
        // Hydro arrays must be computed from particle data before writing.
        // This accumulates current density (J), charge density (ρ), momentum (P),
        // kinetic energy (KE), and stress tensor components (T) for the species.
        
        species_t* sp = find_species(sp_name);
        if (!sp) WRITE_ERROR(("Invalid species \"%s\"", sp_name));

        // Clear grid and accumulate moments from particles
        if (ha->g != grid || sp->g != ha->g || interpolator_array->g != ha->g)
            WRITE_ERROR(("Hydro output requires matching simulation, species and interpolator grids"));
        validate_output("hydro_binary", structured_stem(params, ha->g), ha->g, sp,
                        false, false, output_mask(params, 14));
        Kokkos::deep_copy(ha->k_h_d, 0.0f);
        accumulate_hydro_p_kokkos(sp->k_p_d, sp->k_p_i_d, ha->k_h_d,
                                  interpolator_array->k_i_d, sp);
        
        // Synchronize accumulated data to host
        ha->copy_to_host();
        synchronize_hydro_array(ha);

        grid_t* g = ha->g;

        // =====================================================================
        // VARIABLE DEFINITION TABLE
        // =====================================================================
        // Maps bitmask indices to memory offsets within hydro_t structure.
        // Only variables with their corresponding bit set in params.output_vars
        // will be written to the output file.
        
        // =====================================================================
        // FILE PREPARATION
        // =====================================================================
        
        double io_tic = wallclock();
        
        const std::string stem = structured_stem(params, g);
        prepare_output_parent(stem);

        // Open output file for this rank
        const std::string filename = output_stem(stem, false);
        const char* fname = filename.c_str();
        
        FILE* fp = checked_fopen(fname, "wb");

        // Count requested variables
        int32_t num_vars = 0;
        int32_t var_mask = 0;
        for (const auto& comp : hydro_map) {
            if (params.output_vars.bitset(comp.bit)) { ++num_vars; var_mask |= int32_t(1u << comp.bit); }
        }

        // =====================================================================
        // WRITE HEADER
        // =====================================================================
        
        BinaryHeader header = {
            VPIC_BINARY_MAGIC,
            VPIC_BINARY_VERSION,
            (int32_t)step(),
            g->nx, g->ny, g->nz,
            g->dt,
            g->dx, g->dy, g->dz,
            g->x0, g->y0, g->z0, // Physical subdomain origin
            sp->q / sp->m,     // Charge-to-mass ratio for this species
            num_vars,
            var_mask
        };
        checked_fwrite(&header, sizeof(BinaryHeader), 1, fp);

        // =====================================================================
        // WRITE DATA
        // =====================================================================
        // Iterate through variable map and write only requested components
        
        size_t num_cells = size_t(g->nx) * g->ny * g->nz;
        std::vector<float> buffer(num_cells);
        auto* h_base = ha->h;

        for (const auto& comp : hydro_map) {
            if (params.output_vars.bitset(comp.bit)) {
                // Pack data in column-major order (k->j->i)
                size_t idx = 0;
                for (int k = 1; k <= g->nz; k++) {
                    for (int j = 1; j <= g->ny; j++) {
                        for (int i = 1; i <= g->nx; i++) {
                            int voxel_idx = voxel(i, j, k);
                            float val = *(float*)((char*)&h_base[voxel_idx] + comp.offset);
                            buffer[idx++] = val;
                        }
                    }
                }
                checked_fwrite(buffer.data(), sizeof(float), num_cells, fp);
            }
        }

        checked_fclose(fp);

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        double pure_io_time = wallclock() - io_tic;
        double t_max = 0.0;
        mp_allmax_d(&pure_io_time, &t_max, 1);

        if (rank() == 0 && num_vars > 0) {
            // Calculate total data volume across all ranks
            double mb_total = ((double)(num_cells * num_vars * sizeof(float)) / 
                              (1024.0 * 1024.0)) * nproc();
            double throughput = mb_total / t_max;
            
            MESSAGE(("[METRIC],write_hydro_bin,%ld,%.4f,%.4f,%.4f", 
                     (long)step(), t_max, mb_total, throughput));
        }
        
    } TOC(write_hydro_bin, 1);
}

// =============================================================================
// BINARY PARTICLE WRITER
// =============================================================================

/**
 * @brief Writes particle data to binary format
 * 
 * @param fbase Base filename (without extension)
 * @param species_name Name of species to dump
 * @param options Named coordinates and file layout. Position modes:
 *        - true: Output physical (x,y,z) in simulation coordinates, omit voxel index
 *        - false: Output logical (dx,dy,dz) in [-1,1] cell-local coords, include voxel index
 * 
 * Position Mode Details:
 * 
 * PHYSICAL MODE (compute_physical_position = true):
 * - Converts particle's cell-local coordinates to global simulation coordinates
 * - Transformation: x = x0 + (ix - 1 + 0.5*(1 + dx)) * cell_dx
 * - Output fields: x, y, z, ux, uy, uz, w (7 floats = 28 bytes) - NO voxel index
 * - Use case: Post-processing tools that need absolute positions
 * - File size: ~12.5% SMALLER than logical mode (saves 4 bytes per particle)
 * 
 * LOGICAL MODE (compute_physical_position = false):
 * - Outputs raw particle data as stored during simulation
 * - Output fields: dx, dy, dz, i, ux, uy, uz, w (3 floats + 1 int32 + 4 floats = 32 bytes)
 * - Use case: Diagnostics and visualization that handle cell-local coordinates.
 *   Output momenta are time-centered, so this is not a restart format.
 * - Benefit: Preserves exact simulation state, faster I/O (no coordinate conversion)
 * 
 * Performance characteristics:
 * - Chunked processing (32K particles per chunk) for memory efficiency
 * - Time-centering applied to copied momenta via center_p_dump()
 * - Physical mode reduces file size by 12.5% (28 vs 32 bytes per particle)
 * - Binary format: simple concatenated structs, excellent sequential read performance
 * 
 * @note File format: BinaryHeader (64 bytes) + particle data array
 * @note Header's num_vars field reflects output: 7 (physical) or 8 (logical)
 */
void vpic_simulation::write_particles_binary(const char* fbase, 
                                             const char* species_name,
                                             ParticleWriteOptions options) {
    const bool compute_physical_position = options.coordinates == ParticleCoordinates::Physical;
    if (options.layout != OutputLayout::PerRank)
        WRITE_ERROR(("Binary particle output requires OutputLayout::PerRank"));
    TIC {
        // =====================================================================
        // PHASE 1: SETUP & VALIDATION
        // =====================================================================
        
        species_t* sp = find_species(species_name);
        if(!sp) WRITE_ERROR(("Invalid species: %s", species_name));
        const std::string stem = particle_stem(fbase, sp);
        validate_output("particles_binary", stem, sp->g, sp, false, compute_physical_position);
        prepare_output_parent(stem);
        sp->copy_to_host();
        // Center the output copy with the same interpolator used by this PIC step.
        interpolator_array->copy_to_host();
        Kokkos::fence();
        grid_t* g = sp->g;

        size_t my_count = sp->np;

        // Open output file
        const std::string filename = output_stem(stem, false);
        const char* fname = filename.c_str();
        FILE *fp = checked_fopen(fname, "wb");
        if(!fp) WRITE_ERROR(("Could not open file: %s", fname));

        // =====================================================================
        // PHASE 2: WRITE HEADER
        // =====================================================================
        // Header reflects output format: 7 vars (physical) or 8 vars (logical)
        
        BinaryHeader h = {
            VPIC_BINARY_MAGIC,                            // magic
            VPIC_BINARY_VERSION,                           // version
            (int32_t)g->step,                              // step
            g->nx, g->ny, g->nz,                           // grid dims
            g->dt,                                          // timestep
            g->dx, g->dy, g->dz,                           // cell size
            g->x0, g->y0, g->z0,                           // origin
            sp->q/sp->m,                                    // q/m ratio
            compute_physical_position ? 7 : 8,              // num_vars
            (int32_t)my_count                               // particle count
        };
        checked_fwrite(&h, sizeof(BinaryHeader), 1, fp);

        // =====================================================================
        // PHASE 3: CHUNKED PARTICLE PROCESSING
        // =====================================================================
        // Process in 32K particle chunks for memory efficiency.
        // Each chunk undergoes: Copy → Time-Center → Convert (if physical) → Write
        
        const int PBUF_SIZE = BINARY_PARTICLE_BUFFER_SIZE;
        particle_t* p_buf;
        MALLOC_ALIGNED(p_buf, PBUF_SIZE, 128);

        auto& k_p_h = sp->k_p_h;
        auto& k_p_i_h = sp->k_p_i_h;
        int sp_np = sp->np;
        int sp_max_np = sp->max_np;

        // Precompute grid parameters for physical coordinate conversion
        // UNVOXEL macro will be used: converts voxel index → (i,j,k)
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float dx = g->dx, dy = g->dy, dz = g->dz;
        float x0 = g->x0, y0 = g->y0, z0 = g->z0;

        for(int buf_start = 0; buf_start < sp_np; buf_start += PBUF_SIZE) {
            
            // Adjust species counts for this chunk
            sp->np = sp_np - buf_start;
            if(sp->np > PBUF_SIZE) sp->np = PBUF_SIZE;
            sp->max_np = PBUF_SIZE;

            // -----------------------------------------------------------------
            // Sub-phase 3a: Copy Chunk to Aligned Buffer
            // -----------------------------------------------------------------
            Kokkos::View<particle_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> 
                pbuf_view(p_buf, PBUF_SIZE);
            
            Kokkos::parallel_for("PopulateParticleDumpBuffer",
                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np),
                KOKKOS_LAMBDA(int i) {
                    int idx = buf_start + i;
                    pbuf_view(i).dx = k_p_h(idx, particle_var::dx);
                    pbuf_view(i).dy = k_p_h(idx, particle_var::dy);
                    pbuf_view(i).dz = k_p_h(idx, particle_var::dz);
                    pbuf_view(i).ux = k_p_h(idx, particle_var::ux);
                    pbuf_view(i).uy = k_p_h(idx, particle_var::uy);
                    pbuf_view(i).uz = k_p_h(idx, particle_var::uz);
                    pbuf_view(i).w  = k_p_h(idx, particle_var::w);
                    pbuf_view(i).i  = k_p_i_h(idx);
                });
            Kokkos::fence();

            // -----------------------------------------------------------------
            // Sub-phase 3b: Time-Centering of the Output Copy
            // -----------------------------------------------------------------
            // Center copied momenta without modifying simulation particles.
            center_p_dump(sp, p_buf, interpolator_array);

            // -----------------------------------------------------------------
            // Sub-phase 3c: Coordinate Transformation (Physical Mode Only)
            // -----------------------------------------------------------------
            if (compute_physical_position && sp->np > 0) {
                // Transform (dx,dy,dz,i) → (x,y,z) in-place
                // After this, p_buf[n].dx/dy/dz contain physical coordinates
                // and p_buf[n].i becomes meaningless (won't be written)
                
                for(int n = 0; n < sp->np; ++n) {
                    int voxel_idx = p_buf[n].i;
                    
                    // Convert voxel index to (i,j,k) grid coordinates
                    // UNVOXEL macro: extracts cell indices from flat voxel index
                    int ix, iy, iz;
                    UNVOXEL(voxel_idx, ix, iy, iz, nx, ny, nz);
                    
                    // Transform to physical position
                    // The particle offset is in [-1,1] about its cell center.
                    // Formula: global = origin + (cell_index - 1 + 0.5*(1 + local_offset)) * cell_size
                    // The "-1" accounts for ghost cell at index 0
                    p_buf[n].dx = x0 + (ix - 1 + 0.5f * (1.0f + p_buf[n].dx)) * dx;
                    p_buf[n].dy = y0 + (iy - 1 + 0.5f * (1.0f + p_buf[n].dy)) * dy;
                    p_buf[n].dz = z0 + (iz - 1 + 0.5f * (1.0f + p_buf[n].dz)) * dz;
                }
            }

            // -----------------------------------------------------------------
            // Sub-phase 3d: Write Chunk to File
            // -----------------------------------------------------------------
            if(sp->np > 0) {
                if (compute_physical_position) {
                    // Physical mode: Write only x,y,z,ux,uy,uz,w (skip voxel index)
                    for(int n = 0; n < sp->np; ++n) {
                        checked_fwrite(&p_buf[n].dx, sizeof(float), 3, fp);  // x,y,z
                        checked_fwrite(&p_buf[n].ux, sizeof(float), 4, fp);  // ux,uy,uz,w
                    }
                } else {
                    // Logical mode: Write full struct including voxel index
                    checked_fwrite(p_buf, sizeof(particle_t), sp->np, fp);
                }
            }
        }

        // =====================================================================
        // PHASE 4: CLEANUP & METRICS
        // =====================================================================
        
        // Restore species state
        sp->np = sp_np;
        sp->max_np = sp_max_np;
        FREE_ALIGNED(p_buf);
        checked_fclose(fp);

        // Calculate metrics
        double t_local = wallclock() - _profile_tic;
        size_t bytes_per_particle = compute_physical_position ? (7 * sizeof(float)) : sizeof(particle_t);
        double mb_local = (double)(my_count * bytes_per_particle) / (1024.0 * 1024.0);
        
        double t_max = 0.0, mb_total = 0.0;
        mp_allmax_d(&t_local, &t_max, 1);
        mp_allsum_d(&mb_local, &mb_total, 1);

        if (rank() == 0) {
            double throughput = mb_total / t_max;
            const char* mode = compute_physical_position ? "physical" : "logical";
            MESSAGE(("[METRIC],write_particles_bin_%s,%ld,%.4f,%.4f,%.4f", 
                     mode, (long)step(), t_max, mb_total, throughput));
        }
        
    } TOC(write_particles_bin, 1);
}

#ifdef VPIC_ENABLE_HDF5

// =============================================================================
// XDMF GENERATION HELPER
// =============================================================================

/**
 * @brief Global grid layout information for HDF5 spatial datasets
 * 
 * Contains the total simulation grid dimensions and this rank's offset within
 * the global domain. Used for configuring hyperslab selections in M2O mode.
 */

template<typename T> static T checked_hdf5(T result, const char* operation) {
    if (result < 0) WRITE_ERROR(("HDF5 output failed: %s", operation));
    return result;
}
#define H5_CHECK(call) checked_hdf5((call), #call)

static void require_hdf5_mode(bool single_file) {
    // Writers already require all simulation ranks to participate. Check mode
    // agreement before any directories/files or HDF5 collective operations.
    double local = single_file ? 1.0 : 0.0, lo, hi;
    mp_allmin_d(&local, &lo, 1);
    mp_allmax_d(&local, &hi, 1);
    if (lo != hi) WRITE_ERROR(("HDF5 single_file must agree on all ranks"));
#if !defined(VPIC_HDF5_PARALLEL) || defined(VPIC_HDF5_SERIAL_ONLY)
    if (single_file)
        WRITE_ERROR(("Collective/shared-file HDF5 output requires parallel HDF5; "
                     "this build uses serial mode. Use single_file=false for per-rank output."));
#endif
}

struct GlobalGridInfo {
    hsize_t global_dims[3];  ///< Total simulation grid [nz, ny, nx]
    hsize_t offset[3];       ///< This rank's starting index [z_off, y_off, x_off]
    double origin[3];       ///< Physical origin in XYZ order
};

/**
 * @brief Calculates global grid dimensions and this rank's offset within them
 * 
 * @param g Grid structure containing local dimensions and spatial bounds
 * @param single_file True for M2O (many-to-one), false for M2M (many-to-many)
 * @return GlobalGridInfo containing global dimensions and rank offset
 * 
 * For M2M mode, returns local grid size (each rank writes its own file).
 * For M2O mode, computes where this rank's subdomain fits in the global grid
 * by using MPI reductions to find global min/max coordinates.
 * 
 * @note Dimensions follow HDF5 convention: [nz, ny, nx] (Z slowest, X fastest)
 */
static GlobalGridInfo get_global_grid_info(grid_t* g, bool single_file) {
    GlobalGridInfo info = {{0}, {0}, {g->x0, g->y0, g->z0}};
    
    // Local dimensions in HDF5 convention (Z, Y, X)
    hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};

    // M2M mode: Each file contains only local grid
    if (!single_file) {
        info.global_dims[0] = local_dims[0];
        info.global_dims[1] = local_dims[1];
        info.global_dims[2] = local_dims[2];
        return info;
    }

    // -------------------------------------------------------------------------
    // M2O mode: Compute global grid layout
    // -------------------------------------------------------------------------
    // Use spatial coordinates to determine rank offset in global grid
    
    double local_min[3] = {g->x0, g->y0, g->z0};
    double local_max[3] = {g->x1, g->y1, g->z1};
    double global_min[3], global_max[3];

    mp_allmin_d(local_min, global_min, 3);
    mp_allmax_d(local_max, global_max, 3);

    std::copy(global_min, global_min+3, info.origin);

    // Calculate offsets and dimensions (mapping Z->0, Y->1, X->2)
    info.offset[0] = (hsize_t)((g->z0 - global_min[2]) / g->dz + 0.5);
    info.global_dims[0] = (hsize_t)((global_max[2] - global_min[2]) / g->dz + 0.5);

    info.offset[1] = (hsize_t)((g->y0 - global_min[1]) / g->dy + 0.5);
    info.global_dims[1] = (hsize_t)((global_max[1] - global_min[1]) / g->dy + 0.5);

    info.offset[2] = (hsize_t)((g->x0 - global_min[0]) / g->dx + 0.5);
    info.global_dims[2] = (hsize_t)((global_max[0] - global_min[0]) / g->dx + 0.5);

    return info;
}

/**
 * @brief XDMF grid metadata (pre-computed, no MPI calls)
 */
struct XDMFGridInfo {
    hsize_t dims[3];      // Grid dimensions [nz, ny, nx]
    double origin[3];     // Origin [x, y, z]
    double spacing[3];    // Cell size [dx, dy, dz]
};

static XDMFGridInfo xdmf_grid_info(const GlobalGridInfo& layout, const grid_t* g) {
    XDMFGridInfo info;
    std::copy(layout.global_dims, layout.global_dims+3, info.dims);
    std::copy(layout.origin, layout.origin+3, info.origin);
    info.spacing[0]=g->dx; info.spacing[1]=g->dy; info.spacing[2]=g->dz;
    return info;
}

/**
 * @brief Generates XDMF wrapper for HDF5 structured grid data (fields/hydro)
 * @param xmf_path Output .xmf filename
 * @param h5_filename Relative path to HDF5 file (from .xmf location)
 * @param var_names List of dataset names in HDF5 file
 * @param xinfo Pre-computed grid info (NO MPI CALLS INSIDE THIS FUNCTION)
 * @param g Grid structure for time metadata only
 */

// Offsets from the lower corner of the indexed cell, in cell widths.
// The export contract remains 1..nx, 1..ny, 1..nz for every component.
static void field_sample_offset(const char* name, double* offset) {
    offset[0] = offset[1] = offset[2] = 0.0;
    const char* edges[3][4] = {{"ex","jfx","tcax","ematx"},
                              {"ey","jfy","tcay","ematy"},
                              {"ez","jfz","tcaz","ematz"}};
    const char* faces[3][2] = {{"cbx","fmatx"},{"cby","fmaty"},{"cbz","fmatz"}};
    for (int axis=0; axis<3; ++axis) {
        for (const char* edge : edges[axis])
            if (!strcmp(name, edge)) offset[axis] = 0.5;
        for (const char* face : faces[axis])
            if (!strcmp(name, face)) for (int d=0; d<3; ++d) offset[d] = d==axis ? 0.0 : 0.5;
    }
    if (!strcmp(name,"div_b_err") || !strcmp(name,"cmat"))
        offset[0] = offset[1] = offset[2] = 0.5;
}

static void write_xdmf_structured(const char* xmf_path, const char* h5_filename,
                                const std::vector<const char*>& var_names,
                                const XDMFGridInfo& info, grid_t* g) {
    FILE* xmf = checked_fopen(xmf_path, "w");
    checked_fprintf(xmf, "<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\"><Domain>\n"
                        "<Grid Name=\"VPIC\" GridType=\"Collection\" CollectionType=\"Spatial\">\n"
                        "<Time Value=\"%.17g\"/>\n", g->t0 + g->dt*g->step);
    for (const char* name : var_names) {
        double offset[3];
        field_sample_offset(name, offset); // Hydro names map to nodes (zero offset).
        checked_fprintf(xmf, "<Grid Name=\"%s\" GridType=\"Uniform\">\n"
                            "<Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"%llu %llu %llu\"/>\n"
                            "<Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
                            "<DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">%.17g %.17g %.17g</DataItem>\n"
                            "<DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">%.17g %.17g %.17g</DataItem>\n"
                            "</Geometry>\n<Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Node\">\n"
                            "<DataItem Dimensions=\"%llu %llu %llu\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">%s:/%s</DataItem>\n"
                            "</Attribute></Grid>\n",
                        name, (unsigned long long)info.dims[0], (unsigned long long)info.dims[1], (unsigned long long)info.dims[2],
                        // CoRectMesh geometry uses ZYX order, like its dimensions.
                        // Keep internal coordinates and Yee offsets in XYZ; reverse only for XML.
                        info.origin[2]+offset[2]*info.spacing[2], info.origin[1]+offset[1]*info.spacing[1], info.origin[0]+offset[0]*info.spacing[0],
                        info.spacing[2], info.spacing[1], info.spacing[0], name,
                        (unsigned long long)info.dims[0], (unsigned long long)info.dims[1], (unsigned long long)info.dims[2],
                        xml_escape(h5_filename).c_str(), name);
    }
    checked_fprintf(xmf, "</Grid></Domain></Xdmf>\n");
    checked_fclose(xmf);
}

// XDMF references the three physical datasets directly: no copied payload.
static void write_xdmf_particles(const char* path, const char* file,
                                 hsize_t count, grid_t* g) {
    FILE* xmf = checked_fopen(path, "w");
    checked_fprintf(xmf, "<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\"><Domain>\n"
                        "<Grid Name=\"Particles\" GridType=\"Collection\" CollectionType=\"Spatial\">\n"
                        "<Time Value=\"%.17g\"/>\n", g->t0+g->dt*g->step);
    // Empty collections avoid asking readers to construct a zero-point topology.
    if (count) {
        const auto filename = xml_escape(file);
        checked_fprintf(xmf, "<Grid Name=\"Particles\" GridType=\"Uniform\">\n"
                            "<Topology TopologyType=\"Polyvertex\" NumberOfElements=\"%llu\"/>\n"
                            "<Geometry GeometryType=\"XYZ\">\n"
                            "<DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">%s:/position</DataItem>\n"
                            "</Geometry>\n"
                            "<Attribute Name=\"momentum\" AttributeType=\"Vector\" Center=\"Node\">\n"
                            "<DataItem Dimensions=\"%llu 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">%s:/momentum</DataItem>\n"
                            "</Attribute>\n"
                            "<Attribute Name=\"weight\" AttributeType=\"Scalar\" Center=\"Node\">\n"
                            "<DataItem Dimensions=\"%llu\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">%s:/weight</DataItem>\n"
                            "</Attribute></Grid>\n",
                        (unsigned long long)count,
                        (unsigned long long)count, filename.c_str(),
                        (unsigned long long)count, filename.c_str(),
                        (unsigned long long)count, filename.c_str());
    }
    checked_fprintf(xmf, "</Grid></Domain></Xdmf>\n");
    checked_fclose(xmf);
}

static void write_rank_collection(const char* path, const std::vector<std::string>& children, grid_t* g) {
    FILE* xmf = checked_fopen(path, "w");
    checked_fprintf(xmf, "<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\"><Domain>\n"
                        "<Grid Name=\"Ranks\" GridType=\"Collection\" CollectionType=\"Spatial\">\n"
                        "<Time Value=\"%.17g\"/>\n", g->t0+g->dt*g->step);
    for (const auto& child : children)
        checked_fprintf(xmf, "<xi:include href=\"%s\" xpointer=\"xpointer(/Xdmf/Domain/Grid)\"/>\n", xml_escape(child).c_str());
    checked_fprintf(xmf, "</Grid></Domain></Xdmf>\n");
    checked_fclose(xmf);
}

// Each rank publishes its own geometry; rank zero publishes the spatial index
// only after every child wrapper has been closed successfully.
static void publish_structured_xdmf(DumpParameters& params, bool single_file,
                                    const std::vector<const char*>& names,
                                    const XDMFGridInfo& info, grid_t* g) {
    const auto path = std::filesystem::path(structured_stem(params, g));
    const std::string stem = path.filename().string();
    const std::string dir = path.parent_path().string()+"/";
    if (single_file) {
        if (world_rank == 0) write_xdmf_structured((dir+stem+".xmf").c_str(), (stem+".h5").c_str(), names, info, g);
    } else {
        std::string local = stem+"."+std::to_string(world_rank);
        write_xdmf_structured((dir+local+".xmf").c_str(), (local+".h5").c_str(), names, info, g);
        mp_barrier();
        if (world_rank == 0) {
            std::vector<std::string> children;
            for (int r=0; r<world_size; ++r) children.push_back(stem+"."+std::to_string(r)+".xmf");
            write_rank_collection((dir+stem+".xmf").c_str(), children, g);
        }
    }
}

/**
 * @brief Generates a temporal collection with rebased, relative HDF references
 * @param series_path Output .xmf filename (e.g., "fields_timeseries.xmf")
 * @param steps Vector of timesteps that were dumped
 * @param base_pattern Directory structure with %d placeholder (e.g., "fields/T.%d") 
 * @param base_filename HDF5/XDMF base name (e.g., "fields")
 */

// Expand wrappers generated by these writers and rebase HDF references to
// the series directory. ParaView does not consistently honor XInclude xml:base.
// Only XML metadata is copied; particle/field payload remains in its original HDF5 file.
static std::string xml_unescape(std::string value) {
    const char* entities[] = {"&quot;", "&apos;", "&lt;", "&gt;", "&amp;"};
    const char chars[] = {'"', '\'', '<', '>', '&'};
    for (int i=0; i<5; ++i) {
        size_t pos=0;
        while ((pos=value.find(entities[i],pos)) != std::string::npos) {
            value.replace(pos,strlen(entities[i]),1,chars[i]);
            ++pos;
        }
    }
    return value;
}

static std::string series_grid(const std::filesystem::path& input,
                               const std::filesystem::path& series_directory,
                               std::set<std::string>& active) {
    namespace fs = std::filesystem;
    const fs::path path = fs::absolute(input).lexically_normal();
    if (!active.insert(path.string()).second)
        WRITE_ERROR(("Cyclic XDMF include: %s", path.string().c_str()));
    std::ifstream file(path);
    if (!file) WRITE_ERROR(("Cannot read XDMF snapshot: %s", path.string().c_str()));
    std::ostringstream contents;
    contents << file.rdbuf();
    if (file.bad()) WRITE_ERROR(("Failed reading XDMF snapshot: %s", path.string().c_str()));
    const std::string xml = contents.str();
    size_t begin = xml.find("<Grid "), end = xml.rfind("</Grid>");
    if (begin == std::string::npos || end == std::string::npos || end < begin)
        WRITE_ERROR(("Missing Grid in generated XDMF snapshot: %s", path.string().c_str()));
    std::string grid = xml.substr(begin, end+7-begin);
    const std::regex data(R"re((<DataItem\b[^>]*\bFormat="HDF"[^>]*>)([^<]+)(</DataItem>))re");
    std::string rebased;
    size_t cursor=0;
    for (auto i=std::sregex_iterator(grid.begin(),grid.end(),data); i!=std::sregex_iterator(); ++i) {
        const auto& match=*i;
        rebased.append(grid,cursor,size_t(match.position())-cursor);
        const std::string reference=xml_unescape(match[2].str());
        const size_t separator=reference.rfind(":/");
        if (separator==std::string::npos) WRITE_ERROR(("Invalid generated HDF reference: %s", reference.c_str()));
        const fs::path target=(path.parent_path()/reference.substr(0,separator)).lexically_normal();
        const std::string relative=target.lexically_relative(series_directory).generic_string();
        rebased += match[1].str()+xml_escape(relative+reference.substr(separator))+match[3].str();
        cursor=size_t(match.position()+match.length());
    }
    rebased.append(grid,cursor,std::string::npos);
    const std::regex include(R"re(<xi:include\s+href="([^"]+)"[^>]*/>)re");
    std::string expanded;
    cursor=0;
    for (auto i=std::sregex_iterator(rebased.begin(),rebased.end(),include); i!=std::sregex_iterator(); ++i) {
        const auto& match=*i;
        expanded.append(rebased,cursor,size_t(match.position())-cursor);
        expanded += series_grid(path.parent_path()/xml_unescape(match[1].str()),series_directory,active);
        cursor=size_t(match.position()+match.length());
    }
    expanded.append(rebased,cursor,std::string::npos);
    active.erase(path.string());
    return expanded;
}

void vpic_simulation::write_xdmf_timeseries(const char* series_path,
                                            const std::vector<int>& steps,
                                            const char* base_pattern,
                                            const char* base_filename) {
    if (rank() != 0) return;
    const auto parent = std::filesystem::path(series_path).parent_path();
    if (!parent.empty()) ensure_directory(parent.string().c_str());

    FILE* xmf = checked_fopen(series_path, "w");
    checked_fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    checked_fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    checked_fprintf(xmf, "<Xdmf Version=\"3.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\">\n");
    checked_fprintf(xmf, "  <Domain>\n");
    checked_fprintf(xmf, "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n");

    for (int s : steps) {
        std::string directory(base_pattern);
        for (size_t pos=0; (pos=directory.find("%d", pos)) != std::string::npos;) {
            const std::string value = std::to_string(s);
            directory.replace(pos, 2, value);
            pos += value.size();
        }

        // Resolve includes relative to the series file.
        namespace fs = std::filesystem;
        fs::path xmf_path = fs::path(directory) / (std::string(base_filename) + "." + std::to_string(s) + ".xmf");

        std::set<std::string> active;
        const auto series_directory = fs::absolute(fs::path(series_path)).parent_path().lexically_normal();
        const auto contents = series_grid(xmf_path, series_directory, active);
        checked_fprintf(xmf, "%s\n", contents.c_str());
    }

    checked_fprintf(xmf, "    </Grid>\n");
    checked_fprintf(xmf, "  </Domain>\n");
    checked_fprintf(xmf, "</Xdmf>\n");
    checked_fclose(xmf);

    MESSAGE(("XDMF time series written: %s", series_path));
}

// =============================================================================
// HDF5 HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Writes a scalar attribute to an HDF5 object
 * 
 * @tparam T Attribute data type (int, float, double, etc.)
 * @param loc_id HDF5 object identifier (file, group, or dataset)
 * @param name Attribute name
 * @param type_id HDF5 datatype identifier
 * @param value Value to write
 */
template<typename T>
static void write_scalar_attr(hid_t loc_id, const char* name, hid_t type_id, T value) {
    hid_t space = H5_CHECK(H5Screate(H5S_SCALAR));
    hid_t attr = H5_CHECK(H5Acreate2(loc_id, name, type_id, space, H5P_DEFAULT, H5P_DEFAULT));
    H5_CHECK(H5Awrite(attr, type_id, &value));
    H5_CHECK(H5Aclose(attr));
    H5_CHECK(H5Sclose(space));
}

static void write_text_attr(hid_t object, const char* name, const char* value) {
    hid_t type = H5_CHECK(H5Tcopy(H5T_C_S1));
    H5_CHECK(H5Tset_size(type, strlen(value)+1));
    H5_CHECK(H5Tset_cset(type, H5T_CSET_UTF8));
    hid_t space = H5_CHECK(H5Screate(H5S_SCALAR));
    hid_t attr = H5_CHECK(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT));
    H5_CHECK(H5Awrite(attr, type, value));
    H5_CHECK(H5Aclose(attr)); H5_CHECK(H5Sclose(space)); H5_CHECK(H5Tclose(type));
}

static void write_vector_attr(hid_t object, const char* name, hid_t type, const void* value) {
    const hsize_t count = 3;
    hid_t space = H5_CHECK(H5Screate_simple(1, &count, nullptr));
    hid_t attr = H5_CHECK(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT));
    H5_CHECK(H5Awrite(attr, type, value));
    H5_CHECK(H5Aclose(attr)); H5_CHECK(H5Sclose(space));
}

static void write_output_metadata(hid_t file, const char* kind, const grid_t* g,
                                  const species_t* sp, bool shared, const char* coordinates) {
    write_scalar_attr(file, "schema_version", H5T_NATIVE_INT32, int32_t(2));
    write_text_attr(file, "data_kind", kind);
    write_text_attr(file, "coordinate_system", coordinates);
    write_text_attr(file, "file_layout", shared ? "shared" : "per_rank");
    write_scalar_attr(file, "step", H5T_NATIVE_INT64, int64_t(g->step));
    write_scalar_attr(file, "time", H5T_NATIVE_DOUBLE, double(g->t0+g->dt*g->step));
    write_scalar_attr(file, "dt", H5T_NATIVE_DOUBLE, double(g->dt));
    if (sp) {
        write_text_attr(file, "species", sp->name);
        write_scalar_attr(file, "charge", H5T_NATIVE_DOUBLE, double(sp->q));
        write_scalar_attr(file, "mass", H5T_NATIVE_DOUBLE, double(sp->m));
        write_scalar_attr(file, "q_m", H5T_NATIVE_DOUBLE, double(sp->q/sp->m));
    }
}

static void write_grid_metadata(hid_t file, const GlobalGridInfo& layout, const grid_t* g) {
    const double spacing[3] = {g->dx,g->dy,g->dz};
    const uint64_t dimensions[3] = {layout.global_dims[2],layout.global_dims[1],layout.global_dims[0]};
    write_vector_attr(file, "origin", H5T_NATIVE_DOUBLE, layout.origin);
    write_vector_attr(file, "spacing", H5T_NATIVE_DOUBLE, spacing);
    write_vector_attr(file, "dimensions", H5T_NATIVE_UINT64, dimensions);
    write_text_attr(file, "geometry_axis_order", "xyz");
    write_text_attr(file, "array_axis_order", "zyx");
    write_text_attr(file, "ghost_policy", "exclude_surrounding_layer");
}

static void write_sample_metadata(hid_t dataset, const char* name, const GlobalGridInfo& layout,
                                  const grid_t* g) {
    double offset[3], origin[3];
    field_sample_offset(name, offset);
    const double spacing[3] = {g->dx,g->dy,g->dz};
    for (int i=0; i<3; ++i) origin[i] = layout.origin[i]+offset[i]*spacing[i];
    write_vector_attr(dataset, "sample_offset", H5T_NATIVE_DOUBLE, offset);
    write_vector_attr(dataset, "origin", H5T_NATIVE_DOUBLE, origin);
    write_text_attr(dataset, "centering", "sample_node");
}

/**
 * @brief Creates optimized File Access Property List (FAPL) for parallel I/O
 * 
 * @param single_file True for M2O mode (requires parallel HDF5)
 * @return Configured FAPL handle
 * 
 * Optimizations applied:
 * 1. Collective metadata operations (reduces MPI communication overhead)
 * 2. 16MB stripe alignment (optimizes for parallel filesystems like Lustre)
 * 3. Deferred metadata cache flushes (batches writes, reduces sync operations)
 * 
 * Based on ECP best practices and h5bench benchmarking results.
 */
static hid_t create_optimized_fapl(bool single_file) {
    hid_t fapl = H5_CHECK(H5Pcreate(H5P_FILE_ACCESS));

    #if defined(VPIC_HDF5_PARALLEL) && !defined(VPIC_HDF5_SERIAL_ONLY)
    if (single_file) {
        // Enable MPI-IO driver for parallel access
        H5_CHECK(H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL));

        // Collective metadata operations (reduces per-rank I/O)
        #if (H5_VERS_MAJOR > 1) || (H5_VERS_MAJOR == 1 && H5_VERS_MINOR >= 10)
        H5_CHECK(H5Pset_all_coll_metadata_ops(fapl, 1));
        H5_CHECK(H5Pset_coll_metadata_write(fapl, 1));
        #endif
    }
    #endif

    // Align large objects to 16MB boundaries (optimized for Lustre stripe size)
    H5_CHECK(H5Pset_alignment(fapl, 4096, 16 * 1024 * 1024));

    // Defer metadata flushes until file close (batch B-tree updates)
    H5AC_cache_config_t cache_config;
    cache_config.version = H5AC__CURR_CACHE_CONFIG_VERSION;
    H5_CHECK(H5Pget_mdc_config(fapl, &cache_config));
    cache_config.evictions_enabled = 0;
    cache_config.incr_mode = H5C_incr__off;
    cache_config.flash_incr_mode = H5C_flash_incr__off;
    cache_config.decr_mode = H5C_decr__off;
    H5_CHECK(H5Pset_mdc_config(fapl, &cache_config));

    return fapl;
}

/**
 * @brief Creates HDF5 file with optimized settings and configures dataspaces
 * 
 * @param fname Output filename
 * @param single_file M2O vs M2M mode selector
 * @param ginfo Global grid information
 * @param local_dims Local rank dimensions [nz, ny, nx]
 * @param[out] dataspace_id File dataspace (global or local)
 * @param[out] memspace_id Memory dataspace (local)
 * @return HDF5 file identifier
 * 
 * For M2O mode, configures hyperslab selection for rank's subdomain within
 * global file. For M2M mode, file and memory spaces are identical.
 */
static hid_t setup_hdf5_file(const char* fname, bool single_file, 
                             const GlobalGridInfo& ginfo, const hsize_t* local_dims,
                             hid_t& dataspace_id, hid_t& memspace_id) {
    // Create file with optimized properties
    hid_t fapl_id = create_optimized_fapl(single_file);
    hid_t file_id = H5_CHECK(H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id));
    H5_CHECK(H5Pclose(fapl_id));
    
    if (file_id < 0) WRITE_ERROR(("Failed to create HDF5 file: %s", fname));

    // Configure file dataspace
    dataspace_id = H5_CHECK(H5Screate_simple(3, ginfo.global_dims, NULL));
    
    if (single_file) {
        // M2O: Select hyperslab for this rank's contribution
        H5_CHECK(H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET,
                           ginfo.offset, NULL, local_dims, NULL));
    }
    
    // Memory dataspace always matches local dimensions
    memspace_id = H5_CHECK(H5Screate_simple(3, local_dims, NULL));
    
    return file_id;
}

// =============================================================================
// HDF5 PARTICLE WRITER
// =============================================================================

/**
 * @brief Writes particle data to HDF5 format
 * 
 * @param fbase Base filename (without extension)
 * @param species_name Name of species to dump
 * @param options Named coordinates and file layout. Position modes:
 *        - true: Output physical (x,y,z) in simulation coordinates, omit voxel index
 *        - false: Output logical (dx,dy,dz) in [-1,1] cell-local coords, include voxel index
 * 
 * Position Mode Details:
 * 
 * PHYSICAL MODE (compute_physical_position = true):
 * - Converts to global simulation coordinates before writing
 * - Numeric position[N][3], momentum[N][3], weight[N] arrays (28 bytes/particle)
 * - Advantages: Self-contained coordinates, easier post-processing
 * - File size: ~12.5% SMALLER than logical mode (saves 4 bytes per particle)
 * 
 * LOGICAL MODE (compute_physical_position = false):
 * - Preserves local coordinates and voxel indices
 * - Numeric cell_offset[N][3], cell_index[N], momentum[N][3], weight[N] (32 bytes/particle)
 * - Preserves logical coordinates; copied momenta are time-centered, not restart state
 * - Note: Post-processing must handle cell-local coordinates and voxel indices
 * 
 * Performance characteristics:
 * - Chunked processing (2M particles per chunk) optimized for HDF5 write size
 * - Physical mode reduces file size by 12.5% (28 vs 32 bytes per particle)
 * - M2O mode uses collective I/O with optimized file access properties
 * 
 * Outputs two metrics:
 * - [METRIC]: Aggregated throughput using configurable timing (pure I/O or total)
 * - [DIAGNOSTIC]: Detailed phase timing breakdown
 * 
 * @note TIME_PURE_IO_ONLY flag controls whether metrics measure only H5Dwrite
 *       time or total function time. Set to true for fair I/O benchmarking.
 * @note Physical mode transformation occurs after center_p_dump(), between
 *       t_compute and t_h5dwrite phases.
 */

// One grid record per contributing rank reconstructs logical positions without
// storing another coordinate array. This is small per-rank metadata only.
static void write_particle_grid(hid_t file, hid_t dxpl, grid_t* g,
                                bool single_file, uint64_t offset, uint64_t count) {
    struct RankGrid {
        uint64_t offset, count;
        int32_t nx, ny, nz;
        double x0, y0, z0, dx, dy, dz;
    } row = {offset, count, g->nx, g->ny, g->nz,
             g->x0, g->y0, g->z0, g->dx, g->dy, g->dz};
    hid_t memory_type = H5_CHECK(H5Tcreate(H5T_COMPOUND, sizeof(RankGrid)));
#define GRID_MEMBER(member, type) \
    H5_CHECK(H5Tinsert(memory_type, #member, offsetof(RankGrid, member), type))
    GRID_MEMBER(offset, H5T_NATIVE_UINT64);
    GRID_MEMBER(count, H5T_NATIVE_UINT64);
    GRID_MEMBER(nx, H5T_NATIVE_INT32);
    GRID_MEMBER(ny, H5T_NATIVE_INT32);
    GRID_MEMBER(nz, H5T_NATIVE_INT32);
    GRID_MEMBER(x0, H5T_NATIVE_DOUBLE);
    GRID_MEMBER(y0, H5T_NATIVE_DOUBLE);
    GRID_MEMBER(z0, H5T_NATIVE_DOUBLE);
    GRID_MEMBER(dx, H5T_NATIVE_DOUBLE);
    GRID_MEMBER(dy, H5T_NATIVE_DOUBLE);
    GRID_MEMBER(dz, H5T_NATIVE_DOUBLE);
#undef GRID_MEMBER
    hid_t file_type = H5_CHECK(H5Tcopy(memory_type));
    H5_CHECK(H5Tpack(file_type)); // 76 bytes per rank, independent of particle count.
    hsize_t rows = single_file ? world_size : 1;
    hsize_t index = single_file ? world_rank : 0, one = 1;
    hid_t space = H5_CHECK(H5Screate_simple(1, &rows, NULL));
    hid_t dataset = H5_CHECK(H5Dcreate2(file, "particle_grid", file_type, space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    H5_CHECK(H5Sselect_hyperslab(space, H5S_SELECT_SET, &index, NULL, &one, NULL));
    hid_t memory = H5_CHECK(H5Screate_simple(1, &one, NULL));
    H5_CHECK(H5Dwrite(dataset, memory_type, memory, space, dxpl, &row));
    H5_CHECK(H5Sclose(memory));
    H5_CHECK(H5Dclose(dataset));
    H5_CHECK(H5Sclose(space));
    H5_CHECK(H5Tclose(file_type));
    H5_CHECK(H5Tclose(memory_type));
}

void vpic_simulation::write_particles_hdf5(const char* fbase, 
                                           const char* species_name, 
                                           ParticleWriteOptions options) {
    const bool single_file = options.layout == OutputLayout::Shared;
    const bool compute_physical_position = options.coordinates == ParticleCoordinates::Physical;
    require_hdf5_mode(single_file);
    KOKKOS_TIC() {
        // =====================================================================
        // CONFIGURATION
        // =====================================================================
        // Controls metric calculation: true = H5Dwrite only, false = total time
        const bool TIME_PURE_IO_ONLY = true;
        // =====================================================================

        double t_total_start = wallclock();

        // Phase timing accumulators
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;

        // =====================================================================
        // PHASE 1: SETUP & PARTICLE COUNT AGGREGATION
        // =====================================================================
        
        double t_phase_start = wallclock();

        species_t* sp = find_species(species_name);
        if(!sp) WRITE_ERROR(("Invalid species: %s", species_name));
        const std::string stem = particle_stem(fbase, sp);
        validate_output("particles_hdf5", stem, sp->g, sp, single_file, compute_physical_position);
        prepare_output_parent(stem);
        sp->copy_to_host();
        // Center the output copy with the same interpolator used by this PIC step.
        interpolator_array->copy_to_host();
        Kokkos::fence();
        grid_t* g = sp->g;

        hsize_t my_count = sp->np;
        hsize_t total_count = my_count;
        hsize_t current_file_offset = 0;

        // For M2O mode, compute file offsets via MPI scan
        #if defined(VPIC_HDF5_PARALLEL) && !defined(VPIC_HDF5_SERIAL_ONLY)
        if (single_file) {
            uint64_t local_val = my_count, scan_val = 0, sum_val = 0;
            MPI_Exscan(&local_val, &scan_val, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_val, &sum_val, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
            if (rank() == 0) scan_val = 0;
            current_file_offset = scan_val;
            total_count = sum_val;
        }
        #endif

        t_setup = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 2: FILE & METADATA CREATION
        // =====================================================================
        
        t_phase_start = wallclock();

        const std::string filename = output_stem(stem, single_file)+".h5";
        const char* fname = filename.c_str();

        // Create file with optimized access properties
        hid_t fapl = create_optimized_fapl(single_file);
        hid_t fid = H5_CHECK(H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
        H5_CHECK(H5Pclose(fapl));
        if(fid < 0) WRITE_ERROR(("Failed HDF5 create: %s", fname));

        // ---------------------------------------------------------------------
        // Both coordinate modes share momentum/weight and aligned particle indices.
        hid_t cell_index = -1;
        const size_t particle_size = compute_physical_position ? 28 : 32;
        hid_t fspace = H5_CHECK(H5Screate_simple(1, &total_count, NULL));
        hsize_t dimensions[2] = {total_count, 3};
        hid_t vector_space = H5_CHECK(H5Screate_simple(2, dimensions, NULL));
        hid_t position = H5_CHECK(H5Dcreate2(fid, compute_physical_position ? "position" : "cell_offset",
            H5T_IEEE_F32LE, vector_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        hid_t momentum = H5_CHECK(H5Dcreate2(fid, "momentum", H5T_IEEE_F32LE, vector_space,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        hid_t weight = H5_CHECK(H5Dcreate2(fid, "weight", H5T_IEEE_F32LE, fspace,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        if (!compute_physical_position)
            cell_index = H5_CHECK(H5Dcreate2(fid, "cell_index", H5T_STD_I32LE, fspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        write_output_metadata(fid, "particles", g, sp, single_file, compute_physical_position ? "physical" : "logical");
        write_text_attr(fid, "momentum_centering", "center_p_dump");

        // Configure data transfer properties
        hid_t dxpl = H5_CHECK(H5Pcreate(H5P_DATASET_XFER));
        #if defined(VPIC_HDF5_PARALLEL) && !defined(VPIC_HDF5_SERIAL_ONLY)
        if (single_file) H5_CHECK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));
        #endif

        if (!compute_physical_position)
            write_particle_grid(fid, dxpl, g, single_file, current_file_offset, my_count);

        t_meta_create = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 3: CHUNKED PARTICLE PROCESSING
        // =====================================================================
        // Process in 2M particle chunks. Each chunk undergoes:
        // Copy → Time-Center → Convert (if physical) → Write
        
        const int PBUF_SIZE = HDF5_PARTICLE_BUFFER_SIZE;
        particle_t* p_buf;
        MALLOC_ALIGNED(p_buf, PBUF_SIZE, 128);

        // Bounded numeric bands; no second on-disk particle payload.
        float* output_buf;
        MALLOC_ALIGNED(output_buf, PBUF_SIZE * 7, 128);
        std::vector<int32_t> index_buf(compute_physical_position ? 0 : PBUF_SIZE);

        auto& k_p_h = sp->k_p_h;
        auto& k_p_i_h = sp->k_p_i_h;
        int sp_np = sp->np;
        int sp_max_np = sp->max_np;

        // Precompute grid parameters for coordinate transformation
        int nx = g->nx, ny = g->ny, nz = g->nz;
        float dx = g->dx, dy = g->dy, dz = g->dz;
        float x0 = g->x0, y0 = g->y0, z0 = g->z0;

        // Synchronize loop count for collective I/O
        double local_loops = (double)((size_t(sp_np) + PBUF_SIZE - 1) / PBUF_SIZE);
        if (sp_np == 0) local_loops = 0.0;
        double max_loops_d = 0.0;
        mp_allmax_d(&local_loops, &max_loops_d, 1);
        int total_loops = single_file ? (int)max_loops_d : (int)local_loops;

        for(int loop_idx = 0; loop_idx < total_loops; ++loop_idx) {
            int buf_start = loop_idx * PBUF_SIZE;
            hsize_t chunk_valid_count = 0;

            if (buf_start < sp_np) {
                // Adjust species counts for this chunk
                sp->np = sp_np - buf_start;
                if(sp->np > PBUF_SIZE) sp->np = PBUF_SIZE;
                sp->max_np = PBUF_SIZE;

                // -------------------------------------------------------------
                // Sub-phase 3a: Copy Chunk to Buffer
                // -------------------------------------------------------------
                double t_loop_start = wallclock();
                
                Kokkos::View<particle_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                    pbuf_view(p_buf, PBUF_SIZE);
                
                Kokkos::parallel_for("PopulateParticleDumpBuffer",
                    Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, sp->np),
                    KOKKOS_LAMBDA(int i) {
                        int idx = buf_start + i;
                        pbuf_view(i).dx = k_p_h(idx, particle_var::dx);
                        pbuf_view(i).dy = k_p_h(idx, particle_var::dy);
                        pbuf_view(i).dz = k_p_h(idx, particle_var::dz);
                        pbuf_view(i).ux = k_p_h(idx, particle_var::ux);
                        pbuf_view(i).uy = k_p_h(idx, particle_var::uy);
                        pbuf_view(i).uz = k_p_h(idx, particle_var::uz);
                        pbuf_view(i).w  = k_p_h(idx, particle_var::w);
                        pbuf_view(i).i  = k_p_i_h(idx);
                    });
                Kokkos::fence();
                
                t_buf_pack += (wallclock() - t_loop_start);

                // -------------------------------------------------------------
                // Sub-phase 3b: Time-Centering of the Output Copy
                // -------------------------------------------------------------
                // center_p_dump() changes copied momenta; it does not filter particles.
                t_loop_start = wallclock();
                center_p_dump(sp, p_buf, interpolator_array);
                chunk_valid_count = sp->np;
                t_compute += (wallclock() - t_loop_start);

                // -------------------------------------------------------------
                // Sub-phase 3c: Coordinate Transformation (Physical Mode Only)
                // -------------------------------------------------------------
                if (chunk_valid_count > 0) {
                    // Transform (dx,dy,dz,i) → (x,y,z) and repack into output buffer
                    // Three contiguous bands: position, momentum, weight.
                    
                    float* out = (float*)output_buf;
                    
                    for(hsize_t n = 0; n < chunk_valid_count; ++n) {
                        if (compute_physical_position) {
                            int ix, iy, iz;
                            UNVOXEL(p_buf[n].i, ix, iy, iz, nx, ny, nz);
                            out[n*3]   = x0 + (ix-1 + .5f*(1+p_buf[n].dx))*dx;
                            out[n*3+1] = y0 + (iy-1 + .5f*(1+p_buf[n].dy))*dy;
                            out[n*3+2] = z0 + (iz-1 + .5f*(1+p_buf[n].dz))*dz;
                        } else {
                            out[n*3]   = p_buf[n].dx;
                            out[n*3+1] = p_buf[n].dy;
                            out[n*3+2] = p_buf[n].dz;
                            index_buf[n] = p_buf[n].i;
                        }
                        out[3*PBUF_SIZE + n*3 + 0] = p_buf[n].ux;  // ux
                        out[3*PBUF_SIZE + n*3 + 1] = p_buf[n].uy;  // uy
                        out[3*PBUF_SIZE + n*3 + 2] = p_buf[n].uz;  // uz
                        out[6*PBUF_SIZE + n] = p_buf[n].w;   // w
                    }
                }
            }

            // -----------------------------------------------------------------
            // Sub-phase 3d: Write Chunk to HDF5
            // -----------------------------------------------------------------
            if (single_file || chunk_valid_count > 0) {
                double t_loop_start = wallclock();
                
                hid_t mspace = H5_CHECK(H5Screate_simple(1, &chunk_valid_count, NULL));
                if (chunk_valid_count) {
                    H5_CHECK(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &current_file_offset, NULL, &chunk_valid_count, NULL));
                } else {
                    H5_CHECK(H5Sselect_none(mspace));
                    H5_CHECK(H5Sselect_none(fspace));
                }
                
                {
                    hsize_t dimensions[2] = {chunk_valid_count, 3};
                    hsize_t offset[2] = {current_file_offset, 0};
                    hid_t memory = H5_CHECK(H5Screate_simple(2, dimensions, NULL));
                    if (chunk_valid_count) {
                        H5_CHECK(H5Sselect_hyperslab(vector_space, H5S_SELECT_SET,
                                                   offset, NULL, dimensions, NULL));
                    } else {
                        H5_CHECK(H5Sselect_none(memory));
                        H5_CHECK(H5Sselect_none(vector_space));
                    }
                    H5_CHECK(H5Dwrite(position, H5T_NATIVE_FLOAT, memory, vector_space, dxpl, output_buf));
                    H5_CHECK(H5Dwrite(momentum, H5T_NATIVE_FLOAT, memory, vector_space, dxpl, output_buf+3*PBUF_SIZE));
                    H5_CHECK(H5Dwrite(weight, H5T_NATIVE_FLOAT, mspace, fspace, dxpl, output_buf+6*PBUF_SIZE));
                    H5_CHECK(H5Sclose(memory));
                }
                if (!compute_physical_position)
                    H5_CHECK(H5Dwrite(cell_index, H5T_NATIVE_INT32, mspace, fspace, dxpl, index_buf.data()));

                H5_CHECK(H5Sclose(mspace));
                current_file_offset += chunk_valid_count;
                t_h5dwrite += (wallclock() - t_loop_start);
            }
        }

        // Restore species state
        sp->np = sp_np;
        sp->max_np = sp_max_np;
        FREE_ALIGNED(p_buf);
        FREE_ALIGNED(output_buf);

        // =====================================================================
        // PHASE 4: CLEANUP & CLOSE
        // =====================================================================
        
        t_phase_start = wallclock();

        H5_CHECK(H5Dclose(position));
        H5_CHECK(H5Dclose(momentum));
        H5_CHECK(H5Dclose(weight));
        H5_CHECK(H5Sclose(vector_space));
        if (!compute_physical_position) H5_CHECK(H5Dclose(cell_index));
        H5_CHECK(H5Pclose(dxpl));
        H5_CHECK(H5Sclose(fspace));
        H5_CHECK(H5Fclose(fid));

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        // Find maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute,
               max_h5dwrite, max_meta_close, max_total;
        
        mp_allmax_d(&t_setup, &max_setup, 1);
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        mp_allmax_d(&t_compute, &max_compute, 1);
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        mp_allmax_d(&t_total, &max_total, 1);
        
        // Calculate total data volume
        double mb_total = 0.0;
        if (single_file) {
            mb_total = (double)(total_count * particle_size) / (1024.0 * 1024.0);
        } else {
            double mb_local = (double)(my_count * particle_size) / (1024.0 * 1024.0);
            mp_allsum_d(&mb_local, &mb_total, 1);
        }

        if (rank() == 0) {
            // Determine which time metric to use based on internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            // Generate tag based on mode and file layout
            const char* file_mode = single_file ? "M2O" : "M2M";
            const char* pos_mode = compute_physical_position ? "physical" : "logical";
            char tag[64];
            snprintf(tag, 64, "write_particles_hdf5_%s_%s", file_mode, pos_mode);
            
            // Standard metric output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f",
                     tag, (long)step(), t_metric, mb_total, throughput));
            
            // Detailed diagnostic output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,"
                     "BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f",
                     tag, (long)step(), max_total, max_setup, max_meta_create,
                     max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }
        // =====================================================================
        // XDMF GENERATION (ParaView Support)
        // =====================================================================

        if (compute_physical_position) {
            if (single_file) {
                if (rank() == 0) {
                    std::string file = stem;
                    write_xdmf_particles((file+".xmf").c_str(),
                        std::filesystem::path(file+".h5").filename().string().c_str(), total_count, g);
                }
            } else {
                std::string file = output_stem(stem, false);
                write_xdmf_particles((file+".xmf").c_str(),
                    std::filesystem::path(file+".h5").filename().string().c_str(), my_count, g);
                mp_barrier();
                if (rank() == 0) {
                    std::vector<std::string> children;
                    for (int r=0; r<nproc(); ++r)
                        children.push_back(std::filesystem::path(output_stem(stem, false, r)+".xmf").filename().string());
                    write_rank_collection((stem+".xmf").c_str(), children, g);
                }
            }
        }

    } KOKKOS_TOC(write_particles_hdf5, 1);
}

// =============================================================================
// HDF5 FIELDS WRITER
// =============================================================================

/**
 * @brief Writes electromagnetic field data to HDF5 format
 * 
 * @param params DumpParameters containing output variables and paths
 * @param fa Field array containing E and B field data
 * @param single_file True for M2O (single shared file), false for M2M (per-rank files)
 * 
 * This function performs a critical memory layout transformation:
 * - Input: Array of Structures (AoS) - field_t structs with interleaved components
 * - Output: Structure of Arrays (SoA) - separate datasets per component
 * 
 * The transformation is necessary because:
 * 1. HDF5 performs best with contiguous data per dataset
 * 2. Post-processing tools expect one variable per dataset
 * 3. Enables selective variable loading (don't need to read entire field_t)
 * 
 * Performance characteristics:
 * - Uses per-call Kokkos buffers released before finalization
 * - Single-pass repacking via MDRangePolicy (cache-friendly)
 * - Table-driven variable selection (no conditional logic in inner loops)
 * 
 * Outputs two metrics:
 * - [METRIC]: Aggregated throughput using configurable timing (pure I/O or total)
 * - [DIAGNOSTIC]: Detailed phase timing breakdown
 * 
 * @note TIME_PURE_IO_ONLY flag controls whether metrics measure only H5Dwrite
 *       time or total function time. Set to true for fair I/O benchmarking.
 */
void vpic_simulation::write_fields_hdf5(DumpParameters& params, 
                                        field_array_t* fa, 
                                        bool single_file) {
    require_hdf5_mode(single_file);
    KOKKOS_TIC() {
        // =====================================================================
        // CONFIGURATION
        // =====================================================================
        // Controls metric calculation: true = H5Dwrite only, false = total time
        const bool TIME_PURE_IO_ONLY = true;
        // =====================================================================

        double t_total_start = wallclock();

        // Phase timing accumulators
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;      // Unused for fields (no equivalent to center_p_dump)
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;

        // =====================================================================
        // PHASE 1: SETUP & VARIABLE SELECTION
        // =====================================================================
        
        double t_phase_start = wallclock();

        if (!fa) WRITE_ERROR(("NULL field array"));
        if (step() > fa->last_copied) fa->copy_to_host();
        if (fa->g != grid) WRITE_ERROR(("Field output requires the simulation grid"));
        grid_t* g = fa->g;
        validate_output("fields_hdf5", structured_stem(params, g), g, nullptr,
                        single_file, false, output_mask(params, 24));

        const std::string stem = structured_stem(params, g);
        prepare_output_parent(stem);

        // ---------------------------------------------------------------------
        // Variable Mapping Table
        // ---------------------------------------------------------------------
        // Maps user-selected bit indices to field_t memory layout.
        // Only variables with bitset(bit) == true will be written.

        // Filter to only actively requested variables
        struct ActiveVar { 
            const char* name; 
            size_t offset;
            VarType type;
        };
        std::vector<ActiveVar> active_vars;
        
        for (const auto& m : field_map) {
            if (params.output_vars.bitset(m.bit)) {
                active_vars.push_back({m.name, m.offset, m.type});
            }
        }
        int num_active = active_vars.size();

        t_setup = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 2: FILE & METADATA CREATION
        // =====================================================================
        
        t_phase_start = wallclock();

        const std::string filename = output_stem(stem, single_file)+".h5";
        const char* fname = filename.c_str();

        // Compute grid layout (global dimensions and rank offset)
        hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};
        GlobalGridInfo ginfo = get_global_grid_info(g, single_file);

        // Create file with optimized settings
        hid_t dspace, mspace;
        hid_t fid = setup_hdf5_file(fname, single_file, ginfo, local_dims, dspace, mspace);

        // Write simulation metadata attributes
        write_output_metadata(fid, "fields", g, nullptr, single_file, "physical");
        write_grid_metadata(fid, ginfo, g);

        // Configure data transfer properties
        hid_t dxpl = H5_CHECK(H5Pcreate(H5P_DATASET_XFER));
        #if defined(VPIC_HDF5_PARALLEL) && !defined(VPIC_HDF5_SERIAL_ONLY)
        if (single_file) {
            H5_CHECK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));
        }
        #endif

        t_meta_create = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 3: MEMORY LAYOUT TRANSFORMATION (AoS → SoA)
        // =====================================================================
        // Convert from field_t array (interleaved components) to separate
        // contiguous arrays per component. This requires a full data copy but
        // enables optimal HDF5 write performance and post-processing access.
        
        size_t num_cells = size_t(g->nx) * g->ny * g->nz;
        
        // Per-call ownership releases storage before simulation finalization.
        Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>
            buffer("h5_field_buf", active_vars.size(), num_cells);

        t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Single-Pass Repacking Kernel
        // ---------------------------------------------------------------------
        // Uses MDRangePolicy for cache-friendly 3D iteration.
        // Each thread extracts all requested components from one field_t cell.
        
        auto* f_base = fa->f;
        int nx = g->nx, ny = g->ny, nz = g->nz;
        
        using Policy3D = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, 
                                               Kokkos::Rank<3>>;
        
        Kokkos::parallel_for("PackFieldsSinglePass", 
            Policy3D({1, 1, 1}, {nz+1, ny+1, nx+1}),
            [&](int k, int j, int i) {
                // Compute linear index in output buffer (Z-major ordering)
                size_t buf_idx = ((k-1) * ny * nx) + ((j-1) * nx) + (i-1);
                
                // Get base pointer to this cell's field_t struct
                char* cell_base = (char*)&f_base[voxel(i, j, k)];
                
                // Extract all active components via offset pointer arithmetic
                for (int v = 0; v < num_active; ++v) {
                    const char* p = cell_base + active_vars[v].offset;
                    buffer(v, buf_idx) = (active_vars[v].type == VT_MATID)
                       ? (float)(*(material_id*)p)   // widen int -> float
                       : *(float*)p;
                }
            });
        Kokkos::fence();

        t_buf_pack = wallclock() - t_phase_start;
        t_compute = 0.0;  // No physics computation for fields

        // =====================================================================
        // PHASE 4: WRITE DATA TO HDF5
        // =====================================================================
        // Write each component as a separate 3D dataset.
        // For M2O, all ranks participate in collective write.
        
        t_phase_start = wallclock();

        for (int v = 0; v < num_active; ++v) {
            hid_t dset = H5_CHECK(H5Dcreate2(fid, active_vars[v].name, H5T_NATIVE_FLOAT,
                                   dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            
            // Write from buffer row v (contiguous data for this component)
            H5_CHECK(H5Dwrite(dset, H5T_NATIVE_FLOAT, mspace, dspace, dxpl, &buffer(v, 0)));
            write_sample_metadata(dset, active_vars[v].name, ginfo, g);
            H5_CHECK(H5Dclose(dset));
        }

        t_h5dwrite = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 5: CLEANUP & CLOSE
        // =====================================================================
        
                t_phase_start = wallclock();

        H5_CHECK(H5Pclose(dxpl));
        H5_CHECK(H5Sclose(mspace));
        H5_CHECK(H5Sclose(dspace));
        H5_CHECK(H5Fclose(fid));

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        // Find maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute,
               max_h5dwrite, max_meta_close, max_total;
        
        mp_allmax_d(&t_setup, &max_setup, 1);
        
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        
        mp_allmax_d(&t_compute, &max_compute, 1);
        
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        
        mp_allmax_d(&t_total, &max_total, 1);

        if (rank() == 0 && num_active > 0) {
            // Calculate total data volume across all ranks
            double mb_total = ((double)(num_cells * num_active * sizeof(float)) / 
                              (1024.0 * 1024.0)) * nproc();
            
            // Determine which time metric to use based on internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            const char* tag = single_file ? "write_fields_hdf5_M2O" : "write_fields_hdf5_M2M";
            
            // Standard metric output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f", 
                     tag, (long)step(), t_metric, mb_total, throughput));
            
            // Detailed diagnostic output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,"
                     "BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f",
                     tag, (long)step(), max_total, max_setup, max_meta_create,
                     max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }
        
                // =====================================================================
        // PHASE 6: XDMF GENERATION (ParaView Support)
        // =====================================================================
        
        // **ALL RANKS COMPUTE GRID INFO (MPI COLLECTIVES)**
        const XDMFGridInfo xinfo = xdmf_grid_info(ginfo, g);
        std::vector<const char*> var_list;
        for (const auto& v : active_vars) var_list.push_back(v.name);
        publish_structured_xdmf(params, single_file, var_list, xinfo, g);

    } KOKKOS_TOC(write_fields_hdf5, 1);
}

// =============================================================================
// HDF5 HYDRO WRITER
// =============================================================================

/**
 * @brief Writes hydrodynamic moment data to HDF5 format
 * 
 * @param params DumpParameters containing output variables and paths
 * @param ha Hydro array structure for moment storage
 * @param sp_name Name of species to compute moments for
 * @param single_file True for M2O (single shared file), false for M2M (per-rank files)
 * 
 * This function differs from field writing in a critical way: hydro moments are
 * NOT pre-computed during the simulation. They must be accumulated from particle
 * data on-demand during I/O:
 * 
 * Physics Pipeline:
 * 1. Clear hydro array (J, ρ, pressure tensor, etc.)
 * 2. Accumulate moments from particle distribution via scatter operation
 * 3. Synchronize ghost cells across MPI boundaries
 * 4. Write to disk
 * 
 * This "compute-during-I/O" model means setup time includes physics calculations,
 * making it fundamentally different from field dumps where data already exists.
 * 
 * Memory Layout Transformation (like fields):
 * - Input: Array of Structures (AoS) - hydro_t structs with 14 interleaved moments
 * - Output: Structure of Arrays (SoA) - separate datasets per moment
 * 
 * Performance characteristics:
 * - Physics accumulation dominates setup time (~60-80% for particle-rich cells)
 * - Uses per-call Kokkos buffers released before finalization
 * - Single-pass repacking via MDRangePolicy (cache-friendly)
 * - Table-driven variable selection (no conditional logic in inner loops)
 * 
 * Outputs two metrics:
 * - [METRIC]: Aggregated throughput using configurable timing (pure I/O or total)
 * - [DIAGNOSTIC]: Detailed phase timing breakdown
 * 
 * @note TIME_PURE_IO_ONLY flag controls whether metrics measure only H5Dwrite
 *       time or total function time. Set to true for fair I/O benchmarking.
 * @note Setup timing includes physics accumulation, making it much larger than
 *       for fields/particles where data pre-exists.
 */
void vpic_simulation::write_hydro_hdf5(DumpParameters& params, 
                                       hydro_array_t* ha, 
                                       const char* sp_name, 
                                       bool single_file) {
    require_hdf5_mode(single_file);
    KOKKOS_TIC() {
        // =====================================================================
        // CONFIGURATION
        // =====================================================================
        // Controls metric calculation: true = H5Dwrite only, false = total time
        const bool TIME_PURE_IO_ONLY = true;
        // =====================================================================

        double t_total_start = wallclock();

        // Phase timing accumulators
        double t_setup = 0.0;
        double t_meta_create = 0.0;
        double t_buf_pack = 0.0;
        double t_compute = 0.0;      // Unused (physics is in setup phase)
        double t_h5dwrite = 0.0;
        double t_meta_close = 0.0;

        // =====================================================================
        // PHASE 1: SETUP & PHYSICS ACCUMULATION
        // =====================================================================
        // Unlike fields (pre-computed), hydro moments must be calculated from
        // particles on-demand. This is the most expensive phase.
        
        double t_phase_start = wallclock();

        if (!ha) WRITE_ERROR(("NULL hydro array"));
        
        species_t* sp = find_species(sp_name);
        if (!sp) WRITE_ERROR(("Invalid species: %s", sp_name));

        // ---------------------------------------------------------------------
        // Physics Accumulation: Scatter particles → moments
        // ---------------------------------------------------------------------
        // Computes current density (J), charge density (ρ), momentum (P),
        // kinetic energy (KE), and pressure tensor (T) from particle distribution.
        
        if (ha->g != grid || sp->g != ha->g || interpolator_array->g != ha->g)
            WRITE_ERROR(("Hydro output requires matching simulation, species and interpolator grids"));
        validate_output("hydro_hdf5", structured_stem(params, ha->g), ha->g, sp,
                        single_file, false, output_mask(params, 14));
        Kokkos::deep_copy(ha->k_h_d, 0.0f);
        accumulate_hydro_p_kokkos(sp->k_p_d, sp->k_p_i_d, ha->k_h_d,
                                 interpolator_array->k_i_d, sp);
        ha->copy_to_host();
        synchronize_hydro_array(ha);  // MPI ghost cell exchange
        
        grid_t* g = ha->g;

        const std::string stem = structured_stem(params, g);
        prepare_output_parent(stem);

        // ---------------------------------------------------------------------
        // Variable Mapping Table
        // ---------------------------------------------------------------------
        // Maps user-selected bit indices to hydro_t memory layout.
        // Only variables with bitset(bit) == true will be written.
        
        // Filter to only actively requested variables
        struct ActiveVar {
            const char* name;
            size_t offset;
        };
        std::vector<ActiveVar> active_vars;
        
        for (const auto& m : hydro_map) {
            if (params.output_vars.bitset(m.bit)) {
                active_vars.push_back({m.name, m.offset});
            }
        }
        int num_active = active_vars.size();

        t_setup = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 2: FILE & METADATA CREATION
        // =====================================================================
        
        t_phase_start = wallclock();

        const std::string filename = output_stem(stem, single_file)+".h5";
        const char* fname = filename.c_str();

        // Compute grid layout (global dimensions and rank offset)
        hsize_t local_dims[3] = {(hsize_t)g->nz, (hsize_t)g->ny, (hsize_t)g->nx};
        GlobalGridInfo ginfo = get_global_grid_info(g, single_file);

        // Create file with optimized settings
        hid_t dspace, mspace;
        hid_t fid = setup_hdf5_file(fname, single_file, ginfo, local_dims, dspace, mspace);

        // Write simulation metadata attributes
        write_output_metadata(fid, "hydro", g, sp, single_file, "physical");
        write_grid_metadata(fid, ginfo, g);

        // Configure data transfer properties
        hid_t dxpl = H5_CHECK(H5Pcreate(H5P_DATASET_XFER));
        #if defined(VPIC_HDF5_PARALLEL) && !defined(VPIC_HDF5_SERIAL_ONLY)
        if (single_file) {
            H5_CHECK(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));
        }
        #endif

        t_meta_create = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 3: MEMORY LAYOUT TRANSFORMATION (AoS → SoA)
        // =====================================================================
        // Convert from hydro_t array (14 interleaved moments) to separate
        // contiguous arrays per moment. This enables optimal HDF5 write
        // performance and selective post-processing access.
        
        size_t num_cells = size_t(g->nx) * g->ny * g->nz;
        
        // Per-call ownership releases storage before simulation finalization.
        Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::HostSpace>
            buffer("h5_hydro_buf", active_vars.size(), num_cells);

        t_phase_start = wallclock();

        // ---------------------------------------------------------------------
        // Single-Pass Repacking Kernel
        // ---------------------------------------------------------------------
        // Uses MDRangePolicy for cache-friendly 3D iteration.
        // Each thread extracts all requested moments from one hydro_t cell.
        
        auto* h_base = ha->h;
        int nx = g->nx, ny = g->ny, nz = g->nz;
        
        using Policy3D = Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace,
                                               Kokkos::Rank<3>>;
        
        Kokkos::parallel_for("PackHydroSinglePass",
            Policy3D({1, 1, 1}, {nz+1, ny+1, nx+1}),
            [&](int k, int j, int i) {
                // Compute linear index in output buffer (Z-major ordering)
                size_t buf_idx = ((k-1) * ny * nx) + ((j-1) * nx) + (i-1);
                
                // Get base pointer to this cell's hydro_t struct
                char* cell_base = (char*)&h_base[voxel(i, j, k)];
                
                // Extract all active moments via offset pointer arithmetic
                for (int v = 0; v < num_active; ++v) {
                    buffer(v, buf_idx) = *(float*)(cell_base + active_vars[v].offset);
                }
            });
        Kokkos::fence();

        t_buf_pack = wallclock() - t_phase_start;
        
        // t_compute unused: physics accumulation already happened in setup phase
        t_compute = 0.0;

        // =====================================================================
        // PHASE 4: WRITE DATA TO HDF5
        // =====================================================================
        // Write each moment as a separate 3D dataset.
        // For M2O, all ranks participate in collective write.
        
        t_phase_start = wallclock();

        for (int v = 0; v < num_active; ++v) {
            hid_t dset = H5_CHECK(H5Dcreate2(fid, active_vars[v].name, H5T_NATIVE_FLOAT,
                                   dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            
            // Write from buffer row v (contiguous data for this moment)
            H5_CHECK(H5Dwrite(dset, H5T_NATIVE_FLOAT, mspace, dspace, dxpl, &buffer(v, 0)));
            write_sample_metadata(dset, active_vars[v].name, ginfo, g);
            H5_CHECK(H5Dclose(dset));
        }

        t_h5dwrite = wallclock() - t_phase_start;

        // =====================================================================
        // PHASE 5: CLEANUP & CLOSE
        // =====================================================================
        
        t_phase_start = wallclock();

        H5_CHECK(H5Pclose(dxpl));
        H5_CHECK(H5Sclose(mspace));
        H5_CHECK(H5Sclose(dspace));
        H5_CHECK(H5Fclose(fid));

        t_meta_close = wallclock() - t_phase_start;
        double t_total = wallclock() - t_total_start;

        // =====================================================================
        // PERFORMANCE METRICS
        // =====================================================================
        
        // Find maximum time spent in each phase across all MPI ranks
        double max_setup, max_meta_create, max_buf_pack, max_compute,
               max_h5dwrite, max_meta_close, max_total;
        
        mp_allmax_d(&t_setup, &max_setup, 1);
        mp_allmax_d(&t_meta_create, &max_meta_create, 1);
        mp_allmax_d(&t_buf_pack, &max_buf_pack, 1);
        mp_allmax_d(&t_compute, &max_compute, 1);
        mp_allmax_d(&t_h5dwrite, &max_h5dwrite, 1);
        mp_allmax_d(&t_meta_close, &max_meta_close, 1);
        mp_allmax_d(&t_total, &max_total, 1);

        if (rank() == 0 && num_active > 0) {
            // Calculate total data volume across all ranks
            double mb_total = ((double)(num_cells * num_active * sizeof(float)) /
                              (1024.0 * 1024.0)) * nproc();
            
            // Determine which time metric to use based on internal flag
            double t_metric = TIME_PURE_IO_ONLY ? max_h5dwrite : max_total;
            double throughput = mb_total / t_metric;
            
            const char* tag = single_file ? "write_hydro_hdf5_M2O" : "write_hydro_hdf5_M2M";
            
            // Standard metric output
            MESSAGE(("[METRIC],%s,%ld,%.4f,%.4f,%.4f",
                     tag, (long)step(), t_metric, mb_total, throughput));
            
            // Detailed diagnostic output
            MESSAGE(("[DIAGNOSTIC],%s,%ld,Total:%.4f,Setup:%.4f,MetaCreate:%.4f,"
                     "BufPack:%.4f,Compute:%.4f,H5Dwrite:%.4f,MetaClose:%.4f",
                     tag, (long)step(), max_total, max_setup, max_meta_create,
                     max_buf_pack, max_compute, max_h5dwrite, max_meta_close));
        }
        // Compute geometry on all ranks before publishing child/master wrappers.
        const XDMFGridInfo xinfo = xdmf_grid_info(ginfo, g);
        std::vector<const char*> var_list;
        for (const auto& v : active_vars) var_list.push_back(v.name);
        publish_structured_xdmf(params, single_file, var_list, xinfo, g);

    } KOKKOS_TOC(write_hydro_hdf5, 1);

    
}

#endif // VPIC_ENABLE_HDF5

#undef H5_CHECK
#undef WRITE_ERROR
