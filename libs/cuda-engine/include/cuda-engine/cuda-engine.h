#pragma once
// Standard libs
#include <atomic>
#include <cstdint>
#include <cstdio>

// CUDA libs
#include <cuda_runtime_api.h>
#include <vector_types.h>
// My libs
#include "common/umbrella.h"

// ===== General Structs ===============================================

// Container for objects that define the tick graph runtime
struct GraphEnv
{
    cudaStream_t stream   = nullptr;
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec  = nullptr;
    cudaEvent_t tick_done = nullptr;        // cudaEventDisableTiming
    cudaEvent_t t0 = nullptr, t1 = nullptr; // default flags

    GraphEnv() = default;
    ~GraphEnv();
    GraphEnv(const GraphEnv&)            = delete;
    GraphEnv& operator=(const GraphEnv&) = delete;
};

// ===== Tick Output Objects ===========================================

// Arena for the final simulation state of each tick. This is the location of the primary GPU-side interface
// between the simulation and the renderer.
struct OutputArena
{
    float2* pos        = nullptr;
    float2* vel        = nullptr;
    int32_t* num_atoms = nullptr;

    OutputArena();
    ~OutputArena();

    OutputArena(const OutputArena&)            = delete;
    OutputArena& operator=(const OutputArena&) = delete;

  private:
    uint8_t* base = nullptr;
};

// Container for rotating buffer data
struct View
{
    float2* pos        = nullptr;
    float2* vel        = nullptr;
    int32_t* num_atoms = nullptr;
    int32_t slot       = -1;
};

// A storage class for a rotating triple buffer allowing the tick and frame rates to be independent. The
// renderer reads from buffer i % 3 while the engine writes to buffer (i + 1) % 3. Buffer (i + 2) % 3
// serves as a cushion to keep the writer from outrunning the reader.
class OutputBuffer
{
  public:
    void roll();
    View readTarget();
    View writeTarget();

    OutputBuffer()
        : read_slot(0),
          write_slot(1) {};

    OutputBuffer(const OutputBuffer&)            = delete;
    OutputBuffer& operator=(const OutputBuffer&) = delete;

  private:
    std::atomic<int32_t> read_slot;
    int32_t write_slot;
    OutputArena buffer[3]{};
};

struct TickStaging
{
    TickStaging() = default;
    ~TickStaging()
    {
        if (staged) { cudaFreeHost(staged); }
    }

    void init(
        View initial_target,
        int2 initial_seams,
        cudaStream_t stream
    );

    View& target() { return staged->target; }
    int2& torus() { return staged->torus; }

    void outputTargetUpload(cudaStream_t stream);
    void torusUpload(cudaStream_t stream);

    TickStaging(const TickStaging&)            = delete;
    TickStaging& operator=(const TickStaging&) = delete;

  private:
    struct Staged
    {
        View target;
        int2 torus;
    };
    Staged* staged = nullptr; // one pinned allocation, both fields
};

// ===== PBF declarations ============================================

// Simulation memory arena: hierarchy = data category -> chunk -> atom.
// While chunk data represents the spatial partitioning of the simulation, the physical addresses are not tied
// to specific regions of the simulation. As the user's view pans across the scene, out-of-view chunk data is
// unloaded and new in-view data is loaded into those same memory slots, causing decoherence between the
// data's position in the simulation and their physical addresses. The locations of these axis-bound
// discontinuity "seams" are tracked in a device-resident __constant__ "torus" variable, so named because each
// rolling seam traverses the chunk index space under a toroidal topology - each seam wraps back around to
// the start when it progresses through all the chunk boundaries (along its axis).
struct PBFArena
{
    // Inner chunk data: data of atoms owned by my chunk. Capacity = kMaxAtoms.
    // These data define the bulk of the simulation state. Some of this data is shared between chunks during
    // each tick, but is only ever mutated by the parent chunk.
    uint32_t* offsets = nullptr; // Bit-packed [start, end) indices of each grid cell
    float2* pos[2]{};            // pos[0] = current, pos[1] = predicted
    float2* vel   = nullptr; // Equal to v * dt - i.e. displacement, not true velocity. dt is given by kPBF_dt
    float* lambda = nullptr;
    int32_t* num_atoms = nullptr; // Per chunk

    // Migration data: migration buffers for atoms wishing to enter my chunk. Capacity = kMaxMigrants,
    // drained once per substep.
    // If the migration buffers reach capacity, the original chunks retain their outgoing migrants
    // until queued migrants are drained. If inner atom capacity is reached before the migration buffer fully
    // drains, residual migrants remain queued until space is freed.
    float2* mig_pos    = nullptr; // Migration buffer for inbound atom positions
    float2* mig_vel    = nullptr; // Likewise for velocities
    int32_t* mig_count = nullptr; // Number of inbound atoms in migration queue

    // Halo data: neighbouring chunks' atom data that may interact with my outer atoms. Capacity = kHaloAtoms.
    // The halo is organised into grid cells (max 8 atoms each) which are arranged by neighbour
    // position relative to my chunk: side regions (N, S, E, W) w/ 16 cells each and corner regions (NW, NE,
    // SW, SE) w/ one cell each. Halo data is directly appended to inner chunk data in shared memory for
    // indexing convenience.
    uint32_t* halo_offsets = nullptr; // Offsets begin at kMaxAtoms as halo data is appended to main data
    float2* halo_pos[2]{}; // Double buffered data still being read isn't overwritten during iterations
    float2* halo_vel   = nullptr;
    float* halo_lambda = nullptr;

    PBFArena();
    ~PBFArena();
    PBFArena(const PBFArena&)            = delete;
    PBFArena& operator=(const PBFArena&) = delete;

  private:
    uint8_t* base = nullptr;
};
// Instantiates the complete tick graph starting from substep 0 and ending when the final state is handed over
// to the renderer.
void instantiatePBFGraph(
    GraphEnv& g,
    PBFArena& pbf
);
// Launches a "hosepipe" style fluid emitter kernel to populate the simulation. Emission rate is determined by
// velocity and nozzle width.
int32_t hosePipe(
    const cudaStream_t stream, // PBF graph stream to prevent mid-tick migration channel contention
    const PBFArena& pbf,       // Chunk memory locations
    const float2 origin,       // Nozzle centre, absolute scene coords
    const float2 velocity,     // Base velocity before jitter is added
    const float nozzle_width,  // Number of atoms across the nozzle
    const float spacing,       // Grid spacing between atoms
    const float vel_jitter, // Sets the standard deviation of the velocity randomiser (philox, gaussian dist.)
    const uint64_t tick     // Used to seed the prng
);

// ===== MPM declarations ============================================

// Vehicle to allow the graph instantiation function to return multiple nodes. Only contains nodes that need
// parameter updates at runtime.
struct MPMNodeHandles
{
    cudaGraphNode_t clear;
    cudaGraphNode_t output;
};
struct MPMArena
{
    // Node grid: double buffered so resets can run concurrently with the substep; has a 2-node halo around
    // the simulation bounds to prevent OOB memory reads instead of adding an extra layer of indexing
    // arithmetic.
    float4* grid[2];

    // MP (material point) data: encodes the physical state of the material. Each unique index below kNumAtoms
    // corresponds to an individual simulated MP.
    float2* pos =
        nullptr; // Positions relative to simulation space, where the origin is in the top left corner
    float2* vel = nullptr; // Velocities (actual, not the substep displacements of the PBF implementation)
    float4* C   = nullptr; // Velocity gradient matrices: encodes the rotation/shear at each point
    float4* F   = nullptr; // Deformation gradient tensors, maintained but currently unused
    float* J    = nullptr; // Volumetric Jacobians

    int32_t* num_points = nullptr; // 2 slots, both list the total number of MPs in the simulation. Slot 0 is
                                   // authoritative and used for indexing, slot 1 is used to stage changes to
                                   // the population, after which its value overwrites that of slot 0.
    MPMArena();
    ~MPMArena();
    MPMArena(const MPMArena&)            = delete;
    MPMArena& operator=(const MPMArena&) = delete;

  private:
    uint8_t* base = nullptr;
};
int32_t hosePipe(
    const cudaStream_t stream,
    MPMArena& mpm,
    const float2 origin,
    const float2 velocity,
    const float nozzle_width,
    const float spacing,
    const float vel_jitter,
    const uint64_t tick
);
void instantiateMPMGraph(
    GraphEnv& g_env,
    MPMArena& mpm
);
struct MPMTickStaging
{
    MPMTickStaging() = default;
    ~MPMTickStaging()
    {
        if (staged) { cudaFreeHost(staged); }
    }

    void init(
        View initial_target,
        int2 initial_seams,
        cudaStream_t stream
    );

    View& target() { return staged->target; }
    int2& torus() { return staged->torus; }

    void outputTargetUpload(cudaStream_t stream);
    void torusUpload(cudaStream_t stream);

    MPMTickStaging(const MPMTickStaging&)            = delete;
    MPMTickStaging& operator=(const MPMTickStaging&) = delete;

  private:
    struct Staged
    {
        View target;
        int2 torus;
    };
    Staged* staged = nullptr; // one pinned allocation, both fields
};