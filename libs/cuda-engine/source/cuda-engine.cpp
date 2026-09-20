// Standard libs
#include <cstdio>
#include <stdint.h>
// Cuda libs
#include <cuda_runtime_api.h>
#include <vector_types.h>
// My libs
#include "cuda-engine/cuda-engine.h"

// ===== Helpers ===========================================================

// Outputs the nearest multiple of a >= x, where a is power of 2. Used here to pad allocations to 128 bytes to
// match the GPU global memory transaction granularity.
static constexpr size_t alignUp(
    size_t x,
    size_t a // Must be a power of 2
)
{ return (x + a - 1) & ~(a - 1); }
// Bump-allocator: returns the current offset, advances past `bytes`. `bytes` must be pre-aligned (see the
// alignUp at each *_bytes definition).
static uint8_t* carve(
    uint8_t*& ptr,
    size_t bytes
)
{
    uint8_t* r  = ptr;
    ptr        += bytes;
    return r;
}

// ===== GraphEnv ==========================================================

GraphEnv::~GraphEnv()
{
    if (exec) { cudaGraphExecDestroy(exec); }
    if (graph) { cudaGraphDestroy(graph); }
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (t1) { cudaEventDestroy(t1); }
    if (t0) { cudaEventDestroy(t0); }
    if (tick_done) { cudaEventDestroy(tick_done); }
}

// ===== OutputArena =======================================================

OutputArena::OutputArena()
{
    constexpr size_t float2_bytes  = alignUp((size_t)kNumChunks * kMaxAtoms * sizeof(float2), 128);
    constexpr size_t integer_bytes = alignUp((size_t)kNumChunks * sizeof(int32_t), 128);
    // Must equal the sum of the carve ptr sequence below. Update together
    constexpr size_t total_bytes = 2u * float2_bytes + integer_bytes;

    CUDACHECK(cudaMalloc((void**)&base, total_bytes));
    CUDACHECK(cudaMemset(base, 0, total_bytes));

    uint8_t* ptr = base;
    pos          = (float2*)carve(ptr, float2_bytes);
    vel          = (float2*)carve(ptr, float2_bytes);
    num_atoms    = (int32_t*)carve(ptr, integer_bytes);
}
OutputArena::~OutputArena()
{
    if (base) { cudaFree(base); }
}

// ===== OutputBuffer ======================================================

// Rotates through the triple buffer. Only the read slot needs atomic ops as the writer owns the OutputBuffer
// variable.
void OutputBuffer::roll()
{
    read_slot.store(write_slot, std::memory_order_release);
    write_slot = (write_slot + 1) % 3;
}
// Returns the current renderable tick output location and slot number.
View OutputBuffer::readTarget()
{
    const int32_t s = read_slot.load();
    View v;
    v.pos       = buffer[s].pos;
    v.vel       = buffer[s].vel;
    v.num_atoms = buffer[s].num_atoms;
    v.slot      = s;
    return v;
}
// Returns the next designated tick output location and slot number.
View OutputBuffer::writeTarget()
{
    const int32_t s = write_slot;
    View v;
    v.pos       = buffer[s].pos;
    v.vel       = buffer[s].vel;
    v.num_atoms = buffer[s].num_atoms;
    v.slot      = s;
    return v;
}

// ===== PBF ===============================================================

PBFArena::PBFArena()
{
    // clang-format off
    constexpr size_t inner_offsets_bytes  = alignUp((size_t)kNumChunks * kCellsPerChunk * sizeof(uint32_t), 128u);
    constexpr size_t inner_float2_bytes   = alignUp((size_t)kNumChunks * kMaxAtoms * sizeof(float2), 128u);
    constexpr size_t inner_float_bytes   = alignUp((size_t)kNumChunks * kMaxAtoms * sizeof(float), 128u);
    constexpr size_t num_atoms_bytes      = alignUp((size_t)kNumChunks * sizeof(int32_t), 128u);
    constexpr size_t extended_chunk_grid = (size_t)((kSceneDimX + 2) * (kSceneDimY + 2)); // + 2 for the halo of chunkless migration buffers around the simulation
    constexpr size_t mig_float2_bytes     = alignUp(extended_chunk_grid * (size_t)kMaxMigrants * sizeof(float2), 128u);
    constexpr size_t mig_count_bytes      = alignUp(extended_chunk_grid * sizeof(int32_t), 128u);
    constexpr size_t halo_offsets_bytes   = alignUp((size_t)kNumChunks * kHaloCells * sizeof(uint32_t), 128u);
    constexpr size_t halo_float2_bytes    = alignUp((size_t)kNumChunks * kHaloAtoms * sizeof(float2), 128u);
    constexpr size_t halo_float_bytes    = alignUp((size_t)kNumChunks * kHaloAtoms * sizeof(float), 128u);
    // Must equal the sum of the carve ptr sequence below. Update together
    constexpr size_t total_bytes = inner_float_bytes + inner_float2_bytes * 3u + mig_float2_bytes * 2u +
                                   halo_float_bytes + halo_float2_bytes * 3u + mig_count_bytes +
                                   halo_offsets_bytes + inner_offsets_bytes + num_atoms_bytes;
    // clang-format on

    CUDACHECK(cudaMalloc((void**)&base, total_bytes));
    CUDACHECK(cudaMemset(base, 0, total_bytes));

    uint8_t* ptr = base;
    offsets      = (uint32_t*)carve(ptr, inner_offsets_bytes);
    pos[0]       = (float2*)carve(ptr, inner_float2_bytes);
    pos[1]       = (float2*)carve(ptr, inner_float2_bytes);
    vel          = (float2*)carve(ptr, inner_float2_bytes);
    lambda       = (float*)carve(ptr, inner_float_bytes);
    num_atoms    = (int32_t*)carve(ptr, num_atoms_bytes);
    mig_pos      = (float2*)carve(ptr, mig_float2_bytes);
    mig_vel      = (float2*)carve(ptr, mig_float2_bytes);
    mig_count    = (int32_t*)carve(ptr, mig_count_bytes);
    halo_offsets = (uint32_t*)carve(ptr, halo_offsets_bytes);
    halo_pos[0]  = (float2*)carve(ptr, halo_float2_bytes);
    halo_pos[1]  = (float2*)carve(ptr, halo_float2_bytes);
    halo_vel     = (float2*)carve(ptr, halo_float2_bytes);
    halo_lambda  = (float*)carve(ptr, halo_float_bytes);
}
PBFArena::~PBFArena()
{
    if (base) { cudaFree(base); }
}
// ===== MPM ===============================================================

MPMArena::MPMArena()
{
    constexpr size_t grid_bytes  = alignUp(kGridDimX * kGridDimY * sizeof(float4), 128u);
    constexpr size_t pos_bytes   = alignUp(kNumAtoms * sizeof(float2), 128u);
    constexpr size_t vel_bytes   = alignUp(kNumAtoms * sizeof(float2), 128u);
    constexpr size_t C_bytes     = alignUp(kNumAtoms * sizeof(float4), 128u);
    constexpr size_t F_bytes     = alignUp(kNumAtoms * sizeof(float4), 128u);
    constexpr size_t J_bytes     = alignUp(kNumAtoms * sizeof(float), 128u);
    constexpr size_t count_bytes = 2u * sizeof(int32_t);
    // Must equal the sum of the carve ptr sequence below. Update together
    constexpr size_t total_bytes =
        2u * grid_bytes + pos_bytes + vel_bytes + C_bytes + F_bytes + J_bytes + count_bytes;

    CUDACHECK(cudaMalloc((void**)&base, total_bytes));
    CUDACHECK(cudaMemset(base, 0, total_bytes));

    uint8_t* ptr = base;
    grid[0]      = (float4*)carve(ptr, grid_bytes);
    grid[1]      = (float4*)carve(ptr, grid_bytes);
    pos          = (float2*)carve(ptr, pos_bytes);
    vel          = (float2*)carve(ptr, vel_bytes);
    C            = (float4*)carve(ptr, C_bytes);
    F            = (float4*)carve(ptr, F_bytes);
    J            = (float*)carve(ptr, J_bytes);
    num_points   = (int32_t*)carve(ptr, count_bytes);
}
MPMArena::~MPMArena()
{
    if (base) { cudaFree(base); }
}