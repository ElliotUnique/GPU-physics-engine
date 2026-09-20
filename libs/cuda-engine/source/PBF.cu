// CUDA libs
#include <vector_types.h>
// My libs
#include "common/umbrella.h"
#include "cuda-engine/PBF-internal.cuh"
#include "cuda-engine/cuda-engine.h"
#include "cuda-engine/cuda-internal.cuh"

// ===== Compile time Constants ============================================

// Dials
constexpr int32_t kIters     = 2;
constexpr int32_t kNumSteps  = 20;
constexpr float kEpsilon     = 0.005f;
constexpr float kSCorrK      = 0.005f; // Scale for artificial pressure term
constexpr float kC           = 0.035f;
constexpr float kPBF_dt      = kDt / (float)kNumSteps;
constexpr float kPBFGravity  = 1000.f;
constexpr float kGravityStep = kPBFGravity * kPBF_dt * kPBF_dt;
constexpr float kOmega       = 1.25f; // Jacobi over-relaxation, [1, 2)
constexpr float kMinC        = -0.002f;
constexpr float kMinSpeed    = 0.00005f; // Lowest speed before clamping to 0
constexpr float kMaxSpeed    = 1.f;

// ===== Device global variables ===========================================

// Location of the tick output targets, modified by the host each tick to point to the next output
// destination.
__constant__ View c_output;

// Encodes the positions of the simulation seams in chunk (x, y) coordinates. The host modifies this when old
// scene data is unloaded and new scene data loaded in its place.
__constant__ int2 c_torus;

// ===== TickStaging =======================================================

__host__ void TickStaging::init(
    View initial_target,
    int2 initial_seams,
    cudaStream_t stream
)
{
    CUDACHECK(cudaMallocHost((void**)&staged, sizeof(Staged)));
    staged->target = initial_target;
    staged->torus  = initial_seams;
    outputTargetUpload(stream);
    torusUpload(stream);
}

__host__ void TickStaging::outputTargetUpload(cudaStream_t stream)
{
    CUDACHECK(
        cudaMemcpyToSymbolAsync(c_output, &staged->target, sizeof(View), 0, cudaMemcpyHostToDevice, stream)
    );
}

__host__ void TickStaging::torusUpload(cudaStream_t stream)
{
    CUDACHECK(
        cudaMemcpyToSymbolAsync(c_torus, &staged->torus, sizeof(int2), 0, cudaMemcpyHostToDevice, stream)
    );
}

// ===== PBF ===============================================================

__global__ void haloKernel(
    const uint32_t* offsets,
    const float2* new_pos,
    uint32_t* halo_offsets,
    float2* halo_pos,
    const int32_t* num_atoms
)
{
    auto getChunkIdx = [](int32_t chunk_x, int32_t chunk_y) { return chunk_x + chunk_y * kSceneDimX; };

    const int32_t chunk_x   = (int32_t)(blockIdx.x * blockDim.y + threadIdx.y);
    const int32_t chunk_y   = (int32_t)blockIdx.y;
    const int32_t chunk_idx = getChunkIdx(chunk_x, chunk_y);
    if (chunk_x >= kSceneDimX || chunk_y >= kSceneDimY) { return; }
    const int32_t n    = num_atoms[chunk_idx];
    const int32_t tid  = threadIdx.x;
    const int32_t cell = tid / kCellSlots;
    const int32_t slot = tid & (kCellSlots - 1); // kCellSlots must be a power of two, else replace with %
    const uint32_t* my_offsets = offsets + chunk_idx * kCellsPerChunk;

    // Grid sides
    const uint32_t north         = my_offsets[cell];
    const uint32_t south         = my_offsets[cell + (kGridDim - 1) * kGridDim];
    const uint32_t west          = my_offsets[cell * kGridDim];
    const uint32_t east          = my_offsets[cell * kGridDim + kGridDim - 1];
    const uint32_t cell_range[4] = {north, south, west, east};
    float2 edge_pos[4]{};
    if (n == 0) { return; } // Allow the offsets preloads while n is being lifted from global.

    // Gather outgoing halo data into registers
#pragma unroll
    for (int32_t i = 0; i < 4; ++i) {
        const int32_t start = (cell_range[i] & 0x0000FFFF);
        const int32_t end   = (cell_range[i] >> 16);
        if ((slot) < (end - start)) { edge_pos[i] = new_pos[chunk_idx * kMaxAtoms + start + slot]; }
    }
#pragma unroll
    for (int32_t region = 0; region < 4; ++region) {
        const int32_t start = cell_range[region] & 0x0000FFFF;
        const int32_t end   = cell_range[region] >> 16;
        auto dst            = cardinalDest(chunk_x, chunk_y, region, cell, slot, cell_range[region]);
        if (dst.valid) {
            const float2 shift                         = cardinalShift(region);
            const float new_x                          = edge_pos[region].x + shift.x;
            const float new_y                          = edge_pos[region].y + shift.y;
            halo_pos[dst.chunk * kHaloAtoms + dst.idx] = float2{new_x, new_y};
            if (slot == 0) {
                const int32_t pop       = min(end - start, kCellSlots);
                const int32_t new_start = (kMaxAtoms + dst.idx);
                const int32_t dst_idx   = dst.chunk * kHaloCells + cell + kGridDim * dst.region;
                halo_offsets[dst_idx]   = ((new_start + pop) << 16) | new_start;
            }
        }
        // Dispatch to the ordinal directions
        dst = ordinalDest(chunk_x, chunk_y, region, cell, slot, cell_range[region]);
        if (dst.valid) {
            const float2 shift     = ordinalShift(region, cell);
            const float new_x      = edge_pos[region].x + shift.x;
            const float new_y      = edge_pos[region].y + shift.y;
            const int32_t slot_idx = dst.chunk * kHaloAtoms + dst.idx;
            halo_pos[slot_idx]     = {new_x, new_y};

            constexpr int32_t grid_side_slots = kCellSlots * kGridDim * 4;
            if (slot == 0) {
                const int32_t pop         = min(end - start, kCellSlots);
                const int32_t new_start   = kMaxAtoms + grid_side_slots + dst.region * kCellSlots;
                const int32_t offsets_idx = dst.chunk * kHaloCells + kGridDim * 4 + dst.region;
                halo_offsets[offsets_idx] = ((new_start + pop) << 16) | new_start;
            }
        }
    }
}
__inline__ __device__ float2 cardinalShift(int32_t region)
{
    const float s = (region & 1) ? -kChunkDim : kChunkDim;
    return {(region >= 2) ? s : 0.f, (region < 2) ? s : 0.f};
}
__inline__ __device__ float2 ordinalShift(
    int32_t region,
    int32_t cell
)
{ return {(cell == (kGridDim - 1)) ? -kChunkDim : kChunkDim, (region & 1) ? -kChunkDim : kChunkDim}; }
__inline__ __device__ HaloCompass cardinalDest(
    const int32_t chunk_x,
    const int32_t chunk_y,
    const int32_t region,
    const int32_t cell,
    const int32_t slot,
    const uint32_t cell_range
)
{
    const int32_t start = cell_range & 0xFFFF;
    const int32_t end   = cell_range >> 16;
    const bool axis     = region < 2; // true = y-axis, false = x-axis
    const int32_t dir   = 2 * (region & 1) - 1;

    // Boundary checks in Euclidean index space
    const int32_t dx    = dir * (int32_t)(!axis);
    const int32_t dy    = dir * (int32_t)(axis);
    const int32_t dst_x = stepX(chunk_x, dx);
    const int32_t dst_y = stepY(chunk_y, dy);
    const bool dst_valid =
        !crossesSeam(chunk_x, dst_x, dx, c_torus.x) && !crossesSeam(chunk_y, dst_y, dy, c_torus.y);

    // Compute destination in toroidal index space
    const int32_t dst_chunk  = dst_x + dst_y * kSceneDimX;
    const int32_t dst_region = (~region & 1) + 2 * !axis; // Dest region inverted: N -> S, W -> E, etc
    const int32_t cell_idx   = (cell + kGridDim * dst_region) * kCellSlots;
    const bool in_range      = slot < end - start;
    HaloCompass out;
    out.valid  = dst_valid && in_range;
    out.region = dst_region;
    out.chunk  = dst_chunk;
    out.idx    = slot + cell_idx;
    return out;
}
__inline__ __device__ HaloCompass ordinalDest(
    const int32_t chunk_x,
    const int32_t chunk_y,
    const int32_t region,
    const int32_t cell,
    const int32_t slot,
    const uint32_t cell_range
)
{
    if (region >= 2 || !(cell == 0 || cell == 15)) { return HaloCompass{}; }
    const int32_t start = cell_range & 0xFFFF;
    const int32_t end   = cell_range >> 16;

    // Boundary checks in Euclidean index space
    const int32_t dx    = (cell == 0) ? -1 : 1;
    const int32_t dy    = 2 * (region & 1) - 1;
    const int32_t dst_x = stepX(chunk_x, dx);
    const int32_t dst_y = stepY(chunk_y, dy);
    const bool dst_valid =
        !crossesSeam(chunk_x, dst_x, dx, c_torus.x) && !crossesSeam(chunk_y, dst_y, dy, c_torus.y);

    // Compute destination in toroidal index space
    const int32_t dst_chunk  = dst_x + dst_y * kSceneDimX;
    const int32_t dst_region = (int32_t)(cell == 0) + (int32_t)(region == 0) * 2;
    const int32_t cell_idx   = (dst_region + kGridDim * 4) * kCellSlots;
    const bool in_range      = slot < end - start;
    HaloCompass out;
    out.valid  = dst_valid && in_range;
    out.region = dst_region;
    out.chunk  = dst_chunk;
    out.idx    = slot + cell_idx;
    return out;
}
__global__ void lambdaKernel(
    const uint32_t* offsets,
    const float2* new_pos,
    float* lambda,
    const uint32_t* halo_offsets,
    const float2* halo_pos,
    float* halo_lambda,
    const int32_t* num_atoms
)
{
    constexpr int32_t BLOCKSIZE        = 512;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;

    __shared__ float2 s_pos[kMaxAtoms + kHaloAtoms];
    __shared__ uint32_t s_offsets[kCellsPerChunk + kHaloCells];

    const int32_t chunk_x = blockIdx.x;
    const int32_t chunk_y = blockIdx.y;
    const int32_t chunk   = chunk_x + chunk_y * kSceneDimX;
    const int32_t n       = num_atoms[chunk];
    const int32_t tid     = threadIdx.x;
    if (n == 0) { return; }

    loadPool<BLOCKSIZE>(chunk, n, smemLoad(s_pos, new_pos));
    loadHalo<BLOCKSIZE>(chunk, smemLoad(s_pos, halo_pos));
    loadOffsets<BLOCKSIZE>(chunk, offsets, halo_offsets, s_offsets);
    __syncthreads();

    // Neighbourhood search and lambda computation
#pragma unroll
    for (int32_t a = 0; a < ATOMS_PER_THREAD; ++a) {
        const int32_t atom = tid + BLOCKSIZE * a;
        if (atom >= n) { continue; }
        const float2 p = s_pos[atom];
        float2 del_C{};
        float del_C_sqr      = 0.f;
        float rho            = 0.f;
        const int32_t base_x = clamp<int32_t>((int32_t)floorf(p.x) / kCellDim, 0, kGridDim - 1);
        const int32_t base_y = clamp<int32_t>((int32_t)floorf(p.y) / kCellDim, 0, kGridDim - 1);
#pragma unroll
        for (int32_t y = 0; y < 3; ++y) {
            const int32_t cell_y = base_y + y;
#pragma unroll
            for (int32_t x = 0; x < 3; ++x) {
                const int32_t cell_x = base_x + x;
                const int32_t cell   = cell_x + cell_y * (kGridDim + 2);
                const float2 pA      = s_pos[atom];
                const int32_t start  = s_offsets[cell] & 0x0000FFFF;
                const int32_t end    = s_offsets[cell] >> 16;
                for (int32_t i = start; i < end; ++i) {
                    const float2 pB      = s_pos[i];
                    const float2 r       = {pA.x - pB.x, pA.y - pB.y};
                    const float sqr_dist = r.x * r.x + r.y * r.y;

                    if (sqr_dist > kH * kH) { continue; }

                    const float inv_dist = (sqr_dist > 1e-12f) ? rsqrtf(sqr_dist) : 0.f;
                    const float dist     = (inv_dist > 0.f) ? inv_dist * sqr_dist : 0.f;
                    float2 r_norm;
                    if (pA.x != pB.x || pA.y != pB.y) {
                        r_norm = {r.x * inv_dist, r.y * inv_dist};
                    } else {
                        r_norm = pairDirection(atom, i);
                    }
                    float2 spiky_vec  = spiky(dist, r_norm);
                    spiky_vec.x      *= kMass;
                    spiky_vec.y      *= kMass;
                    del_C.x          += spiky_vec.x;
                    del_C.y          += spiky_vec.y;
                    del_C_sqr        += kInvMass * (spiky_vec.x * spiky_vec.x + spiky_vec.y * spiky_vec.y);
                    rho              += kMass * poly6(sqr_dist);
                }
            }
        }
        // const float C = rho * kInvRho0 - 1.f;
        const float C                    = fmaxf(rho * kInvRho0 - 1.f, kMinC);
        const float self_grad_sqr        = kInvMass * (del_C.x * del_C.x + del_C.y * del_C.y);
        const float grad_sqr             = kInvRho0 * kInvRho0 * (self_grad_sqr + del_C_sqr);
        lambda[chunk * kMaxAtoms + atom] = -C / (grad_sqr + kEpsilon);
    }
    __syncthreads();
    // Dispatch halo lambdas
    const HaloThread h        = haloThreadDecode(tid);
    const int32_t src_cell    = haloSrcCell(h.region, h.cell);
    const uint32_t cell_range = s_offsets[paddedCellIdx(src_cell)];
    const int32_t src_idx     = haloSrcAtom(cell_range, h.slot);
    const float l             = lambda[chunk * kMaxAtoms + src_idx];
    // Cardinals
    auto dst = cardinalDest(chunk_x, chunk_y, h.region, h.cell, h.slot, cell_range);
    if (dst.valid) { halo_lambda[dst.chunk * kHaloAtoms + dst.idx] = l; }
    // Ordinals
    dst = ordinalDest(chunk_x, chunk_y, h.region, h.cell, h.slot, cell_range);
    if (dst.valid) { halo_lambda[dst.chunk * kHaloAtoms + dst.idx] = l; }
}
__device__ inline float2 pairDirection(
    const int32_t a,
    const int32_t b
)
{
    const uint32_t lo = (uint32_t)min(a, b);
    const uint32_t hi = (uint32_t)max(a, b);
    const float ang   = (float)(hash(lo * 0x9E3779B9u ^ hi) & 0xFFFFu) * (6.2831853f / 65536.f);
    float s, c;
    __sincosf(ang, &s, &c);
    const float sign = (a < b) ? 1.f : -1.f;
    return {sign * c, sign * s};
}
__device__ float poly6(float sqr_dist)
{
    float sqr_diff = kH * kH - sqr_dist;
    return kCoeffPoly6 * sqr_diff * sqr_diff * sqr_diff;
}
__device__ float2 spiky(
    float dist,
    float2 r_norm
)
{
    float diff     = kH - dist;
    float diff_sqr = diff * diff;
    float scalar   = -kCoeffSpiky * diff_sqr;
    return {scalar * r_norm.x, scalar * r_norm.y};
}
__device__ __forceinline__ HaloThread haloThreadDecode(const int32_t tid)
{
    static_assert((kCellSlots & (kCellSlots - 1)) == 0, "kCellSlots must be a power of two");
    static_assert((kGridDim & (kGridDim - 1)) == 0, "kGridDim must be a power of two");
    HaloThread h;
    h.region = tid / (kCellSlots * kGridDim);
    h.cell   = (tid & (kCellSlots * kGridDim - 1)) / kCellSlots;
    h.slot   = tid & (kCellSlots - 1);
    return h;
}
__device__ __forceinline__ int32_t haloSrcCell(
    const int32_t region,
    const int32_t cell
)
{
    return (region < 2) ? cell + kGridDim * (kGridDim - 1) * (region & 1) :
                          cell * kGridDim + (kGridDim - 1) * (region & 1);
}
__device__ __forceinline__ int32_t paddedCellIdx(const int32_t src_cell)
{ return ((src_cell / kGridDim) + 1) * (kGridDim + 2) + (src_cell & (kGridDim - 1)) + 1; }
__device__ __forceinline__ int32_t haloSrcAtom(
    const uint32_t cell_range,
    const int32_t slot
)
{
    const int32_t start = (int32_t)(cell_range & 0xFFFFu);
    return clamp<int32_t>(start + slot, 0, kMaxAtoms - 1);
}
__global__ void posCorrectionKernel(
    const uint32_t* offsets,
    float2* new_pos,
    const float* lambda,
    const uint32_t* halo_offsets,
    const float2* halo_pos,
    float2* halo_pos_buf,
    const float* halo_lambda,
    const int32_t* num_atoms
)
{
    constexpr int32_t BLOCKSIZE        = 512;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;
    __shared__ float s_lambda[kMaxAtoms + kHaloAtoms];
    __shared__ float2 s_pos[kMaxAtoms + kHaloAtoms];
    __shared__ uint32_t s_offsets[kCellsPerChunk + kHaloCells];

    const int32_t chunk_x = blockIdx.x;
    const int32_t chunk_y = blockIdx.y;
    const int32_t chunk   = chunk_x + chunk_y * kSceneDimX;
    float2* out_pos       = new_pos + chunk * kMaxAtoms;
    const int32_t n       = num_atoms[chunk];
    const int32_t tid     = threadIdx.x;
    if (n == 0) { return; }

    loadPool<BLOCKSIZE>(chunk, n, smemLoad(s_pos, new_pos), smemLoad(s_lambda, lambda));
    loadHalo<BLOCKSIZE>(chunk, smemLoad(s_pos, halo_pos), smemLoad(s_lambda, halo_lambda));
    loadOffsets<BLOCKSIZE>(chunk, offsets, halo_offsets, s_offsets);
    __syncthreads();

    // Neighbourhood search and lambda computation
    float2 dp[ATOMS_PER_THREAD]{};
#pragma unroll
    for (int32_t a = 0; a < ATOMS_PER_THREAD; ++a) {
        const int32_t atom = tid + BLOCKSIZE * a;
        if (atom >= n) { continue; }
        const float2 pA   = s_pos[atom];
        const float lmbdA = s_lambda[atom];
        // Base cell coordinates stay within 16x16. Extension to 18x18 grid coords handled by loop bounds.
        const int32_t base_x = clamp<int32_t>((int32_t)floorf(pA.x) / kCellDim, 0, kGridDim - 1);
        const int32_t base_y = clamp<int32_t>((int32_t)floorf(pA.y) / kCellDim, 0, kGridDim - 1);
#pragma unroll
        for (int32_t y = 0; y < 3; ++y) {
            const int32_t cell_y = base_y + y;
#pragma unroll
            for (int32_t x = 0; x < 3; ++x) {
                const int32_t cell_x = base_x + x;
                const int32_t cell   = cell_x + cell_y * (kGridDim + 2);
                const int32_t start  = s_offsets[cell] & 0x0000FFFF;
                const int32_t end    = s_offsets[cell] >> 16;
                for (int32_t i = start; i < end; ++i) {
                    const float2 pB      = s_pos[i];
                    const float lmbdB    = s_lambda[i];
                    const float2 r       = {pA.x - pB.x, pA.y - pB.y};
                    const float sqr_dist = r.x * r.x + r.y * r.y;

                    if (sqr_dist > kH * kH) { continue; }

                    const float inv_dist    = (sqr_dist > 1e-12f) ? rsqrtf(sqr_dist) : 0.f;
                    const float2 r_norm     = {r.x * inv_dist, r.y * inv_dist};
                    const float ratio       = poly6(sqr_dist) * kInvPoly6Dq;
                    const float ratio2      = ratio * ratio;
                    const float s_corr      = kSCorrK * ratio2 * ratio2; // kSCorrK > 0
                    const float lmbdABC     = lmbdA * kMass + lmbdB * kMass - s_corr;
                    const float dist        = inv_dist * sqr_dist;
                    const float2 spiky_vec  = spiky(dist, r_norm);
                    dp[a].x                += spiky_vec.x * lmbdABC;
                    dp[a].y                += spiky_vec.y * lmbdABC;
                }
            }
        }
        dp[a].x *= kOmega * kInvMass * kInvRho0;
        dp[a].y *= kOmega * kInvMass * kInvRho0;
    }
    __syncthreads();
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + BLOCKSIZE * i;
        if (atom >= n) { continue; }
        s_pos[atom].x += dp[i].x;
        s_pos[atom].y += dp[i].y;
    }
    handleOOB(chunk_x, chunk_y, s_pos, BLOCKSIZE, ATOMS_PER_THREAD, n);
    __syncthreads();
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + BLOCKSIZE * i;
        if (atom >= n) { continue; }
        out_pos[atom].x = s_pos[atom].x;
        out_pos[atom].y = s_pos[atom].y;
    }
    // Dispatch halo predicted positions
    const HaloThread h        = haloThreadDecode(tid);
    const int32_t src_cell    = haloSrcCell(h.region, h.cell);
    const uint32_t cell_range = s_offsets[paddedCellIdx(src_cell)];
    const int32_t src_idx     = haloSrcAtom(cell_range, h.slot);
    const float2 p            = s_pos[src_idx];
    // Cardinals
    float2 shift = cardinalShift(h.region);
    auto dst     = cardinalDest(chunk_x, chunk_y, h.region, h.cell, h.slot, cell_range);
    if (dst.valid) { halo_pos_buf[dst.chunk * kHaloAtoms + dst.idx] = {p.x + shift.x, p.y + shift.y}; }
    // Ordinals
    shift = ordinalShift(h.region, h.cell);
    dst   = ordinalDest(chunk_x, chunk_y, h.region, h.cell, h.slot, cell_range);
    if (dst.valid) { halo_pos_buf[dst.chunk * kHaloAtoms + dst.idx] = {p.x + shift.x, p.y + shift.y}; }
}
__device__ __forceinline__ void handleOOB(
    const int32_t chunk_x,
    const int32_t chunk_y,
    float2* s_pos,
    const int32_t block_size,
    const int32_t atoms_per_thread,
    const int32_t n
)
{
    const bool N = chunk_y == 0;
    const bool S = chunk_y == kSceneDimY - 1;
    const bool W = chunk_x == 0;
    const bool E = chunk_x == kSceneDimX - 1;

    if (!(N || S || W || E)) { return; }

    const float y_lb = N ? 0.f : -1e9f;
    const float y_ub = S ? kChunkDim - 1e-3f : 1e9f; // 1e-3 to make rounded-down cell membership unambiguous
    const float x_lb = W ? 0.f : -1e9f;
    const float x_ub = E ? kChunkDim - 1e-3f : 1e9f;

#pragma unroll
    for (int32_t i = 0; i < atoms_per_thread; ++i) {
        const int32_t atom = threadIdx.x + block_size * i;
        if (atom >= n) { return; }
        s_pos[atom].y = clamp<float>(s_pos[atom].y, y_lb, y_ub);
        s_pos[atom].x = clamp<float>(s_pos[atom].x, x_lb, x_ub);
    }
}
__global__ void velocityKernel(
    const uint32_t* offsets,
    const float2* old_pos,
    float2* new_pos,
    float2* vel,
    float2* halo_vel,
    const int32_t* num_atoms
)
{
    constexpr int32_t BLOCKSIZE        = 512;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;
    __shared__ float2 s_vel[kMaxAtoms];
    __shared__ uint32_t s_offsets[kCellsPerChunk];

    const int32_t tid   = threadIdx.x;
    int32_t chunk_x     = blockIdx.x;
    int32_t chunk_y     = blockIdx.y;
    const int32_t chunk = chunk_x + chunk_y * kSceneDimX;
    const int32_t n     = num_atoms[chunk];
    if (n == 0) { return; }
    if (tid < kCellsPerChunk) { s_offsets[tid] = offsets[chunk * kCellsPerChunk + tid]; }

    // Compute new velocities
    float2 old_p[ATOMS_PER_THREAD]{};
    float2 new_p[ATOMS_PER_THREAD]{};
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + BLOCKSIZE * i;
        if (atom >= n) { break; }
        old_p[i] = old_pos[chunk * kMaxAtoms + atom];
        new_p[i] = new_pos[chunk * kMaxAtoms + atom];
    }
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + BLOCKSIZE * i;
        if (atom >= n) { break; }

        const int32_t idx  = chunk * kMaxAtoms + atom;
        const float2 new_v = {new_p[i].x - old_p[i].x, new_p[i].y - old_p[i].y};
        s_vel[atom]        = new_v;
        vel[idx]           = new_v;
    }
    __syncthreads();
    // Distribute new halo velocities
    const HaloThread h        = haloThreadDecode(tid);
    const int32_t src_cell    = haloSrcCell(h.region, h.cell);
    const uint32_t cell_range = s_offsets[src_cell];
    const int32_t src_idx     = haloSrcAtom(cell_range, h.slot);

    auto dst = cardinalDest(chunk_x, chunk_y, h.region, h.cell, h.slot, cell_range);
    if (dst.valid) { halo_vel[dst.chunk * kHaloAtoms + dst.idx] = s_vel[src_idx]; }
    dst = ordinalDest(chunk_x, chunk_y, h.region, h.cell, h.slot, cell_range);
    if (dst.valid) { halo_vel[dst.chunk * kHaloAtoms + dst.idx] = s_vel[src_idx]; }
}
__global__ void XSPHKernel(
    const uint32_t* offsets,
    const float2* new_pos,
    float2* vel,
    const uint32_t* halo_offsets,
    const float2* halo_pos,
    const float2* halo_vel,
    const int32_t* num_atoms
)
{
    constexpr int32_t BLOCKSIZE        = 1024;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;
    __shared__ float2 s_vel[kMaxAtoms + kHaloAtoms];
    __shared__ float2 s_pos[kMaxAtoms + kHaloAtoms];
    __shared__ uint32_t s_offsets[kCellsPerChunk + kHaloCells];

    const int32_t chunk_x = blockIdx.x;
    const int32_t chunk_y = blockIdx.y;
    const int32_t chunk   = chunk_x + chunk_y * kSceneDimX;
    const int32_t n       = num_atoms[chunk];
    const int32_t tid     = threadIdx.x;
    if (n == 0) { return; }

    loadPool<BLOCKSIZE>(chunk, n, smemLoad(s_pos, new_pos), smemLoad(s_vel, vel));
    loadHalo<BLOCKSIZE>(chunk, smemLoad(s_pos, halo_pos), smemLoad(s_vel, halo_vel));
    loadOffsets<BLOCKSIZE>(chunk, offsets, halo_offsets, s_offsets);
    __syncthreads();

    // Neighbourhood search and velocity smoothing
    float2 dv[ATOMS_PER_THREAD]{};
#pragma unroll
    for (int32_t a = 0; a < ATOMS_PER_THREAD; ++a) {
        const int32_t atom = tid + BLOCKSIZE * a;
        if (atom >= n) { break; }
        const float2 pA = s_pos[atom];
        const float2 vA = s_vel[atom];
        // Base cell coordinates stay within 16x16. Extension to 18x18 grid coords handled by loop bounds.
        const int32_t base_x = clamp<int32_t>((int32_t)floorf(pA.x) / kCellDim, 0, kGridDim - 1);
        const int32_t base_y = clamp<int32_t>((int32_t)floorf(pA.y) / kCellDim, 0, kGridDim - 1);
#pragma unroll
        for (int32_t y = 0; y < 3; ++y) {
            const int32_t cell_y = base_y + y;
#pragma unroll
            for (int32_t x = 0; x < 3; ++x) {
                const int32_t cell_x = base_x + x;
                const int32_t cell   = cell_x + cell_y * (kGridDim + 2);
                const int32_t start  = s_offsets[cell] & 0x0000FFFF;
                const int32_t end    = s_offsets[cell] >> 16;
                for (int32_t i = start; i < end; ++i) {
                    const float2 pB      = s_pos[i];
                    const float2 vB      = s_vel[i];
                    const float2 r       = {pA.x - pB.x, pA.y - pB.y};
                    const float sqr_dist = r.x * r.x + r.y * r.y;
                    if (sqr_dist > kH * kH) { continue; }
                    float p6  = poly6(sqr_dist);
                    dv[a].x  += (vB.x - vA.x) * p6;
                    dv[a].y  += (vB.y - vA.y) * p6;
                }
            }
        }
        dv[a].x *= kC;
        dv[a].y *= kC;
    }
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + BLOCKSIZE * i;
        if (atom >= n) { break; }
        float2 v  = s_vel[atom];
        v.x      += dv[i].x;
        v.y      += dv[i].y;
        velocityClamp(v);
        float2* my_chunk_vel = vel + chunk * kMaxAtoms;
        my_chunk_vel[atom]   = v;
    }
}
__device__ void velocityClamp(float2& v)
{
    const float v_sqr     = v.x * v.x + v.y * v.y;
    const float inv_abs_v = (v_sqr < kMinSpeed * kMinSpeed) ? 0 : rsqrtf(v_sqr);
    const float2 v_norm   = {v.x * inv_abs_v, v.y * inv_abs_v};
    const float abs_v     = v_sqr * inv_abs_v;
    const float speed     = min(abs_v, kMaxSpeed);
    v.x                   = v_norm.x * speed;
    v.y                   = v_norm.y * speed;
}
__global__ void migrationKernel(
    float2* new_pos,
    float2* vel,
    int32_t* num_atoms,
    float2* mig_pos,
    float2* mig_vel,
    int32_t* mig_count
)
{
    constexpr int32_t BLOCKSIZE        = 512;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;

    // holes = departed slots below new_n, movers = surviving atoms at/above new_n.
    // |holes| == |movers| <= min(d, n - d) <= kMaxAtoms / 2, hence the sizing.
    __shared__ int32_t s_holes[kMaxAtoms / 2];
    __shared__ int32_t s_movers[kMaxAtoms / 2];
    __shared__ int32_t s_dep_count;
    __shared__ int32_t s_hole_cnt;
    __shared__ int32_t s_mover_cnt;

    const int32_t chunk_x    = blockIdx.x;
    const int32_t chunk_y    = blockIdx.y;
    const int32_t chunk      = chunk_x + chunk_y * kSceneDimX;
    const int32_t n          = num_atoms[chunk];
    float2* my_chunk_new_pos = new_pos + chunk * kMaxAtoms;
    float2* my_chunk_vel     = vel + chunk * kMaxAtoms;
    const int32_t tid        = threadIdx.x;
    if (n == 0) { return; }
    if (tid == 0) {
        s_dep_count = 0;
        s_hole_cnt  = 0;
        s_mover_cnt = 0;
    }
    __syncthreads();

    const bool wall_W = (chunk_x == 0);
    const bool wall_E = (chunk_x == kSceneDimX - 1);
    const bool wall_N = (chunk_y == 0);
    const bool wall_S = (chunk_y == kSceneDimY - 1);

    // Phase 1: predicate + export. Departure is recorded per-thread in a register
    // mask rather than recomputed, so the decision and the export always coincide.
    uint32_t departed = 0;
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n) { break; }
        const float2 p    = my_chunk_new_pos[atom];
        const float2 v    = my_chunk_vel[atom];
        const float2 pred = {p.x + v.x, p.y + v.y + kGravityStep};

        int32_t dx = (pred.x < 0.f) ? -1 : ((pred.x >= kChunkDim) ? 1 : 0);
        int32_t dy = (pred.y < 0.f) ? -1 : ((pred.y >= kChunkDim) ? 1 : 0);
        if ((wall_W && dx < 0) || (wall_E && dx > 0)) { dx = 0; }
        if ((wall_N && dy < 0) || (wall_S && dy > 0)) { dy = 0; }

        if (dx || dy) {
            int32_t dst_x = stepX(chunk_x, dx);
            int32_t dst_y = stepY(chunk_y, dy);
            if (crossesSeam(chunk_x, dst_x, dx, c_torus.x) || crossesSeam(chunk_y, dst_y, dy, c_torus.y)) {
                // Outermost destinations are in Euclidean index space as they will be read by CPU processes
                dst_x = wrapX(chunk_x, c_torus) + dx;
                dst_y = wrapY(chunk_y, c_torus) + dy;
            }
            const int32_t dst_chunk = (dst_x + 1) + (dst_y + 1) * (kSceneDimX + 2);
            const int32_t slot      = atomicAdd(&mig_count[dst_chunk], 1);
            if (slot < kMaxMigrants) {
                const int32_t mig_idx  = dst_chunk * kMaxMigrants + slot;
                mig_pos[mig_idx]       = {p.x - dx * kChunkDim, p.y - dy * kChunkDim};
                mig_vel[mig_idx]       = v;
                departed              |= 1u << i;
                atomicAdd(&s_dep_count, 1);
            }
            // else: no slot won -> not a departer; retained in place, retries next substep
        }
    }
    __syncthreads();

    // Phase 2: classify against the new boundary
    const int32_t new_n = n - s_dep_count;
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n) { break; }
        const bool dep = (departed >> i) & 1u;
        if (dep && atom < new_n) { s_holes[atomicAdd(&s_hole_cnt, 1)] = atom; }
        if (!dep && atom >= new_n) { s_movers[atomicAdd(&s_mover_cnt, 1)] = atom; }
    }
    __syncthreads();

    // Phase 3: backfill. Reads and writes are disjoint: reads >= new_n, writes < new_n.
    for (int32_t i = tid; i < s_hole_cnt; i += BLOCKSIZE) {
        const int32_t src     = s_movers[i];
        const int32_t dst     = s_holes[i];
        my_chunk_new_pos[dst] = my_chunk_new_pos[src];
        my_chunk_vel[dst]     = my_chunk_vel[src];
    }
    if (tid == 0) { num_atoms[chunk] = new_n; }
}
__global__ void prepNextSubstepKernel(
    uint32_t* offsets,
    float2* old_pos,
    float2* new_pos,
    const float2* vel,
    int32_t* num_atoms,
    float2* mig_pos,
    float2* mig_vel,
    int32_t* mig_count
)
{
    constexpr int32_t BLOCKSIZE        = 256;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;
    const int32_t tid                  = threadIdx.x;

    __shared__ float2 s_buffer[kMaxAtoms];
    __shared__ uint32_t s_offsets[kGridDim * kGridDim];
    if (tid < kGridDim * kGridDim) { s_offsets[tid] = 0; }
    __syncthreads();
    const int32_t chunk      = blockIdx.x + blockIdx.y * kSceneDimX;
    const int32_t mig_chunk  = blockIdx.x + 1 + (blockIdx.y + 1) * (kSceneDimX + 2);
    int32_t n                = num_atoms[chunk];
    const int32_t mig_n      = min(mig_count[mig_chunk], kMaxMigrants);
    float2* my_chunk_new_pos = new_pos + chunk * kMaxAtoms; // uploaded, then written back after the sort

    const int32_t free_slots = kMaxAtoms - n;
    const int32_t mig_start  = max(mig_n - free_slots, 0);
    float2 p[ATOMS_PER_THREAD]{};
    float2 v[ATOMS_PER_THREAD]{};
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom     = tid + i * BLOCKSIZE;
        const int32_t mig_atom = mig_start + atom - n;
        if (atom < n) {
            p[i] = my_chunk_new_pos[atom];
            v[i] = vel[chunk * kMaxAtoms + atom];
        } else if (mig_atom < mig_n) {
            v[i] = mig_vel[mig_chunk * kMaxMigrants + mig_atom];
            p[i] = mig_pos[mig_chunk * kMaxMigrants + mig_atom];
        }
    }
    const int32_t n_new = n + (mig_n - mig_start);
    if (tid == 0) {
        num_atoms[chunk]     = n_new;
        mig_count[mig_chunk] = mig_start;
    }
    int32_t cells[ATOMS_PER_THREAD]{}; // Holds atoms' Cell IDs and ranks
    // Tally cell populations
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n_new) { break; }
        v[i].x               += p[i].x; // v[i] is now the predicted position
        v[i].y               += p[i].y + kGravityStep;
        const int32_t grid_x  = clamp<int32_t>((int32_t)floorf(v[i].x) / kCellDim, 0, kGridDim - 1);
        const int32_t grid_y  = clamp<int32_t>((int32_t)floorf(v[i].y) / kCellDim, 0, kGridDim - 1);
        cells[i]              = (uint32_t)(grid_x + grid_y * kGridDim);
        const uint32_t rank   = atomicAdd(&s_offsets[cells[i]], 1);
        cells[i]             |= rank << 16;
    }
    __syncthreads();
    incPrefSum256(s_offsets);
    __syncthreads();
    if (tid < kCellsPerChunk) {
        uint32_t start                        = (tid > 0) ? s_offsets[tid - 1] : 0u;
        offsets[chunk * kCellsPerChunk + tid] = (s_offsets[tid] << 16) | start;
    }
    // Rearrange in smem for coalesced gmem writebacks
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n_new) { break; }
        uint32_t base = 0;
        if ((cells[i] & 0xFFFF) > 0) { base = s_offsets[(cells[i] & 0xFFFF) - 1]; }
        cells[i]           = (cells[i] >> 16) + base; // cells[i] now only holds sorted ranks
        s_buffer[cells[i]] = p[i];
    }
    __syncthreads();
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom < n_new) { old_pos[atom + chunk * kMaxAtoms] = s_buffer[atom]; }
    }
    __syncthreads();
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n_new) { break; }
        s_buffer[cells[i]] = v[i];
    }
    __syncthreads();
#pragma unroll
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom < n_new) { new_pos[chunk * kMaxAtoms + atom] = s_buffer[atom]; }
    }
}
__inline__ __device__ void incPrefSum256(uint32_t* s_grid)
{
    int32_t tid = threadIdx.x;
    // Prefix sum within warp-sized segments of the grid
    uint32_t lane      = tid & 31;
    uint32_t my_val    = s_grid[tid];
    uint32_t lower_val = 0;
    if (lane > 0) { lower_val = s_grid[tid - 1]; }
    my_val += lower_val; // The i = 1 step is done manually here as a micro optimisation
#pragma unroll
    for (uint32_t i = 2; i < 32; i <<= 1) {
        lower_val = __shfl_up_sync(0xFFFFFFFF, my_val, i);
        if (lane >= i) { my_val += lower_val; }
    }
    __syncthreads();
    s_grid[tid]         = my_val;
    uint32_t warp_total = __shfl_sync(0xFFFFFFFF, my_val, 31);
    __syncthreads();
    // Sweep across grid to combine warp sum results
    uint32_t my_offset = tid;
#pragma unroll
    for (uint32_t i = 0; i < 7; ++i) { // Loop bounds hardcoded for 256 elements (8 warps)
        my_offset += 32;
        if (my_offset < 256) { atomicAdd(&s_grid[my_offset], warp_total); }
    }
}
__global__ void PBFOutputKernel(
    const float2* pos,
    const float2* vel,
    const int32_t* num_atoms
)
{
    constexpr int32_t BLOCKSIZE        = 256;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;
    const int32_t tid                  = threadIdx.x;

    const int32_t chunk_x = blockIdx.x;
    const int32_t chunk_y = blockIdx.y;
    const int32_t chunk   = chunk_x + chunk_y * kSceneDimX;
    const int32_t n       = num_atoms[chunk];

    if (tid == 0) { c_output.num_atoms[chunk] = n; }
    if (n == 0) { return; }

    float2 p[ATOMS_PER_THREAD]{};
    float2 v[ATOMS_PER_THREAD]{};
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n) { break; }
        p[i]    = pos[chunk * kMaxAtoms + atom];
        p[i].x += chunk_x * kChunkDim;
        p[i].y += chunk_y * kChunkDim;
        v[i]    = vel[chunk * kMaxAtoms + atom];
    }
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = tid + i * BLOCKSIZE;
        if (atom >= n) { break; }
        const int32_t dst_idx = chunk * kMaxAtoms + atom;
        c_output.pos[dst_idx] = p[i];
        c_output.vel[dst_idx] = v[i];
    }
}

// ===== Graph instantiation ========================================

// clang-format off
void instantiatePBFGraph(
    GraphEnv& g,
    PBFArena& pbf
)
{
    CUDACHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&g.tick_done, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreate(&g.t0));
    CUDACHECK(cudaEventCreate(&g.t1));

    const dim3 block_128_2(128,2);
    const dim3 block_256(256);
    const dim3 block_512(512);
    const dim3 block_1024(1024);
    const dim3 chunk_grid(kSceneDimX, kSceneDimY);
    const dim3 halo_grid((kSceneDimX + 1) / 2, kSceneDimY);

    cudaGraphNode_t prev{};
    cudaGraphNode_t halo_reset{};
    CUDACHECK(cudaGraphCreate(&g.graph, 0));
    for (int32_t s = 0; s < kNumSteps; s++) {
        int32_t slot = 0;
        if (s == 0) {
            void* halo_args[] = {&pbf.offsets, &pbf.pos[1], &pbf.halo_offsets, &pbf.halo_pos[slot], &pbf.num_atoms};
            prev = addKernelNode(g.graph, {}, (void*)haloKernel, block_128_2, halo_grid, 0, halo_args);
        } else {
            void* halo_args[] = {&pbf.offsets, &pbf.pos[1], &pbf.halo_offsets, &pbf.halo_pos[slot], &pbf.num_atoms};
            prev = addKernelNode(g.graph, {prev, halo_reset}, (void*)haloKernel, block_128_2, halo_grid, 0, halo_args);
        }

        for (int32_t i = 0; i < kIters; ++i){

            void* lambda_args[] = {&pbf.offsets, &pbf.pos[1], &pbf.lambda, &pbf.halo_offsets, 
                                  &pbf.halo_pos[slot], &pbf.halo_lambda, &pbf.num_atoms};
            prev = addKernelNode(g.graph, {prev}, (void*)lambdaKernel, block_512, chunk_grid, 0, lambda_args);

            void* posCorr_args[] = {&pbf.offsets, &pbf.pos[1], &pbf.lambda, &pbf.halo_offsets, 
                                  &pbf.halo_pos[slot], &pbf.halo_pos[~slot & 1], &pbf.halo_lambda, &pbf.num_atoms};
            prev = addKernelNode(g.graph, {prev}, (void*)posCorrectionKernel, block_512, chunk_grid, 0, posCorr_args);
            slot ^= 1;
        }
        void* velocity_args[] = {&pbf.offsets, &pbf.pos[0], &pbf.pos[1], &pbf.vel, &pbf.halo_vel, &pbf.num_atoms};
        prev = addKernelNode(g.graph, {prev}, (void*)velocityKernel, block_512, chunk_grid, 0, velocity_args);
        
        void* XSPH_args[] = {&pbf.offsets, &pbf.pos[1], &pbf.vel, &pbf.halo_offsets, &pbf.halo_pos[slot], &pbf.halo_vel, &pbf.num_atoms};
        prev = addKernelNode(g.graph, {prev}, (void*)XSPHKernel, block_1024, chunk_grid, 0, XSPH_args);

        halo_reset = addMemsetNode(g.graph, {prev}, pbf.halo_offsets, 4u, (size_t)kNumChunks * kHaloCells);

        void* mig_args[] = {&pbf.pos[1], &pbf.vel, &pbf.num_atoms, &pbf.mig_pos, &pbf.mig_vel, &pbf.mig_count};
        prev = addKernelNode(g.graph, {prev}, (void*)migrationKernel, block_512, chunk_grid, 0, mig_args);

        void* prep_args[] = {&pbf.offsets, &pbf.pos[0], &pbf.pos[1], &pbf.vel, &pbf.num_atoms, &pbf.mig_pos, &pbf.mig_vel, &pbf.mig_count};
        prev = addKernelNode(g.graph, {prev}, (void*)prepNextSubstepKernel, block_256, chunk_grid, 0, prep_args);

    }
    void* output_args[] = {&pbf.pos[0], &pbf.vel, &pbf.num_atoms};
    addKernelNode(g.graph, {prev}, (void*)PBFOutputKernel, block_256, chunk_grid, 0, output_args);

    CUDACHECK(cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0));
}
// clang-format on

// ===== Demo fluid emitter =====================================================

#include <curand_kernel.h>
// Injects atoms through the migration channels once per tick. Number of atoms generated depends on
__global__ void emitterKernel(
    const float2 origin,   // nozzle centre, absolute scene coords
    const float2 velocity, // units per substep
    const int32_t width,   // atoms across the nozzle
    const float spacing,   // ~1.05: just under rest spacing (rho=1 sits at d=1.007)
    const float jitter,    // 0 for a clean stream; <= 0.05f * spacing otherwise
    const int32_t count,   // width * rows, see host side
    const uint64_t seed,
    float2* mig_pos,
    float2* mig_vel,
    int32_t* mig_count
)
{
    if (count == 0 || width <= 0) { return; }
    constexpr int32_t chunk_dim = (int32_t)kChunkDim;

    // Stream frame: t along the flow, nrm across it.
    const float v_sqr = velocity.x * velocity.x + velocity.y * velocity.y;
    const float inv_v = (v_sqr > 1e-12f) ? rsqrtf(v_sqr) : 0.f;
    const float2 t    = {velocity.x * inv_v, velocity.y * inv_v};
    const float2 nrm  = {-t.y, t.x};

    float2 v = velocity;
    velocityClamp(v);

    const int32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = tid; i < count; i += stride) {
        const int32_t col = i % width;
        const int32_t row = i / width;

        const float across = ((float)col - 0.5f * (float)(width - 1)) * spacing;
        const float along  = -((float)row + 0.5f) * spacing; // trails the nozzle

        float2 p = {origin.x + nrm.x * across + t.x * along, origin.y + nrm.y * across + t.y * along};

        if (jitter > 0.f) {
            curandStatePhilox4_32_10_t rng;
            curand_init(seed, (uint64_t)i, 0ull, &rng);
            const float2 j2  = curand_normal2(&rng);
            p.x             += j2.x * jitter;
            p.y             += j2.y * jitter;
        }

        // floorf, not a truncating int divide: p can be negative near the scene edge.
        const int32_t eu_x = (int32_t)floorf(p.x * (1.f / kChunkDim));
        const int32_t eu_y = (int32_t)floorf(p.y * (1.f / kChunkDim));
        if (eu_x < 0 || eu_x >= kSceneDimX || eu_y < 0 || eu_y >= kSceneDimY) { continue; }

        const float local_x = fminf(p.x - (float)(eu_x * chunk_dim), (float)chunk_dim - 1e-3f);
        const float local_y = fminf(p.y - (float)(eu_y * chunk_dim), (float)chunk_dim - 1e-3f);

        const int32_t arr_x     = unwrapX(eu_x, c_torus);
        const int32_t arr_y     = unwrapY(eu_y, c_torus);
        const int32_t mig_chunk = (arr_x + 1) + (arr_y + 1) * (kSceneDimX + 2);

        const int32_t slot = atomicAdd(&mig_count[mig_chunk], 1);
        if (slot < kMaxMigrants) {
            const int32_t idx = mig_chunk * kMaxMigrants + slot;
            mig_pos[idx]      = {local_x, local_y};
            mig_vel[idx]      = v;
        }
    }
}
__host__ int32_t hosePipe(
    const cudaStream_t stream,
    const PBFArena& pbf,
    const float2 origin,
    const float2 velocity,
    const float nozzle_width,
    const float spacing,
    const float vel_jitter,
    const uint64_t tick
)
{
    const float speed   = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);
    const int32_t width = max(1, (int32_t)lroundf(nozzle_width / spacing));
    const int32_t rows  = max(1, (int32_t)lroundf(speed * (float)kNumSteps / spacing));
    const int32_t count = width * rows;
    if (count == 0) { return count; }
    dim3 block{256};
    dim3 grid{(count + 255u) / 256u};
    emitterKernel<<<grid, block, 0, stream>>>(
        origin,
        velocity,
        width,
        spacing,
        vel_jitter,
        count,
        tick,
        pbf.mig_pos,
        pbf.mig_vel,
        pbf.mig_count
    );
    CUDACHECK(cudaGetLastError());
    return count;
}