#pragma once
// Standard libs
#include <cstdint>
// CUDA libs
#include <device_launch_parameters.h>
#include <vector_types.h>
// My libs
#include "common/umbrella.h"

#if defined(__CUDA_ARCH__)
#define DEVICE_UNROLL _Pragma("unroll")
#else
#define DEVICE_UNROLL
#endif

// ===== PBF Kernels ===========================================================

// Substep start

// Copies the boundary cells of each chunk's offsets grid and modifies the bit-packed index ranges (of the
// copy) to fit the destination halo array. These are then broadcast to their destination chunks, along with
// the predicted positions of the atoms they span.
__global__ void haloKernel(
    const uint32_t* offsets,
    const float2* new_pos,
    uint32_t* halo_offsets,
    float2* halo_pos,
    const int32_t* num_atoms
);
// Iteration start

// Computes the lambda (Lagrange multiplier) for each atom by accumulating density and gradient terms
// between itself and each of its neighbours; modelled with standard poly6/spiky kernels (Müller et al.
// 2003), as used in Macklin & Müller's PBF. The derived lambdas of boundary atoms are then broadcast to the
// halo data of bordering chunks.
__global__ void lambdaKernel(
    const uint32_t* offsets,
    const float2* new_pos,
    float* lambda,
    const uint32_t* halo_offsets,
    const float2* halo_pos,
    float* halo_lambda,
    const int32_t* num_atoms
);
// Adjusts the predicted position of each atom using the lambdas computed earlier in the iteration.
// The adjustment is accumulated across the atom's neighbourhood, where the spiky gradient for each
// self/neighbour pair is multiplied by the sum of the two atoms' lambdas (weighted by mass) minus an
// artificial pressure term. The final accumulated value is scaled by Jacobi over-relaxation (kOmega), inverse
// rest density (kInvRho0) and the atom's inverse mass (kInvMass). The corrected predicted positions of
// boundary atoms are then broadcast to the halo data of neighbouring chunks.
__global__ void posCorrectionKernel(
    const uint32_t* offsets,
    float2* new_pos,
    const float* lambda,
    const uint32_t* halo_offsets,
    const float2* halo_pos,
    float2* halo_pos_buf,
    const float* halo_lambda,
    const int32_t* num_atoms
);
// Iteration end

// Computes velocity from the difference between the initial position and the corrected predicted positions.
// The velocities of boundary atoms are then broadcast to the halo data of neighbouring chunks.
__global__ void velocityKernel(
    const uint32_t* offsets,
    const float2* old_pos,
    float2* new_pos,
    float2* vel,
    float2* halo_vel,
    const int32_t* num_atoms
);
// Smooths each atom's velocity using standard XSPH rules: Accumulates each atom's velocity correction by
// taking the differences in velocity between it and its neighbours, scaling them by their poly6 outputs and
// accumulating the results. Also clamps velocity to prevent tunnelling.
__global__ void XSPHKernel(
    const uint32_t* offsets,
    const float2* new_pos,
    float2* vel,
    const uint32_t* halo_offsets,
    const float2* halo_pos,
    const float2* halo_vel,
    const int32_t* num_atoms
);
// Checks each atom's corrected predicted position plus velocity and gravity increments - if the result lies
// outside the bounds of the current host chunk, the atom is moved to a migration buffer owned by the correct
// chunk. If the destination buffer is full, migration is delayed until space becomes available. Gaps left by
// migrants are compacted to preserve contiguity in the main atom pool.
__global__ void migrationKernel(
    float2* new_pos,
    float2* vel,
    int32_t* num_atoms,
    float2* mig_pos,
    float2* mig_vel,
    int32_t* mig_count
);
// Moves inbound atoms from each chunk's migration buffer to its main pool. The corrected predicted
// positions are promoted to current positions and used with a velocity and gravity increment to derive the
// next substep's predicted positions. Atoms are then sorted by grid cell according to their new predictions
// (row-major encoding), and a corresponding offsets grid is produced.
__global__ void prepNextSubstepKernel(
    uint32_t* offsets,
    float2* old_pos,
    float2* new_pos,
    const float2* vel,
    int32_t* num_atoms,
    float2* mig_pos,
    float2* mig_vel,
    int32_t* mig_count
);
// Substep end

// Writes the position and velocity of each atom to an output buffer for the renderer to read at its leisure.
// Positions are transformed to absolute scene coordinates before submission.
__global__ void PBFOutputKernel(
    const float2* pos,
    const float2* vel,
    const int32_t* num_atoms
);

// ===== PBF Shared-memory Staging ======================================

// Uniform gmem->smem loads shared by the search kernels (lambda / posCorrection / XSPH). Must be called
// uniformly by all threads and be immediately followed by a __syncthreads() gate. The loop early exits
// assume threads progress through atom indices monotonically, and that atoms are arranged densely in memory.
// Any modifications to these invariants must be paired with complementary modifications here.

template <typename T>
struct SmemLoad
{
    T* dst;
    const T* src;
}; // shared dst base, global src base

template <typename T>
__device__ __forceinline__ SmemLoad<T> smemLoad(
    T* dst,
    const T* src
)
{ return {dst, src}; }
// Main pool: s_x[atom] = x[chunk*kMaxAtoms + atom], atom in [0, n).
template <
    int32_t BLOCKSIZE,
    typename... Ts>
__device__ __forceinline__ void loadPool(
    int32_t chunk,
    int32_t n,
    SmemLoad<Ts>... loads
)
{
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;

    const int32_t tid = threadIdx.x;
    DEVICE_UNROLL
    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = i * BLOCKSIZE + tid;
        if (atom >= n) { break; }
        ((loads.dst[atom] = loads.src[chunk * kMaxAtoms + atom]), ...);
    }
}
// Halo: s_x[kMaxAtoms + h] = halo_x[chunk*kHaloAtoms + h], h in [0, kHaloAtoms).
template <
    int32_t BLOCKSIZE,
    typename... Ts>
__device__ __forceinline__ void loadHalo(
    int32_t chunk,
    SmemLoad<Ts>... loads
)
{
    constexpr int32_t ITERS = (kHaloAtoms + BLOCKSIZE - 1) / BLOCKSIZE; // collapses to 1 at 1024
    const int32_t tid       = threadIdx.x;
    DEVICE_UNROLL
    for (int32_t i = 0; i < ITERS; ++i) {
        const int32_t h = i * BLOCKSIZE + tid;
        if (h < kHaloAtoms) { ((loads.dst[kMaxAtoms + h] = loads.src[chunk * kHaloAtoms + h]), ...); }
    }
}
// Offsets: interior 16x16 scattered into the (kGridDim+2)^2 extended grid at (1,1),
// plus 64 side + 4 corner halo cells. One pass; needs blockDim.x >= kCellsPerChunk + kHaloCells.
template <int32_t BLOCKSIZE>
__device__ __forceinline__ void loadOffsets(
    int32_t chunk,
    const uint32_t* offsets,
    const uint32_t* halo_offsets,
    uint32_t* s_offsets
)
{
    static_assert(kGridDim == 16, "hardcoded 18 / >>4 / &15 assume kGridDim == 16");
    static_assert(
        BLOCKSIZE >= kCellsPerChunk + kHaloCells,
        "block too small to stage the extended offsets grid in one pass"
    );
    const int32_t tid = threadIdx.x;
    if (tid < kCellsPerChunk) {
        s_offsets[(tid >> 4) * 18 + 18 + (tid & 15) + 1] = offsets[chunk * kCellsPerChunk + tid];
    } else if (tid < kCellsPerChunk + kHaloCells) {
        const int32_t h = tid - kCellsPerChunk;
        int32_t dst_idx;
        if (h < 64) {
            const int32_t region     = h >> 4;
            const int32_t cell_shift = h & 15;
            const int32_t axis       = region >> 1;
            const int32_t far        = region & 1;
            dst_idx                  = axis ? (far * (kGridDim + 1) + (cell_shift + 1) * (kGridDim + 2)) :
                                              (far * (kGridDim + 1) * (kGridDim + 2) + cell_shift + 1);
        } else {
            const int32_t ord = h - 64;
            dst_idx = ((ord & 2) ? (kGridDim + 1) * (kGridDim + 2) : 0) + ((ord & 1) ? (kGridDim + 1) : 0);
        }
        s_offsets[dst_idx] = halo_offsets[chunk * kHaloCells + h];
    }
}

// ===== PBF Helpers =========================================================

__host__ __device__ __forceinline__ int32_t checkBounds(
    int32_t coord,
    int32_t UB
)
{ return ((coord >= 0) && (coord < UB)); }

// Coordinate shift to destination chunk's system
__inline__ __device__ float2 cardinalShift(int32_t region);
__inline__ __device__ float2 ordinalShift(
    int32_t region,
    int32_t cell
);

// Wrapper to allow haloThreadDecode to return internal halo indexing data
struct HaloThread
{
    int32_t region; // N=0, S=1, W=2, E=3
    int32_t cell;   // cell along the region's edge, [0, kGridDim)
    int32_t slot;   // atom within the cell,        [0, kCellSlots)
};

// Returns internal halo indexing data
__device__ __forceinline__ HaloThread haloThreadDecode(const int32_t tid);

// Linear index of the boundary cell drained by a (region, cell) pair. N/S walk a row; W/E walk a column.
__device__ __forceinline__ int32_t haloSrcCell(
    const int32_t region,
    const int32_t cell
);

// Remap a linear kGridDim x kGridDim cell index into the padded (kGridDim+2)^2 shared offsets grid (interior
// inset by one cell on every side).
__device__ __forceinline__ int32_t paddedCellIdx(const int32_t src_cell);

// Clamp (start + slot) to the valid inner-pool range.
__device__ __forceinline__ int32_t haloSrcAtom(
    const uint32_t cell_range,
    const int32_t slot
);
// Wrapper to allow destination data to be returned by cardinalDest and ordinalDest
struct HaloCompass
{
    bool valid     = false; // Does the thread fit all criteria (in-bounds destination, slot in range etc)?
    int32_t region = -1;
    int32_t chunk  = -1;
    int32_t idx    = -1;
};
// Tells caller which chunk its outgoing halo data belongs to
__inline__ __device__ HaloCompass cardinalDest(
    const int32_t chunk_x,
    const int32_t chunk_y,
    const int32_t region,
    const int32_t cell,
    const int32_t slot,
    const uint32_t cell_range
);
__inline__ __device__ HaloCompass ordinalDest(
    const int32_t chunk_x,
    const int32_t chunk_y,
    const int32_t region,
    const int32_t cell,
    const int32_t slot,
    const uint32_t cell_range
);
// Outputs a pseudo-random normal vector that takes the indices of two atoms as seeds. Used to generate
// deterministic position norms in interactions between atoms whose mutual displacement falls below floating
// point precision.
__device__ inline float2 pairDirection(
    const int32_t a,
    const int32_t b
);
// Standard poly6 and spiky gradient functions from Muller et al. 2003
__device__ float poly6(float sqr_dist);
__device__ float2 spiky(
    float dist,
    float2 r_norm
);

// Confines atoms to within simulation bounds. This is a provisional tool for use until rigid bodies and
// static environments are introduced.
__inline__ __device__ void handleOOB(
    const int32_t chunk_x,
    const int32_t chunk_y,
    float2* s_pos,
    const int32_t block_size,
    const int32_t atoms_per_thread,
    const int32_t n
);
__device__ void velocityClamp(float2& v);
// clang-format off

// Translates toroidal --> Euclidean coords
__device__ __forceinline__ int32_t wrapX(const int32_t coord, const int2 seam)
{
    const int32_t r = coord - seam.x;
    return (r < 0) ? r + kSceneDimX : r;
}
__device__ __forceinline__ int32_t wrapY(const int32_t coord, const int2 seam)
{
    const int32_t r = coord - seam.y;
    return (r < 0) ? r + kSceneDimY : r;
}

// Translates Euclidean --> toroidal coords
__device__ __forceinline__ int32_t unwrapX(const int32_t coord, const int2 seam)
{
    const int32_t r = coord + seam.x;
    return (r >= kSceneDimX) ? r - kSceneDimX : r;
}
__device__ __forceinline__ int32_t unwrapY(const int32_t coord, const int2 seam)
{
    const int32_t r = coord + seam.y;
    return (r >= kSceneDimY) ? r - kSceneDimY : r;
}

// toroidal increment: wraps around if OOB in literal index space
__device__ __forceinline__ int32_t stepX(const int32_t x, const int32_t dx)
{
    const int32_t r = x + dx;
    return (r < 0) ? r + kSceneDimX : (r >= kSceneDimX) ? r - kSceneDimX : r;
}
__device__ __forceinline__ int32_t stepY(const int32_t y, const int32_t dy)
{ const int32_t r = y + dy; return (r < 0) ? r + kSceneDimY : (r >= kSceneDimY) ? r - kSceneDimY : r;}

// check Euclidean boundary on an axis from toroidal coord system
__device__ __forceinline__ bool crossesSeam(
    const int32_t src,
    const int32_t dst,
    const int32_t delta,
    const int32_t seam
)
{ return (delta != 0) && (((delta > 0) ? dst : src) == seam); }
// clang-format on

// A parallel prefix sum across exactly 256 elements
__inline__ __device__ void incPrefSum256(uint32_t* s_grid);