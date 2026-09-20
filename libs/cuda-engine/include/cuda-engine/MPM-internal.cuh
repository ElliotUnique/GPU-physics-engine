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

// ===== MPM Kernrels ============================================

// Tares the grid, clearing it of stale data before the substep begins.
__global__ void resetGridKernel(float4* grid);

// Computes and applies each MP's momentum contributions to nodes in its 3x3 grid stencil. Each contribution
// sums the MP's actual momentum (vel[i] * kMass) with its affine momentum (derived from C) and internal
// stress (derived from J). The result is weighted by proximity to the target node, with nearer nodes
// rercieving a stronger signal. The particle's mass is accumulated to the grid separately, weighted by the
// same factor.
__global__ void P2GKernel(
    float4* grid,
    const float2* old_pos,
    const float2* vel,
    const float4* C,
    const float* J,
    int32_t* num_points
);

// Processes each node's data accumulated in P2GKernel, dividing its bulk momentum term by its bulk mass term
// to convert the units to a flat velocity, setting velocity/momentum fields to zero if the node is beyond
// mass term is 0. External forces such as gravity and velocity clamping at the boundaries are applied here.
__global__ void gridKernel(float4* __restrict__ grid);

// Derives each MP's updated state using the new velocity data gathered from nodes in its 3x3 grid stencil:
// vel[i] is resampled as the weighted sum of node velocities; pos[i] is incremented by vel[i] * dt and
// clamped within simulation bounds; new C[i] is derived from B (APIC's affine velocity moment); new F[i]
// becomes A * F[i] (where A = I + C[i] * dt); lastly, J[i] is multiplied by the determinant of A.
__global__ void G2PKernel(
    const float4* __restrict__ grid,
    float2* __restrict__ pos,
    float2* __restrict__ vel,
    float4* __restrict__ C,
    float4* __restrict__ F,
    float* __restrict__ J,
    size_t sim_w,
    size_t sim_h,
    int32_t* num_points
);
// Copies the positions and velocities of the MPs to the tick output buffer for the renderer to consume
__global__ void MPMOutputKernel(
    const float2* pos,
    const float2* vel,
    int32_t* num_points
);
__device__ void particleToGrid(
    const float2 old_pos,
    const float2 vel,
    const float4 Q,
    float4* grid
);
__inline__ __device__ void handleOOB(
    float2& old_pos,
    float2& vel,
    int32_t sim_w,
    int32_t sim_h
);
__device__ void gridToParticle(
    const float4* grid,
    float2& old_pos,
    float2& vel,
    float4& C,
    float4& F,
    float& J,
    int32_t sim_w,
    int32_t sim_h
);