// CUDA libs
#include <vector_types.h>
// My libs
#include "common/umbrella.h"
#include "cuda-engine/MPM-internal.cuh"
#include "cuda-engine/cuda-engine.h"
#include "cuda-engine/cuda-internal.cuh"

// ===== Device global variables ===========================================

// Location of the tick output targets, modified by the host each tick to point to the next output
// destination.
__constant__ View c_output;

__constant__ int2 c_torus; // Unused for MPM

// ===== TickStaging =======================================================

__host__ void MPMTickStaging::init(
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

__host__ void MPMTickStaging::outputTargetUpload(cudaStream_t stream)
{
    CUDACHECK(
        cudaMemcpyToSymbolAsync(c_output, &staged->target, sizeof(View), 0, cudaMemcpyHostToDevice, stream)
    );
}

__host__ void MPMTickStaging::torusUpload(cudaStream_t stream)
{
    CUDACHECK(
        cudaMemcpyToSymbolAsync(c_torus, &staged->torus, sizeof(int2), 0, cudaMemcpyHostToDevice, stream)
    );
}

__global__ void resetGridKernel(float4* grid)
{
    int32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < kGridDimX * kGridDimY; i += stride) { grid[i] = {0.f, 0.f, 0.f, 0.f}; }
}

__global__ void P2GKernel(
    float4* grid,
    const float2* pos,
    const float2* vel,
    const float4* C,
    const float* J,
    int32_t* num_points
)
{
    int32_t tid          = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < *num_points; i += stride) {
        const float4 C_p = C[i];
        const float J_p  = J[i];

        const float p_hat    = kStiffness * (powf(1.f / J_p, kGamma) - 1.f);
        const float pressure = fminf(fmaxf(p_hat, kMinPress), kMaxPress);

        const float4 sigma = {-pressure, 0.f, 0.f, -pressure};

        const float V_p   = kRestVol * J_p;
        const float coeff = -dt * V_p * 4 * inv_dx * inv_dx;
        const float4 Q    = {
            kMass * C_p.x + coeff * sigma.x,
            kMass * C_p.y + coeff * sigma.y,
            kMass * C_p.z + coeff * sigma.z,
            kMass * C_p.w + coeff * sigma.w
        };
        particleToGrid(pos[i], vel[i], Q, grid);
    }
}
__device__ inline float gridCoord(float x) { return x * inv_dx + (float)kGridHalo; }
__device__ void particleToGrid(
    const float2 pos,
    const float2 vel,
    const float4 Q,
    float4* grid
)
{
    const float px   = pos.x;
    const float gx   = gridCoord(px);
    const int32_t bx = (int32_t)floorf(gx - 0.5f);
    const float fx   = gx - (float)bx;

    float wx[3];
    wx[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
    wx[1] = 0.75f - (fx - 1.0f) * (fx - 1.0f);
    wx[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);

    const float py   = pos.y;
    const float gy   = gridCoord(py);
    const int32_t by = (int32_t)floorf(gy - 0.5f);
    const float fy   = gy - (float)by;

    float wy[3];
    wy[0] = 0.5f * (1.5f - fy) * (1.5f - fy);
    wy[1] = 0.75f - (fy - 1.0f) * (fy - 1.0f);
    wy[2] = 0.5f * (fy - 0.5f) * (fy - 0.5f);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float w     = wx[i] * wy[j];
            const float2 dpos = {
                ((float)i - fx) * kDx,
                ((float)j - fy) * kDx
            }; // kDx covers both x and y dims as grid is composed of uniform squares
            const float2 affine = {(dpos.x * Q.x + dpos.y * Q.y), (dpos.x * Q.z + dpos.y * Q.w)};
            const uint32_t node = (by + j) * (uint32_t)kGridDimX + (bx + i);

            atomicAdd(&grid[node].x, w * (vel.x * kMass + affine.x));
            atomicAdd(&grid[node].y, w * (vel.y * kMass + affine.y));
            atomicAdd(&grid[node].z, w * kMass);
        }
    }
}

constexpr int32_t n_cells = kGridDimX * kGridDimY;
__global__ void gridKernel(float4* __restrict__ grid)
{
    const int32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;
    for (int32_t base = tid; base < n_cells; base += stride * 4) {
        float4 node[4];
        int32_t idx[4];

#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            idx[j] = base + j * stride;
            if (idx[j] < n_cells) { node[j] = grid[idx[j]]; }
        }

#pragma unroll
        for (int32_t j = 0; j < 4; ++j) {
            if (idx[j] >= n_cells) { continue; }

            float4 v = node[j];
            if (v.z > 0.f) {
                const float inv_m  = 1.f / v.z;
                v.x               *= inv_m;
                v.y                = v.y * inv_m + kGravity * dt;

                const int32_t x = idx[j] % kGridDimX;
                const int32_t y = idx[j] / kGridDimX;

                if (x <= (int32_t)kGridHalo && v.x < 0.f) {
                    v.x = 0.f;
                } else if (x >= (int32_t)kGridDimX - 1 - (int32_t)kGridHalo && v.x > 0.f) {
                    v.x = 0.f;
                }
                if (y <= (int32_t)kGridHalo && v.y < 0.f) {
                    v.y = 0.f;
                } else if (y >= (int32_t)kGridDimY - 1 - (int32_t)kGridHalo && v.y > 0.f) {
                    v.y = 0.f;
                }
            } else {
                v.x = 0.f;
                v.y = 0.f;
            }

            grid[idx[j]] = v;
        }
    }
}
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
)
{
    const int32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < *num_points; i += stride) {
        gridToParticle(grid, pos[i], vel[i], C[i], F[i], J[i], sim_w, sim_h);
    }
}
__device__ void gridToParticle(
    const float4* grid,
    float2& pos,
    float2& vel,
    float4& C,
    float4& F,
    float& J,
    int32_t sim_w,
    int32_t sim_h
)
{
    const float px   = pos.x;
    const float gx   = gridCoord(px);
    const int32_t bx = (int32_t)floorf(gx - 0.5f);
    const float fx   = gx - (float)bx;

    float wx[3];
    wx[0] = 0.5f * (1.5f - fx) * (1.5f - fx);
    wx[1] = 0.75f - (fx - 1.0f) * (fx - 1.0f);
    wx[2] = 0.5f * (fx - 0.5f) * (fx - 0.5f);

    const float py   = pos.y;
    const float gy   = gridCoord(py);
    const int32_t by = (int32_t)floorf(gy - 0.5f);
    const float fy   = gy - (float)by;

    float wy[3];
    wy[0] = 0.5f * (1.5f - fy) * (1.5f - fy);
    wy[1] = 0.75f - (fy - 1.0f) * (fy - 1.0f);
    wy[2] = 0.5f * (fy - 0.5f) * (fy - 0.5f);

    vel      = {0.f, 0.f};
    float4 B = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const float w     = wx[i] * wy[j];
            const float2 dpos = {
                ((float)i - fx) * kDx,
                ((float)j - fy) * kDx
            }; // kDx covers both x and y dims as grid is composed of uniform squares
            const uint32_t node = (by + j) * (uint32_t)kGridDimX + (bx + i);
            float2 v_i          = {grid[node].x, grid[node].y};

            vel.x += w * v_i.x;
            vel.y += w * v_i.y;

            const float2 wv = {w * v_i.x, w * v_i.y};

            B.x += wv.x * dpos.x;
            B.y += wv.x * dpos.y;
            B.z += wv.y * dpos.x;
            B.w += wv.y * dpos.y;
        }
    }
    const float s  = 4.f * inv_dx * inv_dx;
    pos.x         += vel.x * dt;
    pos.y         += vel.y * dt;
    C              = {s * B.x, s * B.y, s * B.z, s * B.w};
    handleOOB(pos, vel, sim_w, sim_h);

    float4 A;
    A.x  = 1 + C.x * dt;
    A.y  = C.y * dt;
    A.z  = C.z * dt;
    A.w  = 1 + C.w * dt;
    F    = matMul2D(A, F);
    J   *= det2D(A);
    J    = fminf(fmaxf(J, 0.1f), 10.f);
}
__device__ void handleOOB(
    float2& pos,
    float2& vel,
    int32_t sim_w,
    int32_t sim_h
)
{
    const float w = (float)sim_w;
    const float h = (float)sim_h;

    if (!isfinite(pos.x) || !isfinite(vel.x)) {
        pos.x = 0.5f * w;
        vel.x = 0.f;
    }
    if (!isfinite(pos.y) || !isfinite(vel.y)) {
        pos.y = 0.5f * h;
        vel.y = 0.f;
    }
    if (pos.x < 0.f) {
        pos.x = 0.f;
        vel.x = fmaxf(vel.x, 0.f);
    }
    if (pos.x > w) {
        pos.x = w;
        vel.x = fminf(vel.x, 0.f);
    }
    if (pos.y < 0.f) {
        pos.y = 0.f;
        vel.y = fmaxf(vel.y, 0.f);
    }
    if (pos.y > h) {
        pos.y = h;
        vel.y = fminf(vel.y, 0.f);
    }
}
__global__ void MPMOutputKernel(
    const float2* pos,
    const float2* vel,
    int32_t* num_points
)
{
    int32_t n = num_points[0];

    const int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    for (int32_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        const int32_t chunk = i / kMaxAtoms;
        c_output.pos[i]     = pos[i];
        c_output.vel[i]     = vel[i];
        if (i + 1 == n) {
            c_output.num_atoms[chunk] = (i % kMaxAtoms) + 1;
            num_points[1]             = n; // Updating slot 1 in case emitter kernel added anything to slot 0
        } else if ((i % kMaxAtoms) + 1 == kMaxAtoms) {
            c_output.num_atoms[chunk] = kMaxAtoms;
        }
    }
}

// ===== Graph instantiation ========================================

void instantiateMPMGraph(
    GraphEnv& g,
    MPMArena& mpm
)
{
    CUDACHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    CUDACHECK(cudaEventCreateWithFlags(&g.tick_done, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreate(&g.t0));
    CUDACHECK(cudaEventCreate(&g.t1));

    int32_t sim_w = kSceneDimX * (int32_t)kChunkDim;
    int32_t sim_h = kSceneDimY * (int32_t)kChunkDim;
    float2* pos   = mpm.pos;
    float2* vel   = mpm.vel;
    float4* C     = mpm.C;
    float4* F     = mpm.F;
    float* J      = mpm.J;

    const size_t block_size = kCudaBlockX * kCudaBlockY;
    const size_t grid_size  = kCudaGridX * kCudaGridY;
    dim3 initBlock(block_size);
    const dim3 initGrid((kNumAtoms + block_size - 1) / block_size);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaStreamSynchronize(g.stream));

    // Graph instantiation
    auto flip = [](int32_t x) { return x & 1; };
    CUDACHECK(cudaGraphCreate(&g.graph, 0));
    cudaGraphNode_t prev{};
    const dim3 BD(block_size);
    const dim3 GD(grid_size);
    void* arg[] = {&mpm.grid[0]};
    auto reset  = addKernelNode(g.graph, {}, (void*)resetGridKernel, BD, GD, 0, arg);
    for (int32_t s = 0; s < kSubsteps; ++s) {
        void* P2G_args[] = {&mpm.grid[flip(s)], &pos, &vel, &C, &J, &mpm.num_points};
        prev = (s) ? addKernelNode(g.graph, {prev, reset}, (void*)P2GKernel, BD, GD, 0, P2G_args) :
                     addKernelNode(g.graph, {reset}, (void*)P2GKernel, BD, GD, 0, P2G_args);
        void* reset_args[] = {&mpm.grid[flip(s + 1)]};
        reset              = addKernelNode(g.graph, {prev}, (void*)resetGridKernel, BD, GD, 0, reset_args);
        void* grid_args[]  = {&mpm.grid[flip(s)]};
        prev               = addKernelNode(g.graph, {prev}, (void*)gridKernel, BD, GD, 0, grid_args);
        void* G2P_args[]   = {&mpm.grid[flip(s)], &pos, &vel, &C, &F, &J, &sim_w, &sim_h, &mpm.num_points};
        prev               = addKernelNode(g.graph, {prev}, (void*)G2PKernel, BD, GD, 0, G2P_args);
    }
    void* output_args[] = {&pos, &vel, &mpm.num_points};
    addKernelNode(g.graph, {prev}, (void*)MPMOutputKernel, BD, GD, 0, output_args);

    CUDACHECK(cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0));
}

// ===== Demo fluid emitter =====================================================

#include <curand_kernel.h>
// Injects atoms through the migration channels once per tick. Number of atoms generated depends on
__global__ void MPMEmitterKernel(
    const float2 origin,
    const float2 velocity,
    const int32_t width,
    const float spacing,
    const float jitter,
    const int32_t count,
    const uint64_t seed,
    float2* vel,
    float2* pos,
    float4* C,
    float4* F,
    float* J,
    int32_t* num_points
)
{
    if (count == 0 || width <= 0) { return; }

    // Stream frame: t along the flow, nrm across it.
    const float v_sqr = velocity.x * velocity.x + velocity.y * velocity.y;
    const float inv_v = (v_sqr > 1e-12f) ? rsqrtf(v_sqr) : 0.f;
    const float2 t    = {velocity.x * inv_v, velocity.y * inv_v};
    const float2 nrm  = {-t.y, t.x};

    float2 v = {velocity.x / kDt, velocity.y / kDt};

    const int32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int32_t i = tid; i < count; i += stride) {
        const int32_t col  = i % width;
        const int32_t row  = i / width;
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
        const int32_t idx         = i + num_points[1];
        const bool contributor    = idx < kNumAtoms;
        const uint32_t active     = __activemask();
        const uint32_t contr_mask = __ballot_sync(active, contributor);
        if (contributor) {
            pos[idx] = p;
            vel[idx] = v;
            C[idx]   = {0, 0, 0, 0};
            F[idx]   = {1, 0, 0, 1};
            J[idx]   = 1;
        }
        if ((tid & (warpSize - 1)) == __ffs(active) - 1) {
            const int32_t warp_contribution = __popc(contr_mask);
            atomicAdd(&num_points[0], warp_contribution);
        }
    }
}
__host__ int32_t hosePipe(
    const cudaStream_t stream,
    MPMArena& mpm,
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
    const int32_t rows  = max(1, (int32_t)lroundf(speed / spacing));
    const int32_t count = width * rows;
    if (count == 0) { return count; }
    dim3 block{256};
    dim3 grid{(count + 255u) / 256u};
    MPMEmitterKernel<<<grid, block, 0, stream>>>(
        origin,
        velocity,
        width,
        spacing,
        vel_jitter,
        count,
        tick,
        mpm.vel,
        mpm.pos,
        mpm.C,
        mpm.F,
        mpm.J,
        mpm.num_points
    );
    return count;
}