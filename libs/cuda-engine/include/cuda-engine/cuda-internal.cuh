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

// ===== General Helpers ===============================================

// Rough-and-ready randomizer
__host__ __device__ __forceinline__ uint32_t hash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
// Applies inclusive bounding clamps to any type with well-defined pairwise ordering
template <typename T>
__inline__ __host__ __device__ T clamp(
    T val,
    T min,
    T max
)
{
    if (val > max) {
        return max;
    } else if (val < min) {
        return min;
    } else {
        return val;
    }
}
__inline__ __host__ __device__ float det2D(const float4 M) { return M.x * M.w - M.y * M.z; }
__inline__ __host__ __device__ float4 matMul2D(
    const float4 A,
    const float4 B
)
{
    float4 C{0};
    C.x = A.x * B.x + A.y * B.z;
    C.y = A.x * B.y + A.y * B.w;
    C.z = A.z * B.x + A.w * B.z;
    C.w = A.z * B.y + A.w * B.w;
    return C;
}
// Graph node construction functions
#pragma warning(push)
#pragma warning(disable : 4505)
static cudaGraphNode_t addKernelNode(
    cudaGraph_t graph,
    std::initializer_list<cudaGraphNode_t> deps,
    void* func,
    dim3 block,
    dim3 grid,
    size_t smem,
    void** argv
)
{
    cudaKernelNodeParams p{};
    p.func           = func;
    p.gridDim        = grid;
    p.blockDim       = block;
    p.sharedMemBytes = (unsigned)smem;
    p.kernelParams   = argv;
    p.extra          = nullptr;

    cudaGraphNode_t n{};
    CUDACHECK(cudaGraphAddKernelNode(&n, graph, deps.begin(), deps.size(), &p));
    return n;
}
static cudaGraphNode_t addMemsetNode(
    cudaGraph_t graph,
    std::initializer_list<cudaGraphNode_t> deps,
    void* dst,
    uint32_t element_size,
    size_t num_elements
)
{
    cudaMemsetParams p{};
    p.dst         = dst;
    p.value       = 0u;
    p.pitch       = 0;            // Ignored when height == 1
    p.elementSize = element_size; // Size in bytes. be 1, 2 or 4
    p.width       = num_elements;
    p.height      = 1;

    cudaGraphNode_t n{};
    CUDACHECK(cudaGraphAddMemsetNode(&n, graph, deps.begin(), deps.size(), &p));
    return n;
}
static void modifyKernelNode(
    cudaGraphExec_t exec,
    const cudaGraphNode_t& node,
    void* kernel,
    dim3 block,
    dim3 grid,
    size_t smem,
    void** argv
)
{
    cudaKernelNodeParams p{};
    p.func           = kernel;
    p.gridDim        = grid;
    p.blockDim       = block;
    p.sharedMemBytes = (unsigned)smem;
    p.kernelParams   = argv;
    p.extra          = nullptr;

    CUDACHECK(cudaGraphExecKernelNodeSetParams(exec, node, &p));
}
#pragma warning(pop)
