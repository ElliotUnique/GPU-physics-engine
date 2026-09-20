// Standard libs
#include <cstring>
#include <stdint.h>
// My libs
#include "common/umbrella.h"
#include "cuda-engine/cuda-engine.h"
#include "render/cuda-interface.h"
#include "render/cuda-internal.cuh"

static struct
{
    cudaExternalMemory_t mem[CUDA_ENGINE_MAX_BUFFERS];
    uint8_t* pixels[CUDA_ENGINE_MAX_BUFFERS];
    uint32_t buffer_count;
    cudaExternalSemaphore_t sem_signal; // CUDA signals: "kernel N done"
    cudaExternalSemaphore_t sem_wait;   // CUDA waits:   "copy N done"
    cudaStream_t stream;
    uint32_t buf_row_pitch, width, height;
    int32_t initialised;
    int32_t last_error; // Needs thread_local when multithreading
} g{};

static int32_t record(cudaError_t e)
{
    g.last_error = (int32_t)e; // Careful when multithreading
    return (e == cudaSuccess) ? 0 : -1;
}

int32_t cuda_engine_init(const CudaEngineDeviceDesc* d)
{
    if (!d || !d->cuda_fence_handle || !d->d3d_fence_handle) { return -4; }
    if (g.initialised) { return -5; }

    int32_t count = 0, chosen = -1;
    if (record(cudaGetDeviceCount(&count)) != 0) { return -1; }

    for (int32_t i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) { continue; }
        if (p.luidDeviceNodeMask == 0) { continue; }
        if (memcmp(p.luid, d->adapter_luid, sizeof(p.luid)) == 0) {
            chosen = i;
            break;
        }
    }
    if (chosen < 0) { return -2; }

    if (record(cudaSetDevice(chosen)) != 0) { return -1; }
    if (record(cudaFree(0)) != 0) { return -1; } // force context creation

    cudaExternalSemaphoreHandleDesc sd{};
    sd.type  = cudaExternalSemaphoreHandleTypeD3D12Fence;
    sd.flags = 0;

    sd.handle.win32.handle = d->cuda_fence_handle;
    if (record(cudaImportExternalSemaphore(&g.sem_signal, &sd)) != 0) { return -1; }

    sd.handle.win32.handle = d->d3d_fence_handle;
    if (record(cudaImportExternalSemaphore(&g.sem_wait, &sd)) != 0) {
        cudaDestroyExternalSemaphore(g.sem_signal);
        g.sem_signal = nullptr;
        return -1;
    }

    if (record(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking)) != 0) {
        cudaDestroyExternalSemaphore(g.sem_signal);
        cudaDestroyExternalSemaphore(g.sem_wait);
        g.sem_signal = nullptr;
        g.sem_wait   = nullptr;
        return -1;
    }

    g.initialised = 1;
    return 0;
}

int32_t cuda_engine_bind_buffers(const CudaEngineBufferDesc* b)
{
    if (!g.initialised) { return -3; }
    if (!b || b->buffer_count == 0 || b->buffer_count > CUDA_ENGINE_MAX_BUFFERS) { return -4; }
    if (g.buffer_count != 0) {
        return -5; // unbind first
    }

    int32_t rc = -1;
    for (uint32_t i = 0; i < b->buffer_count; ++i) {
        if (!b->resource_handles[i]) {
            rc = -4;
            goto fail;
        }

        cudaExternalMemoryHandleDesc md{};
        md.type                = cudaExternalMemoryHandleTypeD3D12Resource;
        md.handle.win32.handle = b->resource_handles[i];
        md.size                = b->allocation_bytes;
        md.flags               = cudaExternalMemoryDedicated;
        if (record(cudaImportExternalMemory(&g.mem[i], &md)) != 0) { goto fail; }

        cudaExternalMemoryBufferDesc bd{};
        bd.offset = 0;
        bd.size   = b->buf_bytes;
        bd.flags  = 0;
        if (record(cudaExternalMemoryGetMappedBuffer((void**)&g.pixels[i], g.mem[i], &bd)) != 0) {
            goto fail;
        }
    }

    g.buffer_count  = b->buffer_count;
    g.buf_row_pitch = b->buf_row_pitch;
    g.width         = b->width;
    g.height        = b->height;
    return 0;

fail:
    for (uint32_t i = 0; i < CUDA_ENGINE_MAX_BUFFERS; ++i) {
        if (g.pixels[i]) {
            cudaFree(g.pixels[i]);
            g.pixels[i] = nullptr;
        }
        if (g.mem[i]) {
            cudaDestroyExternalMemory(g.mem[i]);
            g.mem[i] = nullptr;
        }
    }
    return rc;
}

void cuda_engine_unbind_buffers(void)
{
    if (g.buffer_count == 0) { return; }

    cudaStreamSynchronize(g.stream); // queued kernels may still write these

    for (uint32_t i = 0; i < CUDA_ENGINE_MAX_BUFFERS; ++i) {
        if (g.pixels[i]) {
            cudaFree(g.pixels[i]);
            g.pixels[i] = nullptr;
        }
        if (g.mem[i]) {
            cudaDestroyExternalMemory(g.mem[i]);
            g.mem[i] = nullptr;
        }
    }
    g.buffer_count  = 0;
    g.buf_row_pitch = g.width = g.height = 0;
}

void cuda_engine_shutdown(void)
{
    if (!g.initialised) { return; }
    cuda_engine_unbind_buffers();
    if (g.sem_signal) { cudaDestroyExternalSemaphore(g.sem_signal); }
    if (g.sem_wait) { cudaDestroyExternalSemaphore(g.sem_wait); }
    if (g.stream) { cudaStreamDestroy(g.stream); }
    g = {};
}

int32_t cuda_engine_last_error(void) { return g.last_error; }
const char* cuda_engine_error_name(int32_t err) { return cudaGetErrorName((cudaError_t)err); }

#pragma warning(push)
#pragma warning(disable : 4100)
int32_t cuda_engine_render_frame(
    const View tick_output,
    uint64_t wait_value,
    uint64_t signal_value,
    uint64_t frame,
    uint32_t buffer_index
)
{
    if (g.buffer_count == 0) { return -3; }

    cudaExternalSemaphoreWaitParams wp{};
    wp.params.fence.value = wait_value;
    CUDACHECK(cudaWaitExternalSemaphoresAsync(&g.sem_wait, &wp, 1, g.stream));

    launchFill(tick_output, g.pixels[buffer_index], g.buf_row_pitch, g.width, g.height, g.stream);

    CUDACHECK(cudaStreamSynchronize(g.stream));
    CUDACHECK(cudaGetLastError());

    cudaExternalSemaphoreSignalParams sp{};
    sp.params.fence.value = signal_value;
    CUDACHECK(cudaSignalExternalSemaphoresAsync(&g.sem_signal, &sp, 1, g.stream));
    return 0;
}
#pragma warning(pop)

__host__ void launchFill(
    const View tick_output,
    uint8_t* buf_base,
    const size_t buf_row_pitch,
    const size_t rect_w,
    const size_t rect_h,
    const cudaStream_t stream
)
{
    dim3 block(32, 16);
    dim3 grid((kDispWidth + block.x - 1) / block.x, (kDispHeight + block.y - 1) / block.y);
    backgroundKernel<<<grid, block, 0, stream>>>(buf_base, buf_row_pitch, rect_w, rect_h);

    block = dim3(256);
    grid  = dim3(kSceneDimX, kSceneDimY);
    drawKernel<<<grid, block, 0, stream>>>(
        tick_output.pos,
        tick_output.vel,
        tick_output.num_atoms,
        buf_base,
        buf_row_pitch,
        rect_w,
        rect_h
    );
}
__global__ void backgroundKernel(
    uint8_t* buf_base,
    const size_t buf_row_pitch,
    const size_t rect_w,
    const size_t rect_h
)
{
    auto pixel = [&](int32_t X, int32_t Y)
    { return reinterpret_cast<uchar4*>(buf_base + (size_t)Y * buf_row_pitch) + X; };

    const int32_t rect_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t rect_y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((size_t)rect_x >= rect_w || (size_t)rect_y >= rect_h) { return; }
    constexpr float inv_scale = 1 / kSimScale;
    const bool chunk_bound_x  = (rect_x % (int32_t)(kChunkDim * kSimScale)) == 0;
    const bool chunk_bound_y  = (rect_y % (int32_t)(kChunkDim * kSimScale)) == 0;
    if (chunk_bound_x || chunk_bound_y) {
        uchar4* pxl = pixel(rect_x, rect_y);
        pxl->x      = 0;
        pxl->y      = 0;
        pxl->z      = 0;
        pxl->w      = 255;
    } else {
        uchar4* pxl = pixel(rect_x, rect_y);
        pxl->x      = 255u - (uint8_t)(100.f * ((float)rect_y / (float)rect_h));
        pxl->y      = 160u - (uint8_t)(100.f * ((float)rect_y / (float)rect_h));
        pxl->z      = 80;
        pxl->w      = 255;
    }
}
__device__ __forceinline__ uchar4 alpha(
    uchar4 over_layer,
    uchar4 under_layer
)
{
    constexpr float c = 1.f / 255.f;
    const float a     = (float)over_layer.w * c;
    const float ia    = 1.f - a;

    const uint8_t R = (uint8_t)((float)over_layer.x * a + (float)under_layer.x * ia + 0.5f);
    const uint8_t G = (uint8_t)((float)over_layer.y * a + (float)under_layer.y * ia + 0.5f);
    const uint8_t B = (uint8_t)((float)over_layer.z * a + (float)under_layer.z * ia + 0.5f);

    return {R, G, B, 255u};
}
__global__ void drawKernel(
    const float2* pos,
    const float2* vel,
    const int32_t* num_atoms,
    uint8_t* buf_base,
    const size_t buf_row_pitch,
    const size_t rect_w,
    const size_t rect_h
)
{
    auto pixel = [&](int32_t X, int32_t Y)
    { return reinterpret_cast<uchar4*>(buf_base + (size_t)Y * buf_row_pitch) + X; };

    constexpr int32_t BLOCKSIZE        = 256;
    constexpr int32_t ATOMS_PER_THREAD = (kMaxAtoms + BLOCKSIZE - 1) / BLOCKSIZE;

    const int32_t chunk_x = blockIdx.x;
    const int32_t chunk_y = blockIdx.y;
    const int32_t chunk   = chunk_x + chunk_y * kSceneDimX;
    const int32_t n       = min(num_atoms[chunk], kMaxAtoms);

    for (int32_t i = 0; i < ATOMS_PER_THREAD; ++i) {
        const int32_t atom = threadIdx.x + i * BLOCKSIZE;
        if (atom >= n) { break; }
        const float2 p       = pos[chunk * kMaxAtoms + atom];
        const int32_t rect_x = (int32_t)(p.x * kSimScale);
        const int32_t rect_y = (int32_t)(p.y * kSimScale);
        if ((size_t)rect_x >= rect_w || (size_t)rect_y >= rect_h) { continue; }
        const float2 v = vel[chunk * kMaxAtoms + atom];

        const uint8_t min_RGB = 150u;
        const float mult      = 25.f;

        const uint8_t R = 140u; // 0u;
        const uint8_t G = 210u; // min_RGB + (uint8_t)fminf(mult * abs(v.x) * 255.f, 255.f - (float)min_RGB);
        const uint8_t B = 255u; // min_RGB + (uint8_t)fminf(mult * abs(v.y) * 255.f, 255.f - (float)min_RGB);
        const uint8_t A = 100;
        uchar4 new_pxl  = {R, G, B, A};

        constexpr int32_t lo = -2;
        constexpr int32_t hi = 2;
        for (int32_t i = lo; i <= hi; ++i) {
            if (rect_y + i >= rect_h || rect_y + i < 0) { continue; }
            for (int32_t j = lo; j <= hi; ++j) {
                if (rect_x + j >= rect_w || rect_x + j < 0) { continue; }
                if (abs(i) == hi && abs(j) == hi) { continue; }
                uchar4* pxl = pixel(rect_x + j, rect_y + i);
                *pxl        = alpha(new_pxl, *pxl);
            }
        }
    }
}
