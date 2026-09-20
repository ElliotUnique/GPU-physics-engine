#pragma once
#include "common/umbrella.h"
#include <stdint.h>

#define CUDA_ENGINE_MAX_BUFFERS 4

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct CudaEngineDeviceDesc
    {
        void* cuda_fence_handle; /* CUDA signals this   */
        void* d3d_fence_handle;  /* CUDA waits on this  */
        uint8_t adapter_luid[8];
    } CudaEngineDeviceDesc;

    typedef struct CudaEngineBufferDesc
    {
        void* resource_handles[CUDA_ENGINE_MAX_BUFFERS];
        uint32_t buffer_count;
        uint64_t allocation_bytes;
        uint64_t buf_bytes;
        uint32_t buf_row_pitch;
        uint32_t width;
        uint32_t height;
    } CudaEngineBufferDesc;

    int32_t cuda_engine_init(const CudaEngineDeviceDesc* d);
    int32_t cuda_engine_bind_buffers(const CudaEngineBufferDesc* b); /* imports, pitch, w, h */
    void cuda_engine_unbind_buffers(void); /* cudaFree + cudaDestroyExternalMemory */
    int32_t cuda_engine_render_frame(
        const View tick_output,
        uint64_t wait_value,
        uint64_t signal_value,
        uint64_t frame,
        uint32_t buffer_index
    );
    void cuda_engine_shutdown(void);
    int32_t cuda_engine_last_error(void);
    const char* cuda_engine_error_name(int32_t err);

#ifdef __cplusplus
}
#endif
