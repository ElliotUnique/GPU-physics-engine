#pragma once
#include "common/umbrella.h"
#include <vector_types.h>

__host__ void launchFill(
    const View tick_output,
    uint8_t* buf_base,
    const size_t buf_row_pitch,
    const size_t rect_w,
    const size_t rect_h,
    const cudaStream_t stream
);
__global__ void backgroundKernel(
    uint8_t* buf_base,
    const size_t buf_row_pitch,
    const size_t rect_w,
    const size_t rect_h
);
__global__ void drawKernel(
    const float2* pos,
    const float2* vel,
    const int32_t* num_atoms,
    uint8_t* buf_base,
    const size_t buf_row_pitch,
    const size_t rect_w,
    const size_t rect_h
);