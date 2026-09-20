#pragma once
#include "constants.h"
#include <math.h>
#include <stdint.h>

#ifndef WIN_API
#define WIN_API 0
#endif

#ifdef __CUDACC__
#if __CUDACC__
template <size_t N>
class Fixed
{
  public:
    using rep = int32_t;

    __host__ __device__ constexpr Fixed() : value_(0) {}

    __host__ __device__ static constexpr Fixed from_raw(rep r)
    {
        Fixed f;
        f.value_ = r;
        return f;
    }
    __host__ __device__ static constexpr Fixed from_int(int32_t x)
    { return from_raw(static_cast<rep>(static_cast<uint32_t>(x) << N)); }
    __host__ __device__ static Fixed from_float(float x)
    { return from_raw(static_cast<rep>(lrintf(x * kScale))); }

    __host__ __device__ constexpr rep raw() const { return value_; }
    __host__ __device__ float to_float() const { return static_cast<float>(value_) * kInvScale; }

    __host__ __device__ constexpr Fixed operator+(Fixed o) const { return from_raw(value_ + o.value_); }
    __host__ __device__ constexpr Fixed operator-(Fixed o) const { return from_raw(value_ - o.value_); }
    __host__ __device__ constexpr Fixed operator-() const { return from_raw(-value_); }

    __host__ __device__ Fixed operator*(Fixed o) const
    {
        int64_t wide  = static_cast<int64_t>(value_) * o.value_;
        wide         += (int64_t{1} << (N - 1));
        return from_raw(static_cast<rep>(wide >> N));
    }

    __host__ __device__ constexpr bool operator<(Fixed o) const { return value_ < o.value_; }

  private:
    static constexpr float kScale    = static_cast<float>(int64_t{1} << N);
    static constexpr float kInvScale = 1.0f / kScale;
    rep value_;
};
#else
template <size_t N>
class Fixed
{
    int32_t value_ = (int32_t)N;
};
#endif
#endif

// template <size_t N>
// struct alignas(8) Fixed2
//{
//     Fixed<N> x = 0;
//     Fixed<N> y = 0;
// };

// #if WIN_API
// struct alignas(8) float2
//{
//     float x;
//     float y;
// };
// struct alignas(16) float4
//{
//     float x;
//     float y;
//     float z;
//     float w;
// };
// #endif

// Tells the renderer what part of the sim output to capture, where x and y describe the top left corner of
// the camera capture region in sim output space.
struct CameraSpan
{
    int32_t x   = 0;
    int32_t y   = 0;
    size_t w    = kCamWidth;
    size_t h    = kCamHeight;
    double zoom = 1. / (double)sim_rect_ratio; // Inverted - i.e. x2 zoom is expressed as 0.5, x4 as 0.25 etc
};