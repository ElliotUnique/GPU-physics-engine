#pragma once
#include <stdio.h>
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(const char*);
#ifndef ENGINE_ASSERTS
#define ENGINE_ASSERTS 0
#endif

#define CUDACHECK(x)                                                                               \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            char buf_[512];                                                                        \
            snprintf(                                                                              \
                buf_,                                                                              \
                sizeof buf_,                                                                       \
                "%s:%d %s -> %s (%s)\n",                                                           \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                #x,                                                                                \
                cudaGetErrorName(e_),                                                              \
                cudaGetErrorString(e_)                                                             \
            );                                                                                     \
            OutputDebugStringA(buf_);                                                              \
            __debugbreak();                                                                        \
        }                                                                                          \
    } while (0)
/*
#if ENGINE_ASSERTS

#if defined(__CUDA_ARCH__)
#define ASSERT(cond)                                                                               \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf("assert: %s  %s:%d\n", #cond, __FILE__, __LINE__);                              \
            __trap();                                                                              \
        }                                                                                          \
    } while (0)
#else
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <windows.h>

namespace diag
{

bool assertFailed(
    const char* expr,
    const char* file,
    int line
)
{
    char msg[1024];
    std::snprintf(
        msg,
        sizeof msg,
        "ASSERT [tid %zu] %s\n  %s:%d\n",
        std::hash<std::thread::id>{}(std::this_thread::get_id()),
        expr,
        file,
        line
    );

    OutputDebugStringA(msg);
    std::fputs(msg, stderr);
    std::fflush(stderr);

    if (IsDebuggerPresent()) {
        return true; // caller breaks, in its own frame
    }

    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    std::abort();
}

} // namespace diag
#define ASSERT(cond)                                                                               \
    do {                                                                                           \
        if (!(cond)) [[unlikely]] {                                                                \
            if (::diag::assertFailed(#cond, __FILE__, __LINE__)) { __debugbreak(); }               \
        }                                                                                          \
    } while (0)
#endif

#else
#define ASSERT(cond)                                                                               \
    do {                                                                                           \
        (void)sizeof((cond) ? 1 : 0);                                                              \
    } while (0)
#endif
*/