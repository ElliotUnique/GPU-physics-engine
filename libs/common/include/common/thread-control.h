#pragma once
#if defined(__CUDACC__)
#error "Host-only header: do not include from .cu translation units"
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdint.h>

// Will eventually turn into proper user settings with a permanent home in secondary memory.
struct GUISettings
{
    size_t disp_width  = 0; // The resolution X dim of the user's monitor
    size_t disp_height = 0; // The resolution Y dim of the user's monitor
    int32_t FPS        = 30;
    bool vsync         = true;
    bool fullscreen    = false; // Whether the application is in borderless fullscreen (true) or
                                // windowed mode (false)
};

enum class Status : int32_t
{
    NEUTRAL,
    READY,
    EXIT = -1,
};
#pragma warning(push)
#pragma warning(disable : 4324) // Stops the compiler from warning about the padding
// Wrapper for atomic Status enum class. Owned by the master thread, used by workers to signal their current
// status to the master.
struct alignas(64) ThreadSlot
{
    std::atomic<Status> status = Status::NEUTRAL;
    Status load() const { return status.load(); }
    void store(Status s) { status.store(s); }
    bool operator==(const Status rhs) const { return (status.load() == rhs); }
};
#pragma warning(pop)

struct Bell
{
    std::mutex m;
    std::condition_variable cv;
    uint64_t board = 0;

    void ring()
    {
        std::lock_guard lk(m);
        ++board;
        cv.notify_one();
    }
};
