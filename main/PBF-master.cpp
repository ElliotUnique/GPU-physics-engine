// Standard libs
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <math.h>
#include <mutex>
#include <stddef.h>
#include <stdint.h>
#include <thread>
// CUDA libs
#include <cuda_runtime_api.h>
#include <vector_types.h>
// My libs
#include "common/umbrella.h"
#include "cuda-engine/cuda-engine.h"
#include "render/renderer.h"

// Hosepipe settings
constexpr float2 origin        = {(float)kSceneDimX * float(kChunkDim) / 3.f, kChunkDim * 2.f};
constexpr float2 velocity      = {1.f, -0.3f};
constexpr float nozzle_width   = 8.f;
constexpr float spacing        = 1.05f * 1.f;
constexpr float jitter         = spacing * 0.05f;
constexpr int32_t duration     = 120;
constexpr int32_t TARGET_COUNT = 40000;

void recordSample(
    float ms,
    float& cum_tot,
    float& longest
);

int32_t main()
{
    // cudaDeviceProp p;
    // cudaGetDeviceProperties(&p, 0);
    // const auto d_props = p;
    GUISettings settings = {kDispWidth, kDispHeight, 60, false};

    PBFArena pbf;
    OutputBuffer buff;

    CUDACHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32u << 20));
    GraphEnv g{};
    instantiatePBFGraph(g, pbf);

    const View initial_output_target = buff.writeTarget();
    const int2 initial_torus_seams   = {0, 0};
    TickStaging stg;
    stg.init(initial_output_target, initial_torus_seams, g.stream);

    Bell bell{};

    ThreadSlot slots[kWorkerCount];

    std::jthread gui(
        rend::winEngineGUI,
        std::ref(settings),
        std::ref(buff),
        std::ref(slots[0].status),
        std::ref(bell)
    );

    slots[0].status.wait(Status::NEUTRAL, std::memory_order_acquire);
    slots[0].store(Status::NEUTRAL);

    auto nextTick              = std::chrono::steady_clock::now();
    uint64_t seen              = 0;
    uint64_t tick              = 0;
    uint64_t tick_last_sampled = 0;
    float cum_tot              = 0.f;
    float longest              = 0.f;
    int32_t atoms              = 0;
    for (;;) {
        std::unique_lock lk(bell.m);
        bell.cv.wait_until(lk, nextTick, [&] { return bell.board != seen; });
        seen = bell.board;
        lk.unlock();

        if (slots[0].load() == Status::EXIT) { break; }

        if (std::chrono::steady_clock::now() >= nextTick) {
            const cudaError_t st = cudaEventQuery(g.tick_done);
            if (st == cudaErrorNotReady) { continue; }
            if (tick > 0) {
                buff.roll();
                const View tgt = buff.writeTarget();
                stg.target()   = tgt;
                stg.outputTargetUpload(g.stream);
                float ms = 0.f;
                CUDACHECK(cudaEventElapsedTime(&ms, g.t0, g.t1));
                recordSample(ms, cum_tot, longest);
            }
            if (tick - tick_last_sampled == 150) {
                printf("\n------- Tick %u -------", (uint32_t)tick);
                printf("\nAverage duration = %f ms", cum_tot / 150.f);
                printf("\nLongest tick = %f ms", longest);
                printf("\nSimulating %i atoms", atoms);
                cum_tot           = 0.f;
                longest           = 0.f;
                tick_last_sampled = tick;
            }
            cudaMemsetAsync(
                pbf.halo_offsets,
                0,
                (size_t)kNumChunks * kHaloCells * sizeof(uint32_t),
                g.stream
            );
            if (atoms < TARGET_COUNT) {
                atoms += hosePipe(g.stream, pbf, origin, velocity, nozzle_width, spacing, jitter, tick);
            }
            CUDACHECK(cudaEventRecord(g.t0, g.stream));
            CUDACHECK(cudaGraphLaunch(g.exec, g.stream));
            CUDACHECK(cudaEventRecord(g.t1, g.stream));
            CUDACHECK(cudaEventRecord(g.tick_done, g.stream));

            nextTick += kTickPeriod;
            ++tick;
        }
    }
    return 0;
}

void recordSample(
    float ms,
    float& cum_tot,
    float& longest
)
{
    longest  = std::max(ms, longest);
    cum_tot += ms;
}