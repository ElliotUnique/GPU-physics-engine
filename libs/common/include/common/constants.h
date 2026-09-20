#pragma once
// Standard libs
#include <chrono>
#include <stdint.h>
// CUDA libs
// My libs

// ===== General =================================
constexpr uint32_t sign_mask = 0x80000000;
constexpr size_t kNumSMs     = 9;
constexpr int32_t kSceneDimX = 12; // In chunk widths
constexpr int32_t kSceneDimY = 6;  // In chunk heights

// ===== Parallelism =============================
// CPU presets
constexpr uint32_t kWorkerCount = 1;
// GPU presets
constexpr size_t kCudaBlockX = 32;
constexpr size_t kCudaBlockY = 8;
constexpr size_t kCudaGridX  = 6;
constexpr size_t kCudaGridY  = 6;

// ===== Core Physics ============================
constexpr auto kTickPeriod = std::chrono::milliseconds(32);
constexpr float kDt        = std::chrono::duration<float>(kTickPeriod).count();
constexpr float kGravity   = 700.f;
constexpr float kMass      = 1.f;
constexpr float kInvMass   = 1.f / kMass;
constexpr float kPi        = 3.14159f;

// ===== PBF =====================================

// Physical constants
constexpr float kAtomDiam   = 1.f;
constexpr float kH          = 2.f;
constexpr float kRho0       = 1.015f;
constexpr float kInvRho0    = 1.f / kRho0;
constexpr float kCoeffPoly6 = 1.f / (64.f * kPi);
constexpr float kCoeffSpiky = 30.f / (32.f * kPi);
constexpr float kDq         = 0.2f * kH;
constexpr float kInvPoly6Dq =
    1.f / (kCoeffPoly6 * (kH * kH - kDq * kDq) * (kH * kH - kDq * kDq) * (kH * kH - kDq * kDq));

// Structural
constexpr int32_t kGridDim       = 16; // In cells widths/heights
constexpr int32_t kCellDim       = 2;  // In atom diameters
constexpr int32_t kCellSlots     = 8;
constexpr float kChunkDim        = 32.f;
constexpr int32_t kNumChunks     = kSceneDimX * kSceneDimY;
constexpr int32_t kMaxAtoms      = 128 * 12; // Per chunk
constexpr int32_t kMaxMigrants   = 320;
constexpr int32_t kCellsPerChunk = kGridDim * kGridDim;     // 256 sort cells per chunk
constexpr int32_t kHaloCells     = 4 * kGridDim + 4;        // 68: 4 edges + 4 corners of imported cells
constexpr int32_t kHaloAtoms     = kHaloCells * kCellSlots; // 544: 4 edges + 4 corners of imported atoms
constexpr float kCoincSqr        = 1e-6f;                   // For locked pair disentanglement

// ===== Renderer ================================
// Client rectangle
constexpr float kSimScale    = 5.f;
constexpr size_t kDispWidth  = (size_t)((float)kSceneDimX * kChunkDim * kSimScale);
constexpr size_t kDispHeight = (size_t)((float)kSceneDimY * kChunkDim * kSimScale);
// Sim
constexpr double sim_rect_ratio = 1.;
constexpr size_t kSimWidth      = (size_t)(kSceneDimX * (int32_t)kChunkDim);
constexpr size_t kSimHeight     = (size_t)(kSceneDimY * (int32_t)kChunkDim);
// Camera
constexpr size_t kCamWidth  = (size_t)((double)kSimWidth * sim_rect_ratio);
constexpr size_t kCamHeight = (size_t)((double)kSimHeight * sim_rect_ratio);

// ===== MLS-MPM =================================
// Structural
constexpr int32_t kSubsteps = 20;
// Metadata sizes
constexpr size_t kGridHalo     = 2;
constexpr size_t kGridCellDim  = 2;
constexpr size_t kGridDimX     = (kSimWidth / kGridCellDim) + kGridHalo * 2;
constexpr size_t kGridDimY     = (kSimHeight / kGridCellDim) + kGridHalo * 2;
constexpr size_t kAlignPadding = 2 << 10;
// Physical constants
constexpr float kDx         = (float)kGridCellDim;
constexpr float inv_dx      = 1.f / kDx;
constexpr float kRestVol    = 1.2f;
constexpr int32_t kNumAtoms = 400 * 1000;
constexpr float kStiffness  = 8.2e4f;
constexpr float kGamma      = 3.2f;
constexpr float kMinPress   = -0.f * kStiffness;
constexpr float kMaxPress   = 10.f * kStiffness;
constexpr float dt          = kDt / kSubsteps;
