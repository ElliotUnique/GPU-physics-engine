# GPU Physics Engine

A real-time 2D physics engine written in C++ and CUDA. The design goal is for simulation to run primarily on the GPU, with the CPU in a purely coordinating role.

Two solvers are implemented: Position-Based Fluids (PBF) and the Material Point Method (MLS-MPM).

## Layout

```
main/               Application entry points / drivers
libs/common/        Shared types, constants, macros, threading utilities
libs/cuda-engine/   Simulation core — PBF and MLS-MPM solvers
libs/render/        CUDA-interop rendering and windowing (Win32)
```

## Starting points

- `main/PBF-master.cpp` — PBF driver
- `main/MPM-master.cpp` — MLS-MPM driver
- `libs/cuda-engine/source/PBF.cu`, `MPM.cu` — solver implementations
- `libs/cuda-engine/include/cuda-engine/cuda-engine.h` — the engine's public interface

## Building

Windows only. Requires the CUDA Toolkit and the Windows SDK (Win32 and Direct3D 12).

Configured with Ninja Multi-Config:

```
cmake -B build -G "Ninja Multi-Config" -DCMAKE_CONFIGURATION_TYPES="RelWithDebInfo;Release"
cmake --build build --config Release
```
