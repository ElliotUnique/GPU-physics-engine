cmake_minimum_required(VERSION 3.25...4.3)

set(CMAKE_CONFIGURATION_TYPES "Debug;RelWithDebInfo;Release" CACHE STRING "" FORCE)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

project(physics-engine LANGUAGES CXX CUDA)
message(STATUS "config types: [${CMAKE_CONFIGURATION_TYPES}]")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61)

set(CMAKE_DEFAULT_BUILD_TYPE RelWithDebInfo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CXX>:/WX>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive-> # Disables MSVC's non-conforming lang extensions
    $<$<COMPILE_LANGUAGE:CXX>:/FC>) # Makes diagnostics print full source paths for errors
add_compile_options(
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-lineinfo> # Conflicts with Debug's -G
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/WX,/Z7,/FC,/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

add_subdirectory(libs)
add_subdirectory(main)