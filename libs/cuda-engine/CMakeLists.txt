add_library(cuda-engine STATIC
    source/MPM.cu
    source/PBF.cu
    source/cuda-engine.cpp)

target_sources(cuda-engine PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda-engine/cuda-engine.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda-engine/cuda-internal.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda-engine/PBF-internal.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda-engine/MPM-internal.cuh)

target_include_directories(cuda-engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(cuda-engine PUBLIC
    CUDA::cudart
    engine::common)