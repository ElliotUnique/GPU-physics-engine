add_library(render STATIC
    source/cuda-render.cu
    source/renderer.cpp
    source/window-thread.cpp)

set_target_properties(render PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

target_compile_definitions(render PUBLIC
    $<$<CONFIG:RelWithDebInfo>:ENGINE_ASSERTS=1>
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    UNICODE
    UNICODE_)

target_sources(render PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/render/cuda-interface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/render/renderer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/render/cuda-internal.cuh)


target_include_directories(render PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(render PUBLIC
    d3d12
    dxgi
    dxguid
    engine::common
    cuda-engine
)
