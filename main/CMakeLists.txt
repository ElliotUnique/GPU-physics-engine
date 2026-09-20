add_executable(PBF-master
    master.manifest
    PBF-master.cpp)

target_link_libraries(PBF-master PRIVATE 
    CUDA::cudart
    cuda-engine
    render
    engine::common)

target_compile_definitions(PBF-master PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX 
    WIN32_LEAN_AND_MEAN)

add_executable(MPM-master
    MPM-master.cpp
    master.manifest)

target_link_libraries(MPM-master PRIVATE 
    CUDA::cudart
    cuda-engine
    render
    engine::common)

target_compile_definitions(MPM-master PRIVATE
    UNICODE
    _UNICODE
    NOMINMAX 
    WIN32_LEAN_AND_MEAN)