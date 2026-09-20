add_library(common INTERFACE)

target_include_directories(common INTERFACE include)
target_compile_definitions(common INTERFACE
    $<$<CONFIG:RelWithDebInfo>:ENGINE_ASSERTS=1>)

target_sources(common PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/common/umbrella.h)

add_library(engine::common ALIAS common)