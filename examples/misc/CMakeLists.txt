
add_example(NAME "lut" WEB LINK_IMGUI)
add_example(NAME "raycast")
add_example(NAME "mouse_key_listener")
add_example(NAME "clipping" LINK_IMGUI)
add_example(NAME "morphtargets" LINK_IMGUI)
add_example(NAME "morphtargets_sphere" LINK_ASSIMP)
add_example(NAME "multiple_scenes")

if (THREEPP_WITH_DAWN)
    add_example(NAME "dawn_triangle")

    # Standalone raw wgpu-native examples (do not use DawnRenderer)
    add_executable(dawn_ocean standalone/dawn_ocean.cpp)
    target_link_libraries(dawn_ocean PRIVATE threepp ${WGPU_LIBRARY} glfw)
    target_include_directories(dawn_ocean PRIVATE
        "${WGPU_INCLUDE_DIR}"
        "${PROJECT_SOURCE_DIR}/src/external/glfw/include"
    )
    target_compile_definitions(dawn_ocean PRIVATE
        DATA_FOLDER="${PROJECT_SOURCE_DIR}/data"
        NOMINMAX
    )
endif ()
