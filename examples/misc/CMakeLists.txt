
add_example(NAME "lut" WEB LINK_IMGUI)
add_example(NAME "raycast")
add_example(NAME "mouse_key_listener")
add_example(NAME "clipping" LINK_IMGUI)
add_example(NAME "morphtargets" LINK_IMGUI)
add_example(NAME "morphtargets_sphere" LINK_ASSIMP)
add_example(NAME "multiple_scenes")

if (THREEPP_WITH_DAWN)
    add_example(NAME "dawn_triangle")
    add_example(NAME "dawn_ocean")
    target_include_directories(dawn_ocean PRIVATE
        "${WGPU_INCLUDE_DIR}"
        "${PROJECT_SOURCE_DIR}/src/external/glfw/include"
    )
    target_link_libraries(dawn_ocean PRIVATE "${WGPU_LIBRARY}" glfw)
endif ()
