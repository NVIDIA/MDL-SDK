# name of the target and the resulting library
set(PROJECT_NAME imgui)
set(PROJECT_VERSION 1.54)

# collect sources
set(PROJECT_SOURCES
    "include/imconfig.h"
    "include/imgui.h"
    "include/imgui_impl_glfw_gl3.h"
    "src/imgui.cpp"
    "src/imgui_demo.cpp"
    "src/imgui_draw.cpp"
    "src/imgui_impl_glfw_gl3.cpp"
    "src/imgui_internal.h"
    "src/stb_rect_pack.h"
    "src/stb_textedit.h"
    "src/stb_truetype.h"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
# to avoid redundant definitions for the samples we use this helper function
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        opengl 
    # NO_LINKING      # the consumer will also add the opengl dependency
    # NO_RUNTIME_COPY # the consumer will also add the opengl dependency
    )

# add include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
