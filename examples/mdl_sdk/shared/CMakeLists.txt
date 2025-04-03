#*****************************************************************************
# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

# collect sources
set(PROJECT_SOURCES
    "example_shared.cpp"
    "example_shared.h"
    "example_shared_dump.cpp"
    "example_shared_dump.h"
    "compiled_material_traverser_base.h"
    "compiled_material_traverser_base.cpp"
    "lpe.cpp"
    "lpe.h"
    "texture_support_native.h"
    "utils/enums.h"
    "utils/io.h"
    "utils/mdl.h"
    "utils/mdl.cpp"
    "utils/os.h"
    "utils/strings.h"
    )

if (MDL_BUILD_SDK_EXAMPLES)
    list(APPEND PROJECT_SOURCES
        "gui/gui.h"
        "gui/gui.cpp"
        "gui/gui_material_properties.h"
        "gui/gui_material_properties.cpp"
        )

    if(MDL_ENABLE_AXF_EXAMPLES)
        list(APPEND PROJECT_SOURCES
            "utils/ospath.h"
            "utils/ospath.cpp"
            )
    endif()

    if(MDL_ENABLE_CUDA_EXAMPLES)
        list(APPEND PROJECT_SOURCES_NON_DXR
            "example_cuda_shared.h"
            "texture_support_cuda.h"
            )
    endif()

    if(MDL_ENABLE_D3D11_EXAMPLES)
        list(APPEND PROJECT_SOURCES
            "gui/gui_api_interface_dx11.h"
            )
    endif()

    if(MDL_ENABLE_D3D12_EXAMPLES)
        list(APPEND PROJECT_SOURCES
            "gui/gui_api_interface_dx12.h"
            )
    endif()

    if(MDL_ENABLE_OPENGL_EXAMPLES)
        list(APPEND PROJECT_SOURCES_NON_DXR
            "utils/gl_display.h"
            "gui/gui_api_interface_glfw.h"
            "example_glsl_shared.h"
            )
    endif()

    if(MDL_ENABLE_VULKAN_EXAMPLES)
        list(APPEND PROJECT_SOURCES_NON_DXR
            "example_vulkan_shared.h"
            "example_vulkan_shared.cpp"
            )
    endif()
endif()


# helper function for common shared target properties, dependencies, etc.
function(ADD_COMMON_TARGET_PROPERTIES TARGET_NAME)
    # inject the MDL_SAMPLES_ROOT path
    target_compile_definitions(${TARGET_NAME}
        PRIVATE
            "MDL_SAMPLES_ROOT=\"${MDL_EXAMPLES_FOLDER}\""
        )

    if(WINDOWS)
        # MS deprecated the codecvt_utf8 functions without providing a replacement
        # the deprecated function will be available until there is such a replacement
        target_compile_definitions(${TARGET_NAME}
            PUBLIC
                "_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING"
            )
    endif()

    if(MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS)
        # make sure the plugins are build before samples are build
        if(TARGET shaders-plugin-dds)
            add_dependencies(${TARGET_NAME} shaders-plugin-dds)
        endif()
        if(TARGET shaders-plugin-openimageio)
            add_dependencies(${TARGET_NAME} shaders-plugin-openimageio)
        endif()
        if(TARGET shaders-plugin-mdl_distiller)
            add_dependencies(${TARGET_NAME} shaders-plugin-mdl_distiller)
        endif()
    endif()

    # add include directories
    target_include_directories(${TARGET_NAME}
        INTERFACE
            # the template makes non public headers private by default
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        )

    # add dependencies
    target_add_dependencies(TARGET ${TARGET_NAME}
        DEPENDS
            mdl::mdl_sdk
        NO_RUNTIME_COPY
        )
endfunction()



# name of the target and the resulting library
#
# This variant is used by all examples except the DXR example.
set(PROJECT_NAME examples-mdl_sdk-shared)

# Select MSVC runtime based on MDL_MSVC_DYNAMIC_RUNTIME. This is the default behavior from
# target_build_setup(), just for consistency with DXR case below.
if(MDL_MSVC_DYNAMIC_RUNTIME)
    set(_SHARED_MSVC_RUNTIME "DYNAMIC_MSVC_RUNTIME")
else()
    set(_SHARED_MSVC_RUNTIME "STATIC_MSVC_RUNTIME")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE mdl_sdk
    OUTPUT_NAME "shared"
    SOURCES ${PROJECT_SOURCES} ${PROJECT_SOURCES_NON_DXR}
    EXAMPLE
    ${_SHARED_MSVC_RUNTIME}
)

add_library(mdl_sdk::shared ALIAS ${PROJECT_NAME})

add_common_target_properties(${PROJECT_NAME})

# add dependencies
if(MDL_BUILD_SDK_EXAMPLES)
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            imgui
        NO_RUNTIME_COPY
        )

    if(MDL_ENABLE_OPENGL_EXAMPLES OR MDL_ENABLE_VULKAN_EXAMPLES)
        target_add_dependencies(TARGET ${PROJECT_NAME}
            DEPENDS
                glfw
            NO_RUNTIME_COPY
            )
    endif()

    if(MDL_ENABLE_OPENGL_EXAMPLES)
        target_add_dependencies(TARGET ${PROJECT_NAME}
            DEPENDS
                opengl
            NO_RUNTIME_COPY
            )
    endif()

    if(MDL_ENABLE_VULKAN_EXAMPLES)
        target_add_dependencies(TARGET ${PROJECT_NAME}
            DEPENDS
                vulkan
            NO_RUNTIME_COPY
            )
    endif()
endif()



if(MDL_ENABLE_D3D12_EXAMPLES)

# name of the target and the resulting library
#
# This variant is used by the MDL SDK DXR example.
set(PROJECT_NAME examples-mdl_sdk-shared_dxr)

# Select MSVC runtime based on MDL_MSVC_DYNAMIC_RUNTIME_DXR_EXAMPLE.
if(MDL_MSVC_DYNAMIC_RUNTIME_DXR_EXAMPLE)
    set(_SHARED_DXR_MSVC_RUNTIME "DYNAMIC_MSVC_RUNTIME")
else()
    set(_SHARED_DXR_MSVC_RUNTIME "STATIC_MSVC_RUNTIME")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE mdl_sdk
    OUTPUT_NAME "shared_dxr"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
    ${_SHARED_DXR_MSVC_RUNTIME}
)

add_library(mdl_sdk::shared_dxr ALIAS ${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES PROJECT_LABEL "shared_dxr")

add_common_target_properties(${PROJECT_NAME})

# add dependencies
if(MDL_BUILD_SDK_EXAMPLES)
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            imgui_dxr
        NO_RUNTIME_COPY
        )

    # OpenGL/Vulkan dependencies not relevant for the MDL SDK DXR example.
endif()

endif()
