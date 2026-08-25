#*****************************************************************************
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

# name of the target and the resulting library
set(PROJECT_NAME examples-mdl_sdk-shared)

# collect sources
set(PROJECT_SOURCES
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
    "utils/loading.h"
    "utils/loading.cpp"
    "utils/main_utf8.h"
    "utils/mdl.h"
    "utils/mdl.cpp"
    "utils/os.h"
    "utils/sdk_strings.h"
    )

if (MDL_BUILD_SDK_EXAMPLES)
    list(APPEND PROJECT_SOURCES
        "gui/gui.h"
        "gui/gui.cpp"
        "gui/gui_material_properties.h"
        "gui/gui_material_properties.cpp"
        )

    if(MDL_ENABLE_CUDA_EXAMPLES)
        list(APPEND PROJECT_SOURCES
            "example_cuda_shared.h"
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

    if(MDL_ENABLE_MATERIALX)
        list(APPEND PROJECT_SOURCES
            "example_materialx_shared.h"
            "example_materialx_shared.cpp"
            )
    endif()

    if(MDL_ENABLE_OPENGL_EXAMPLES)
        list(APPEND PROJECT_SOURCES
            "utils/gl_display.h"
            "gui/gui_api_interface_glfw.h"
            "example_glsl_shared.h"
            )
    endif()

    if(MDL_ENABLE_VULKAN_EXAMPLES)
        list(APPEND PROJECT_SOURCES
            "example_vulkan_shared.h"
            "example_vulkan_shared.cpp"
            )
    endif()
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    OUTPUT_NAME "shared"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
)

add_library(mdl_sdk::shared ALIAS ${PROJECT_NAME})

# inject the MDL_EXAMPLES_ROOT path
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        "MDL_EXAMPLES_ROOT=\"${MDL_EXAMPLES_FOLDER}\""
    )

if(WINDOWS)
    # MS deprecated the codecvt_utf8 functions without providing a replacement
    # the deprecated function will be available until there is such a replacement
    target_compile_definitions(${PROJECT_NAME}
        PUBLIC
            "_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING"
        )
endif()

if(MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS)
    # make sure the plugins are build before examples are build
    if(TARGET shaders-plugin-dds)
        add_dependencies(${PROJECT_NAME} shaders-plugin-dds)
    endif()
    if(TARGET shaders-plugin-openimageio)
        add_dependencies(${PROJECT_NAME} shaders-plugin-openimageio)
    endif()
    if(TARGET shaders-plugin-mdl_distiller)
        add_dependencies(${PROJECT_NAME} shaders-plugin-mdl_distiller)
    endif()
endif()

# add include directories
target_include_directories(${PROJECT_NAME}
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MDL_SRC_FOLDER}/render/baker/baker
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
    NO_RUNTIME_COPY
    )

# add dependencies
if(MDL_BUILD_SDK_EXAMPLES)
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            imgui
        NO_RUNTIME_COPY
        )

    if(MDL_ENABLE_MATERIALX)
        target_add_dependencies(TARGET ${PROJECT_NAME}
            DEPENDS
                materialx
            )
    endif()

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
