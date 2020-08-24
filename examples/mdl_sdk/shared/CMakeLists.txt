#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

# name of the target and the resulting library
set(PROJECT_NAME examples-mdl_sdk-shared)

# collect sources
set(PROJECT_SOURCES
    "example_cuda_shared.h"
    "example_shared.cpp"
    "example_shared.h"
    "texture_support_cuda.h"
    "compiled_material_traverser_base.h"
    "compiled_material_traverser_base.cpp"
    "lpe.cpp"
    "lpe.h"
    "gui/gui.h"
    "gui/gui.cpp"
    "gui/gui_api_interface_dx11.h"
    "gui/gui_api_interface_dx12.h"
    "gui/gui_material_properties.h"
    "gui/gui_material_properties.cpp"
    "utils/enums.h"
    "utils/io.h"
    "utils/mdl.h"
    "utils/os.h"
    "utils/strings.h"
    )

if(MDL_ENABLE_OPENGL_EXAMPLES)
    set(PROJECT_SOURCES
        ${PROJECT_SOURCES}
        "utils/gl_display.h"
        )
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE mdl_sdk
    OUTPUT_NAME "shared"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
)
add_library(mdl_sdk::shared ALIAS ${PROJECT_NAME})

# inject the MDL_SAMPLES_ROOT path
target_compile_definitions(${PROJECT_NAME} 
    PUBLIC
        "MDL_SAMPLES_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/../..\""
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        imgui
    NO_RUNTIME_COPY     # header only -> the consumer will also add this dependency
    )

# make sure the plugins are build before samples are build
if(TARGET shaders-plugin-dds)
    add_dependencies(${PROJECT_NAME} shaders-plugin-dds)
endif()

if(TARGET shaders-plugin-freeimage)
    add_dependencies(${PROJECT_NAME} shaders-plugin-freeimage)
endif()

# add include directories
target_include_directories(${PROJECT_NAME}
    INTERFACE
        # the template makes non public headers private by default
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
    )
