#*****************************************************************************
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

# name of the target and the resulting example
set(PROJECT_NAME examples-mdl_sdk-dxr)

# additional third-party dependencies
set(THIRDPARTY_SOURCES
    "../../thirdparty/d3dx12/d3dx12.h"
    "../../thirdparty/gltf/fx/gltf.h"
    "../../thirdparty/gltf/nlohmann/json.hpp"
    )

# collect headers
set(PROJECT_HEADERS
    "mdl_d3d12/base_application.h"
    "mdl_d3d12/buffer.h"
    "mdl_d3d12/camera_controls.h"
    "mdl_d3d12/command_queue.h"
    "mdl_d3d12/common.h"
    "mdl_d3d12/descriptor_heap.h"
    "mdl_d3d12/gltf.h"
    "mdl_d3d12/mdl_d3d12.h"
    "mdl_d3d12/mdl_material.h"
    "mdl_d3d12/mdl_material_description.h"
    "mdl_d3d12/mdl_material_library.h"
    "mdl_d3d12/mdl_material_target.h"
    "mdl_d3d12/mdl_sdk.h"
    "mdl_d3d12/raytracing_pipeline.h"
    "mdl_d3d12/scene.h"
    "mdl_d3d12/shader.h"
    "mdl_d3d12/texture.h"
    "mdl_d3d12/utils.h"
    "mdl_d3d12/window.h"
    "mdl_d3d12/window_win32.h"
    "mdl_d3d12/window_image_file.h"
    "example_dxr.h"
    "example_dxr_gui.h"
    "example_dxr_options.h"
    )

# collect sources
set(PROJECT_SOURCES
    "mdl_d3d12/base_application.cpp"
    "mdl_d3d12/buffer.cpp"
    "mdl_d3d12/camera_controls.cpp"
    "mdl_d3d12/command_queue.cpp"
    "mdl_d3d12/descriptor_heap.cpp"
    "mdl_d3d12/gltf.cpp"
    "mdl_d3d12/mdl_material.cpp"
    "mdl_d3d12/mdl_material_description.cpp"
    "mdl_d3d12/mdl_material_library.cpp"
    "mdl_d3d12/mdl_material_target.cpp"
    "mdl_d3d12/mdl_sdk.cpp"
    "mdl_d3d12/raytracing_pipeline.cpp"
    "mdl_d3d12/scene.cpp"
    "mdl_d3d12/shader.cpp"
    "mdl_d3d12/texture.cpp"
    "mdl_d3d12/utils.cpp"
    "mdl_d3d12/window_win32.cpp"
    "mdl_d3d12/window_image_file.cpp"
    "example_dxr_gui.cpp"

    ${PROJECT_HEADERS}
    ${THIRDPARTY_SOURCES}
    )

if(MDL_ENABLE_MATERIALX)
    set(PROJECT_SOURCES
        ${PROJECT_SOURCES}
        "mdl_d3d12/materialx/mdl_generator.h"
        "mdl_d3d12/materialx/mdl_generator.cpp"
        "mdl_d3d12/materialx/mdl_material_description_loader_mtlx.h"
        "mdl_d3d12/materialx/mdl_material_description_loader_mtlx.cpp"
    )
endif()

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME} WIN32
    TYPE WIN_EXECUTABLE
    NAMESPACE mdl_sdk
    OUTPUT_NAME "dxr"
    SOURCES
        ${PROJECT_SOURCES}
         "example_dxr.cpp"
    ADDITIONAL_INCLUDE_DIRS
        "../../thirdparty/d3dx12"
        "../../thirdparty/gltf"
    EXAMPLE
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        d3d12
        imgui
        mdl::mdl_sdk
        mdl_sdk::shared
    )

# copy content
target_add_content(TARGET ${PROJECT_NAME}
    FILES
        "content/common.hlsl"
        "content/mdl_renderer_runtime.hlsl"
        "content/mdl_target_code_types.hlsl"
        "content/mdl_hit_programs.hlsl"
        "content/miss_programs.hlsl"
        "content/ray_gen_program.hlsl"
        "content/gltf/sphere/sphere.bin"
        "content/gltf/sphere/sphere.gltf"
        "content/hdri/hdrihaven_teufelsberg_inner_2k.exr"
    )

# content potentially shared between examples
target_add_content(TARGET ${PROJECT_NAME}
    FILE_BASE "../../thirdparty/"
    DEP_NAME "content_shared"
    FILES
        "content/fonts/LICENSE.txt"
        "content/fonts/NotoSans-Medium.ttf"
        "content/fonts/NotoSansArabic-Medium.ttf"
        "content/fonts/NotoSansCJK-Medium.ttc"
        "content/fonts/NotoSansHebrew-Medium.ttf"
        "content/fonts/NotoSansThai-Medium.ttf"
    )

if(MDL_ENABLE_MATERIALX)
# add material support
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            materialx
        )

    target_compile_definitions(${PROJECT_NAME}
        PRIVATE
            "MDL_ENABLE_MATERIALX"
        )

    target_add_content(TARGET ${PROJECT_NAME}
        FILE_BASE "${MATERIALX_REPOSITORY}/"
        TARGET_BASE "autodesk_materialx/"
        DEP_NAME "content_mtlx"
        FILES
            "libraries/bxdf/disney_brdf_2012.mtlx"
            "libraries/bxdf/disney_brdf_2015.mtlx"
            "libraries/bxdf/standard_surface.mtlx"
            "libraries/bxdf/usd_preview_surface.mtlx"
            "libraries/pbrlib/genmdl/pbrlib_genmdl_impl.mtlx"
            "libraries/pbrlib/pbrlib_defs.mtlx"
            "libraries/pbrlib/pbrlib_ng.mtlx"
            "libraries/stdlib/genmdl/stdlib_genmdl_cm_impl.mtlx"
            "libraries/stdlib/genmdl/stdlib_genmdl_impl.mtlx"
            "libraries/stdlib/genmdl/stdlib_genmdl_unit_impl.mtlx"
            "libraries/stdlib/stdlib_defs.mtlx"
            "libraries/stdlib/stdlib_ng.mtlx"
            "source/MaterialXGenMdl/mdl/materialx/cm.mdl"
            "source/MaterialXGenMdl/mdl/materialx/core.mdl"
            "source/MaterialXGenMdl/mdl/materialx/hsv.mdl"
            "source/MaterialXGenMdl/mdl/materialx/noise.mdl"
            "source/MaterialXGenMdl/mdl/materialx/pbrlib.mdl"
            "source/MaterialXGenMdl/mdl/materialx/sampling.mdl"
            "source/MaterialXGenMdl/mdl/materialx/stdlib.mdl"
            "source/MaterialXGenMdl/mdl/materialx/swizzle.mdl"
        )

endif()

# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

# add tests if available
add_tests()
