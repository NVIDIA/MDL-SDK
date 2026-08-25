#*****************************************************************************
# Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME render-baker-baker)

# collect sources
set(PROJECT_HEADERS
    "baker.h"
    "i_baker.h"
    "baker_shared.h"
    )

set(PROJECT_SOURCES
    "baker.cpp"
    ${PROJECT_HEADERS}
    )

# The GPU baker is controlled by MDL_ENABLE_GPU_BAKER (declared in the top-level
# CMakeLists.txt). When ON, the baker compiles the kernel to PTX, embeds it into
# the library, and links against libcuda and libcudart at build time.
if(MDL_ENABLE_GPU_BAKER)
    list(APPEND PROJECT_HEADERS
        "baker_cuda_oss.h"
        "baker_kernel_oss_params.h"
        "texture_support_cuda.h"
        )
    list(APPEND PROJECT_SOURCES
        "baker_cuda_oss.cpp"
        )
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        boost
    )

if(MDL_ENABLE_GPU_BAKER)
    # PTX compilation of the kernel — a separate object library target,
    # produced by target_add_cuda_ptx_rule from utilities.cmake.
    target_add_cuda_ptx_rule(TARGET ${PROJECT_NAME}
        CUDA_SOURCES
            "baker_kernel_oss.cu"
        FAST_MATH
        )

    # Force the PTX (and thus the embedded baker_kernel_oss_ptx.h) to rebuild
    # when the device-side headers change. CMake's CUDA header-dependency
    # scanning is unreliable on the VS/MSBuild generator, so without this an
    # edit to texture_support_cuda.h (or the shared layout/constant headers it
    # pulls in) can leave a STALE embedded kernel that disagrees with
    # baker_cuda_oss.cpp at runtime.
    set_source_files_properties("baker_kernel_oss.cu" PROPERTIES
        OBJECT_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/texture_support_cuda.h;${CMAKE_CURRENT_SOURCE_DIR}/baker_kernel_oss_params.h;${CMAKE_CURRENT_SOURCE_DIR}/baker_shared.h"
        )

    # NVCC needs to see <mi/neuraylib/target_code_types.h>, which lives in
    # the public MDL include tree. The helper above doesn't add this by
    # default, so wire it explicitly onto the PTX target.
    target_include_directories(${PROJECT_NAME}_PTX PRIVATE
        ${MDL_INCLUDE_FOLDER}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

    # Embed the resulting PTX as a C unsigned-char array into a generated
    # header. The PTX file is copied into a per-config subdir of
    # ${CMAKE_CURRENT_BINARY_DIR} by the helper.
    # Resolve the Python interpreter (sets ${python_PATH}).
    target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)

    set(_PTX_LOC "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/baker_kernel_oss.ptx")
    set(GEN_PTX_HEADER "${CMAKE_CURRENT_BINARY_DIR}/baker_kernel_oss_ptx.h")
    set(BIN2C_SCRIPT   "${MDL_BASE_FOLDER}/cmake/bin2c.py")

    # DEPENDS on the device headers as well, so that editing them re-runs bin2c
    # (re-embedding the freshly recompiled PTX). Without this, an incremental
    # build on the VS generator can leave a STALE embedded kernel that no longer
    # matches baker_cuda_oss.cpp at runtime.
    add_custom_command(
        OUTPUT  "${GEN_PTX_HEADER}"
        COMMAND ${python_PATH} "${BIN2C_SCRIPT}"
                --input  "${_PTX_LOC}"
                --output "${GEN_PTX_HEADER}"
                --symbol g_baker_kernel_oss_ptx
        DEPENDS ${PROJECT_NAME}_PTX_COPY
                "${BIN2C_SCRIPT}"
                "${CMAKE_CURRENT_SOURCE_DIR}/texture_support_cuda.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/baker_kernel_oss_params.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/baker_shared.h"
        COMMAND_EXPAND_LISTS
        VERBATIM
        )
    target_sources(${PROJECT_NAME} PRIVATE "${GEN_PTX_HEADER}")
    add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_PTX_COPY)

    target_compile_definitions(${PROJECT_NAME} PRIVATE MDL_ENABLE_GPU_BAKER)

    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS cuda
        )
    # for the generated baker_kernel_oss_ptx.h
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()
