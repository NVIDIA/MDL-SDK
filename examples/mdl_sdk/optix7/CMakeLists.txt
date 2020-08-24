#*****************************************************************************
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME examples-mdl_sdk-optix7)

# -------------------------------------------------------------------------------------------------
# Main Target
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_SOURCES
    "optix7_mdl.cpp"
    "optix7_mdl.h"
    "optix7_mdl_helper.h"
    "material_info_helper.h"
    )

set(_GENERATED_SOURCES
    "optix7_mdl_closest_hit_radiance.bc"
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk
    OUTPUT_NAME "optix7"
    SOURCES
        ${PROJECT_SOURCES}
        ${_GENERATED_SOURCES}
    EXAMPLE
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
        opengl
        imgui
        cuda
        optix7
    )

# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})


# -------------------------------------------------------------------------------------------------
# CLANG Build Step
# -------------------------------------------------------------------------------------------------

# get clang
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL clang)

# on Linux, we need to set C++11
if(LINUX)
    set(_CLANG_FLAGS --std=c++11)
endif()

# Clang 7 needs CUDA 8 to compile the closest hit shader
set(CUDA8_PATH "NOT-SPECIFIED" CACHE PATH "Directory that contains the CUDA 8 include folder")
if(EXISTS ${CUDA8_PATH}/include/cuda.h)
    set(_CLANG_FLAGS ${_CLANG_FLAGS} --cuda-path=${CUDA8_PATH})
else()
    message(FATAL_ERROR "Please specify CUDA8_PATH which contains \"include/cuda.h\"")
endif()

# generate bytecode for closest hit radiance shader
add_custom_command(
    OUTPUT
        optix7_mdl_closest_hit_radiance.d
        optix7_mdl_closest_hit_radiance.bc
    COMMAND ${CMAKE_COMMAND} -E echo "Compile optix7_mdl_closest_hit_radiance bytecode using clang..."
    COMMAND ${clang_PATH}
        -I ${OPTIX7_DIR}/include -I ${MDL_INCLUDE_FOLDER}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/../shared
        ${_CLANG_FLAGS}
        -emit-llvm -c -O3 -ffast-math -fcuda-flush-denormals-to-zero -fno-vectorize
        --cuda-device-only ${CMAKE_CURRENT_SOURCE_DIR}/optix7_mdl_closest_hit_radiance.cu
        -o optix7_mdl_closest_hit_radiance.bc
        -MD -MT optix7_mdl_closest_hit_radiance.bc
        -MP -MF optix7_mdl_closest_hit_radiance.d
    WORKING_DIRECTORY
        ${CMAKE_CFG_INTDIR}
    DEPENDS
        "optix7_mdl_closest_hit_radiance.cu"
        ${PROJECT_SOURCES}
    VERBATIM
)

# -------------------------------------------------------------------------------------------------
# PTX Target
# -------------------------------------------------------------------------------------------------

target_add_cuda_ptx_rule(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
        optix7
    CUDA_SOURCES
        "optix7_mdl.cu"
        "optix7_mdl_closest_hit_radiance.cu"
    FAST_MATH
    )

# add tests if available
add_tests()
