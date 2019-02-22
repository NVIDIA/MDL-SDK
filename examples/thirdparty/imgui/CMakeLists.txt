#*****************************************************************************
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
