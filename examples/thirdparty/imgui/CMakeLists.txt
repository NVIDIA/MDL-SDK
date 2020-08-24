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
set(PROJECT_NAME imgui)
set(PROJECT_VERSION 1.75) # 58b3e02b95b4c7c5bb9128a28c6d55546501bf93

# collect sources
set(PROJECT_SOURCES
    "imconfig.h"
    "imgui.h"
    "imgui_internal.h"
    "imstb_rectpack.h"
    "imstb_textedit.h"
    "imstb_truetype.h"
    "imgui.cpp"
    "imgui_demo.cpp"
    "imgui_draw.cpp"
    "imgui_widgets.cpp"
    )

if(WINDOWS)
    list(APPEND PROJECT_SOURCES 
        "imgui_impl_win32.h"
        "imgui_impl_win32.cpp"
        )

    if(MDL_ENABLE_D3D11_EXAMPLES)
        list(APPEND PROJECT_SOURCES 
            "imgui_impl_dx11.cpp"
            "imgui_impl_dx11.h"
        )
    endif()

    if(MDL_ENABLE_D3D12_EXAMPLES)
        list(APPEND PROJECT_SOURCES 
            "imgui_impl_dx12.cpp"
            "imgui_impl_dx12.h"
        )
		include_directories(
			AFTER ${MDL_DEPENDENCY_D3D12_INCLUDE} ${MDL_DEPENDENCY_DXGI_INCLUDE}
		)
    endif()
endif()

if(MACOSX)
    list(APPEND PROJECT_SOURCES 
        "imgui_impl_osx.h"
        "imgui_impl_osx.mm"
        )
endif()

if(MDL_ENABLE_OPENGL_EXAMPLES)
    list(APPEND PROJECT_SOURCES 
        "imgui_impl_glfw.cpp"
        "imgui_impl_glfw.h"
        "imgui_impl_opengl3.cpp"
        "imgui_impl_opengl3.h"
    )
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
)

if(MDL_ENABLE_OPENGL_EXAMPLES)
    # add dependencies
    target_add_dependencies(TARGET ${PROJECT_NAME} 
        DEPENDS 
            opengl 
        )

    # additional definitions
    target_compile_definitions(${PROJECT_NAME} 
        PRIVATE 
            "IMGUI_IMPL_OPENGL_LOADER_GLEW" 
        )
endif()

# add include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
