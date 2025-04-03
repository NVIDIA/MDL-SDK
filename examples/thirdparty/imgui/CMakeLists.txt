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

set(PROJECT_VERSION 1.90.9) # cb16be3a3fc1f9cd146ae24d52b615f8a05fa93d

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
    "imgui_tables.cpp"
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

if(MDL_ENABLE_OPENGL_EXAMPLES OR MDL_ENABLE_VULKAN_EXAMPLES)
    list(APPEND PROJECT_SOURCES_NON_DXR
        "imgui_impl_glfw.cpp"
        "imgui_impl_glfw.h"
    )
endif()

if(MDL_ENABLE_OPENGL_EXAMPLES)
    list(APPEND PROJECT_SOURCES_NON_DXR
        "imgui_impl_opengl3.cpp"
        "imgui_impl_opengl3.h"
        "imgui_impl_opengl3_loader.h"
    )
endif()

if(MDL_ENABLE_VULKAN_EXAMPLES)
    list(APPEND PROJECT_SOURCES_NON_DXR
        "imgui_impl_vulkan.cpp"
        "imgui_impl_vulkan.h"
    )
endif()



# name of the target and the resulting library
#
# This variant is used by all examples except the MDL SDK DXR example.
set(PROJECT_NAME imgui)

# Select MSVC runtime based on MDL_MSVC_DYNAMIC_RUNTIME. This is the default behavior from
# target_build_setup(), just for consistency with DXR case below.
if(MDL_MSVC_DYNAMIC_RUNTIME)
    set(_IMGUI_MSVC_RUNTIME "DYNAMIC_MSVC_RUNTIME")
else()
    set(_IMGUI_MSVC_RUNTIME "STATIC_MSVC_RUNTIME")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOURCES ${PROJECT_SOURCES} ${PROJECT_SOURCES_NON_DXR}
    EXAMPLE
    ${_IMGUI_MSVC_RUNTIME}
)

# add include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )

# suppress MacOS/clang specific warning
if(MACOSX)
    target_compile_options(${PROJECT_NAME}
        PRIVATE
            "-Wno-arc-bridge-casts-disallowed-in-nonarc"
        )
endif()

# add dependencies
if(MDL_ENABLE_OPENGL_EXAMPLES OR MDL_ENABLE_VULKAN_EXAMPLES)
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            glfw
        )
endif()

if(MDL_ENABLE_OPENGL_EXAMPLES)
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            opengl
        )
endif()

if(MDL_ENABLE_VULKAN_EXAMPLES)
    target_add_dependencies(TARGET ${PROJECT_NAME}
        DEPENDS
            vulkan
        )

    target_compile_definitions(${PROJECT_NAME}
        PRIVATE
            "IMGUI_IMPL_VULKAN_USE_VOLK"
        )
endif()



if(MDL_ENABLE_D3D12_EXAMPLES)

# name of the target and the resulting library
#
# This variant is used by the MDL SDK DXR example.
set(PROJECT_NAME imgui_dxr)

# Select MSVC runtime based on MDL_MSVC_DYNAMIC_RUNTIME_DXR_EXAMPLE.
if(MDL_MSVC_DYNAMIC_RUNTIME_DXR_EXAMPLE)
    set(_IMGUI_DXR_MSVC_RUNTIME "DYNAMIC_MSVC_RUNTIME")
else()
    set(_IMGUI_DXR_MSVC_RUNTIME "STATIC_MSVC_RUNTIME")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
    ${_IMGUI_DXR_MSVC_RUNTIME}
)

set_target_properties(${PROJECT_NAME} PROPERTIES PROJECT_LABEL "imgui_dxr")

# add include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )

# MacOS/clang specific warning and OpenGL/Vulkan dependencies not relevant for
# the MDL SDK DXR example.

endif()
