#*****************************************************************************
# Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_VULKAN_EXT)

    if(NOT MDL_ENABLE_VULKAN_EXAMPLES)
        message(WARNING "Examples that require Vulkan are disabled. Enable the option 'MDL_ENABLE_VULKAN_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    find_package(VulkanHeaders QUIET CONFIG)
    if(VulkanHeaders_FOUND)
        get_target_property(_VULKAN_INCLUDE Vulkan::Headers INTERFACE_INCLUDE_DIRECTORIES)
    else()
        message(STATUS "_VULKAN_INCLUDE: ${_VULKAN_INCLUDE}")
        message(FATAL_ERROR "The dependency \"VulkanHeaders\" could not be resolved. Please specify "
            "'CMAKE_TOOLCHAIN_FILE'. Alternatively, you can disable the option 'MDL_ENABLE_VULKAN_EXAMPLES'.")
    endif()

    find_package(volk QUIET CONFIG)
    if(volk_FOUND)
        get_target_property(_VOLK_INCLUDE volk::volk INTERFACE_INCLUDE_DIRECTORIES)
    else()
        message(STATUS "_VOLK_INCLUDE: ${_VOLK_INCLUDE}")
        message(FATAL_ERROR "The dependency \"volk\" could not be resolved. Please specify "
            "'CMAKE_TOOLCHAIN_FILE'. Alternatively, you can disable the option 'MDL_ENABLE_VULKAN_EXAMPLES'.")
    endif()

    find_path(_VULKAN_LAYERS_DIR
        NAMES VkLayer_khronos_validation.json
        PATHS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
        NO_DEFAULT_PATH
        )

    find_package(glslang QUIET CONFIG)
    if(glslang_FOUND)
        get_target_property(_GLSLANG_INCLUDE glslang::glslang INTERFACE_INCLUDE_DIRECTORIES)
        set(_GLSLANG_LIB glslang::glslang glslang::glslang-default-resource-limits glslang::SPIRV)
    else()
        message(STATUS "_GLSLANG_INCLUDE: ${_GLSLANG_INCLUDE}")
        message(STATUS "_GLSLANG_LIB: ${_GLSLANG_LIB}")
        message(FATAL_ERROR "The dependency \"glslang\" could not be resolved. Please specify "
            "'CMAKE_TOOLCHAIN_FILE'. Alternatively, you can disable the option 'MDL_ENABLE_VULKAN_EXAMPLES'.")
    endif()

    find_package(SPIRV-Tools-opt QUIET CONFIG)
    if(SPIRV-Tools-opt_FOUND)
        get_target_property(_SPIRV_TOOLS_OPT_INCLUDE SPIRV-Tools-opt INTERFACE_INCLUDE_DIRECTORIES)
        set(_SPIRV_TOOLS_OPT_LIB SPIRV-Tools-opt)
    else()
        message(STATUS "_SPIRV_TOOLS_OPT_INCLUDE: ${_SPIRV_TOOLS_OPT_INCLUDE}")
        message(STATUS "_SPIRV_TOOLS_OPT_LIB: ${_SPIRV_TOOLS_OPT_LIB}")
        message(FATAL_ERROR "The dependency \"SPIRV-Tools-opt\" could not be resolved. Please specify "
            "'CMAKE_TOOLCHAIN_FILE'. Alternatively, you can disable the option 'MDL_ENABLE_VULKAN_EXAMPLES'.")
    endif()
    
    # store paths that are later used in add_vulkan.cmake
    set(MDL_DEPENDENCY_VULKAN_INCLUDE ${_VULKAN_INCLUDE} CACHE INTERNAL "vulkan headers")
    set(MDL_DEPENDENCY_VOLK_INCLUDE ${_VOLK_INCLUDE} CACHE INTERNAL "volk headers")
    if(WIN32)
        set(MDL_DEPENDENCY_VULKAN_LAYERS_DIR ${_VULKAN_LAYERS_DIR} CACHE INTERNAL "vulkan layers dir")
    endif()
    set(MDL_DEPENDENCY_GLSLANG_INCLUDE ${_GLSLANG_INCLUDE} CACHE INTERNAL "glslang headers")
    set(MDL_DEPENDENCY_GLSLANG_LIBS ${_GLSLANG_LIB} CACHE INTERNAL "glslang libs")
    set(MDL_DEPENDENCY_SPIRV_TOOLS_INCLUDE ${_SPIRV_TOOLS_OPT_INCLUDE} CACHE INTERNAL "SPIRV-Tools-opt headers")
    set(MDL_DEPENDENCY_SPIRV_TOOLS_LIBS ${_SPIRV_TOOLS_OPT_LIB} CACHE INTERNAL "SPIRV-Tools-opt libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_VULKAN_INCLUDE:            ${MDL_DEPENDENCY_VULKAN_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_VOLK_INCLUDE:              ${MDL_DEPENDENCY_VOLK_INCLUDE}")
        if(WIN32)
           message(STATUS "[INFO] MDL_DEPENDENCY_VULKAN_LAYERS_DIR:         ${MDL_DEPENDENCY_VULKAN_LAYERS_DIR}")
        endif()
        message(STATUS "[INFO] MDL_DEPENDENCY_GLSLANG_INCLUDE:           ${MDL_DEPENDENCY_GLSLANG_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLSLANG_LIBS:              ${MDL_DEPENDENCY_GLSLANG_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_SPIRV_TOOLS_INCLUDE:       ${MDL_DEPENDENCY_SPIRV_TOOLS_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_SPIRV_TOOLS_LIBS:          ${MDL_DEPENDENCY_SPIRV_TOOLS_LIBS}")
    endif()

endfunction()
