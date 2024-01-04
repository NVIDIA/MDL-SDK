#*****************************************************************************
# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_GLFW_EXT)

    if(NOT MDL_ENABLE_OPENGL_EXAMPLES AND NOT MDL_ENABLE_VULKAN_EXAMPLES)
        message(WARNING "Examples that require GLFW are disabled. Enable the options 'MDL_ENABLE_OPENGL_EXAMPLES' and/or 'MDL_ENABLE_VULKAN_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    find_package(glfw3 QUIET)

    if(glfw3_FOUND)
        get_target_property(_GLFW_INCLUDE glfw INTERFACE_INCLUDE_DIRECTORIES)
        set(_GLFW_LIB "glfw")
    else()
        message(STATUS "_GLFW_INCLUDE: ${_GLFW_INCLUDE}")
        message(STATUS "_GLFW_LIB: ${_GLFW_LIB}")
        message(FATAL_ERROR "The dependency \"glfw\" could not be resolved. Please specify 'CMAKE_TOOLCHAIN_FILE'. Alternatively, you can disable the options 'MDL_ENABLE_OPENGL_EXAMPLES' and 'MDL_ENABLE_VULKAN_EXAMPLES'.")
    endif()

    # store paths that are later used in the add_glfw.cmake
    set(MDL_DEPENDENCY_GLFW_INCLUDE ${_GLFW_INCLUDE} CACHE INTERNAL "glfw headers")
    set(MDL_DEPENDENCY_GLFW_LIBS ${_GLFW_LIB} CACHE INTERNAL "glfw libs")
    set(MDL_GLFW_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_INCLUDE:          ${MDL_DEPENDENCY_GLFW_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_LIBS:             ${MDL_DEPENDENCY_GLFW_LIBS}")
    endif()

endfunction()
