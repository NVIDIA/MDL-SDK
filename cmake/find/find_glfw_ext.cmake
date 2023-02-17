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

    set(GLFW_DIR "" CACHE PATH "Directory that contains the glfw include dir, libs and binaries")

    if(NOT MDL_ENABLE_OPENGL_EXAMPLES AND NOT MDL_ENABLE_VULKAN_EXAMPLES)
        message(WARNING "Examples that require GLFW are disabled. Enable the option 'MDL_ENABLE_OPENGL_EXAMPLES' or/and 'MDL_ENABLE_VULKAN_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    find_package(glfw3 QUIET)
    if(glfw3_FOUND)
        set(_GLFW_INCLUDE ${GLFW_INCLUDE_DIR})
        set(_GLFW_SHARED glfw)
    else()
        # try to find GLFW manually
        set(_GLFW_INCLUDE "NOTFOUND")
        set(_GLFW_LIB "NOTFOUND")
        set(_GLFW_SHARED "NOTFOUND")
        mark_as_advanced(glfw3_DIR)

        if(EXISTS ${GLFW_DIR})
            set(_GLFW_INCLUDE ${GLFW_DIR}/include)

            # assuming that the windows (x64) binaries from http://www.glfw.org/download.html are used
            if(WINDOWS)
                set(_GLFW_LIB "${GLFW_DIR}/lib-vc2015/glfw3dll.lib")
                set(_GLFW_SHARED "${GLFW_DIR}/lib-vc2015/glfw3.dll")

            else()
                # link dynamic
                set(_GLFW_LIB "")  # not used
                find_file(_GLFW_SHARED "${CMAKE_SHARED_LIBRARY_PREFIX}glfw${CMAKE_SHARED_LIBRARY_SUFFIX}"
                    HINTS 
                        ${GLFW_DIR}
                        ${GLFW_DIR}/lib64
                        ${GLFW_DIR}/lib
                        /usr/lib64/
                        /usr/lib/x86_64-linux-gnu
                        /usr/lib
                    )
            endif()
        endif()

        # error if dependencies can not be resolved
        if(NOT EXISTS ${_GLFW_INCLUDE} OR (WINDOWS AND NOT EXISTS ${_GLFW_LIB}) OR NOT EXISTS ${_GLFW_SHARED})
            message(STATUS "GLFW_DIR: ${GLFW_DIR}")
            message(STATUS "_GLFW_INCLUDE: ${_GLFW_INCLUDE}")
            message(STATUS "_GLFW_LIB: ${_GLFW_LIB}")
            message(STATUS "_GLFW_SHARED: ${_GLFW_SHARED}")
            message(FATAL_ERROR "The dependency \"glfw\" could not be resolved. Please specify GLFW_DIR. Alternatively, you can disable the option 'MDL_ENABLE_OPENGL_EXAMPLES'.")
        endif()
    endif()

    # store path that are later used in the add_opengl.cmake
    set(MDL_DEPENDENCY_GLFW_INCLUDE ${_GLFW_INCLUDE} CACHE INTERNAL "glfw headers")
    set(MDL_DEPENDENCY_GLFW_LIBS ${_GLFW_LIB} CACHE INTERNAL "glfw libs")
    set(MDL_DEPENDENCY_GLFW_SHARED ${_GLFW_SHARED} CACHE INTERNAL "glfw shared libs")
    set(MDL_GLFW_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_INCLUDE:        ${MDL_DEPENDENCY_GLFW_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_LIBS:           ${MDL_DEPENDENCY_GLFW_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_SHARED:         ${MDL_DEPENDENCY_GLFW_SHARED}")
    endif()

endfunction()
