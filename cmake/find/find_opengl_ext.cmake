#*****************************************************************************
# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_OPENGL_EXT)

    if(NOT MDL_ENABLE_OPENGL_EXAMPLES)
        message(WARNING "Examples that require OpenGL are disabled. Enable the option 'MDL_ENABLE_OPENGL_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    #-----------------------------------------------------------------------------------------------

    # find the required packages
    set(OpenGL_GL_PREFERENCE "GLVND")
    find_package(OpenGL QUIET)
    if(OPENGL_FOUND)
        set(_GL_SHARED OpenGL::GL OpenGL::GLU)
    else()
        message(STATUS "OPENGL_FOUND: ${OPENGL_FOUND}")
        message(STATUS "OPENGL_XMESA_FOUND: ${OPENGL_XMESA_FOUND}")
        message(STATUS "OPENGL_GLU_FOUND: ${OPENGL_GLU_FOUND}")
        message(STATUS "OPENGL_INCLUDE_DIR: ${OPENGL_INCLUDE_DIR}")
        message(STATUS "OPENGL_LIBRARIES: ${OPENGL_LIBRARIES}")
        if(NOT WINDOWS AND EXISTS ${OPENGL_INCLUDE_DIR})
            find_file(_LIB_OPENGL 
                NAMES 
                    "libOpenGL${CMAKE_SHARED_LIBRARY_SUFFIX}"
                    "libGL${CMAKE_SHARED_LIBRARY_SUFFIX}"
                HINTS
                    /usr/lib64/
                    /usr/lib/x86_64-linux-gnu
                    /usr/lib
                )

            find_file(_LIB_GLU 
                NAMES
                    "libGLU${CMAKE_SHARED_LIBRARY_SUFFIX}"
                HINTS
                    /usr/lib64/
                    /usr/lib/x86_64-linux-gnu
                    /usr/lib
                )
        endif()

        if(EXISTS ${_LIB_OPENGL} AND EXISTS ${_LIB_GLU})
            set(_GL_SHARED ${_LIB_OPENGL} ${_LIB_GLU})
        else()
            # no further fall-back here
            message(FATAL_ERROR "The dependency \"gl\" could not be resolved. They usually come with the graphics driver. On Linux, try to install 'libglu-dev' and 'libx11-dev' packages. Alternatively, you can disable to option 'MDL_ENABLE_OPENGL_EXAMPLES'.")
        endif()

    endif()

    # store paths that are later used in the add_opengl.cmake
    set(MDL_DEPENDENCY_GL_INCLUDE ${OPENGL_INCLUDE_DIR} CACHE INTERNAL "gl headers")
    set(MDL_DEPENDENCY_GL_SHARED ${_GL_SHARED} CACHE INTERNAL "gl libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GL_INCLUDE:            ${MDL_DEPENDENCY_GL_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GL_SHARED:             ${MDL_DEPENDENCY_GL_SHARED}")
    endif()

    #-----------------------------------------------------------------------------------------------

    find_package(GLEW QUIET)

    if(GLEW_FOUND)
        get_target_property(_GLEW_INCLUDE GLEW::GLEW INTERFACE_INCLUDE_DIRECTORIES)
        set(_GLEW_LIB "GLEW::GLEW")
    else()
        message(STATUS "_GLEW_INCLUDE: ${_GLEW_INCLUDE}")
        message(STATUS "_GLEW_LIB: ${_GLEW_LIB}")
        message(FATAL_ERROR "The dependency \"glew\" could not be resolved. Please specify 'CMAKE_TOOLCHAIN_FILE'. Alternatively, you can disable the option 'MDL_ENABLE_OPENGL_EXAMPLES'.")
    endif()

    # store paths that are later used in the add_opengl.cmake
    set(MDL_DEPENDENCY_GLEW_INCLUDE ${_GLEW_INCLUDE} CACHE INTERNAL "glew headers")
    set(MDL_DEPENDENCY_GLEW_LIBS ${_GLEW_LIB} CACHE INTERNAL "glew libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GLEW_INCLUDE:          ${MDL_DEPENDENCY_GLEW_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLEW_LIBS:             ${MDL_DEPENDENCY_GLEW_LIBS}")
    endif()

endfunction()
