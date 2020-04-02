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

function(FIND_OPENGL_EXT)

    set(GLEW_DIR "" CACHE PATH "Directory that contains the glew include dir, libs and binaries")
    set(GLFW_DIR "" CACHE PATH "Directory that contains the glfw include dir, libs and binaries")

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

    # store path that are later used in the add_opengl.cmake
    set(MDL_DEPENDENCY_GL_INCLUDE ${OPENGL_INCLUDE_DIR} CACHE INTERNAL "gl headers")
    set(MDL_DEPENDENCY_GL_SHARED ${_GL_SHARED} CACHE INTERNAL "gl libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GL_INCLUDE:          ${MDL_DEPENDENCY_GL_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GL_SHARED:           ${MDL_DEPENDENCY_GL_SHARED}")
    endif()
    #-----------------------------------------------------------------------------------------------

    # extend find GLEW
    set(_SAVED_GLEW_DIR ${GLEW_DIR}) # since cmake 3.15 FindGLEW overrides our GLEW_DIR variable
    find_package(GLEW QUIET)
    if(GLEW_FOUND)
        set(_GLEW_INCLUDE ${GLEW_INCLUDE_DIR})
        set(_GLEW_SHARED GLEW::GLEW)
    else()
        # restore saved dir
        set(GLEW_DIR ${_SAVED_GLEW_DIR} CACHE PATH "Directory that contains the glew include dir, libs and binaries" FORCE)

        # try to find GLEW manually
        set(_GLEW_INCLUDE "NOTFOUND")
        set(_GLEW_LIB "NOTFOUND")
        set(_GLEW_SHARED "NOTFOUND")

        if(EXISTS ${GLEW_DIR})
            set(_GLEW_INCLUDE ${GLEW_DIR}/include)

            # now, look for binaries and libs
            if(WINDOWS)
                # assuming that the windows binaries from http://glew.sourceforge.net/ are used
                set(_GLEW_LIB "${GLEW_DIR}/lib/Release/x64/glew32.lib")
                set(_GLEW_SHARED "${GLEW_DIR}/bin/Release/x64/glew32.dll")

            else()
                # link dynamic
                set(_GLEW_LIB "")  # not used
                find_file(_GLEW_SHARED "${CMAKE_SHARED_LIBRARY_PREFIX}GLEW${CMAKE_SHARED_LIBRARY_SUFFIX}"
                    HINTS 
                        ${GLEW_DIR}
                        ${GLEW_DIR}/lib64
                        ${GLEW_DIR}/lib
                        /usr/lib64/
                        /usr/lib/x86_64-linux-gnu
                        /usr/lib
                    )
            endif()
        endif()

        # error if dependencies can not be resolved
        if(NOT EXISTS ${_GLEW_INCLUDE} OR (WINDOWS AND NOT EXISTS ${_GLEW_LIB}) OR NOT EXISTS ${_GLEW_SHARED})
            message(STATUS "GLEW_DIR: ${GLEW_DIR}")
            message(STATUS "_GLEW_INCLUDE: ${_GLEW_INCLUDE}")
            message(STATUS "_GLEW_LIB: ${_GLEW_LIB}")
            message(STATUS "_GLEW_SHARED: ${_GLEW_SHARED}")
            message(FATAL_ERROR "The dependency \"glew\" could not be resolved. Please specify GLEW_DIR. Alternatively, you can disable the option 'MDL_ENABLE_OPENGL_EXAMPLES'.")
        endif()    
    endif()

    # store path that are later used in the add_opengl.cmake
    set(MDL_DEPENDENCY_GLEW_INCLUDE ${_GLEW_INCLUDE} CACHE INTERNAL "glew headers")
    set(MDL_DEPENDENCY_GLEW_LIBS ${_GLEW_LIB} CACHE INTERNAL "glew libs")
    set(MDL_DEPENDENCY_GLEW_SHARED ${_GLEW_SHARED} CACHE INTERNAL "glew shared libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GLEW_INCLUDE:        ${MDL_DEPENDENCY_GLEW_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLEW_LIBS:           ${MDL_DEPENDENCY_GLEW_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLEW_SHARED:         ${MDL_DEPENDENCY_GLEW_SHARED}")
    endif()
    #-----------------------------------------------------------------------------------------------

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

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_INCLUDE:        ${MDL_DEPENDENCY_GLFW_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_LIBS:           ${MDL_DEPENDENCY_GLFW_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_GLFW_SHARED:         ${MDL_DEPENDENCY_GLFW_SHARED}")
    endif()

endfunction()
