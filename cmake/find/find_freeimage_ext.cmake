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

function(FIND_FREEIMAGE_EXT)

    set(FREEIMAGE_DIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the freeimage library and the corresponding headers.")
    #-----------------------------------------------------------------------------------------------

    # try to find FreeImage manually
    set(_FREEIMAGE_INCLUDE "NOTFOUND")
    set(_FREEIMAGE_LIB "NOTFOUND")
    set(_FREEIMAGE_SHARED "NOTFOUND")

    find_file(_FREEIMAGE_HEADER_FILE "FreeImage.h" 
        HINTS 
            ${FREEIMAGE_DIR}
            ${FREEIMAGE_DIR}/include
            ${FREEIMAGE_DIR}/Dist/x64
            /usr/include
        )
    mark_as_advanced(_FREEIMAGE_HEADER_FILE)
    mark_as_advanced(_FREEIMAGE_SHARED)
    mark_as_advanced(_FREEIMAGE_LIB)

    if(EXISTS ${_FREEIMAGE_HEADER_FILE})
        get_filename_component(_FREEIMAGE_INCLUDE ${_FREEIMAGE_HEADER_FILE} PATH)

        if(WINDOWS)
            # assuming that the windows (x64) binaries from http://freeimage.sourceforge.net/download.html are used
            find_file(_FREEIMAGE_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}freeimage${CMAKE_STATIC_LIBRARY_SUFFIX}" 
                HINTS 
                    ${FREEIMAGE_DIR}
                    ${FREEIMAGE_DIR}/Dist/x64
                )

            find_file(_FREEIMAGE_SHARED "${CMAKE_SHARED_LIBRARY_PREFIX}freeimage${CMAKE_SHARED_LIBRARY_SUFFIX}" 
                HINTS 
                    ${FREEIMAGE_DIR}
                    ${FREEIMAGE_DIR}/Dist/x64
                )

        elseif(LINUX OR MACOSX)
            # assuming the 'freeimage-dev' package is installed
            # or freeimage was build manually and follows a common folder structure
            set(_FREEIMAGE_LIB "") # not used
            find_file(_FREEIMAGE_SHARED
                NAMES
                    "${CMAKE_SHARED_LIBRARY_PREFIX}freeimage${CMAKE_SHARED_LIBRARY_SUFFIX}"
                    "libfreeimage.so"
                HINTS 
                    ${FREEIMAGE_DIR}
                    ${FREEIMAGE_DIR}/lib64
                    ${FREEIMAGE_DIR}/lib
                    /usr/lib64
                    /usr/lib/x86_64-linux-gnu
                    /usr/lib
                    /usr/local/lib
                )

            if(NOT EXISTS ${_FREEIMAGE_SHARED})
                set(_OS_MESSAGE " install the 'libfreeimage-dev' package or")
            endif()
        endif()
    endif()

    # error if dependencies can not be resolved
    if(NOT EXISTS ${_FREEIMAGE_INCLUDE} OR (WINDOWS AND NOT EXISTS ${_FREEIMAGE_LIB}) OR NOT EXISTS ${_FREEIMAGE_SHARED})
        message(STATUS "FREEIMAGE_DIR: ${FREEIMAGE_DIR}")
        message(STATUS "_FREEIMAGE_HEADER_FILE: ${_FREEIMAGE_HEADER_FILE}")
        message(STATUS "_FREEIMAGE_INCLUDE: ${_FREEIMAGE_INCLUDE}")
        message(STATUS "_FREEIMAGE_LIB: ${_FREEIMAGE_LIB}")
        message(STATUS "_FREEIMAGE_SHARED: ${_FREEIMAGE_SHARED}")
        message(FATAL_ERROR "The dependency \"freeimage\" could not be resolved. Please${_OS_MESSAGE} specify 'FREEIMAGE_DIR'.")
    endif()

    # store path that are later used in the add_freeimage.cmake
    set(MDL_DEPENDENCY_FREEIMAGE_INCLUDE ${_FREEIMAGE_INCLUDE} CACHE INTERNAL "freeimage headers")
    set(MDL_DEPENDENCY_FREEIMAGE_LIBS ${_FREEIMAGE_LIB} CACHE INTERNAL "freeimage libs")
    set(MDL_DEPENDENCY_FREEIMAGE_SHARED ${_FREEIMAGE_SHARED} CACHE INTERNAL "freeimage shared libs")
    set(MDL_FREEIMAGE_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_FREEIMAGE_INCLUDE:   ${MDL_DEPENDENCY_FREEIMAGE_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_FREEIMAGE_LIBS:      ${MDL_DEPENDENCY_FREEIMAGE_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_FREEIMAGE_SHARED:    ${MDL_DEPENDENCY_FREEIMAGE_SHARED}")
    endif()

endfunction()
