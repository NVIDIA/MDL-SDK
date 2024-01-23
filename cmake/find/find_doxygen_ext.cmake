#*****************************************************************************
# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_DOXYGEN_EXT)

    if(NOT DOXYGEN_DIR)
        set(DOXYGEN_DIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the Doxygen 'bin' "
            "directory.")
    endif()
    if(NOT GRAPHVIZ_DIR)
        set(GRAPHVIZ_DIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the Graphviz include "
            "dir, libs, and binaries.")
    endif()

    find_program(_DOXYGEN_PATH doxygen HINTS ${DOXYGEN_DIR}/bin ${DOXYGEN_DIR})

    if(NOT _DOXYGEN_PATH)
        message(FATAL_ERROR "Doxygen was not found. Please specify 'DOXYGEN_DIR' or disable "
            "'MDL_BUILD_DOCUMENTATION'.")
        return()
    endif()

    # call --version
    execute_process(COMMAND "${_DOXYGEN_PATH}" "--version"
        OUTPUT_VARIABLE _DOXYGEN_VERSION_STRING
        ERROR_VARIABLE _DOXYGEN_VERSION_STRING
    )
    if(NOT _DOXYGEN_VERSION_STRING)
        message(STATUS "_DOXYGEN_PATH: ${_DOXYGEN_PATH}")
        message(FATAL_ERROR "Doxygen version could not be determined.")
        return()
    endif()

    # check version
    STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _DOXYGEN_VERSION_STRING ${_DOXYGEN_VERSION_STRING})
    if(NOT ${_DOXYGEN_VERSION_STRING} STREQUAL "1.9.4")
        message(WARNING "Doxygen version ${_DOXYGEN_VERSION_STRING} is different from recommended"
            " version 1.9.4.")
    endif()

    find_program(_DOT_PATH NAMES dot HINTS ${GRAPHVIZ_DIR}/bin)

    set(MDL_DEPENDENCY_DOXYGEN_PATH ${_DOXYGEN_PATH} CACHE INTERNAL "doxygen binary")
    set(MDL_DEPENDENCY_DOT_PATH ${_DOT_PATH} CACHE INTERNAL "dot binary")
    if(EXISTS ${_DOT_PATH})
        set(MDL_DEPENDENCY_DOT_FOUND ON CACHE INTERNAL "")
    else()
        # No warning/error message if not found.
        set(MDL_DEPENDENCY_DOT_FOUND OFF CACHE INTERNAL "")
    endif()

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_DOXYGEN_PATH:          ${MDL_DEPENDENCY_DOXYGEN_PATH}")
        message(STATUS "[INFO] MDL_DEPENDENCY_DOT_PATH:              ${MDL_DEPENDENCY_DOT_PATH}")
    endif()

endfunction()
