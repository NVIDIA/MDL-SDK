#*****************************************************************************
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_PANTORA_AXF_EXT)

    set(PANTORA_AXF_DIR "" CACHE PATH "Directory that contains the PANTORA_AXF include dir, libs and binaries")

    # this path has to be specified by the user
    if(NOT EXISTS ${PANTORA_AXF_DIR})
        message(FATAL_ERROR "The dependency \"Pantora AXF\" could not be resolved. Please specify 'PANTORA_AXF_DIR' or disable 'MDL_ENABLE_AXF_EXAMPLES'")
        return()
    endif()

    # assuming that the binaries are downloaded from ____
    set(_PANTORA_AXF_INCLUDE "${PANTORA_AXF_DIR}/../include")
    if(WINDOWS)
        file(GLOB _PANTORA_AXF_SHARED "${PANTORA_AXF_DIR}/bin/AxFDecoding.[0-9].[0-9].[0-9].dll")
        set(_PANTORA_AXF_LIBS "${PANTORA_AXF_DIR}/lib/AxFDecoding.lib")
    elseif(LINUX)
        file(GLOB _PANTORA_AXF_SHARED "${PANTORA_AXF_DIR}/lib/libAxFDecoding.so.[0-9].[0-9].[0-9]")
        set(_PANTORA_AXF_LIBS "${_PANTORA_AXF_SHARED}")
    elseif(MACOSX)
        file(GLOB _PANTORA_AXF_SHARED "${PANTORA_AXF_DIR}/lib/libAxFDecoding.[0-9].[0-9].[0-9].dylib")
        set(_PANTORA_AXF_LIBS "${_PANTORA_AXF_SHARED}")
    endif()
    foreach(_SHARED ${_PANTORA_AXF_SHARED})
        if(NOT EXISTS ${_SHARED})
            message(FATAL_ERROR "The dependency \"Pantora AXF\" could not be resolved. The following library does not exist: \"${_SHARED}\". To continue without AFX, you can disable the option 'MDL_ENABLE_AXF_EXAMPLES'.")
        endif()
    endforeach()
    foreach(_LIB ${_PANTORA_AXF_LIBS})
        if(NOT EXISTS ${_LIB})
            message(FATAL_ERROR "The dependency \"Pantora AXF\" could not be resolved. The following library does not exist: \"${_LIB}\". To continue without AFX, you can disable the option 'MDL_ENABLE_AXF_EXAMPLES'.")
        endif()
    endforeach()

    # store paths that are later used in add_pantora_axf.cmake
    set(MDL_DEPENDENCY_PANTORA_AXF_INCLUDE ${_PANTORA_AXF_INCLUDE} CACHE INTERNAL "PANTORA_AXF includes")
    set(MDL_DEPENDENCY_PANTORA_AXF_LIBS ${_PANTORA_AXF_LIBS} CACHE INTERNAL "PANTORA_AXF static libs")
    set(MDL_DEPENDENCY_PANTORA_AXF_SHARED ${_PANTORA_AXF_SHARED} CACHE INTERNAL "PANTORA_AXF shared libs")
    set(MDL_PANTORA_AXF_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_PANTORA_AXF_SHARED:    ${MDL_DEPENDENCY_PANTORA_AXF_SHARED}")
    endif()

endfunction()
