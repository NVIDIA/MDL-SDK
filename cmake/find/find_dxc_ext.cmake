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

function(FIND_DXC_EXT)

    set(DXC_DIR "" CACHE PATH "Directory that contains the DXC include dir, libs and binaries")

    # this path has to be specified by the user
    if(NOT EXISTS ${DXC_DIR})
        message(FATAL_ERROR "The dependency \"DirectX Shader Compiler\" could not be resolved. Please specify 'DXC_DIR' or disable 'MDL_ENABLE_D3D12_EXAMPLES'")
        return()
    endif()

    # assuming that the windows (x64) binaries from https://github.com/microsoft/DirectXShaderCompiler/releases are used
    if(WINDOWS)
        set(_DXC_INCLUDE "${DXC_DIR}/inc")
        set(_DXC_LIBS "${DXC_DIR}/lib/x64/dxcompiler.lib")
        set(_DXC_SHARED "${DXC_DIR}/bin/x64/dxcompiler.dll"
                        "${DXC_DIR}/bin/x64/dxil.dll"
        )
    endif()
    foreach(_SHARED ${_DXC_INCLUDE} ${_DXC_SHARED}   ${MDL_DEPENDENCY_DXC_LIBS})
        if(NOT EXISTS ${_SHARED})
            message(FATAL_ERROR "The dependency \"DirectX Shader Compiler\" could not be resolved. The following library does not exist: \"${_SHARED}\". To continue without D3D12, you can disable the option 'MDL_ENABLE_D3D12_EXAMPLES'.")
        endif()
    endforeach()

    # store paths that are later used
    set(MDL_DEPENDENCY_DXC_INCLUDE ${_DXC_INCLUDE} CACHE INTERNAL "DXC includes")
    set(MDL_DEPENDENCY_DXC_SHARED ${_DXC_SHARED} CACHE INTERNAL "DXC shared libs")
    set(MDL_DEPENDENCY_DXC_LIBS ${_DXC_LIBS} CACHE INTERNAL "DXC libs")
    set(MDL_DXC_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_DXC_INCLUDE:           ${MDL_DEPENDENCY_DXC_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_DXC_LIBS:              ${MDL_DEPENDENCY_DXC_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_DXC_SHARED:            ${MDL_DEPENDENCY_DXC_SHARED}")
    endif()

endfunction()
