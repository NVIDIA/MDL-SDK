#*****************************************************************************
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

# Instructions to get the SDK can be found here: https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#installb
# Assuming the nuget package is download and extracted like described above using:
#   Invoke-WebRequest -Uri https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.711.3-preview -OutFile agility-1.711.3-preview.zip
#   Expand-Archive agility-1.711.3-preview.zip -DestinationPath d3d

function(FIND_AGILITY_SDK_EXT)

    set(AGILITY_SDK_DIR "" CACHE PATH "Directory that contains the `include` and `bin` dir. When extracting the nuget package, it's the `build/native` subfolder.")
    option(AGILITY_SDK_ENABLED "Enable the \"Microsoft Agility SDK\" for advanced D3D features." OFF)

    if(NOT AGILITY_SDK_ENABLED)
        message(STATUS "[INFO] The optional dependency \"Microsoft Agility SDK\" was disabled'")
        return()
    endif()

    # this path has to be specified by the user
    if(NOT EXISTS ${DXC_DIR})
        message(FATAL_ERROR "The dependency \"Microsoft Agility SDK\" could not be resolved. Please specify 'AGILITY_SDK_DIR' or disable 'AGILITY_SDK_ENABLED'")
        return()
    endif()

    if(WINDOWS)
        set(_AGILITY_SDK_INCLUDE "${AGILITY_SDK_DIR}/include")
        set(_AGILITY_SDK_SHARED
            "${AGILITY_SDK_DIR}/bin/x64/D3D12Core.dll"
            "${AGILITY_SDK_DIR}/bin/x64/d3d12SDKLayers.dll"
        )
    endif()
    foreach(_SHARED ${_AGILITY_SDK_INCLUDE} ${_AGILITY_SDK_SHARED} )
        if(NOT EXISTS ${_SHARED})
            message(FATAL_ERROR "The dependency \"Microsoft Agility SDK\" could not be resolved. The following library does not exist: \"${_SHARED}\". To continue without the Agility SDK, you can disable the option 'AGILITY_SDK_ENABLED'.")
        endif()
    endforeach()

    # store paths that are later used
    set(MDL_DEPENDENCY_AGILITY_SDK_INCLUDE ${_AGILITY_SDK_INCLUDE} CACHE INTERNAL "AGILITY_SDK includes")
    set(MDL_DEPENDENCY_AGILITY_SDK_SHARED ${_AGILITY_SDK_SHARED} CACHE INTERNAL "AGILITY_SDK shared libs")
    set(MDL_AGILITY_SDK_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_AGILITY_SDK_INCLUDE:   ${MDL_DEPENDENCY_AGILITY_SDK_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_AGILITY_SDK_SHARED:    ${MDL_DEPENDENCY_AGILITY_SDK_SHARED}")
    endif()

endfunction()
