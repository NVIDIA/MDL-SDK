#*****************************************************************************
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_D3D12_EXT)

    if(NOT WINDOWS)
        return()
    endif()

    set(WINDOWS_SDK_DIR "C:/Program Files (x86)/Windows Kits/10/Include/10.0.17763.0" 
        CACHE PATH "Directory that contains the header, e.g.: \"C:/Program Files (x86)/Windows Kits/10/Include/10.0.17763.0\"")

    if(NOT MDL_ENABLE_D3D12_EXAMPLES)
        message(WARNING "Examples that require D3D12 are disabled. Enable the option 'MDL_ENABLE_D3D12_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    # find headers
    find_file(_D3D12_HEADER "D3d12.h"
        HINTS
            ${WINDOWS_SDK_DIR}/um
        )
    get_filename_component(_D3D12_INCLUDE_DIR "${_D3D12_HEADER}" DIRECTORY)

    if(NOT EXISTS ${_D3D12_INCLUDE_DIR})
        message(FATAL_ERROR "The dependency \"d3d12\" could not be resolved. Please specify WINDOWS_SDK_DIR. Alternatively, you can disable the option 'MDL_ENABLE_D3D12_EXAMPLES'.")
    endif()  

    # libs
    string(REPLACE "Windows Kits/10/Include" "Windows Kits/10/lib" _D3D12_LIBRARY_DIR ${_D3D12_INCLUDE_DIR})
    set(_D3D12_LIBRARY_DIR ${_D3D12_LIBRARY_DIR}/x64)
    set(_D3D12_LIBS 
        ${_D3D12_LIBRARY_DIR}/d3d12.lib
        ${_D3D12_LIBRARY_DIR}/dxgi.lib
        ${_D3D12_LIBRARY_DIR}/dxguid.lib
        ${_D3D12_LIBRARY_DIR}/dxcompiler.lib
        ${_D3D12_LIBRARY_DIR}/D3DCompiler.lib
    )

    foreach(_LIB ${_D3D12_LIBS})
        if(NOT EXISTS ${_LIB})
            message(STATUS "D3D12_INCLUDE_DIR: ${_D3D12_INCLUDE_DIR}")
            message(STATUS "D3D12_LIBRARY_DIR: ${_D3D12_LIBRARY_DIR}")
            message(FATAL_ERROR "The dependency \"d3d12\" could not be resolved. The following library does not exist: \"${_LIB}\". To continue without D3D12, you can disable the option 'MDL_ENABLE_D3D12_EXAMPLES'.")
        endif()   
    endforeach()

    # runtime libraries
    string(REPLACE "Windows Kits/10/lib" "Windows Kits/10/bin" _D3D12_BIN_DIR ${_D3D12_LIBRARY_DIR})
    string(REPLACE "/um/x64" "/x64" _D3D12_BIN_DIR ${_D3D12_BIN_DIR})
	find_file(_D3D12_DLL "D3D12.dll" PATHS ${_D3D12_BIN_DIR} $ENV{WINDIR}/System32)
	find_file(_D3D12_SDK_LAYER_DLL "d3d12SDKLayers.dll" PATHS ${_D3D12_BIN_DIR} $ENV{WINDIR}/System32)
	find_file(_D3D12_COMPILER_DLL "d3dcompiler_47.dll" PATHS ${_D3D12_BIN_DIR} $ENV{WINDIR}/System32)
	find_file(_DX_COMPILER_DLL "dxcompiler.dll" PATHS ${_D3D12_BIN_DIR} $ENV{WINDIR}/System32)
	find_file(_DX_IL_DLL "dxil.dll" PATHS ${_D3D12_BIN_DIR} $ENV{WINDIR}/System32)
    set(_D3D12_SHARED
        ${_D3D12_DLL}
        ${_D3D12_SDK_LAYER_DLL}
        ${_D3D12_COMPILER_DLL}
        ${_DX_COMPILER_DLL}
        ${_DX_IL_DLL}
    )

    foreach(_SHARED ${_D3D12_SHARED})
        if(NOT EXISTS ${_SHARED})
            message(STATUS "D3D12_DLL: ${_D3D12_DLL}")
            message(STATUS "D3D12_SDK_LAYER_DLL: ${_D3D12_SDK_LAYER_DLL}")
            message(STATUS "D3D12_COMPILER_DLL: ${_D3D12_COMPILER_DLL}")
            message(STATUS "DX_COMPILER_DLL: ${_DX_COMPILER_DLL}")
            message(STATUS "DX_IL_DLL: ${_DX_IL_DLL}")
            message(FATAL_ERROR "The dependency \"d3d12\" could not be resolved. The following library does not exist: \"${_SHARED}\". To continue without D3D12, you can disable the option 'MDL_ENABLE_D3D12_EXAMPLES'.")
        endif()   
    endforeach()

    # store path that are later used in the add_opengl.cmake
    set(MDL_DEPENDENCY_D3D12_INCLUDE ${_D3D12_INCLUDE_DIR} CACHE INTERNAL "d3d12 headers")
    set(MDL_DEPENDENCY_D3D12_LIBS ${_D3D12_LIBS} CACHE INTERNAL "d3d12 libs")
    set(MDL_DEPENDENCY_D3D12_SHARED ${_D3D12_SHARED} CACHE INTERNAL "d3d12 shared libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_D3D12_INCLUDE:       ${MDL_DEPENDENCY_D3D12_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_D3D12_LIBS:          ${MDL_DEPENDENCY_D3D12_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_D3D12_SHARED:        ${MDL_DEPENDENCY_D3D12_SHARED}")
    endif()

endfunction()
