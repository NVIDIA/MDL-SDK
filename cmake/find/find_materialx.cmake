#*****************************************************************************
# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_MATERIALX)

    set(MATERIALX_DIR "NOT-SPECIFIED" CACHE PATH "Path to a downloaded and extracted MaterialX pre-built package.")

    # check if the dependencies are available, if not disable
    #-----------------------------------------------------------------------------------------------
    if(NOT EXISTS ${MATERIALX_DIR})
        message(FATAL_ERROR "The dependency \"MaterialX\" could not be resolved. Please specify 'MATERIALX_DIR' or disable 'MDL_ENABLE_MATERIALX'")
        return()
    endif()

    # collect information required for the build
    #-----------------------------------------------------------------------------------------------

    # set include dir
    set(_MX_INCLUDE ${MATERIALX_DIR}/include)
    
    # set libs
    set(_MX_LIBS
        ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXCore$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXFormat$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXGenShader$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXGenMdl$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    
    # store paths that are later used in the add_materialx.cmake
    set(MDL_DEPENDENCY_MATERIALX_INCLUDE ${_MX_INCLUDE} CACHE INTERNAL "materialx headers" FORCE)
    set(MDL_DEPENDENCY_MATERIALX_LIBS ${_MX_LIBS} CACHE INTERNAL "materialx libs" FORCE)
  
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_INCLUDE:     ${MDL_DEPENDENCY_MATERIALX_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_LIBS:        ${MDL_DEPENDENCY_MATERIALX_LIBS}")
    endif()

endfunction()
