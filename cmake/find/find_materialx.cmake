#*****************************************************************************
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

    set(MATERIALX_REPOSITORY "NOT-SPECIFIED" CACHE PATH "Checked out MaterialX git repository.")
    set(MATERIALX_BUILD "NOT-SPECIFIED" CACHE PATH "CMake build folder with MaterialX libraries compiled from the git repository.")
    
    if(NOT MDL_ENABLE_MATERIALX)
        message(WARNING "Support of MaterialX in Examples is disabled. Enable the option 'MDL_ENABLE_MATERIALX' and resolve the required dependencies to re-enable them.")
        return()
    endif()
    
    # check if the dependencies are available, if not disable
    #-----------------------------------------------------------------------------------------------
    foreach(_TO_CHECK ${MATERIALX_REPOSITORY}/source ${MATERIALX_BUILD}/source)
        if(NOT EXISTS ${_TO_CHECK})
            message(WARNING "The dependency \"MaterialX\" could not be resolved and the support is disabled now. Please specify 'MATERIALX_REPOSITORY' as well as 'MATERIALX_BUILD' and enable 'MDL_ENABLE_MATERIALX'")
            set(MDL_ENABLE_MATERIALX OFF CACHE BOOL "Enable MaterialX in examples that support it." FORCE)
            return()
        endif()
    endforeach()

    # collect information required for the build
    #-----------------------------------------------------------------------------------------------

    # set include dir
    set(_MX_INCLUDE ${MATERIALX_REPOSITORY}/source)
    
    # set libs
    set(_MX_LIBS
        ${MATERIALX_BUILD}/source/MaterialXCore/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXCore${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${MATERIALX_BUILD}/source/MaterialXFormat/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXFormat${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${MATERIALX_BUILD}/source/MaterialXGenShader/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXGenShader${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${MATERIALX_BUILD}/source/MaterialXGenMdl/$<CONFIG>/${CMAKE_STATIC_LIBRARY_PREFIX}MaterialXGenMdl${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    
    # store path that are later used in the add_materialx.cmake
    set(MDL_DEPENDENCY_MATERIALX_INCLUDE ${_MX_INCLUDE} CACHE INTERNAL "materialx headers" FORCE)
    set(MDL_DEPENDENCY_MATERIALX_LIBS ${_MX_LIBS} CACHE INTERNAL "materialx libs" FORCE)
  
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_INCLUDE:   ${MDL_DEPENDENCY_MATERIALX_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_LIBS:      ${MDL_DEPENDENCY_MATERIALX_LIBS}")
    endif()

endfunction()
