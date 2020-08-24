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

function(FIND_ARNOLDSDK_EXT)

    # Dependencies to resolve by the user
    set(ARNOLD_SDK_DIR "NOT-SPECIFIED" CACHE PATH "Main directory of the extracted Arnold SDK package which contains the 'include', 'lib', and 'bin' folders.")

    # Skip if he user disabled the projects that require this dependency
    if(NOT MDL_BUILD_ARNOLD_PLUGIN)
        message(WARNING "Build of the MDL Arnold Plugin is disabled. Enable the option 'MDL_BUILD_ARNOLD_PLUGIN' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    # collect information required for the build
    #-----------------------------------------------------------------------------------------------

    # get include dir
    set(_ASDK_INCLUDE ${ARNOLD_SDK_DIR}/include)

    # save the binary folder, this will be required to run kick for instance
    # applications that renderer with arnold, e.g. Maya or Max will have their own ai.dll so we don't copy the runtime libraries
    set(_ASDK_BIN ${ARNOLD_SDK_DIR}/bin)

    # get libs
    if(WINDOWS)
        set(_ASDK_LIBS
            ${ARNOLD_SDK_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}ai${CMAKE_STATIC_LIBRARY_SUFFIX}
        )
    elseif(MACOSX)
        set(_ASDK_LIBS
            ${ARNOLD_SDK_DIR}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}ai${CMAKE_SHARED_LIBRARY_SUFFIX}
        )
    endif()

    # check if the dependencies are available, if not disable
    #-----------------------------------------------------------------------------------------------
    foreach(_TO_CHECK ${_ASDK_INCLUDE} ${_ASDK_LIBS} ${_ASDK_BIN})
        if(NOT EXISTS ${_TO_CHECK})
            message(WARNING "The dependency \"Arnold SDK\" could not be resolved because of missing \"${_TO_CHECK}\" and the build is disabled now. Please specify 'ARNOLD_SDK_DIR' and enable 'MDL_BUILD_ARNOLD_PLUGIN'")
            set(MDL_BUILD_ARNOLD_PLUGIN OFF CACHE BOOL "Enable the build of the MDL Arnold plugin." FORCE)
            return()
        endif()
    endforeach()

    # store path that are later used in the add_arnoldsdk.cmake
    set(MDL_DEPENDENCY_ARNOLDSDK_INCLUDE ${_ASDK_INCLUDE} CACHE INTERNAL "Arnold SDK headers" FORCE)
    set(MDL_DEPENDENCY_ARNOLDSDK_LIBS ${_ASDK_LIBS} CACHE INTERNAL "Arnold SDK libs" FORCE)
    set(MDL_DEPENDENCY_ARNOLDSDK_BIN ${_ASDK_BIN} CACHE INTERNAL "Arnold SDK binary directory" FORCE)
  
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_ARNOLDSDK_INCLUDE:   ${MDL_DEPENDENCY_ARNOLDSDK_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_ARNOLDSDK_LIBS:      ${MDL_DEPENDENCY_ARNOLDSDK_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_ARNOLDSDK_BIN:       ${MDL_DEPENDENCY_ARNOLDSDK_BIN}")
    endif()

endfunction()
