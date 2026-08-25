#*****************************************************************************
# Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

# Checks whether debug and/or release variant of LIB_NAME exist in ${MATERIALX_DIR}/lib
# and adds them correspondingly to _MX_LIBS from the parent scope.
function(ADD_MATERIALX_LIBRARY_TO_MX_LIBS LIB_NAME)

    set(_LIB_DEBUG   ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(_LIB_RELEASE ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(_LIB_BOTH    ${MATERIALX_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${LIB_NAME}$<$<CONFIG:Debug>:d>${CMAKE_STATIC_LIBRARY_SUFFIX})

    if(EXISTS ${_LIB_DEBUG})
        if(EXISTS ${_LIB_RELEASE})
            set(_LIB ${_LIB_BOTH})
        else()
            set(_LIB ${_LIB_DEBUG})
        endif()
    else()
        if(EXISTS ${_LIB_RELEASE})
            set(_LIB ${_LIB_RELEASE})
        else()
            message(FATAL_ERROR "The dependency \"MaterialX\" could not be resolved. Expected library ${LIB_NAME} not found in ${MATERIALX_DIR}/lib.")
            return()
        endif()
    endif()

    list(APPEND _MX_LIBS ${_LIB})
    set(_MX_LIBS ${_MX_LIBS} PARENT_SCOPE)
endfunction()

# Try IMPLIB first for DLLs on DLL platforms. Otherwise, use LOCATION.
function(GET_IMPLIB_OR_LOCATION_PROPERTY LIB_NAME _CONFIG)
    get_target_property(_TMP ${LIB_NAME} IMPORTED_IMPLIB_${_CONFIG})
    if(NOT _TMP)
        get_target_property(_TMP ${LIB_NAME} IMPORTED_LOCATION_${_CONFIG})
    endif()
    set(${LIB_NAME}_${_CONFIG} ${_TMP} PARENT_SCOPE)
endfunction()

function(FIND_MATERIALX)

    set(MATERIALX_DIR "NOT-SPECIFIED" CACHE PATH "Path to a downloaded and extracted MaterialX pre-built package.")

    if(EXISTS ${MATERIALX_DIR})
        # collect information required for the build from MATERIALX_DIR
        #-----------------------------------------------------------------------------------------------

        # set include dir
        set(_MX_INCLUDE ${MATERIALX_DIR}/include)

        # set libs
        add_materialx_library_to_mx_libs(MaterialXGenMdl)
        add_materialx_library_to_mx_libs(MaterialXGenShader)
        add_materialx_library_to_mx_libs(MaterialXFormat)
        add_materialx_library_to_mx_libs(MaterialXCore)

        # base dir to reference the standard libraries
        set(_MX_BASE_DIR ${MATERIALX_DIR})

    else()
        # try if we can find it without the MATERIALX_DIR, e.g. because it's installed by vcpkg
        #-----------------------------------------------------------------------------------------------

        find_package(MaterialX)
        # mark as advanced to avoid confusion with MATERIALX_DIR
        mark_as_advanced(MaterialX_DIR FORCE)

        if(NOT ${MaterialX_FOUND})
            # Failure
            message(FATAL_ERROR "The dependency \"MaterialX\" could not be resolved. Please install using 'vcpkg', specify 'MATERIALX_DIR' or disable 'MDL_ENABLE_MATERIALX'")
            return()
        endif()

        # collect information required for the build from imported targets
        get_target_property(_MX_INCLUDE MaterialXCore INTERFACE_INCLUDE_DIRECTORIES)

        # need CMake >= 3.28.0 on non-DLL platforms
        get_implib_or_location_property(MaterialXCore DEBUG)
        get_implib_or_location_property(MaterialXCore RELEASE)
        get_implib_or_location_property(MaterialXFormat DEBUG)
        get_implib_or_location_property(MaterialXFormat RELEASE)
        get_implib_or_location_property(MaterialXGenShader DEBUG)
        get_implib_or_location_property(MaterialXGenShader RELEASE)
        get_implib_or_location_property(MaterialXGenMdl DEBUG)
        get_implib_or_location_property(MaterialXGenMdl RELEASE)

        set(_MX_LIBS
            $<IF:$<CONFIG:Debug>,${MaterialXGenMdl_DEBUG},${MaterialXGenMdl_RELEASE}>
            $<IF:$<CONFIG:Debug>,${MaterialXGenShader_DEBUG},${MaterialXGenShader_RELEASE}>
            $<IF:$<CONFIG:Debug>,${MaterialXFormat_DEBUG},${MaterialXFormat_RELEASE}>
            $<IF:$<CONFIG:Debug>,${MaterialXCore_DEBUG},${MaterialXCore_RELEASE}>
        )

        set(_MX_BASE_DIR ${MATERIALX_BASE_DIR})
    endif()

    # store paths that are later used in the add_materialx.cmake
    set(MDL_DEPENDENCY_MATERIALX_INCLUDE ${_MX_INCLUDE} CACHE INTERNAL "materialx headers" FORCE)
    set(MDL_DEPENDENCY_MATERIALX_LIBS ${_MX_LIBS} CACHE INTERNAL "materialx libs" FORCE)
    set(MDL_DEPENDENCY_MATERIALX_BASE_DIR ${_MX_BASE_DIR} CACHE INTERNAL "materialx base dir" FORCE)

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_INCLUDE:         ${MDL_DEPENDENCY_MATERIALX_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_LIBS:            ${MDL_DEPENDENCY_MATERIALX_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_MATERIALX_BASE_DIR:        ${MDL_DEPENDENCY_MATERIALX_BASE_DIR}")
    endif()

endfunction()


# Finds all mtlx libraries and MDL support modules in the MaterialX SDK and returns them as
# _MATERIALX_MTLX_FILENAMES and _MATERIALX_MDL_MODULE_FILENAMES in the parent scope.
function(GATHER_MATERIALX_CONTENT)

    set(_MATERIALX_MTLX_FILENAMES "")
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/bxdf/*.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/bxdf/genmdl/*.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/bxdf/lama/*.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/bxdf/translation/*.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/*/*_defs.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/*/*_ng.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    file(GLOB _PATHS RELATIVE ${MDL_DEPENDENCY_MATERIALX_BASE_DIR} "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/*/genmdl/*_impl.mtlx")
    list(APPEND _MATERIALX_MTLX_FILENAMES ${_PATHS})
    list(APPEND _MATERIALX_MTLX_FILENAMES "libraries/targets/genmdl.mtlx")
    set(_MATERIALX_MTLX_FILENAMES ${_MATERIALX_MTLX_FILENAMES} PARENT_SCOPE)

    file(GLOB _MATERIALX_MDL_MODULE_FILENAMES RELATIVE "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/mdl/materialx" "${MDL_DEPENDENCY_MATERIALX_BASE_DIR}/libraries/mdl/materialx/*.mdl")
    set(_MATERIALX_MDL_MODULE_FILENAMES ${_MATERIALX_MDL_MODULE_FILENAMES} PARENT_SCOPE)

endfunction()
