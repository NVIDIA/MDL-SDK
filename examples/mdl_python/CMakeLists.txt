#*****************************************************************************
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

# create Visual Studio code launch.json
get_property(BINDING_MODULE_PATH TARGET "mdl::mdl_python" PROPERTY BINARY_DIR)

# Depending on the product and it's distribution we need set paths and environment variables slightly different
if("MDL_BINARY_RELEASE" IN_LIST MDL_ADDITIONAL_COMPILER_DEFINES)
    if(MDL_INTERNAL_BUILD)
        set(_VS_CODE_FILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/mdl_python_examples.examples_internal.code-workspace)
        set(_SETUP_ENV_FILE_IN_SH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/setup_env.examples_internal.sh)
        set(_SETUP_ENV_FILE_IN_BAT ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/setup_env.examples_internal.bat)
    else()
        set(_VS_CODE_FILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/mdl_python_examples.binary_release.code-workspace)
        set(_SETUP_ENV_FILE_IN_SH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/setup_env.binary_release.sh)
        set(_SETUP_ENV_FILE_IN_BAT ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/setup_env.binary_release.bat)
    endif()
elseif ("MDL_SOURCE_RELEASE" IN_LIST MDL_ADDITIONAL_COMPILER_DEFINES)
    get_property(MDL_SDK_PATH TARGET "mdl::mdl_sdk" PROPERTY BINARY_DIR)
    if(TARGET "mdl::nv_openimageio")
        get_property(OPENIMAGEIO_PATH TARGET "mdl::nv_openimageio" PROPERTY BINARY_DIR)
    endif()
    if(TARGET "mdl::dds")
        get_property(DDS_PATH TARGET "mdl::dds" PROPERTY BINARY_DIR)
    endif()
    if(TARGET "mdl::mdl_distiller")
        get_property(MDLDISTILLER_PATH TARGET "mdl::mdl_distiller" PROPERTY BINARY_DIR)
    endif()
    set(ENV_MDL_SDK_PATH ${MDL_SDK_PATH})
    set(ENV_OPENIMAGEIO_PATH ${OPENIMAGEIO_PATH})
    set(ENV_DDS_PATH ${DDS_PATH})
    set(ENV_MDLDISTILLER_PATH ${MDLDISTILLER_PATH})
    if(WINDOWS)
        set(ENV_MDL_SDK_PATH_BAT ${MDL_SDK_PATH})
        set(ENV_OPENIMAGEIO_PATH_BAT ${OPENIMAGEIO_PATH})
        set(ENV_DDS_PATH_BAT ${DDS_PATH})
        set(ENV_MDLDISTILLER_PATH_BAT ${MDLDISTILLER_PATH})
        # for running the shell scripts on windows (e.g. using git bash or mingw)
        # we need to adjust the strings that go into the PATH to use a unix-like format
        if(MDL_SDK_PATH)
            string(SUBSTRING ${MDL_SDK_PATH} 1 1 _CHAR_AT_1)
            if(_CHAR_AT_1 STREQUAL ":")
                string(SUBSTRING ${MDL_SDK_PATH} 0 1 _CHAR_AT_0)
                string(SUBSTRING ${MDL_SDK_PATH} 2 -1 ENV_MDL_SDK_PATH)
                set(ENV_MDL_SDK_PATH "/${_CHAR_AT_0}${ENV_MDL_SDK_PATH}")
            endif()
        endif()
        if(OPENIMAGEIO_PATH)
            string(SUBSTRING ${OPENIMAGEIO_PATH} 1 1 _CHAR_AT_1)
            if(_CHAR_AT_1 STREQUAL ":")
                string(SUBSTRING ${OPENIMAGEIO_PATH} 0 1 _CHAR_AT_0)
                string(SUBSTRING ${OPENIMAGEIO_PATH} 2 -1 ENV_OPENIMAGEIO_PATH)
                set(ENV_OPENIMAGEIO_PATH "/${_CHAR_AT_0}${ENV_OPENIMAGEIO_PATH}")
            endif()
        endif()
        if(DDS_PATH)
            string(SUBSTRING ${DDS_PATH} 1 1 _CHAR_AT_1)
            if(_CHAR_AT_1 STREQUAL ":")
                string(SUBSTRING ${DDS_PATH} 0 1 _CHAR_AT_0)
                string(SUBSTRING ${DDS_PATH} 2 -1 ENV_DDS_PATH)
                set(ENV_DDS_PATH "/${_CHAR_AT_0}${ENV_DDS_PATH}")
            endif()
        endif()
        if(MDLDISTILLER_PATH)
            string(SUBSTRING ${MDLDISTILLER_PATH} 1 1 _CHAR_AT_1)
            if(_CHAR_AT_1 STREQUAL ":")
                string(SUBSTRING ${MDLDISTILLER_PATH} 0 1 _CHAR_AT_0)
                string(SUBSTRING ${MDLDISTILLER_PATH} 2 -1 ENV_MDLDISTILLER_PATH)
                set(ENV_MDLDISTILLER_PATH "/${_CHAR_AT_0}${ENV_MDLDISTILLER_PATH}")
            endif()
        endif()
    endif()
    set(_VS_CODE_FILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/mdl_python_examples.open_source.code-workspace)
    set(_SETUP_ENV_FILE_IN_SH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/setup_env.open_source.sh)
    set(_SETUP_ENV_FILE_IN_BAT ${CMAKE_CURRENT_SOURCE_DIR}/cmake_templates/setup_env.open_source.bat)
endif()

# create the IDE or shell support files
get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_GENERATOR_IS_MULTI_CONFIG)
    foreach(_CONFIG ${CMAKE_CONFIGURATION_TYPES})
        set(_MI_OBJ ${MI_OBJ_${_CONFIG}})
        make_directory(${CMAKE_CURRENT_BINARY_DIR}/env/${_CONFIG})
        configure_file(${_SETUP_ENV_FILE_IN_SH} ${CMAKE_CURRENT_BINARY_DIR}/env/${_CONFIG}/setup_env.sh @ONLY)
        if(WINDOWS)
            configure_file(${_SETUP_ENV_FILE_IN_BAT} ${CMAKE_CURRENT_BINARY_DIR}/env/${_CONFIG}/setup_env.bat @ONLY)
        endif()
    endforeach()
else()
    set(_CONFIG ${CMAKE_BUILD_TYPE})
    set(_MI_OBJ ${MI_OBJ_${_CONFIG}})
    make_directory(${CMAKE_CURRENT_BINARY_DIR}/env/${_CONFIG})
    configure_file(${_SETUP_ENV_FILE_IN_SH} ${CMAKE_CURRENT_BINARY_DIR}/env/${_CONFIG}/setup_env.sh @ONLY)
    if(WINDOWS)
        configure_file(${_SETUP_ENV_FILE_IN_BAT} ${CMAKE_CURRENT_BINARY_DIR}/env/${_CONFIG}/setup_env.bat @ONLY)
    endif()
endif()
configure_file(${_VS_CODE_FILE_IN} ${CMAKE_BINARY_DIR}/mdl_python_examples.code-workspace @ONLY)
