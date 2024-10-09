#*****************************************************************************
# Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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

set(PROJECT_NAME prod-lib-mdl_arnold)

# collect sources
set(SOURCES_HEADERS
    "mdl_arnold.h"
    "mdl_arnold_bsdf.h"
    "mdl_arnold_interface.h"
    "mdl_arnold_utils.h"
    )
set(PROJECT_SOURCES
    "mdl_arnold.cpp"
    "mdl_arnold_edf.cpp"
    "mdl_arnold_bsdf.cpp"
    "mdl_arnold_interface.cpp"
    "mdl_arnold_opacity.cpp"
    )
set(PROJECT_ADDITIONAL_FILES
    "readme.md"
    "aiMDLTemplate.py"
    "example_scene/cornell_single.ass"
    "example_scene/mdl/simple.mdl"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    SOURCES
        ${SOURCES_HEADERS}
        ${PROJECT_SOURCES}
        ${PROJECT_ADDITIONAL_FILES}
        copied_files.d
    EXPORTED_SYMBOLS NodeLoader
    SKIP_UNDEFINED_SYMBOL_CHECK
    )
set_source_files_properties(${PROJECT_ADDITIONAL_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "mdl_arnold")
if(MACOSX)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so") # corresponding to the binary release
endif()

# target alias for the custom name
add_library(mdl_arnold ALIAS ${PROJECT_NAME})
add_library(mdl::mdl_arnold ALIAS ${PROJECT_NAME})

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
    )

# add Arnold SDK but without binary dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS arnoldsdk NO_RUNTIME
    )

# create a special output folder structure to match the format we want to use for a release
if(WINDOWS)
    set(_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/lib)
    set(_SHARED_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
elseif(MACOSX)
    set(_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
    set(_SHARED_SUFFIX ".so")
else() # LINUX
    set(_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
    set(_SHARED_SUFFIX ".so")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${_OUTPUT_DIRECTORY}
    PDB_OUTPUT_DIRECTORY_DEBUG ${_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${_OUTPUT_DIRECTORY}
    PDB_OUTPUT_DIRECTORY_RELEASE ${_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${_OUTPUT_DIRECTORY}
    PDB_OUTPUT_DIRECTORY_MINSIZEREL ${_OUTPUT_DIRECTORY}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_OUTPUT_DIRECTORY}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_OUTPUT_DIRECTORY}
    PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO ${_OUTPUT_DIRECTORY}
)

# create a list of files to copy
list(APPEND _COMMANDS
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_OUTPUT_DIRECTORY}/../ae
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/aiMDLTemplate.py ${_OUTPUT_DIRECTORY}/../ae
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/readme.md ${_OUTPUT_DIRECTORY}/..
)
list(APPEND _DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/aiMDLTemplate.py
    ${CMAKE_CURRENT_SOURCE_DIR}/readme.md
)

# add renamed SDK libraries and plugins
# in case this example is used with the open source release of the MDL SDK
if(TARGET mdl::mdl_sdk)
    list(APPEND _COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mdl::mdl_sdk> ${_OUTPUT_DIRECTORY}/libmdl_sdk_ai${_SHARED_SUFFIX}
    )
    list(APPEND _DEPENDS
        mdl::mdl_sdk
    )
    if(TARGET mdl::nv_openimageio)
        list(APPEND _COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mdl::nv_openimageio> ${_OUTPUT_DIRECTORY}/nv_openimageio_ai${_SHARED_SUFFIX}
        )
        list(APPEND _DEPENDS
            mdl::nv_openimageio
        )
    endif()
    if(TARGET mdl::dds)
        list(APPEND _COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mdl::dds> ${_OUTPUT_DIRECTORY}/dds_ai${_SHARED_SUFFIX}
        )
        list(APPEND _DEPENDS
            mdl::dds
        )
    endif()
# in case this example is used with the binary release of the MDL SDK
else()
    # TODO remove this from the binary release
    if(MDL_INTERNAL_BUILD)
        # copy and rename libraries from the internal build output
        set(_SDK_BINARY_DIR ${MI_OBJ_${CMAKE_BUILD_TYPE}})
        set(_OPENIMAGEIO_BINARY_DIR ${_SDK_BINARY_DIR})
        set(_DDS_BINARY_DIR ${_SDK_BINARY_DIR})

        # prefer the src structure over the tape structure
        if(EXISTS ${_SDK_BINARY_DIR}/shaders/plugin/openimageio/nv_openimageio${_SHARED_SUFFIX} )
            set(_OPENIMAGEIO_BINARY_DIR ${_SDK_BINARY_DIR}/shaders/plugin/openimageio)
        endif()

        if(EXISTS ${_SDK_BINARY_DIR}/shaders/plugin/dds/dds${_SHARED_SUFFIX}  )
            set(_DDS_BINARY_DIR ${_SDK_BINARY_DIR}/shaders/plugin/dds)
        endif()

        if(EXISTS ${_SDK_BINARY_DIR}/prod/lib/mdl_sdk/libmdl_sdk${_SHARED_SUFFIX} )
            set(_SDK_BINARY_DIR ${_SDK_BINARY_DIR}/prod/lib/mdl_sdk)
        endif()
    else()
        # copy and rename libraries shipped in the release bundle
        set(_OPENIMAGEIO_BINARY_DIR ${MDL_BASE_FOLDER}/${MI_PLATFORM_NAME}/lib/)
        set(_DDS_BINARY_DIR ${MDL_BASE_FOLDER}/${MI_PLATFORM_NAME}/lib/)
        set(_SDK_BINARY_DIR ${MDL_BASE_FOLDER}/${MI_PLATFORM_NAME}/lib/)
    endif()

    list(APPEND _COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_SDK_BINARY_DIR}/libmdl_sdk${_SHARED_SUFFIX} ${_OUTPUT_DIRECTORY}/libmdl_sdk_ai${_SHARED_SUFFIX}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_OPENIMAGEIO_BINARY_DIR}/nv_openimageio${_SHARED_SUFFIX} ${_OUTPUT_DIRECTORY}/nv_openimageio_ai${_SHARED_SUFFIX}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_DDS_BINARY_DIR}/dds${_SHARED_SUFFIX} ${_OUTPUT_DIRECTORY}/dds_ai${_SHARED_SUFFIX}
    )
    list(APPEND _DEPENDS
        ${_SDK_BINARY_DIR}/libmdl_sdk${_SHARED_SUFFIX}
        ${_OPENIMAGEIO_BINARY_DIR}/nv_openimageio${_SHARED_SUFFIX}
        ${_DDS_BINARY_DIR}/dds${_SHARED_SUFFIX}
    )
endif()

# copy the files collected, if not stated otherwise
if(NOT DEFINED MDL_BUILD_ARNOLD_COPY_ADDITIONAL_FILES OR MDL_BUILD_ARNOLD_COPY_ADDITIONAL_FILES)
    add_custom_command(
        OUTPUT copied_files.d
        DEPENDS ${_DEPENDS}
        COMMAND ${CMAKE_COMMAND} -E echo "Copy SDK libraries and additional files to the binary folder..."
        ${_COMMANDS}
        COMMAND ${CMAKE_COMMAND} -E touch copied_files.d
    )
else()
    add_custom_command(
        OUTPUT copied_files.d
        COMMAND ${CMAKE_COMMAND} -E echo "\"SDK libraries and additional files are NOT copied to the binary folder (disabled by MDL_BUILD_ARNOLD_COPY_ADDITIONAL_FILES) ...\""
        COMMAND ${CMAKE_COMMAND} -E touch copied_files.d
    )
endif()

# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
set(_VS_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/lib)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        VS_DEBUGGER_COMMAND             "${MDL_DEPENDENCY_ARNOLDSDK_BIN}/kick${CMAKE_BINARY_LIBRARY_SUFFIX}"
        VS_DEBUGGER_COMMAND_ARGUMENTS   "-t 0 -v 4 -nostdin -i ${CMAKE_CURRENT_SOURCE_DIR}/example_scene/cornell_single.ass -ipr q"
    )
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_ ${MDL_DEPENDENCY_ARNOLDSDK_BIN}/kick${CMAKE_BINARY_LIBRARY_SUFFIX})
target_add_vs_debugger_env_var(TARGET ${PROJECT_NAME}
    VARS
        "ARNOLD_PLUGIN_PATH=${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/lib"
        "MDL_PATHS=${CMAKE_CURRENT_SOURCE_DIR}/example_scene/mdl"
    )

target_create_vs_user_settings(TARGET ${PROJECT_NAME})
