#*****************************************************************************
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

# name of the target and the resulting example
set(PROJECT_NAME examples-mdl_sdk-distilling-target)

# collect sources
set(PROJECT_SOURCES
    "example_distilling_target.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk
    OUTPUT_NAME "distilling_target"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
    )

# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})

# -------------------------------------------------------------------------------------------------
# Create installation rules to copy the build directory
# -------------------------------------------------------------------------------------------------
add_target_install(
    TARGET ${PROJECT_NAME}
    DESTINATION "examples/mdl_sdk/distilling_target"
    )

# -------------------------------------------------------------------------------------------------
# Plugin target
# -------------------------------------------------------------------------------------------------

# name of the target and the resulting example
set(PLUGIN_PROJECT_NAME examples-mdl_sdk-distilling-target-plugin)

# generated Files
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES
    ${_GENERATED_DIR}/distilling_target_plugin_rules.h
    ${_GENERATED_DIR}/distilling_target_plugin_rules.cpp
    )

# scripts/files used to generate the sources above
set(_GENERATOR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/distilling_target_plugin_rules.mdltl
    )

# collect sources
set(PLUGIN_PROJECT_SOURCES
    "mdl_assert.h"
    "mdl_assert.cpp"
    "distilling_target_plugin.h"
    "distilling_target_plugin.cpp"
    ${_GENERATED_SOURCES}
    ${_GENERATOR_FILES}
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# add the generator files to the solution
set_source_files_properties(${_GENERATOR_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("generator" FILES ${_GENERATOR_FILES})

# create target from template
create_from_base_preset(
    TARGET ${PLUGIN_PROJECT_NAME}
    TYPE SHARED
    SOURCES ${PLUGIN_PROJECT_SOURCES}
    OUTPUT_NAME "distilling_target_plugin"
    EXPORTED_SYMBOLS mi_plugin_factory
    ADDITIONAL_INCLUDE_DIRS ${_GENERATED_DIR}
    )

# customize name
set_target_properties(${PLUGIN_PROJECT_NAME} PROPERTIES PREFIX "")

if(MACOSX)
    set_target_properties(${PLUGIN_PROJECT_NAME} PROPERTIES SUFFIX ".so") # corresponding to the binary release
endif()

# add mdl and other dependencies
target_add_dependencies(TARGET ${PLUGIN_PROJECT_NAME}
    DEPENDS
        ${LINKER_WHOLE_ARCHIVE}
        ${LINKER_START_GROUP}
        mdl::mdl_sdk
        ${LINKER_END_GROUP}
        ${LINKER_NO_WHOLE_ARCHIVE}
    )

# extend to project names
get_target_property(_PROJECT_LABEL ${PROJECT_NAME} PROJECT_LABEL)
set_target_properties(${PROJECT_NAME} PROPERTIES
    PROJECT_LABEL   "${_PROJECT_LABEL} (main)"      # project name
    )
set_target_properties(${PLUGIN_PROJECT_NAME} PROPERTIES
    PROJECT_LABEL   "${_PROJECT_LABEL} (plugin)"    # project name
    )

# -------------------------------------------------------------------------------------------------
# Code Generation Step
# -------------------------------------------------------------------------------------------------

# get and run mdltlc

set(_platform_dir ${MI_PLATFORM_NAME})
if(_platform_dir STREQUAL "macosx-x86-64" OR _platform_dir STREQUAL "macosx-aarch64")
    set(_platform_dir "macosx-uni")
endif()
set(_mdltlc_path ${MDL_BASE_FOLDER}/${_platform_dir}/bin/mdltlc${CMAKE_EXECUTABLE_SUFFIX})

message(STATUS "[DEBUG] Looking for mdltlc at: ${_mdltlc_path}")
if(EXISTS ${_mdltlc_path})
    message(STATUS "[DEBUG] Found mdltlc at: ${_mdltlc_path}")
    set(mdltlc_PATH ${_mdltlc_path} CACHE FILEPATH "")
else()
    set(_mdltlc_path ${MI_OBJ}/prod/bin/mdltlc/mdltlc)
    message(STATUS "[DEBUG] Looking for mdltlc at: ${_mdltlc_path}")
    if(EXISTS ${_mdltlc_path})
        message(STATUS "[DEBUG] Found mdltlc at: ${_mdltlc_path}")
        set(mdltlc_PATH ${_mdltlc_path} CACHE FILEPATH "")
    else()
        message(STATUS "[DEBUG] Looking for mdltlc via tool dependency")
        target_add_tool_dependency(TARGET ${PLUGIN_PROJECT_NAME} TOOL mdltlc)
    endif()
endif()

add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/distilling_target_plugin_rules.h
        ${_GENERATED_DIR}/distilling_target_plugin_rules.cpp
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Files with mdltlc ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${mdltlc_PATH} --generate --output-dir ${_GENERATED_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/distilling_target_plugin_rules.mdltl
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/distilling_target_plugin_rules.mdltl
    VERBATIM
    )


# -------------------------------------------------------------------------------------------------
# Create installation rules to copy the build directory
# -------------------------------------------------------------------------------------------------
add_target_install(
    TARGET ${PLUGIN_PROJECT_NAME}
    DESTINATION "examples/mdl_sdk/distilling_target"
    )
