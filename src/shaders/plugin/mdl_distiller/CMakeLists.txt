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

# name of the target and the resulting library
set(PROJECT_NAME shaders-plugin-mdl_distiller)

# generated Files
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES
    ${_GENERATED_DIR}/dist_rules.h
    ${_GENERATED_DIR}/dist_rules.cpp
    ${_GENERATED_DIR}/dist_rules_transmissive_pbr.h
    ${_GENERATED_DIR}/dist_rules_transmissive_pbr.cpp
    ${_GENERATED_DIR}/dist_rules_ue.h
    ${_GENERATED_DIR}/dist_rules_ue.cpp
    )

# scripts/files used to generate the sources above
set(_GENERATOR_FILES_ORIGINAL
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_rules.mdltl
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_rules_transmissive_pbr.mdltl
    ${CMAKE_CURRENT_SOURCE_DIR}/dist_rules_ue.mdltl
    )

set(_GENERATOR_FILES ${_GENERATOR_FILES_ORIGINAL})


# collect sources
set(PROJECT_SOURCES
    "mdl_assert.h"
    "mdl_assert.cpp"
    "mdl_distiller.h"
    "mdl_distiller.cpp"
    ${_GENERATED_SOURCES}
    ${_GENERATOR_FILES_ORIGINAL}
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# add the generator files to the solution
set_source_files_properties(${_GENERATOR_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("generator" FILES ${_GENERATOR_FILES})

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    SOURCES ${PROJECT_SOURCES}
    EMBED_RC "mdl_distiller.rc"
    OUTPUT_NAME "mdl_distiller"
    EXPORTED_SYMBOLS mi_plugin_factory
    ADDITIONAL_INCLUDE_DIRS ${_GENERATED_DIR}
    )

# -------------------------------------------------------------------------------------------------
# Code Generation Step
# -------------------------------------------------------------------------------------------------

# get and run mdltlc
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL mdltlc)

add_custom_command(
    OUTPUT
        ${_GENERATED_SOURCES}
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Files with mdltlc ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}


    COMMAND ${CMAKE_COMMAND} -E echo ${mdltlc_PATH} --generate --all-errors --output-dir ${_GENERATED_DIR} ${_GENERATOR_FILES}
    COMMAND ${mdltlc_PATH} --generate --all-errors --output-dir ${_GENERATED_DIR} ${_GENERATOR_FILES}
    DEPENDS
        ${_GENERATOR_FILES_ORIGINAL}
    VERBATIM
    )


# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

if(MACOSX)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so") # corresponding to the binary release
endif()

# add mdl and other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        boost
        ${LINKER_WHOLE_ARCHIVE}
        ${LINKER_START_GROUP}
        mdl::base-system-version
        ${LINKER_END_GROUP}
        ${LINKER_NO_WHOLE_ARCHIVE}
    )

# target alias for the custom name
add_library(mdl_distiller ALIAS ${PROJECT_NAME})
add_library(mdl::mdl_distiller ALIAS ${PROJECT_NAME})

# -------------------------------------------------------------------------------------------------
# Create installation rules to copy the build directory
# -------------------------------------------------------------------------------------------------
add_target_install(
    TARGET ${PROJECT_NAME}
    )
