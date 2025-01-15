#*****************************************************************************
# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME mdl-integration-mdlnr)

# generated Files
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES
    ${_GENERATED_DIR}/mdlnr_module.cpp
    ${_GENERATED_DIR}/mdlnr_module.h
    )

# collect sources
set(PROJECT_HEADERS
    "i_mdlnr.h"
    "mdlnr.h"
    "mdlnr_search_path.h"
    )

set(PROJECT_SOURCES
    "mdlnr.cpp"
    "mdlnr_search_path.cpp"
    ${_GENERATED_SOURCES}
    ${PROJECT_HEADERS}
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    ADDITIONAL_INCLUDE_DIRS ${_GENERATED_DIR}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        boost
    )

set(_STANDARD_MDL
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../shaders/mdl/nvidia/distilling_support.mdl
    )

target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/mdlnr_module.cpp
        ${_GENERATED_DIR}/mdlnr_module.h
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Standard Module Header ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../../../mdl/compiler/compilercore/generate_stdmodule.py -E P4ssW0rT -e -d ${_GENERATED_DIR}/mdlnr_module.cpp ${_STANDARD_MDL}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../mdl/compiler/compilercore/generate_stdmodule.py
        ${_STANDARD_MDL}
    VERBATIM
    )
