#*****************************************************************************
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME prod-lib-mdl_python)

# Files to generate
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES
    ${_GENERATED_DIR}/mdl_python_swig.cpp
    ${_GENERATED_DIR}/swig.d
    )

# scripts/files used to generate the sources above
set(_GENERATOR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/instantiate_templates_inline.py
    ${CMAKE_CURRENT_SOURCE_DIR}/mdl_python_swig.i
    )

# -------------------------------------------------------------------------------------------------
# Main Target
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_SOURCES
    "mdl_python.cpp"
    "mdl_python.h"
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
    DYNAMIC_MSVC_RUNTIME
    SOURCES
        ${PROJECT_SOURCES}
        ${_GENERATED_SOURCES}
        ${_GENERATOR_FILES}
    )

# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "_pymdlsdk")
if(WINDOWS)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")
endif()

# target alias for the custom name
add_library(mdl_python ALIAS ${PROJECT_NAME})
add_library(mdl::mdl_python ALIAS ${PROJECT_NAME})

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
        python_dev
    )

# disable some warnings in generated code
if(UNIX AND NOT APPLE)
    target_compile_options(${PROJECT_NAME}
        PUBLIC
            -Wno-delete-non-virtual-dtor
    )
endif()


# -------------------------------------------------------------------------------------------------
# Code Generation Step
# -------------------------------------------------------------------------------------------------

# get tools
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL swig)

# run template processing for the interface classes
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/$(configuration)/processed_headers.d
    COMMAND ${CMAKE_COMMAND} -E echo "SWIG template instantiation ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}/processed_headers
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}/$(configuration)
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/instantiate_templates_inline.py
        ${CMAKE_CURRENT_SOURCE_DIR}/mdl_python_swig.i
        ""
        ${MDL_INCLUDE_FOLDER}
        ${_GENERATED_DIR}/processed_headers
        ${_GENERATED_DIR}/$(configuration)/processed_headers.d
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/instantiate_templates_inline.py
        ${CMAKE_CURRENT_SOURCE_DIR}/mdl_python_swig.i
    VERBATIM
    )

# run swig to generate the bindings
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/mdl_python_swig.cpp
    COMMAND ${CMAKE_COMMAND} -E echo "SWIG binding code generation ..."
    COMMAND ${swig_PATH}
        -c++ -python -doxygen
        -DSWIGWORDSIZE64
        -I-
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -I${_GENERATED_DIR}/processed_headers
        -I${MDL_INCLUDE_FOLDER}
        -MMD -MP -MF ${_GENERATED_DIR}/mdl_python_swig.cpp.tmp.d
        -cppext cpp
        -o ${_GENERATED_DIR}/mdl_python_swig.cpp
        -outdir $<TARGET_FILE_DIR:${PROJECT_NAME}>
        -module pymdlsdk
        ${CMAKE_CURRENT_SOURCE_DIR}/mdl_python_swig.i
    DEPENDS
        ${_GENERATED_DIR}/$(configuration)/processed_headers.d
        ${CMAKE_CURRENT_SOURCE_DIR}/mdl_python_swig.i
        ${PROJECT_SOURCES}
    VERBATIM
    )

# make sure the .pyd and the .py file are up to date when switching build configurations
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/swig.d
    COMMAND ${CMAKE_COMMAND} -E echo "SWIG touch dependency file ..."
    COMMAND ${CMAKE_COMMAND} -E touch ${_GENERATED_DIR}/swig.d
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/mdl_python_swig.i
        ${_GENERATED_DIR}/mdl_python_swig.cpp
        ${_GENERATED_DIR}/$(configuration)/processed_headers.d
    VERBATIM
    )

# add tests if available
add_tests(POST)
