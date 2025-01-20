#*****************************************************************************
# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_LIB_NAME prod-bin-mdltlclib)
set(PROJECT_BIN_NAME prod-bin-mdltlc)

# generated Files
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES
    ${_GENERATED_DIR}/Parser.h
    ${_GENERATED_DIR}/Parser.cpp
    ${_GENERATED_DIR}/Scanner.h
    ${_GENERATED_DIR}/Scanner.cpp
    ${_GENERATED_DIR}/mdltl_enum_names.i
    ${_GENERATED_DIR}/mdltlc_module.cpp
    ${_GENERATED_DIR}/mdltlc_module.h
    )

# scripts/files used to generate the sources above
set(_GENERATOR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mdltlc.atg
    ${CMAKE_CURRENT_SOURCE_DIR}/Scanner.frame
    ${CMAKE_CURRENT_SOURCE_DIR}/Parser.frame
    ${CMAKE_CURRENT_SOURCE_DIR}/Copyright.frame
    )

# collect sources
set(PROJECT_LIB_HEADERS
    "mdltlc_analysis.h"
    "mdltlc_codegen.h"
    "mdltlc_compilation_unit.h"
    "mdltlc_compiler.h"
    "mdltlc_compiler_options.h"
    "mdltlc_env.h"
    "mdltlc_exprs.h"
    "mdltlc_expr_walker.h"
    "mdltlc.h"
    "mdltlc_locations.h"
    "mdltlc_message.h"
    "mdltlc_pprint.h"
    "mdltlc_rules.h"
    "mdltlc_symbols.h"
    "mdltlc_types.h"
    "mdltlc_values.h"
    )

set(PROJECT_LIB_SOURCES
    "mdltlc_analysis.cpp"
    "mdltlc_codegen.cpp"
    "mdltlc_compilation_unit.cpp"
    "mdltlc_compiler.cpp"
    "mdltlc_compiler_options.cpp"
    "mdltlc.cpp"
    "mdltlc_env.cpp"
    "mdltlc_tc.cpp"
    "mdltlc_exprs.cpp"
    "mdltlc_expr_walker.cpp"
    "mdltlc_message.cpp"
    "mdltlc_pprint.cpp"
    "mdltlc_rules.cpp"
    "mdltlc_symbols.cpp"
    "mdltlc_types.cpp"
    "mdltlc_values.cpp"
    ${_GENERATED_SOURCES}
    ${_GENERATOR_FILES}
    ${PROJECT_LIB_HEADERS}
    )

set(PROJECT_BIN_HEADERS
    "getopt.h"
    )

set(PROJECT_BIN_SOURCES
    "getopt.cpp"
    "main.cpp"
    ${PROJECT_BIN_HEADERS}
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# add the generator files to the solution
set_source_files_properties(${_GENERATOR_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("generator" FILES ${_GENERATOR_FILES})

# create mdltlc library target from template
create_from_base_preset(
    TARGET ${PROJECT_LIB_NAME}
    TYPE STATIC
    OUTPUT_NAME "libmdltlc"
    SOURCES ${PROJECT_LIB_SOURCES}
    ADDITIONAL_INCLUDE_DIRS ${_GENERATED_DIR}
)

# add mdl and other dependencies
target_add_dependencies(TARGET ${PROJECT_LIB_NAME}
    DEPENDS
        ${LINKER_START_GROUP}
        mdl::mdl-compiler-compilercore
        mdl::mdl-codegenerators-generator_dag
        mdl::mdl-codegenerators-generator_code
        mdl::mdl-runtime
        mdl::mdl-no_jit-generator_stub
        ${LINKER_DEPENDENCIES_BASE_WITHOUT_BASE_DATA}
        ${LINKER_END_GROUP}
    )

# create mdltlc binary target from template
create_from_base_preset(
    TARGET ${PROJECT_BIN_NAME}
    TYPE EXECUTABLE
    OUTPUT_NAME "mdltlc"
    SOURCES ${PROJECT_BIN_SOURCES}
)

# add  dependencies to mdltlc binary
target_add_dependencies(TARGET ${PROJECT_BIN_NAME}
    DEPENDS
        mdl::base-system-version
        mdl::prod-bin-mdltlclib
    )

# extend to project names
get_target_property(_PROJECT_LABEL ${PROJECT_BIN_NAME} PROJECT_LABEL)
set_target_properties(${PROJECT_BIN_NAME} PROPERTIES
    PROJECT_LABEL   "${_PROJECT_LABEL} (main)"      # project name
    )
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES
    PROJECT_LABEL   "${_PROJECT_LABEL} (lib)"       # project name
    )

# -------------------------------------------------------------------------------------------------
# Code Generation Step
# -------------------------------------------------------------------------------------------------

# get and run coco
target_add_tool_dependency(TARGET ${PROJECT_LIB_NAME} TOOL coco)
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/Parser.h
        ${_GENERATED_DIR}/Parser.cpp
        ${_GENERATED_DIR}/Scanner.h
        ${_GENERATED_DIR}/Scanner.cpp
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Files with COCO ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${coco_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/mdltlc.atg -no_misplaced_resolver -frames . -trace FX -o ${_GENERATED_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/mdltlc.atg
        ${CMAKE_CURRENT_SOURCE_DIR}/Scanner.frame
        ${CMAKE_CURRENT_SOURCE_DIR}/Parser.frame
        ${CMAKE_CURRENT_SOURCE_DIR}/Copyright.frame
    VERBATIM
    )


set(_STANDARD_MDL
    ${MDL_SRC_FOLDER}/shaders/mdl/nvidia/distilling_support.mdl
    )

target_add_tool_dependency(TARGET ${PROJECT_LIB_NAME} TOOL python)
add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/mdltl_enum_names.i
        ${_GENERATED_DIR}/mdltlc_module.cpp
        ${_GENERATED_DIR}/mdltlc_module.h
    COMMAND ${CMAKE_COMMAND} -E echo "Generate Standard Module Header ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${MDL_SRC_FOLDER}/mdl/compiler/compilercore/generate_stdmodule.py -E P4ssW0rT -e -d ${_GENERATED_DIR}/mdltlc_module.cpp ${_STANDARD_MDL}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_mdltl_enum_names.py ${MDL_INCLUDE_FOLDER} ${_GENERATED_DIR}/mdltl_enum_names.i
    DEPENDS
        ${MDL_SRC_FOLDER}/mdl/compiler/compilercore/generate_stdmodule.py
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_mdltl_enum_names.py
        ${_STANDARD_MDL}
        ${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_expressions.h
        ${MDL_INCLUDE_FOLDER}/mi/mdl/mdl_definitions.h
    VERBATIM
    )
# create installation rules
add_target_install(
    TARGET ${PROJECT_BIN_NAME}
    )


# add unit tests
add_unit_tests(POST)
