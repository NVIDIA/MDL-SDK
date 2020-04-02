#*****************************************************************************
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME mdl-compiler-compiler_hlsl)

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

# generated Files
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES 
    ${_GENERATED_DIR}/hlsl_intrinsics.h
    ${_GENERATED_DIR}/hlsl_intrinsics.i
    )

# scripts/files used to generate the sources above
set(_GENERATOR_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_hlsl_intrinsics.py
    ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrin_main.txt
    )

# collect sources
set(PROJECT_HEADERS 
    "compiler_hlsl_analysis.h"
    "compiler_hlsl_assert.h"
    "compiler_hlsl_builtin_types.h"
    "compiler_hlsl_cc_conf.h"
    "compiler_hlsl_compilation_unit.h"
    "compiler_hlsl_compiler.h"
    "compiler_hlsl_declarations.h"
    "compiler_hlsl_definitions.h"
    "compiler_hlsl_exprs.h"
    "compiler_hlsl_half.h"
    "compiler_hlsl_locations.h"
    "compiler_hlsl_messages.h"
    "compiler_hlsl_optimizer.h"
    "compiler_hlsl_predefined_symbols.h"
    "compiler_hlsl_printers.h"
    "compiler_hlsl_stmts.h"
    "compiler_hlsl_symbols.h"
    "compiler_hlsl_tools.h"
    "compiler_hlsl_types.h"
    "compiler_hlsl_values.h"
    )

set(PROJECT_SOURCES 
    "compiler_hlsl_analysis.cpp"
    "compiler_hlsl_compilation_unit.cpp"
    "compiler_hlsl_compiler.cpp"
    "compiler_hlsl_declarations.cpp"
    "compiler_hlsl_definitions.cpp"
    "compiler_hlsl_errors.cpp"
    "compiler_hlsl_exprs.cpp"
    "compiler_hlsl_half.cpp"
    "compiler_hlsl_intrinsics.cpp"
    "compiler_hlsl_messages.cpp"
    "compiler_hlsl_optimizer.cpp"
    "compiler_hlsl_printers.cpp"
    "compiler_hlsl_stmts.cpp"
    "compiler_hlsl_symbols.cpp"
    "compiler_hlsl_types.cpp"
    "compiler_hlsl_values.cpp"
    "compiler_hlsl_visitor.cpp"
    ${_GENERATED_SOURCES}
    ${_GENERATOR_FILES}
    ${PROJECT_HEADERS}
    )

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# add the generator files to the solution
set_source_files_properties(${_GENERATOR_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("generator" FILES ${_GENERATOR_FILES})

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

# -------------------------------------------------------------------------------------------------
# Code Generation Step 
# -------------------------------------------------------------------------------------------------

target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL python)
add_custom_command(
    OUTPUT ${_GENERATED_DIR}/hlsl_intrinsics.h
    OUTPUT ${_GENERATED_DIR}/hlsl_intrinsics.i
    COMMAND ${CMAKE_COMMAND} -E echo "Generate intrinsic Header ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${python_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/gen_hlsl_intrinsics.py ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrin_main.txt ${_GENERATED_DIR}/hlsl_intrinsics.h ${_GENERATED_DIR}/hlsl_intrinsics.i
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_hlsl_intrinsics.py 
        ${CMAKE_CURRENT_SOURCE_DIR}/gen_intrin_main.txt
    VERBATIM
    )
