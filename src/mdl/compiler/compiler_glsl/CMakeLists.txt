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
set(PROJECT_NAME mdl-compiler-compiler_glsl)

# -------------------------------------------------------------------------------------------------
# Main Target
# -------------------------------------------------------------------------------------------------

# collect sources
set(PROJECT_HEADERS
    "compiler_glsl_analysis.h"
    "compiler_glsl_assert.h"
    "compiler_glsl_builtins.h"
    "compiler_glsl_builtin_types.h"
    "compiler_glsl_cc_conf.h"
    "compiler_glsl_compilation_unit.h"
    "compiler_glsl_compiler.h"
    "compiler_glsl_declarations.h"
    "compiler_glsl_definitions.h"
    "compiler_glsl_errors.h"
    "compiler_glsl_exprs.h"
    "compiler_glsl_keywords.h"
    "compiler_glsl_locations.h"
    "compiler_glsl_messages.h"
    "compiler_glsl_optimizer.h"
    "compiler_glsl_predefined_symbols.h"
    "compiler_glsl_printers.h"
    "compiler_glsl_runtime.h"
    "compiler_glsl_stmts.h"
    "compiler_glsl_symbols.h"
    "compiler_glsl_tools.h"
    "compiler_glsl_types.h"
    "compiler_glsl_type_cache.h"
    "compiler_glsl_values.h"
    "compiler_glsl_version.h"
    "compiler_glsl_visitor.h"
    )

set(PROJECT_SOURCES
    "compiler_glsl_analysis.cpp"
    "compiler_glsl_builtins.cpp"
    "compiler_glsl_compilation_unit.cpp"
    "compiler_glsl_compiler.cpp"
    "compiler_glsl_declarations.cpp"
    "compiler_glsl_definitions.cpp"
    "compiler_glsl_errors.cpp"
    "compiler_glsl_exprs.cpp"
    "compiler_glsl_keywords.cpp"
    "compiler_glsl_messages.cpp"
    "compiler_glsl_optimizer.cpp"
    "compiler_glsl_printers.cpp"
    "compiler_glsl_stmts.cpp"
    "compiler_glsl_symbols.cpp"
    "compiler_glsl_types.cpp"
    "compiler_glsl_values.cpp"
    "compiler_glsl_version.cpp"
    "compiler_glsl_visitor.cpp"
    ${PROJECT_HEADERS}
    )

# add the generator files to the solution
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)
source_group("generator" FILES ${_GENERATED_SOURCES})


# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        boost
    )

