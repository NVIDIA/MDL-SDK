#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME mdl-codegenerators-generator_dag)

# collect sources
set(PROJECT_HEADERS 
    "generator_dag.h"
    "generator_dag_builder.h"
    "generator_dag_code_printer.h"
    "generator_dag_dep_graph.h"
    "generator_dag_derivatives.h"
    "generator_dag_dumper.h"
    "generator_dag_generated_dag.h"
    "generator_dag_ir.h"
    "generator_dag_ir_checker.h"
    "generator_dag_lambda_function.h"
    "generator_dag_serializer.h"
    "generator_dag_tools.h"
    "generator_dag_walker.h"
    )

set(PROJECT_SOURCES 
    "generator_dag.cpp"
    "generator_dag_builder.cpp"
    "generator_dag_code_printer.cpp"
    "generator_dag_dep_graph.cpp"
    "generator_dag_derivatives.cpp"
    "generator_dag_dumper.cpp"
    "generator_dag_generated_dag.cpp"
    "generator_dag_ir.cpp"
    "generator_dag_ir_checker.cpp"
    "generator_dag_lambda_function.cpp"
    "generator_dag_serializer.cpp"
    "generator_dag_thin_walled.cpp"
    "generator_dag_opacity.cpp"
    "generator_dag_walker.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )

# additional definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        "_USE_MATH_DEFINES" # to get M_PI
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )