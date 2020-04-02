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
set(PROJECT_NAME mdl-jit-libmdlrt)

# create the project
project(${PROJECT_NAME})

# Files to generate
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES 
    ${_GENERATED_DIR}/libmdlrt.bc
    ${_GENERATED_DIR}/libmdlrt.d     # dependency file
    ${_GENERATED_DIR}/libmdlrt.ll    # useful for debugging
    )

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS
    "libmdlrt.h"
    )

set(PROJECT_SOURCES
    "libmdlrt.cpp"
    ${PROJECT_HEADERS}
    )

# disable compiling
set_source_files_properties("libmdlrt.cpp" PROPERTIES HEADER_FILE_ONLY ON)

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES 
        ${PROJECT_SOURCES}
        ${_GENERATED_SOURCES}
        ${DUMMY_CPP}
    )

target_print_log_header(TARGET ${PROJECT_NAME})
setup_ide(TARGET ${PROJECT_NAME} SOURCES ${PROJECT_SOURCES})

# -------------------------------------------------------------------------------------------------
# CLANG Build Step
# -------------------------------------------------------------------------------------------------

# get clang
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL clang)

add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/libmdlrt.bc
        ${_GENERATED_DIR}/libmdlrt.d
        ${_GENERATED_DIR}/libmdlrt.ll
    COMMAND ${CMAKE_COMMAND} -E echo "Compile libmdlrt bytecode using clang ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${clang_PATH} 
        -emit-llvm -c -O3 -ffast-math -fno-vectorize -target nvptx-pc-cuda-gnu ${CMAKE_CURRENT_SOURCE_DIR}/libmdlrt.cpp
        -o ${_GENERATED_DIR}/libmdlrt.bc -MD -MT ${_GENERATED_DIR}/libmdlrt.bc -MP -MF ${_GENERATED_DIR}/libmdlrt.d.tmp
    COMMAND ${CMAKE_COMMAND} -E copy ${_GENERATED_DIR}/libmdlrt.d.tmp ${_GENERATED_DIR}/libmdlrt.d
    COMMAND ${CMAKE_COMMAND} -E remove ${_GENERATED_DIR}/libmdlrt.d.tmp
    COMMAND ${clang_PATH} 
        -emit-llvm -S -O3 -ffast-math -fno-vectorize -target nvptx-pc-cuda-gnu ${CMAKE_CURRENT_SOURCE_DIR}/libmdlrt.cpp
        -o ${_GENERATED_DIR}/libmdlrt.ll
    DEPENDS     
        ${PROJECT_SOURCES}
    VERBATIM
    )
