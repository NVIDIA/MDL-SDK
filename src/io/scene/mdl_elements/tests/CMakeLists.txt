#*****************************************************************************
# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME io-scene-mdl_elements)

function(CREATE_UNIT_TEST_TEMPLATE)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs)
    cmake_parse_arguments(CREATE_UNIT_TEST_TEMPLATE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    create_unit_test(
        NAME
            ${CREATE_UNIT_TEST_TEMPLATE_NAME}
        SOURCES
            ../${CREATE_UNIT_TEST_TEMPLATE_NAME}.cpp
        HEADERS
            ../test_shared.h
        DEPENDS
            ${LINKER_START_GROUP}
            ${LINKER_DEPENDENCIES_IO}
            ${LINKER_DEPENDENCIES_MDL_JIT}
            ${LINKER_DEPENDENCIES_BASE}
            ${LINKER_END_GROUP}
            llvm
            boost
        # Only used ty test_misc.
        RUNTIME_DEPENDS
            shaders-plugin-openimageio
        # Only used ty test_misc.
        LIBRARY_PATHS
            ${CMAKE_CURRENT_BINARY_DIR}/../../../../shaders/plugin/openimageio
        )
endfunction()

# add unit tests
create_unit_test_template(NAME test_types)
create_unit_test_template(NAME test_values)
create_unit_test_template(NAME test_expressions)
create_unit_test_template(NAME test_misc)
