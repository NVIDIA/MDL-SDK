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
set(PROJECT_NAME io-image-image)

# add unit tests
function(CREATE_UNIT_TEST_TEMPLATE)
    set(options USES_IDIFF)
    set(oneValueArgs NAME)
    set(multiValueArgs LINK_LIBRARIES)
    cmake_parse_arguments(CREATE_UNIT_TEST_TEMPLATE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(CREATE_UNIT_TEST_TEMPLATE_USES_IDIFF)
        set(_USES_IDIFF "USES_IDIFF")
    endif()

    create_unit_test(
        NAME
            ${CREATE_UNIT_TEST_TEMPLATE_NAME}
        SOURCES
            ../${CREATE_UNIT_TEST_TEMPLATE_NAME}.cpp
        HEADERS
            ../test_shared.h
        DEPENDS
            ${LINKER_START_GROUP}
            mdl::io-image-image
            ${LINKER_DEPENDENCIES_BASE}
            ${LINKER_END_GROUP}
            boost
        LINK_LIBRARIES
            ${CREATE_UNIT_TEST_TEMPLATE_LINK_LIBRARIES}
        # Not all tests need all runtime dependencies.
        RUNTIME_DEPENDS
            shaders-plugin-dds
            shaders-plugin-openimageio
        # Not all tests need all paths.
        LIBRARY_PATHS
            ${CMAKE_CURRENT_BINARY_DIR}/../../../../shaders/plugin/dds
            ${CMAKE_CURRENT_BINARY_DIR}/../../../../shaders/plugin/openimageio
        ${_USES_IDIFF}
        )
endfunction()

# add unit tests
create_unit_test_template(NAME test_access_canvas USES_IDIFF)
create_unit_test_template(NAME test_access_mipmap USES_IDIFF)
create_unit_test_template(NAME test_dds USES_IDIFF)
create_unit_test_template(NAME test_huge_tiles)
create_unit_test_template(NAME test_import_export USES_IDIFF)
create_unit_test_template(NAME test_mipmap USES_IDIFF)
create_unit_test_template(NAME test_module)
create_unit_test_template(NAME test_pixel_conversion)
create_unit_test_template(NAME test_pixel_conversion_sse)
create_unit_test_template(NAME test_quantization)
