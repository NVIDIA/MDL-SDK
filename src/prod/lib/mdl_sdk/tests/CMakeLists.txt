#*****************************************************************************
# Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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
set(PROJECT_NAME prod-lib-mdl_sdk)

function(CREATE_UNIT_TEST_TEMPLATE)
    set(options)
    set(oneValueArgs NAME)
    set(multiValueArgs LINK_LIBRARIES)
    cmake_parse_arguments(CREATE_UNIT_TEST_TEMPLATE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    create_unit_test(
        NAME
            ${CREATE_UNIT_TEST_TEMPLATE_NAME}
        SOURCES
            ../../neuray/${CREATE_UNIT_TEST_TEMPLATE_NAME}.cpp
        HEADERS
            ../../neuray/test_shared.h
        DEPENDS
            boost
        LINK_LIBRARIES
            ${CREATE_UNIT_TEST_TEMPLATE_LINK_LIBRARIES}
        # Not all tests need all runtime dependencies.
        RUNTIME_DEPENDS
            prod-lib-mdl_sdk
            shaders-plugin-dds
            shaders-plugin-mdl_distiller
            shaders-plugin-openimageio
        # Not all tests need all paths.
        LIBRARY_PATHS
            ${CMAKE_CURRENT_BINARY_DIR}/..
            ${CMAKE_CURRENT_BINARY_DIR}/../../../../shaders/plugin/dds
            ${CMAKE_CURRENT_BINARY_DIR}/../../../../shaders/plugin/mdl_distiller
            ${CMAKE_CURRENT_BINARY_DIR}/../../../../shaders/plugin/openimageio
        )
endfunction()

# add unit tests using the template
create_unit_test_template(NAME test_class_factory)
create_unit_test_template(NAME test_db_elements)
create_unit_test_template(NAME test_idebug_configuration)
create_unit_test_template(NAME test_ifactory)
create_unit_test_template(NAME test_i18n)
create_unit_test_template(NAME test_iimage)
create_unit_test_template(NAME test_ilogging_configuration)
create_unit_test_template(NAME test_imdl_configuration)
create_unit_test_template(NAME test_imdl_module)
create_unit_test_template(NAME test_ineuray)
create_unit_test_template(NAME test_itransaction)
create_unit_test_template(NAME test_neuray_factory)
create_unit_test_template(NAME test_set_get)
create_unit_test_template(NAME test_target_materials LINK_LIBRARIES Boost::filesystem)
create_unit_test_template(NAME test_types)
create_unit_test_template(NAME test_types_array)
create_unit_test_template(NAME test_types_compound)
create_unit_test_template(NAME test_types_map)

# add unit tests via Python scripts
target_add_tool_dependency(TARGET prod-lib-mdl_sdk-test_types_at_api_boundary TOOL python)
add_test(
    NAME prod-lib-mdl_sdk-test_types_at_api_boundary
    COMMAND
        ${python_PATH}
        ${MDL_SRC_FOLDER}/prod/lib/neuray/test_types_at_api_boundary.py
        ${MDL_INCLUDE_FOLDER}
        ${MDL_SRC_FOLDER}/prod/lib/neuray/test_types_at_api_boundary.txt
)
set_property(
    TEST prod-lib-mdl_sdk-test_types_at_api_boundary
    PROPERTY LABELS "unit_test"
)

target_add_tool_dependency(TARGET prod-lib-mdl_sdk-test_unique_IIDs TOOL python)
add_test(
    NAME prod-lib-mdl_sdk-test_unique_IIDs
    COMMAND
        ${python_PATH}
        ${MDL_SRC_FOLDER}/prod/lib/neuray/test_unique_IIDs.py
        ${MDL_SRC_FOLDER}
)
set_property(
    TEST prod-lib-mdl_sdk-test_unique_IIDs
    PROPERTY LABELS "unit_test"
)
