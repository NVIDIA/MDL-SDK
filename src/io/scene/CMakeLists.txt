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
set(PROJECT_NAME io-scene)

# collect sources
set(PROJECT_HEADERS
    "bsdf_measurement/i_bsdf_measurement.h"
    "dbimage/i_dbimage.h"
    "lightprofile/i_lightprofile.h"
    "mdl_elements/i_mdl_elements_compiled_material.h"
    "mdl_elements/i_mdl_elements_expression.h"
    "mdl_elements/i_mdl_elements_function_call.h"
    "mdl_elements/i_mdl_elements_function_definition.h"
    "mdl_elements/i_mdl_elements_material_definition.h"
    "mdl_elements/i_mdl_elements_material_instance.h"
    "mdl_elements/i_mdl_elements_module.h"
    "mdl_elements/i_mdl_elements_module_builder.h"
    "mdl_elements/i_mdl_elements_module_transformer.h"
    "mdl_elements/i_mdl_elements_resource_map.h"
    "mdl_elements/i_mdl_elements_type.h"
    "mdl_elements/i_mdl_elements_utilities.h"
    "mdl_elements/i_mdl_elements_value.h"
    "mdl_elements/mdl_elements_annotation_definition_proxy.h"
    "mdl_elements/mdl_elements_ast_builder.h"
    "mdl_elements/mdl_elements_detail.h"
    "mdl_elements/mdl_elements_expression.h"
    "mdl_elements/mdl_elements_type.h"
    "mdl_elements/mdl_elements_utilities.h"
    "mdl_elements/mdl_elements_value.h"
    "scene/i_scene_attr_resv_id.h"
    "scene/i_scene_journal_types.h"
    "scene/i_scene_scene_element.h"
    "scene/i_scene_scene_element_base.h"
    "scene/scene_scene_element_inline.h"
    "texture/i_texture.h"
    )

set(PROJECT_SOURCES 
    "bsdf_measurement/bsdf_measurement.cpp"
    "dbimage/dbimage_image.cpp"
    "lightprofile/lightprofile.cpp"
    "lightprofile/lightprofile_ies_parser.cpp"
    "mdl_elements/mdl_elements_annotation_definition_proxy.cpp"
    "mdl_elements/mdl_elements_ast_builder.cpp"
    "mdl_elements/mdl_elements_compiled_material.cpp"
    "mdl_elements/mdl_elements_detail.cpp"
    "mdl_elements/mdl_elements_expression.cpp"
    "mdl_elements/mdl_elements_function_call.cpp"
    "mdl_elements/mdl_elements_function_definition.cpp"
    "mdl_elements/mdl_elements_material_definition.cpp"
    "mdl_elements/mdl_elements_material_instance.cpp"
    "mdl_elements/mdl_elements_module.cpp"
    "mdl_elements/mdl_elements_module_builder.cpp"
    "mdl_elements/mdl_elements_module_transformer.cpp"
    "mdl_elements/mdl_elements_type.cpp"
    "mdl_elements/mdl_elements_utilities.cpp"
    "mdl_elements/mdl_elements_value.cpp"
    "texture/texture.cpp"
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

# add tests if available
add_tests(POST)
