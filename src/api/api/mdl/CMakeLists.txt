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
set(PROJECT_NAME api-api-mdl)

# collect sources
set(PROJECT_HEADERS
    "../mdl/mdl_logger.h"
    "../mdl/mdl_mdl_compiler_impl.h"
    "../mdl/mdl_neuray_impl.h"
    "../neuray/i_neuray_attribute_context.h"
    "../neuray/i_neuray_transaction.h"
    "../neuray/i_neuray_db_element.h"
    "../neuray/i_neuray_proxy.h"
    "../neuray/neuray_array_impl.h"
    "../neuray/neuray_array_impl_proxy.h"
    "../neuray/neuray_attribute_container_impl.h"
    "../neuray/neuray_attribute_set_impl_helper.h"
    "../neuray/neuray_bsdf_measurement_impl.h"
    "../neuray/neuray_class_factory.h"
    "../neuray/neuray_class_registration.h"
    "../neuray/neuray_compiled_material_impl.h"
    "../neuray/neuray_compound_impl.h"
    "../neuray/neuray_database_impl.h"
    "../neuray/neuray_db_element_tracker.h"
    "../neuray/neuray_debug_configuration_impl.h"
    "../neuray/neuray_enum_decl_impl.h"
    "../neuray/neuray_enum_impl.h"
    "../neuray/neuray_export_result_ext_impl.h"
    "../neuray/neuray_expression_impl.h"
    "../neuray/neuray_factory_impl.h"
    "../neuray/neuray_function_call_impl.h"
    "../neuray/neuray_function_definition_impl.h"
    "../neuray/neuray_image_api_impl.h"
    "../neuray/neuray_image_impl.h"
    "../neuray/neuray_impexp_utilities.h"
    "../neuray/neuray_lightprofile_impl.h"
    "../neuray/neuray_map_impl.h"
    "../neuray/neuray_material_instance_impl.h"
    "../neuray/neuray_mdl_archive_api_impl.h"
    "../neuray/neuray_mdl_backend_api_impl.h"
    "../neuray/neuray_mdl_backend_impl.h"
    "../neuray/neuray_mdl_compatibility_api_impl.h"
    "../neuray/neuray_mdl_configuration_impl.h"
    "../neuray/neuray_mdl_discovery_api_impl.h"
    "../neuray/neuray_mdl_entity_resolver_impl.h"
    "../neuray/neuray_mdl_evaluator_api_impl.h"
    "../neuray/neuray_mdl_execution_context_impl.h"
    "../neuray/neuray_mdl_factory_impl.h"
    "../neuray/neuray_mdl_i18n_configuration_impl.h"
    "../neuray/neuray_mdl_impexp_api_impl.h"
    "../neuray/neuray_mdl_module_transformer_impl.h"
    "../neuray/neuray_mdl_resource_callback.h"
    "../neuray/neuray_mdle_api_impl.h"
    "../neuray/neuray_module_impl.h"
    "../neuray/neuray_number_impl.h"
    "../neuray/neuray_plugin_configuration_impl.h"
    "../neuray/neuray_pointer_impl.h"
    "../neuray/neuray_recording_transaction.h"
    "../neuray/neuray_ref_impl.h"
    "../neuray/neuray_scope_impl.h"
    "../neuray/neuray_string_impl.h"
    "../neuray/neuray_structure_decl_impl.h"
    "../neuray/neuray_structure_impl.h"
    "../neuray/neuray_texture_impl.h"
    "../neuray/neuray_transaction_impl.h"
    "../neuray/neuray_type_impl.h"
    "../neuray/neuray_type_utilities.h"
    "../neuray/neuray_uri.h"
    "../neuray/neuray_uuid_impl.h"
    "../neuray/neuray_value_impl.h"
    "../neuray/neuray_version_impl.h"
    "../neuray/neuray_void_impl.h"
    "../neuray/neuray_attribute_set_impl.h"
    "../neuray/neuray_db_element_impl.h"
    "../neuray/neuray_material_definition_impl.h"
)

set(PROJECT_SOURCES 
    "../mdl/mdl_logger.cpp"
    "../mdl/mdl_mdl_compiler_impl.cpp"
    "../mdl/mdl_neuray_impl.cpp"
    "../neuray/neuray_array_impl.cpp"
    "../neuray/neuray_array_impl_proxy.cpp"
    "../neuray/neuray_attribute_container_impl.cpp"
    "../neuray/neuray_attribute_set_impl_helper.cpp"
    "../neuray/neuray_bsdf_measurement_impl.cpp"
    "../neuray/neuray_class_factory.cpp"
    "../neuray/neuray_class_registration.cpp"
    "../neuray/neuray_compiled_material_impl.cpp"
    "../neuray/neuray_compound_impl.cpp"
    "../neuray/neuray_db_element_impl_base.cpp"
    "../neuray/neuray_database_impl.cpp"
    "../neuray/neuray_db_element_tracker.cpp"
    "../neuray/neuray_debug_configuration_impl.cpp"
    "../neuray/neuray_enum_impl.cpp"
    "../neuray/neuray_enum_decl_impl.cpp"
    "../neuray/neuray_export_result_ext_impl.cpp"
    "../neuray/neuray_expression_impl.cpp"
    "../neuray/neuray_factory_impl.cpp"
    "../neuray/neuray_function_call_impl.cpp"
    "../neuray/neuray_function_definition_impl.cpp"
    "../neuray/neuray_image_impl.cpp"
    "../neuray/neuray_image_api_impl.cpp"
    "../neuray/neuray_impexp_utilities.cpp"
    "../neuray/neuray_lightprofile_impl.cpp"
    "../neuray/neuray_map_impl.cpp"
    "../neuray/neuray_material_definition_impl.cpp"
    "../neuray/neuray_material_instance_impl.cpp"
    "../neuray/neuray_mdl_archive_api_impl.cpp"
    "../neuray/neuray_mdl_backend_api_impl.cpp"
    "../neuray/neuray_mdl_backend_impl.cpp"
    "../neuray/neuray_mdl_compatibility_api_impl.cpp"
    "../neuray/neuray_mdl_configuration_impl.cpp"
    "../neuray/neuray_mdl_discovery_api_impl.cpp"
    "../neuray/neuray_mdl_entity_resolver_impl.cpp"
    "../neuray/neuray_mdl_evaluator_api_impl.cpp"
    "../neuray/neuray_mdl_execution_context_impl.cpp"
    "../neuray/neuray_mdl_factory_impl.cpp"
    "../neuray/neuray_mdl_i18n_configuration_impl.cpp"
    "../neuray/neuray_mdl_impexp_api_impl.cpp"
    "../neuray/neuray_mdl_module_transformer_impl.cpp"
    "../neuray/neuray_mdl_resource_callback.cpp"
    "../neuray/neuray_mdle_api_impl.cpp"
    "../neuray/neuray_module_impl.cpp"
    "../neuray/neuray_number_impl.cpp"
    "../neuray/neuray_plugin_configuration_impl.cpp"
    "../neuray/neuray_pointer_impl.cpp"
    "../neuray/neuray_ref_impl.cpp"
    "../neuray/neuray_recording_transaction.cpp"
    "../neuray/neuray_scope_impl.cpp"
    "../neuray/neuray_string_impl.cpp"
    "../neuray/neuray_structure_impl.cpp"
    "../neuray/neuray_structure_decl_impl.cpp"
    "../neuray/neuray_texture_impl.cpp"
    "../neuray/neuray_transaction_impl.cpp"
    "../neuray/neuray_type_impl.cpp"
    "../neuray/neuray_type_utilities.cpp"
    "../neuray/neuray_uuid_impl.cpp"
    "../neuray/neuray_uri.cpp"
    "../neuray/neuray_value_impl.cpp"
    "../neuray/neuray_version_impl.cpp"
    "../neuray/neuray_void_impl.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    ADDITIONAL_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/../neuray"
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
        )