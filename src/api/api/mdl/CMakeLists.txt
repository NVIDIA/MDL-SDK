# name of the target and the resulting library
set(PROJECT_NAME api-api-mdl)

# collect sources
set(PROJECT_SOURCES 
    "mdl_logger.cpp"
    "mdl_mdl_backend_impl.cpp"
    "mdl_mdl_compiler_impl.cpp"
    "mdl_neuray_impl.cpp"
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
    "../neuray/neuray_mdl_discovery_api_impl.cpp"
    "../neuray/neuray_mdl_evaluator_api_impl.cpp"
    "../neuray/neuray_mdl_factory_impl.cpp"
    "../neuray/neuray_mdl_resource_callback.cpp"
    "../neuray/neuray_module_impl.cpp"
    "../neuray/neuray_number_impl.cpp"
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