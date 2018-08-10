# name of the target and the resulting library
set(PROJECT_NAME io-scene)

# collect sources
set(PROJECT_SOURCES 
    "bsdf_measurement/bsdf_measurement.cpp"
    "dbimage/dbimage_image.cpp"
    "lightprofile/lightprofile.cpp"
    "lightprofile/lightprofile_ies_parser.cpp"
    "mdl_elements/mdl_elements_ast_builder.cpp"
    "mdl_elements/mdl_elements_compiled_material.cpp"
    "mdl_elements/mdl_elements_detail.cpp"
    "mdl_elements/mdl_elements_expression.cpp"
    "mdl_elements/mdl_elements_function_call.cpp"
    "mdl_elements/mdl_elements_function_definition.cpp"
    "mdl_elements/mdl_elements_material_definition.cpp"
    "mdl_elements/mdl_elements_material_instance.cpp"
    "mdl_elements/mdl_elements_module.cpp"
    "mdl_elements/mdl_elements_type.cpp"
    "mdl_elements/mdl_elements_utilities.cpp"
    "mdl_elements/mdl_elements_value.cpp"
    "texture/texture.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )

# additional options
target_compile_options(${PROJECT_NAME} 
    PRIVATE 
        "-D_USE_MATH_DEFINES" # to get M_PI
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )