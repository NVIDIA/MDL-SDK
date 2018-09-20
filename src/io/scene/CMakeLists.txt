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
    "mdl_elements/i_mdl_elements_type.h"
    "mdl_elements/i_mdl_elements_utilities.h"
    "mdl_elements/i_mdl_elements_value.h"
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