# name of the target and the resulting library
set(PROJECT_NAME base-data-attr)

# collect sources
set(PROJECT_HEADERS
    "attr.h"
    "attr_attribute_list_inline.h"
    "attr_inline_attr.h"
    "attr_inline_attrset.h"
    "attr_inline_type.h"
    "attr_inline_type_iterator.h"
    "attr_module.h"
    "i_attr_attribute.h"
    "i_attr_attribute_list.h"
    "i_attr_registry.h"
    "i_attr_type.h"
    "i_attr_type_code_traits.h"
    "i_attr_type_named_value_iterator.h"
    "i_attr_type_value_iterator.h"
    "i_attr_types.h"
    "i_attr_utilities.h"
)

set(PROJECT_SOURCES 
    "attr.cpp"
    "attr_attribute.cpp"
    "attr_attribute_list.cpp"
    "attr_attrset.cpp"
    "attr_module.cpp"
    "attr_registry.cpp"
    "attr_type.cpp"
    "attr_type_value_iterator.cpp"
    "attr_utilities.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
    )