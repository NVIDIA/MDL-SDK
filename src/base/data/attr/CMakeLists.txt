# name of the target and the resulting library
set(PROJECT_NAME base-data-attr)

# collect sources
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