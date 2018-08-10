# name of the target and the resulting library
set(PROJECT_NAME base-util-string_utils)

# collect sources
set(PROJECT_SOURCES
    "string_utils.cpp"
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