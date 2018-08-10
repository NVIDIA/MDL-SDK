# name of the target and the resulting example
set(PROJECT_NAME mdl_sdk_example-calls)

# collect sources
set(PROJECT_SOURCES
    "example_calls.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk_examples
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk_examples::mdl_sdk_shared
    )

# link system libraries
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        system
    COMPONENTS
        ld
    )
