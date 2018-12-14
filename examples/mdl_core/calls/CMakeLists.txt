# name of the target and the resulting example
set(PROJECT_NAME examples-mdl_core-calls)

# collect sources
set(PROJECT_SOURCES
    "example_calls.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_core
    OUTPUT_NAME "calls"
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS
        mdl::mdl_core
        mdl_core::shared
        )
        
# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})

# add tests if available
add_tests()
