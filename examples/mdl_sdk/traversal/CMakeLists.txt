# name of the target and the resulting example
set(PROJECT_NAME examples-mdl_sdk-traversal)

# collect sources
set(PROJECT_SOURCES
    "example_traversal.cpp"
    "compiled_material_traverser_base.h"
    "compiled_material_traverser_base.cpp"
    "compiled_material_traverser_print.h"
    "compiled_material_traverser_print.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk
    OUTPUT_NAME "traversal"
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
    )
    
# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})

# add tests if available
add_tests()
