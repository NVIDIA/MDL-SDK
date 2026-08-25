set(PROJECT_NAME examples-mdl_sdk-baking)

set(PROJECT_SOURCES
    "example_baking.cpp"
    )

create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    OUTPUT_NAME "baking"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
)

target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
    )

target_create_vs_user_settings(TARGET ${PROJECT_NAME})

add_target_install(
    TARGET ${PROJECT_NAME}
    DESTINATION "examples/mdl_sdk/baking"
    )
