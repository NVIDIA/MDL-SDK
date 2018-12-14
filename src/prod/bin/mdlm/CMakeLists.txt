# name of the target and the resulting example
set(PROJECT_NAME prod-bin-mdlm)

# collect sources
set(PROJECT_SOURCES
    "application.cpp"
    "archive.cpp"
    "command.cpp"
    "errors.cpp"
    "logger.cpp"
    "main.cpp"
    "options.cpp"
    "search_path.cpp"
    "util.cpp"
    "version.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    OUTPUT_NAME "mdlm"
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        boost
        ${LINKER_START_GROUP}
        mdl::api-api-mdl
        mdl::base-hal-disk
        mdl::base-hal-hal
        mdl::base-lib-log
        mdl::base-lib-path
        mdl::base-system-main
        mdl::base-util-string_utils
        ${LINKER_END_GROUP}
        mdl::mdl_sdk
    )
    
# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})
