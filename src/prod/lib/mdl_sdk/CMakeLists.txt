# name of the target and the resulting library
set(PROJECT_NAME prod-lib-mdl_sdk)

# collect sources
set(PROJECT_SOURCES 
    "mdl_sdk_factory.cpp" 
    "mdl_sdk_required_modules.cpp"
    "mdl_sdk_stubs.cpp"
    "mdl_sdk.rc"
    )

if(WINDOWS)
    list(APPEND PROJECT_SOURCES "mdl_sdk.rc")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    SOURCES ${PROJECT_SOURCES}
    )

# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "lib")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "mdl_sdk")

# target alias for the custom name
add_library(mdl_sdk ALIAS ${PROJECT_NAME})
add_library(mdl::mdl_sdk ALIAS ${PROJECT_NAME})

# add mdl and other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
        ${LINKER_WHOLE_ARCHIVE}
        ${LINKER_START_GROUP}
        mdl::api-api-mdl
        mdl::base-data-attr
        mdl::base-data-dblight
        mdl::base-data-serial
        mdl::base-hal-disk
        mdl::base-hal-hal
        mdl::base-hal-link
        mdl::base-hal-time
        mdl::base-lib-config
        mdl::base-lib-mem
        mdl::base-lib-path
        mdl::base-lib-plug
        mdl::base-lib-libzip
        mdl::base-lib-zlib
        mdl::base-util-string_utils
        mdl::base-util-registry
        mdl::base-system-main
        mdl::base-system-version
        mdl::io-image
        mdl::io-scene
        mdl::mdl-codegenerators-generator_dag
        mdl::mdl-codegenerators-generator_code
        mdl::mdl-compiler-compilercore
        mdl::mdl-integration-mdlnr
        mdl::mdl-jit-generator_jit
        mdl::mdl-no_glsl-generator_stub
        mdl::mdl-runtime
        mdl::render-mdl-backends
        mdl::render-mdl-runtime
        ${LINKER_END_GROUP}
        llvm
        ${LINKER_NO_WHOLE_ARCHIVE}
    )

# add system dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} DEPENDS system
    COMPONENTS
        ld threads
    )