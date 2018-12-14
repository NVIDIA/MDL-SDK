# name of the target and the resulting library
set(PROJECT_NAME prod-bin-i18n)

# collect sources
set(PROJECT_HEADERS
    "application.h"
    "command.h"
    "errors.h"
    "logger.h"
    "options.h"
    "neuray_factory.h"
    "search_path.h"
    "traversal.h"
    "util.h"
    "version.h"
    "xliff.h"
    )

set(PROJECT_SOURCES
    "main.cpp"
    "application.cpp"
    "command.cpp"
    "errors.cpp"
    "logger.cpp"
    "options.cpp"
    "search_path.cpp"
    "traversal.cpp"
    "util.cpp"
    "version.cpp"
    "xliff.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    OUTPUT_NAME "i18n"
    SOURCES ${PROJECT_SOURCES}
)

# add mdl and other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        boost
        llvm
        ${LINKER_START_GROUP}
        mdl::api-api-mdl
        mdl::base-hal-disk
        mdl::base-hal-hal
        mdl::base-lib-tinyxml2
        mdl::base-system-main
        base-util-string_utils
        ${LINKER_END_GROUP}
    )
    