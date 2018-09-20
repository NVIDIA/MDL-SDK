# name of the target and the resulting library
set(PROJECT_NAME base-lib-config)

# collect sources
set(PROJECT_HEADERS
    "config.h"
    "config_impl.h"
    )

set(PROJECT_SOURCES 
    "config.cpp"
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