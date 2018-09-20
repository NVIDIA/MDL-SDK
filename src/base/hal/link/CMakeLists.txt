# name of the target and the resulting library
set(PROJECT_NAME base-hal-link)

# collect sources
set(PROJECT_HEADERS
    "i_link.h"
    "link_impl.h"
    )

set(PROJECT_SOURCES 
    "link_impl.cpp"
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