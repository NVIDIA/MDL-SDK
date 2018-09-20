# name of the target and the resulting library
set(PROJECT_NAME render-mdl-backends)

# collect sources
set(PROJECT_HEADERS
    "backends_backends.h"
    "backends_link_unit.h"
    "backends_target_code.h"
    )

set(PROJECT_SOURCES 
    "backends_backends.cpp" 
    "backends_target_code.cpp"
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