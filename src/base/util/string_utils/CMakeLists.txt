# name of the target and the resulting library
set(PROJECT_NAME base-util-string_utils)

# collect sources
set(PROJECT_HEADERS
    "i_string_lexicographic_cast.h"
    "i_string_utils.h"
    )

set(PROJECT_SOURCES
    "string_utils.cpp"
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