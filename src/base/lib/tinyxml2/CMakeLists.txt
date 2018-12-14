# name of the target and the resulting library
set(PROJECT_NAME base-lib-tinyxml2)
set(PROJECT_VERSION 6.2.0)

# collect sources
set(PROJECT_SOURCES
    "tinyxml2.h"
    "tinyxml2.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOURCES ${PROJECT_SOURCES}
)
