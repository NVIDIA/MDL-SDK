# name of the target and the resulting library
set(PROJECT_NAME base-system-version)

# collect sources
set(PROJECT_SOURCES
    "version.h"
    "version.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )