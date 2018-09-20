# name of the target and the resulting library
set(PROJECT_NAME base-hal-hal)

# collect sources
set(PROJECT_HEADERS
    "hal.h"
    "i_hal_ospath.h"
    )

set(PROJECT_SOURCES 
    "hal.cpp"
    "hal_color.cpp"
    "hal_ospath.cpp"
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