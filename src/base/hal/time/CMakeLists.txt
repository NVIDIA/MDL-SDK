# name of the target and the resulting library
set(PROJECT_NAME base-hal-time)

# collect sources
set(PROJECT_HEADERS
    "i_time.h"
    "time_inline.h"
    "time_stopwatch.h"
    "time_stopwatch_inline.h"
    )

set(PROJECT_SOURCES 
    "time.cpp"
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