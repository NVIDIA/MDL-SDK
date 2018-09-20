# name of the target and the resulting library
set(PROJECT_NAME base-lib-mem)

# collect sources
set(PROJECT_HEADERS
    "i_mem_aligned.h"
    "i_mem_allocatable.h"
    "i_mem_consumption.h"
    "mem.h"
    "mem_debug_alloc.h"
    "mem_memtrack.h"
    )

set(PROJECT_SOURCES 
    "mem.cpp"
    "mem_debug_alloc.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
