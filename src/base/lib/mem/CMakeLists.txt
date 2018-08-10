# name of the target and the resulting library
set(PROJECT_NAME base-lib-mem)

# collect sources
set(PROJECT_SOURCES 
    "mem.cpp"
    "mem_debug_alloc.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
