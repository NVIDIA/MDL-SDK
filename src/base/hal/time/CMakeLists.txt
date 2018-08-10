# name of the target and the resulting library
set(PROJECT_NAME base-hal-time)

# collect sources
#file(GLOB_RECURSE PROJECT_SOURCES "*.h" "*.cpp")
set(PROJECT_SOURCES 
    "time.cpp"
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