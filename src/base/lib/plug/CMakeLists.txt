# name of the target and the resulting library
set(PROJECT_NAME base-lib-plug)

# collect sources
#file(GLOB_RECURSE PROJECT_SOURCES "*.h" "*.cpp")
set(PROJECT_SOURCES 
    "plug_impl.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
