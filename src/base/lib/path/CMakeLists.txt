# name of the target and the resulting library
set(PROJECT_NAME base-lib-path)

# collect sources
set(PROJECT_HEADERS
    "i_path.h"
    "path_module_impl.h"
    )

set(PROJECT_SOURCES 
    "path_module_impl.cpp"
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