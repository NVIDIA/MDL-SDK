# name of the target and the resulting library
set(PROJECT_NAME base-data-dblight)

# collect sources
set(PROJECT_HEADERS
    "dblight_database.h"
    "dblight_scope.h"
    "dblight_transaction.h"
    "i_dblight.h"
    )

set(PROJECT_SOURCES 
    "dblight_access.cpp"
    "dblight_database.cpp"
    "dblight_info.cpp"
    "dblight_scope.cpp"
    "dblight_transaction.cpp"
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