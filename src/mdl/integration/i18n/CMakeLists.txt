# name of the target and the resulting library
set(PROJECT_NAME mdl-integration-i18n)

# collect sources
set(PROJECT_HEADERS
    "i_i18n.h"
    "i18n_db.h"
    "i18n_translator.h"
)

set(PROJECT_SOURCES 
    "i18n_db.cpp"
    "i18n_translator.cpp"
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