# name of the target and the resulting library
set(PROJECT_NAME mdl-codegenerators-generator_code)

# collect sources
set(PROJECT_HEADERS 
    "generator_code.h"
    "generator_code_hash.h"
    )

set(PROJECT_SOURCES 
    "generator_code.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
