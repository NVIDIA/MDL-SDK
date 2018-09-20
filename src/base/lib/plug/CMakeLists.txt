# name of the target and the resulting library
set(PROJECT_NAME base-lib-plug)

# collect sources
set(PROJECT_HEADERS
    "i_plug.h"
    "plug_impl.h"
    )

set(PROJECT_SOURCES 
    "plug_impl.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
