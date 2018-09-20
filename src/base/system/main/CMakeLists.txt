# name of the target and the resulting library
set(PROJECT_NAME base-system-main)

# collect sources
set(PROJECT_HEADERS
    "access_module.h"
    "i_assert.h"
    "i_module.h"
    "i_module_id.h"
    "module.h"
    "module_registration.h"
    "module_registration_entry.h"
    "neuray_cc_conf.h"
    "platform.h"
    "types.h"
    )

set(PROJECT_SOURCES
    "module.cpp"
    "module_registration_entry.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )