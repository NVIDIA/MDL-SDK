# name of the target and the resulting library
set(PROJECT_NAME base-lib-log)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS
    "i_log_assert.h"
    "i_log_friendly.h"
    "i_log_logger.h"
    "i_log_macros.h"
    "i_log_module.h"
    "i_log_stream.h"
    "i_log_target.h"
    "log.h"
    )

set(PROJECT_SOURCES
     ${PROJECT_HEADERS}
     ${DUMMY_CPP}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
