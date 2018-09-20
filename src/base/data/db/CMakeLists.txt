# name of the target and the resulting library
set(PROJECT_NAME base-data-db)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS
    "i_db_access.h"
    "i_db_cacheable.h"
    "i_db_database.h"
    "i_db_element.h"
    "i_db_info.h"
    "i_db_journal_type.h"
    "i_db_scope.h"
    "i_db_tag.h"
    "i_db_transaction.h"
    "i_db_transaction_id.h"
    "i_db_transaction_wrapper.h"
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
