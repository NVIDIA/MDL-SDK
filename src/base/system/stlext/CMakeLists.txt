# name of the target and the resulting library
set(PROJECT_NAME base-system-stlext)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS
    "i_stlext_any.h"
    "i_stlext_atomic_counter.h"
    "i_stlext_binary_cast.h"
    "i_stlext_concepts.h"
    "i_stlext_cv_traits.h"
    "i_stlext_intrusive_ptr.h"
    "i_stlext_iterator_facade.h"
    "i_stlext_likely.h"
    "i_stlext_no_unused_variable_warning.h"
    "i_stlext_restore.h"
    "i_stlext_safe_cast.h"
    "i_stlext_type_traits.h"
    "i_stlext_type_traits_base_types.h"
    "stlext_atomic_counter_inline.h"
    "stlext_mpl.h"
    "stlext_type_traits_defs.h"
    "stlext_type_traits_helper.h"
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
