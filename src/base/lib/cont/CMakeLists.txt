# name of the target and the resulting library
set(PROJECT_NAME base-lib-cont)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS
    "cont_allocators.h"
    "cont_allocators_inline.h"
    "cont_array_inline.h"
    "cont_bitset_inline.h"
    "cont_bitvector_inline.h"
    "cont_dlist_inline.h"
    "cont_rle_array_inline.h"
    "cont_set_inline.h"
    "cont_tree.h"
    "i_cont_array.h"
    "i_cont_bitset.h"
    "i_cont_bitvector.h"
    "i_cont_dlist.h"
    "i_cont_rle_array.h"
    "i_cont_set.h"
    "i_cont_traits.h"
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
