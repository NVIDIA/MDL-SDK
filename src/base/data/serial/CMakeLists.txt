# name of the target and the resulting library
set(PROJECT_NAME base-data-serial)

# collect sources
set(PROJECT_HEADERS
    "i_compressed_serialization.h"
    "i_serial_buffer_serializer.h"
    "i_serial_buffered_file_serializer.h"
    "i_serial_classid.h"
    "i_serial_file_serializer.h"
    "i_serial_serializable.h"
    "i_serializer.h"
    "serial.h"
    "serial_inline.h"
    "serial_marker_helpers.h"
    )

set(PROJECT_SOURCES 
    "serial.cpp"
    "serial_buffer_serializer.cpp"
    "serial_file_serializer.cpp"
    "serial_marker_helpers.cpp"
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