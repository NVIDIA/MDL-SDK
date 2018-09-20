# name of the target and the resulting library
set(PROJECT_NAME base-hal-disk)

# collect sources
set(PROJECT_HEADERS
    "disk.h"
    "disk_file_reader_writer_impl.h"
    "disk_inline.h"
    "disk_memory_reader_writer_impl.h"
    "disk_stream_position_impl.h"
    "i_disk_buffered_reader.h"
    "i_disk_file.h"
    "i_disk_ifile.h"
    "i_disk_zip_file.h"
    )

set(PROJECT_SOURCES 
    "disk.cpp"
    "diskdirectory.cpp"
    "diskfile.cpp"
    "disk_file_reader_writer_impl.cpp"
    "disk_memory_reader_writer_impl.cpp"
    "disk_zip_file.cpp"
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