# name of the target and the resulting library
set(PROJECT_NAME shaders-plugin-dds)

# collect sources
set(PROJECT_HEADERS
    "dds_decompress.h"
    "dds_half_to_float.h"
    "dds_image.h"
    "dds_image_file_reader_impl.h"
    "dds_image_file_writer_impl.h"
    "dds_image_plugin_impl.h"
    "dds_surface.h"
    "dds_texture.h"
    "dds_types.h"
    "dds_utilities.h"
    )

set(PROJECT_SOURCES 
    "dds_decompress.cpp"
    "dds_image.cpp"
    "dds_image_plugin_impl.cpp"
    "dds_image_file_reader_impl.cpp"
    "dds_image_file_writer_impl.cpp"
    "dds_utilities.cpp"
    ${PROJECT_HEADERS}
    )

if(WINDOWS)
    list(APPEND PROJECT_SOURCES "dds.rc")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    SOURCES ${PROJECT_SOURCES}
)

# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "dds")

if(MACOSX)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so") # corresponding to the binary release
endif()

# target alias for the custom name
add_library(dds ALIAS ${PROJECT_NAME})
add_library(mdl::dds ALIAS ${PROJECT_NAME})

# add dependencies other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        mdl::base-system-version
    )