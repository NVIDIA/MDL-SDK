# name of the target and the resulting library
set(PROJECT_NAME shaders-plugin-dds)

# collect sources
set(PROJECT_SOURCES 
    "dds_decompress.cpp"
    "dds_image.cpp"
    "dds_image_plugin_impl.cpp"
    "dds_image_file_reader_impl.cpp"
    "dds_image_file_writer_impl.cpp"
    "dds_utilities.cpp"
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

# target alias for the custom name
add_library(dds ALIAS ${PROJECT_NAME})
add_library(mdl::dds ALIAS ${PROJECT_NAME})

# add dependencies other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS 
        mdl::base-system-version
    )