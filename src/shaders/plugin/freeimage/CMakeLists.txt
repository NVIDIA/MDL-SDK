# name of the target and the resulting library
set(PROJECT_NAME shaders-plugin-freeimage)

# collect sources
set(PROJECT_SOURCES 
    "freeimage_image_plugin_impl.cpp"
    "freeimage_image_file_reader_impl.cpp"
    "freeimage_image_file_writer_impl.cpp"
    "freeimage_utilities.cpp"
    )

if(WINDOWS)
    list(APPEND PROJECT_SOURCES "freeimage.rc")
endif()

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    SOURCES ${PROJECT_SOURCES}
)

# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "nv_freeimage")

# target alias for the custom name
add_library(nv_freeimage ALIAS ${PROJECT_NAME})
add_library(mdl::nv_freeimage ALIAS ${PROJECT_NAME})

# add dependencies other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS
        freeimage
        mdl::base-system-version
    )