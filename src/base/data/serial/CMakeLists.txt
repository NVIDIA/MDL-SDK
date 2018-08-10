# name of the target and the resulting library
set(PROJECT_NAME base-data-serial)

# collect sources
set(PROJECT_SOURCES 
    "serial.cpp"
    "serial_buffer_serializer.cpp"
    "serial_file_serializer.cpp"
    "serial_marker_helpers.cpp"
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