# name of the target and the resulting library
set(PROJECT_NAME io-image)

# collect sources
set(PROJECT_SOURCES 
    "image/image_module_impl.cpp"
    "image/image_canvas_impl.cpp"
    "image/image_tile_impl.cpp"
    "image/image_access_canvas.cpp"
    "image/image_mipmap_impl.cpp"
    "image/image_access_mipmap.cpp"
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