# name of the target and the resulting library
set(PROJECT_NAME mdl-runtime)

# collect sources
set(PROJECT_SOURCES 
    "spectral/chroma_grid_d60.cpp"
    "spectral/chroma_grid_d65.cpp"
    "spectral/chroma_grid_e.cpp"
    "spectral/spectral.cpp"
    "spectral/spectral_tables.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
)
