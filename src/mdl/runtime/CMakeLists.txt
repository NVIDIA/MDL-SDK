# name of the target and the resulting library
set(PROJECT_NAME mdl-runtime)

# collect sources
set(PROJECT_HEADERS 
    "spectral/chroma_grid.h"
    "spectral/i_spectral.h"
    "spectral/spectral_tables.h"
    )
    
set(PROJECT_SOURCES 
    "spectral/chroma_grid_d60.cpp"
    "spectral/chroma_grid_d65.cpp"
    "spectral/chroma_grid_e.cpp"
    "spectral/spectral.cpp"
    "spectral/spectral_tables.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
)
