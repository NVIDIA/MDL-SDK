# name of the target and the resulting library
set(PROJECT_NAME render-mdl-runtime)

# collect sources
set(PROJECT_HEADERS
    "i_mdlrt_bsdf_measurement.h"
    "i_mdlrt_light_profile.h"
    "i_mdlrt_resource_handler.h"
    "i_mdlrt_texture.h"
    )

set(PROJECT_SOURCES 
    "mdlrt_bsdf_measurement.cpp" 
    "mdlrt_light_profile.cpp"
    "mdlrt_resource_handler.cpp"
    "mdlrt_texture.cpp"
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