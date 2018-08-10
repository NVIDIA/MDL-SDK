# name of the target and the resulting library
set(PROJECT_NAME mdl-integration-mdlnr)

# collect sources
set(PROJECT_SOURCES
    "mdlnr.cpp"
    "mdlnr_search_path.cpp"
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