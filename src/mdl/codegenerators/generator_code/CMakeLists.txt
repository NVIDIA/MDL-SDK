# name of the target and the resulting library
set(PROJECT_NAME mdl-codegenerators-generator_code)

# collect sources
set(PROJECT_SOURCES 
    "generator_code.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
