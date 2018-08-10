# name of the target and the resulting library
set(PROJECT_NAME base-system-main)

# collect sources
set(PROJECT_SOURCES
    "module.cpp"
    "module_registration_entry.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )