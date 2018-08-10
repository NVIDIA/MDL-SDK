# name of the target and the resulting library
set(PROJECT_NAME tinyxml2)
set(PROJECT_VERSION 6.2.0)

# collect sources
set(PROJECT_SOURCES
    "include/tinyxml2.h"
    "src/tinyxml2.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    SOURCES ${PROJECT_SOURCES}
)

# add include directories
target_include_directories(${PROJECT_NAME} 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )