# name of the target and the resulting library
set(PROJECT_NAME mdl-jit-libbsdf)

# create the project
project(${PROJECT_NAME})

# Files to generate
set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_GENERATED_SOURCES 
    ${_GENERATED_DIR}/libbsdf.bc
    ${_GENERATED_DIR}/libbsdf.d     # dependency file
    ${_GENERATED_DIR}/libbsdf.ll    # useful for debugging
    )

# collect sources
set(PROJECT_HEADERS
    "libbsdf.h"
    "libbsdf_internal.h"
    "libbsdf_runtime.h"
    "libbsdf_utilities.h"
    )

set(PROJECT_SOURCES
    "libbsdf.cpp"
    ${PROJECT_HEADERS}
    )


# mark files as generated to disable the check for existence during configure
set_source_files_properties(${_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)

# create target
add_custom_target(${PROJECT_NAME} 
    DEPENDS
        ${_GENERATED_DIR}/libbsdf.bc
        ${_GENERATED_DIR}/libbsdf.d
    SOURCES 
        ${PROJECT_SOURCES}
    )
target_print_log_header(TARGET ${PROJECT_NAME})
setup_ide(TARGET ${PROJECT_NAME} SOURCES ${PROJECT_SOURCES})

# -------------------------------------------------------------------------------------------------
# CLANG Build Step
# -------------------------------------------------------------------------------------------------

# get clang
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL clang)

add_custom_command(
    OUTPUT
        ${_GENERATED_DIR}/libbsdf.bc
        ${_GENERATED_DIR}/libbsdf.d
        ${_GENERATED_DIR}/libbsdf.ll
    COMMAND ${CMAKE_COMMAND} -E echo "Compile libbsdf bytecode using clang ..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
    COMMAND ${clang_PATH} 
        -emit-llvm -c -O2 -ffast-math -target x86_64-pc-win32 ${CMAKE_CURRENT_SOURCE_DIR}/libbsdf.cpp
        -o ${_GENERATED_DIR}/libbsdf.bc -MD -MT ${_GENERATED_DIR}/libbsdf.bc -MP -MF ${_GENERATED_DIR}/libbsdf.d.tmp
    COMMAND ${CMAKE_COMMAND} -E copy ${_GENERATED_DIR}/libbsdf.d.tmp ${_GENERATED_DIR}/libbsdf.d
    COMMAND ${CMAKE_COMMAND} -E remove ${_GENERATED_DIR}/libbsdf.d.tmp
    COMMAND ${clang_PATH} 
        -emit-llvm -S -O2 -ffast-math -target x86_64-pc-win32 ${CMAKE_CURRENT_SOURCE_DIR}/libbsdf.cpp
        -o ${_GENERATED_DIR}/libbsdf.ll
    DEPENDS     
        ${PROJECT_SOURCES}
    VERBATIM
    )
