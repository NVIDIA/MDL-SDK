# name of the target and the resulting library
set(PROJECT_NAME mdl-jit-libbsdf)

# create the project
project(${PROJECT_NAME})

# create target
add_custom_target(${PROJECT_NAME} 
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.bc
        ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.d.tmp
    )

# log message
target_print_log_header(TARGET ${PROJECT_NAME})

# configure visual studio and maybe other IDEs
setup_ide(TARGET ${PROJECT_NAME})

# -------------------------------------------------------------------------------------------------
# CLANG Build Step
# -------------------------------------------------------------------------------------------------

# get clang
target_add_tool_dependency(TARGET ${PROJECT_NAME} TOOL clang)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.bc
        ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.d.tmp
    COMMAND ${CMAKE_COMMAND} -E echo "Compile libbsdf bytecode using clang ..."
    COMMAND ${clang_PATH} 
        -emit-llvm -c -O2 -ffast-math -target x86_64-pc-win32 ${CMAKE_CURRENT_SOURCE_DIR}/libbsdf.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.bc -MD -MT
           ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.bc -MP -MF ${CMAKE_CURRENT_BINARY_DIR}/libbsdf.d.tmp
    DEPENDS     
        ${CMAKE_CURRENT_SOURCE_DIR}/libbsdf.cpp
    VERBATIM
    )