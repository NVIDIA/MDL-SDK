# -------------------------------------------------------------------------------------------------
# script expects the following variables:
    # - TARGET_ADD_TOOL_DEPENDENCY_TARGET
    # - TARGET_ADD_TOOL_DEPENDENCY_TOOL
# -------------------------------------------------------------------------------------------------

# use a default fallback
find_program(clang_PATH clang)
if(NOT clang_PATH)
    MESSAGE(FATAL_ERROR "The tool dependency \"${TARGET_ADD_TOOL_DEPENDENCY_TOOL}\" for target \"${TARGET_ADD_TOOL_DEPENDENCY_TARGET}\" could not be resolved.")
endif()

# call --version
execute_process(COMMAND "${clang_PATH}" "--version" 
    OUTPUT_VARIABLE 
        _CLANG_VERSION_STRING 
    ERROR_VARIABLE 
        _CLANG_VERSION_STRING
    )

# parse version number
STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _CLANG_VERSION_STRING ${_CLANG_VERSION_STRING})

# check version
if(${_CLANG_VERSION_STRING} VERSION_GREATER_EQUAL "3.5.0" OR ${_CLANG_VERSION_STRING} VERSION_LESS "3.4.0")
    message(FATAL_ERROR "Clang 3.4 is required but Clang ${_CLANG_VERSION_STRING} was found instead. Please set the CMake option 'clang_PATH' that needs to point to a clang 3.4.x compiler.")
endif()
