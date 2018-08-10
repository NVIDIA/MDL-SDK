# -------------------------------------------------------------------------------------------------
# script expects the following variables:
    # - TARGET_ADD_TOOL_DEPENDENCY_TARGET
    # - TARGET_ADD_TOOL_DEPENDENCY_TOOL
# -------------------------------------------------------------------------------------------------

# use a default fallback
find_program(python_PATH python)
if(NOT python_PATH)
    MESSAGE(FATAL_ERROR "The tool dependency \"${TARGET_ADD_TOOL_DEPENDENCY_TOOL}\" for target \"${TARGET_ADD_TOOL_DEPENDENCY_TARGET}\" could not be resolved.")
endif()

# call --version
execute_process(COMMAND "${python_PATH}" "--version" 
    OUTPUT_VARIABLE 
        _PYTHON_VERSION_STRING 
    ERROR_VARIABLE 
        _PYTHON_VERSION_STRING
    )

# parse version number
STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _PYTHON_VERSION_STRING ${_PYTHON_VERSION_STRING})

# check version
if(${_PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL "3.0.0" OR ${_PYTHON_VERSION_STRING} VERSION_LESS "2.7.0")
    message(FATAL_ERROR "Python 2.7 is required but Python ${_PYTHON_VERSION_STRING} was found instead. Please set the CMake option 'python_PATH' that needs to point to a python 2.7 interpreter.")
endif()
