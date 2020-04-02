#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

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
get_filename_component(python_PATH_ABS ${python_PATH} REALPATH)
set(python_PATH ${python_PATH_ABS} CACHE FILEPATH "Path of the Python 2.7+ binary." FORCE)
execute_process(COMMAND "${python_PATH}" "--version" 
    OUTPUT_VARIABLE 
        _PYTHON_VERSION_STRING 
    ERROR_VARIABLE 
        _PYTHON_VERSION_STRING
    )

# check version
if(NOT _PYTHON_VERSION_STRING)
    message(STATUS "python_PATH: ${python_PATH}")
    message(FATAL_ERROR "Python version could not be determined.")
else()
    # parse version number
    STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _PYTHON_VERSION_STRING ${_PYTHON_VERSION_STRING})
    if((${_PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL "3.0.0" OR ${_PYTHON_VERSION_STRING} VERSION_LESS "2.7.0") AND (${_PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL "4.0.0" OR ${_PYTHON_VERSION_STRING} VERSION_LESS "3.7.0"))
        message(FATAL_ERROR "Python 2.7 or 3.7 is required but Python ${_PYTHON_VERSION_STRING} was found instead. Please set the CMake option 'python_PATH' that needs to point to a Python 2.7 or 3.7 interpreter.")
    endif()
endif()
