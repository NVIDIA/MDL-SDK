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
find_program(clang_PATH clang)
if(NOT clang_PATH)
    MESSAGE(FATAL_ERROR "The tool dependency \"${TARGET_ADD_TOOL_DEPENDENCY_TOOL}\" for target \"${TARGET_ADD_TOOL_DEPENDENCY_TARGET}\" could not be resolved.")
endif()

# call --version
get_filename_component(clang_PATH_ABS ${clang_PATH} REALPATH)
set(clang_PATH ${clang_PATH_ABS} CACHE FILEPATH "Path of the Clang 7.0 binary." FORCE)
execute_process(COMMAND "${clang_PATH}" "--version" 
    OUTPUT_VARIABLE 
        _CLANG_VERSION_STRING 
    ERROR_VARIABLE 
        _CLANG_VERSION_STRING
    )

# check version
if(NOT _CLANG_VERSION_STRING)
    message(STATUS "clang_PATH: ${clang_PATH}")
    message(FATAL_ERROR "Clang version could not be determined.")
else()
    # parse version number
    STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _CLANG_VERSION_STRING ${_CLANG_VERSION_STRING})
    if(${_CLANG_VERSION_STRING} VERSION_GREATER_EQUAL "7.1.0" OR ${_CLANG_VERSION_STRING} VERSION_LESS "7.0.0")
        message(FATAL_ERROR "Clang 7.0 is required but Clang ${_CLANG_VERSION_STRING} was found instead. Please set the CMake option 'clang_PATH' that needs to point to a clang 7.0.x compiler.")
    endif()
endif()