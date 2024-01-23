#*****************************************************************************
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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
find_program(swig_PATH swig)
if(NOT swig_PATH)
    MESSAGE(FATAL_ERROR "The tool dependency \"${TARGET_ADD_TOOL_DEPENDENCY_TOOL}\" for target \"${TARGET_ADD_TOOL_DEPENDENCY_TARGET}\" could not be resolved.")
endif()

# call --version
get_filename_component(swig_PATH_ABS ${swig_PATH} REALPATH)
set(swig_PATH ${swig_PATH_ABS} CACHE FILEPATH "Path of the SWIG binary." FORCE)
execute_process(COMMAND "${swig_PATH}" "-version" 
    OUTPUT_VARIABLE 
        _SWIG_VERSION_STRING 
    ERROR_VARIABLE 
        _SWIG_VERSION_STRING
    )

# check version
if(NOT _SWIG_VERSION_STRING)
    message(STATUS "swig_PATH: ${swig_PATH}")
    message(FATAL_ERROR "SWIG version could not be determined.")
else()
    # parse version number
    STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _SWIG_VERSION_STRING ${_SWIG_VERSION_STRING})
    if(${_SWIG_VERSION_STRING} VERSION_LESS "4.0.0")
        message(FATAL_ERROR "SWIG 4.0.0 is required but SWIG ${_SWIG_VERSION_STRING} was found instead. Please set the CMake option 'swig_PATH' that needs to point to a more recent version.")
    endif()
endif()
