#*****************************************************************************
# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
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

macro(CUDA_MESSAGE HEADER)
    message(WARNING
        " ${HEADER} Possible solutions:\n"
        " - Install the CUDA SDK and/or set 'CMAKE_CUDA_COMPILER' (or the CMake generator option for the Visual Studio generators).\n"
        " - Disable the CMake options 'MDL_ENABLE_GPU_BAKER' and 'MDL_ENABLE_CUDA_EXAMPLES'.\n")
endmacro()

# The enable_language() call needs to be at file scope, not inside a function.
if(MDL_ENABLE_GPU_BAKER OR MDL_ENABLE_CUDA_EXAMPLES)

    # use the c++ compiler as host compiler (setting this does not work with Visual Studio or Apple clang 9.x)
    if(LINUX)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
    endif()

    # use check_language() such that we can provide our own error message pointing to the
    # corresponding CMake flags
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        cuda_message("Enabling CUDA language support failed.")
    endif()

endif()

function(FIND_CUDA_EXT)

    if(NOT CMAKE_CUDA_COMPILER)
        # warning was already emitted by the code above
        return()
    endif()

    # we don't use findCUDA here, we assume we can find all our dependencies relative to nvcc
    # find_package(Cuda REQUIRED)
    if(EXISTS ${CMAKE_CUDA_COMPILER})
        get_filename_component(_CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} PATH)
        set(_CUDA_SDK_DIR ${_CUDA_BIN_DIR}/..)
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "Found CUDA using the compiler.")
        endif()
    else()
        find_file(_CUDA_HEADER "include/cuda.h")
        if(_CUDA_HEADER)
            get_filename_component(_CUDA_INCLUDE_DIR ${_CUDA_HEADER} PATH)
            set(_CUDA_SDK_DIR ${_CUDA_INCLUDE_DIR}/..)
            if(MDL_LOG_DEPENDENCIES)
                message(STATUS "Found CUDA using 'cuda.h')")
            endif()
        endif()
    endif()

    if(NOT _CUDA_SDK_DIR)
        cuda_message("The dependency 'CUDA' could not be resolved (headers).")
        return()
    endif()

    # add include directories
    list(APPEND _CUDA_INCLUDE "${_CUDA_SDK_DIR}/include")
    list(APPEND _CUDA_INCLUDE "${_CUDA_SDK_DIR}/curand_dev/include")

    if(WINDOWS)

        set(_CUDA_LIB_DIRECTORY ${_CUDA_SDK_DIR}/lib/x64)
        set(_CUDA_LIB "${_CUDA_LIB_DIRECTORY}/cuda.lib")
        set(_CUDART_LIB "${_CUDA_LIB_DIRECTORY}/cudart_static.lib")

        # warning if dependencies can not be resolved
        if(NOT EXISTS ${_CUDA_LIB} OR NOT EXISTS ${_CUDART_LIB})
            message(STATUS "_CUDA_SDK_DIR: ${_CUDA_SDK_DIR}")
            message(STATUS "_CUDA_LIB: ${_CUDA_LIB}")
            message(STATUS "_CUDART_LIB: ${_CUDART_LIB}")
            cuda_message("The dependency 'CUDA' could not be resolved (libraries).")
        endif()

        list(APPEND _CUDA_LIBS ${_CUDA_LIB})
        list(APPEND _CUDA_LIBS ${_CUDART_LIB})

    else()

        find_file(_CUDA_SO
            NAMES
                ${CMAKE_SHARED_LIBRARY_PREFIX}cuda${CMAKE_SHARED_LIBRARY_SUFFIX}
            HINTS
                ${_CUDA_SDK_DIR}/lib64/stubs   # Linux
                ${_CUDA_SDK_DIR}/lib64
                /usr/local/cuda/lib            # MacOS
                ${_CUDA_SDK_DIR}/lib
            )
        find_file(_CUDART_A
            NAMES
                ${CMAKE_STATIC_LIBRARY_PREFIX}cudart_static${CMAKE_STATIC_LIBRARY_SUFFIX}
            HINTS
                ${_CUDA_SDK_DIR}/lib64         # Linux
                /usr/local/cuda/lib            # MacOS
                ${_CUDA_SDK_DIR}/lib
            )

        # warning if dependencies can not be resolved
        if(NOT EXISTS ${_CUDA_SO} OR NOT EXISTS ${_CUDART_A})
            message(STATUS "_CUDA_SDK_DIR: ${_CUDA_SDK_DIR}")
            message(STATUS "_CUDA_SO: ${_CUDA_SO}")
            message(STATUS "_CUDART_A: ${_CUDART_A}")
            cuda_message("The dependency 'CUDA' could not be resolved (libraries).")
        endif()

        list(APPEND _CUDA_LIBS ${_CUDA_SO})
        list(APPEND _CUDA_LIBS ${_CUDART_A})
        list(APPEND _CUDA_LIBS -lrt)

        if(MACOSX)
            list(APPEND _CUDA_LIBS "-F${_CUDA_SDK_DIR}/lib/stubs -Xlinker -framework -Xlinker CUDA")
        endif()

    endif()

    # store paths that are later used in add_cuda.cmake
    set(MDL_DEPENDENCY_CUDA_INCLUDE ${_CUDA_INCLUDE} CACHE INTERNAL "cuda headers")
    set(MDL_DEPENDENCY_CUDA_LIBS ${_CUDA_LIBS} CACHE INTERNAL "cuda libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_CUDA_INCLUDE:              ${MDL_DEPENDENCY_CUDA_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_CUDA_LIBS:                 ${MDL_DEPENDENCY_CUDA_LIBS}")
    endif()

endfunction()
