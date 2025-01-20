#*****************************************************************************
# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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

# check if CUDA is available
# Note, this enables CUDA for all projects (only of concern for Visual Studio)
if(MDL_ENABLE_CUDA_EXAMPLES)

    # use the c++ compiler as host compiler (setting this does not work with Visual Studio or Apple clang 9.x)
    if(LINUX)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
    endif()

    # try to enable cuda, can fail even though the compiler is found (see error message of second call)
    enable_language(CUDA OPTIONAL)
    if(NOT EXISTS ${CMAKE_CUDA_COMPILER})
        message(warning "Failed enable CUDA. Please install the CUDA SDK or disable 'MDL_ENABLE_CUDA_EXAMPLES'.")
        enable_language(CUDA) # call again to get an error
        return()
    endif()

    # Note, that the nvcc needs to support the host compiler (see <CUDA_PATH>/include/crt/host_config.h)
    # Here are some examples:
    # - CUDA 9.0 supports _MSC_VER up to 1911 which equals to VS2017, version 15.3 & 15.4 (tools version 14.11)
    # - CUDA 9.2 supports _MSC_VER up to 1913 which equals to VS2017, version 15.6 (tools version 14.13)
    #
    # since CMake 3.12.1 you can specify the minor versions using -T version=14.11,cuda=9.0
    # before, only the major version was available using -T v140,cuda=9.0 (which uses the VS 2015 tools with CUDA 9.0)
endif()

function(FIND_CUDA_EXT)

    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "Failed enable CUDA. Please install the CUDA SDK or disable 'MDL_ENABLE_CUDA_EXAMPLES'.")
        return()
    endif()

    set(_DEFAULT_ERROR_MESSAGE "The dependency \"cuda\" could not be resolved. Please install the CUDA SDK on your system or deactivate the option 'MDL_ENABLE_CUDA_EXAMPLES'")

    # we don't use findCUDA here, we assume we can find all our dependencies relative to nvcc
    # find the required packages
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
        message(FATAL_ERROR ${_DEFAULT_ERROR_MESSAGE})
        return()
    endif()

    # add include directories
    list(APPEND _CUDA_INCLUDE "${_CUDA_SDK_DIR}/include")
    list(APPEND _CUDA_INCLUDE "${_CUDA_SDK_DIR}/curand_dev/include")

    if(WINDOWS)
        set(_CUDA_LIB_DIRECTORY ${_CUDA_SDK_DIR}/lib/x64)
        set(_CUDA_LIB "${_CUDA_LIB_DIRECTORY}/cuda.lib")
        set(_CUDART_LIB "${_CUDA_LIB_DIRECTORY}/cudart_static.lib")

        # error if dependencies can not be resolved
        if(NOT EXISTS ${_CUDA_LIB} OR NOT EXISTS ${_CUDART_LIB})
            message(STATUS "_CUDA_SDK_DIR: ${_CUDA_SDK_DIR}")
            message(STATUS "_CUDA_LIB: ${_CUDA_LIB}")
            message(STATUS "_CUDART_LIB: ${_CUDART_LIB}")
            message(FATAL_ERROR ${_DEFAULT_ERROR_MESSAGE})
        endif()

        list(APPEND _CUDA_LIBS ${_CUDA_LIB})
        list(APPEND _CUDA_LIBS ${_CUDART_LIB})

    else() # Linux and MacOSX
        find_file(_CUDA_SO
            NAMES
                ${CMAKE_SHARED_LIBRARY_PREFIX}cuda${CMAKE_SHARED_LIBRARY_SUFFIX}
            HINTS
                # linux
                ${_CUDA_SDK_DIR}/lib64/stubs
                ${_CUDA_SDK_DIR}/lib64
                # macosx
                /usr/local/cuda/lib
                ${_CUDA_SDK_DIR}/lib
            )
        find_file(_CUDART_A
            NAMES
                ${CMAKE_STATIC_LIBRARY_PREFIX}cudart_static${CMAKE_STATIC_LIBRARY_SUFFIX}
            HINTS
                # linux
                ${_CUDA_SDK_DIR}/lib64
                # macosx
                /usr/local/cuda/lib
                ${_CUDA_SDK_DIR}/lib
            )

        # error if dependencies can not be resolved
        if(NOT EXISTS ${_CUDA_SO} OR NOT EXISTS ${_CUDART_A})
            message(STATUS "_CUDA_SDK_DIR: ${_CUDA_SDK_DIR}")
            message(STATUS "_CUDA_SO: ${_CUDA_SO}")
            message(STATUS "_CUDART_A: ${_CUDART_A}")
            message(FATAL_ERROR ${_DEFAULT_ERROR_MESSAGE})
        endif()

        list(APPEND _CUDA_LIBS ${_CUDA_SO})
        list(APPEND _CUDA_LIBS ${_CUDART_A})
        list(APPEND _CUDA_LIBS -lrt)

        if(MACOSX)
            list(APPEND _CUDA_LIBS "-F${_CUDA_SDK_DIR}/lib/stubs -Xlinker -framework -Xlinker CUDA")
        endif()
    endif()

    # store paths that are later used in the add_cuda.cmake
    set(MDL_DEPENDENCY_CUDA_INCLUDE ${_CUDA_INCLUDE} CACHE INTERNAL "cuda headers")
    set(MDL_DEPENDENCY_CUDA_LIBS ${_CUDA_LIBS} CACHE INTERNAL "cuda libs")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_CUDA_INCLUDE:              ${MDL_DEPENDENCY_CUDA_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_CUDA_LIBS:                 ${MDL_DEPENDENCY_CUDA_LIBS}")
    endif()

endfunction()
