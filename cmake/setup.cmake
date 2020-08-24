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

#if nothing is specified, we assume a debug build
set(_DEFAULT_BUILD_TYPE "Debug")

# no command line argument and no multi-config generator
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting default build type: '${_DEFAULT_BUILD_TYPE}'")
    set(CMAKE_BUILD_TYPE ${_DEFAULT_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)

    # define possible build types for cmake-gui and to be able to iterate over it during configuration
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo") # "MinSizeRel" "RelWithDebInfo"
else()
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Build types available to IDEs." FORCE)
endif()

# Set the C/C++ specified in the projects as requirements
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)

# IDE Setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # Generate folders for IDE targets
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "_cmake")
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run not moc automatically (Qt)
# We select the files manually and add the generated files to the project and the IDEs. 
# set(CMAKE_AUTOMOC ON)

# custom properties
define_property(TARGET PROPERTY VS_DEBUGGER_PATHS
    BRIEF_DOCS "List of paths that are added to the Visual Studio debugger environment PATH."
    FULL_DOCS "List of paths that are added to the Visual Studio debugger environment PATH. Usually added by dependency scripts. Requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' after adding all dependencies."
    )
define_property(TARGET PROPERTY VS_DEBUGGER_PATHS_DEBUG
    BRIEF_DOCS "List of paths that are added to the Visual Studio debugger environment PATH for Debug builds only."
    FULL_DOCS "List of paths that are added to the Visual Studio debugger environment PATH for Debug builds only. Usually added by dependency scripts. Requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' after adding all dependencies."
    )
define_property(TARGET PROPERTY VS_DEBUGGER_PATHS_RELEASE
    BRIEF_DOCS "List of paths that are added to the Visual Studio debugger environment PATH for Release builds only."
    FULL_DOCS "List of paths that are added to the Visual Studio debugger environment PATH for Release builds only. Usually added by dependency scripts. Requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' after adding all dependencies."
    )
define_property(TARGET PROPERTY VS_DEBUGGER_PATHS_RELWITHDEBINFO
    BRIEF_DOCS "List of paths that are added to the Visual Studio debugger environment PATH for Release with debug info builds only."
    FULL_DOCS "List of paths that are added to the Visual Studio debugger environment PATH for Release with debug info builds only. Usually added by dependency scripts. Requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' after adding all dependencies."
    )
define_property(TARGET PROPERTY VS_DEBUGGER_ENV_VARS
    BRIEF_DOCS "List of environment variables that are added to the Visual Studio debugger."
    FULL_DOCS "List of environment variables that are added to the Visual Studio debugger. Usually added by dependency scripts. Requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' after adding all dependencies."
    )

define_property(TARGET PROPERTY VS_DEBUGGER_COMMAND
    BRIEF_DOCS "Executable that is called when running the debugger in Visual Studio."
    FULL_DOCS "Executable that is called when running the debugger in Visual Studio. Usually added by dependency scripts. Requires a call to 'set_property' after adding all dependencies."
    )

define_property(TARGET PROPERTY VS_DEBUGGER_COMMAND_ARGUMENTS
    BRIEF_DOCS "List of arguments that are passed the debugging command in Visual Studio."
    FULL_DOCS "List of arguments that are passed the debugging command in Visual Studio. Usually added by dependency scripts. Requires a call to 'set_property' after adding all dependencies."
    )

# set platform variable
set(WINDOWS FALSE)
set(LINUX FALSE)
set(MACOSX FALSE)

if(WIN32 AND NOT UNIX)
    set(WINDOWS TRUE)
elseif(UNIX AND NOT WIN32 AND NOT APPLE)
    set(LINUX TRUE)
elseif(APPLE AND UNIX)
    set(MACOSX TRUE)
else()
    MESSAGE(AUTHOR_WARNING "System is currently not supported explicitly. Assuming Linux.")
    set(LINUX TRUE)
endif()

# remove the /MD flag cmake sets by default
set(CompilerFlags
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    CMAKE_C_FLAGS_RELWITHDEBINFO
    )

foreach(CompilerFlag ${CompilerFlags})
  string(REPLACE "/MDd" "" ${CompilerFlag} "${${CompilerFlag}}")
  string(REPLACE "/MD" "" ${CompilerFlag} "${${CompilerFlag}}")
endforeach()

# mark internally used variables as advanced (since the are not supposed to be changed in CMake-Gui)
mark_as_advanced(MDL_BASE_FOLDER)
mark_as_advanced(MDL_INCLUDE_FOLDER)
mark_as_advanced(MDL_SRC_FOLDER)
mark_as_advanced(MDL_EXAMPLES_FOLDER)

# print system/build information for error reporting
if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_BASE_FOLDER:                    " ${MDL_BASE_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_INCLUDE_FOLDER:                 " ${MDL_INCLUDE_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_SRC_FOLDER:                     " ${MDL_SRC_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_EXAMPLES_FOLDER:                " ${MDL_EXAMPLES_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_ADDITIONAL_COMPILER_OPTIONS:    " ${MDL_ADDITIONAL_COMPILER_OPTIONS})
    MESSAGE(STATUS "[INFO] CMAKE_VERSION:                      " ${CMAKE_VERSION})
    MESSAGE(STATUS "[INFO] CMAKE_SYSTEM_NAME:                  " ${CMAKE_SYSTEM_NAME})
    MESSAGE(STATUS "[INFO] WINDOWS:                            " ${WINDOWS})
    MESSAGE(STATUS "[INFO] LINUX:                              " ${LINUX})
    MESSAGE(STATUS "[INFO] MACOSX:                             " ${MACOSX})
    MESSAGE(STATUS "[INFO] CMAKE_BUILD_TYPE:                   " ${CMAKE_BUILD_TYPE})
    MESSAGE(STATUS "[INFO] CMAKE_GENERATOR:                    " ${CMAKE_GENERATOR})
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    MESSAGE(STATUS "[INFO] GENERATOR_IS_MULTI_CONFIG:          " ${_GENERATOR_IS_MULTI_CONFIG})
    
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_ID:              " ${CMAKE_CXX_COMPILER_ID})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_VERSION:         " ${CMAKE_CXX_COMPILER_VERSION})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER:                 " ${CMAKE_CXX_COMPILER})
    if(MSVC)
        MESSAGE(STATUS "[INFO] MSVC_VERSION:                       " ${MSVC_VERSION})
        MESSAGE(STATUS "[INFO] MSVC_IDE:                           " ${MSVC_IDE})
    endif()
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS:                    " ${CMAKE_CXX_FLAGS})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_DEBUG:              " ${CMAKE_CXX_FLAGS_DEBUG})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_RELEASE:            " ${CMAKE_CXX_FLAGS_RELEASE})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_RELWITHDEBINFO:     " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
endif()

option(MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES "Links the MSCV dynamic runtime (\\MD) instead of static (\\MT)." OFF)

# check for dependencies
# pre-declare all options that are used
# in order to show them in CMake-Gui, even the script stops because of an error.
option(MDL_ENABLE_CUDA_EXAMPLES "Enable examples that require CUDA." ON)
option(MDL_ENABLE_OPENGL_EXAMPLES "Enable examples that require OpenGL." ON)
option(MDL_ENABLE_QT_EXAMPLES "Enable examples that require Qt." ON)
option(MDL_ENABLE_D3D12_EXAMPLES "Enable examples that require Direct3D and DirectX 12." ${WINDOWS})
option(MDL_ENABLE_OPTIX7_EXAMPLES "Enable examples that require OptiX 7." OFF)
option(MDL_ENABLE_MATERIALX "Enable MaterialX in examples that support it." OFF)

if(EXISTS ${MDL_BASE_FOLDER}/cmake/tests/CMakeLists.txt)
    option(MDL_ENABLE_TESTS "Generates unit and example tests." ON)
else()
    set(MDL_ENABLE_TESTS OFF CACHE INTERNAL "Generates unit and example tests." FORCE)
endif()

# list of tests that can be defined only after all other targets are setup (clear that list here)
set(MDL_TEST_LIST_POST "" CACHE INTERNAL "list of test directories to add after regular targets are defined")

include(${MDL_BASE_FOLDER}/cmake/find/find_cuda_ext.cmake)
find_cuda_ext()

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER_ID:             " ${CMAKE_CUDA_COMPILER_ID})
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER_VERSION:        " ${CMAKE_CUDA_COMPILER_VERSION})
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER:                " ${CMAKE_CUDA_COMPILER})
endif()

include(${MDL_BASE_FOLDER}/cmake/find/find_opengl_ext.cmake)
find_opengl_ext()

include(${MDL_BASE_FOLDER}/cmake/find/find_qt_ext.cmake)
find_qt_ext()

include(${MDL_BASE_FOLDER}/cmake/find/find_d3d12_ext.cmake)
find_d3d12_ext()

include(${MDL_BASE_FOLDER}/cmake/find/find_optix7_ext.cmake)
find_optix7_ext()

include(${MDL_BASE_FOLDER}/cmake/find/find_arnoldsdk_ext.cmake)
find_arnoldsdk_ext()

# examples could potentially use FreeImage directly
if(EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_freeimage_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_freeimage_ext.cmake)
    find_freeimage_ext()
endif()

if(EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    find_boost_ext()
endif()

if(MDL_ENABLE_MATERIALX AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_materialx.cmake)
include(${MDL_BASE_FOLDER}/cmake/find/find_materialx.cmake)
    find_materialx()
endif()

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_ENABLE_OPENGL_EXAMPLES:         " ${MDL_ENABLE_OPENGL_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_CUDA_EXAMPLES:           " ${MDL_ENABLE_CUDA_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_D3D12_EXAMPLES:          " ${MDL_ENABLE_D3D12_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_OPTIX7_EXAMPLES:         " ${MDL_ENABLE_OPTIX7_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_QT_EXAMPLES:             " ${MDL_ENABLE_QT_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_TESTS:                   " ${MDL_ENABLE_TESTS})
endif()
