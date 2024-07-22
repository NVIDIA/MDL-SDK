#*****************************************************************************
# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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

define_property(TARGET PROPERTY VS_DEBUGGER_WORKING_DIRECTORY
    BRIEF_DOCS "Working directory that is used when running the debugging command in Visual Studio."
    FULL_DOCS "Working directory that is used when running the debugging command in Visual Studio. Usually added by dependency scripts. Requires a call to 'set_property' after adding all dependencies."
    )

# set platform variable
set(WINDOWS FALSE)
set(NOT_WINDOWS TRUE)
set(LINUX FALSE)
set(NOT_LINUX TRUE)
set(MACOSX FALSE)
set(NOT_MACOSX TRUE)
set(ARCH_X64 FALSE)
set(ARCH_ARM FALSE)
set(ARCH_NAME "")
set(OS_NAME "")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
    set(ARCH_ARM TRUE)
    set(ARCH_NAME "aarch64")
else()
    set(ARCH_X64 TRUE)
    set(ARCH_NAME "x86-64")
endif()

if(WIN32 AND NOT UNIX)
    set(WINDOWS TRUE)
    set(NOT_WINDOWS FALSE)
    set(OS_NAME "nt")
elseif(UNIX AND NOT WIN32 AND NOT APPLE)
    set(LINUX TRUE)
    set(NOT_LINUX FALSE)
    set(OS_NAME "linux")
elseif(APPLE AND UNIX)
    set(MACOSX TRUE)
    set(NOT_MACOSX FALSE)
    set(OS_NAME "macosx")
else()
    message(WARNING "System is currently not supported explicitly. Assuming Linux.")
    set(LINUX TRUE)
    set(OS_NAME "linux")
endif()

# CMAKE_TOOLCHAIN_FILE is handled incorrectly in older versions (at least on Windows)
if(WINDOWS AND ${CMAKE_VERSION} VERSION_LESS "3.21")
    message(FATAL_ERROR "CMake >= 3.21 is required on Windows.")
endif()

set(MI_PLATFORM_NAME ${OS_NAME}-${ARCH_NAME} CACHE INTERNAL "Name of the platform in the MI build system." FORCE)

# Platform depended symbols and keywords
if(WINDOWS)
    set(ENV_SEP ";")
    set(ENV_LIB_PATH "PATH")
elseif(MACOSX)
    set(ENV_SEP ":")
    set(ENV_LIB_PATH "DYLD_LIBRARY_PATH")
else() # LINUX
    set(ENV_SEP ":")
    set(ENV_LIB_PATH "LD_LIBRARY_PATH")
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
    MESSAGE(STATUS "[INFO] MDL_BASE_FOLDER:                      ${MDL_BASE_FOLDER}")
    MESSAGE(STATUS "[INFO] MDL_INCLUDE_FOLDER:                   ${MDL_INCLUDE_FOLDER}")
    MESSAGE(STATUS "[INFO] MDL_SRC_FOLDER:                       ${MDL_SRC_FOLDER}")
    MESSAGE(STATUS "[INFO] MDL_EXAMPLES_FOLDER:                  ${MDL_EXAMPLES_FOLDER}")
    MESSAGE(STATUS "[INFO] MDL_LOG_PLATFORM_INFOS:               ${MDL_LOG_PLATFORM_INFOS}")
    MESSAGE(STATUS "[INFO] MDL_LOG_DEPENDENCIES:                 ${MDL_LOG_DEPENDENCIES}")
    MESSAGE(STATUS "[INFO] MDL_LOG_FILE_DEPENDENCIES:            ${MDL_LOG_FILE_DEPENDENCIES}")
    MESSAGE(STATUS "[INFO] MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS: ${MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS}")
    MESSAGE(STATUS "[INFO] MDL_ADDITIONAL_COMPILER_OPTIONS:      ${MDL_ADDITIONAL_COMPILER_OPTIONS}")

    MESSAGE(STATUS "[INFO] CMAKE_VERSION:                        ${CMAKE_VERSION}")
    MESSAGE(STATUS "[INFO] CMAKE_SYSTEM_NAME:                    ${CMAKE_SYSTEM_NAME}")
    MESSAGE(STATUS "[INFO] CMAKE_SYSTEM_PROCESSOR:               ${CMAKE_SYSTEM_PROCESSOR}")
    MESSAGE(STATUS "[INFO] WINDOWS:                              ${WINDOWS}")
    MESSAGE(STATUS "[INFO] LINUX:                                ${LINUX}")
    MESSAGE(STATUS "[INFO] MACOSX:                               ${MACOSX}")
    MESSAGE(STATUS "[INFO] ARCH_ARM:                             ${ARCH_ARM}")
    MESSAGE(STATUS "[INFO] ARCH_X64:                             ${ARCH_X64}")
    MESSAGE(STATUS "[INFO] MI_PLATFORM_NAME:                     ${MI_PLATFORM_NAME}")

    MESSAGE(STATUS "[INFO] CMAKE_BUILD_TYPE:                     ${CMAKE_BUILD_TYPE}")
    MESSAGE(STATUS "[INFO] CMAKE_GENERATOR:                      ${CMAKE_GENERATOR}")
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    MESSAGE(STATUS "[INFO] GENERATOR_IS_MULTI_CONFIG:            ${_GENERATOR_IS_MULTI_CONFIG}")

    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_ID:                ${CMAKE_CXX_COMPILER_ID}")
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_VERSION:           ${CMAKE_CXX_COMPILER_VERSION}")
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER:                   ${CMAKE_CXX_COMPILER}")
    if(MSVC)
        MESSAGE(STATUS "[INFO] MSVC_VERSION:                         ${MSVC_VERSION}")
        MESSAGE(STATUS "[INFO] MSVC_IDE:                             ${MSVC_IDE}")
    endif()
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS:                      ${CMAKE_CXX_FLAGS}")
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_DEBUG:                ${CMAKE_CXX_FLAGS_DEBUG}")
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_RELEASE:              ${CMAKE_CXX_FLAGS_RELEASE}")
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_RELWITHDEBINFO:       ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

    MESSAGE(STATUS "[INFO] CMAKE_EXE_LINKER_FLAGS:               ${CMAKE_EXE_LINKER_FLAGS}")

    MESSAGE(STATUS "[INFO] CMAKE_TOOLCHAIN_FILE:                 ${CMAKE_TOOLCHAIN_FILE}")
    MESSAGE(STATUS "[INFO] VCPKG_TARGET_TRIPLET:                 ${VCPKG_TARGET_TRIPLET}")
    MESSAGE(STATUS "[INFO] Qt5_DIR:                              ${Qt5_DIR}")
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER:                  ${CMAKE_CUDA_COMPILER}")
    MESSAGE(STATUS "[INFO] PYTHON_DIR:                           ${PYTHON_DIR}")
    MESSAGE(STATUS "[INFO] python_PATH:                          ${python_PATH}")
    MESSAGE(STATUS "[INFO] clang_PATH:                           ${clang_PATH}")
    MESSAGE(STATUS "[INFO] swig_PATH:                            ${swig_PATH}")
    MESSAGE(STATUS "[INFO] ARNOLD_SDK_DIR:                       ${ARNOLD_SDK_DIR}")
    MESSAGE(STATUS "[INFO] MATERIALX_DIR:                        ${MATERIALX_DIR}")
    MESSAGE(STATUS "[INFO] SLANG_DIR:                            ${SLANG_DIR}")
    MESSAGE(STATUS "[INFO] OPTIX7_DIR:                           ${OPTIX7_DIR}")
    MESSAGE(STATUS "[INFO] CUDA8_PATH:                           ${CUDA8_PATH}")
    MESSAGE(STATUS "[INFO] PANTORA_AXF_DIR:                      ${PANTORA_AXF_DIR}")
    MESSAGE(STATUS "[INFO] VULKAN_SDK_DIR:                       ${VULKAN_SDK_DIR}")
    if(WINDOWS)
        MESSAGE(STATUS "[INFO] DXC_DIR:                              ${DXC_DIR}")
        MESSAGE(STATUS "[INFO] AGILITY_SDK_ENABLED:                  ${AGILITY_SDK_ENABLED}")
        MESSAGE(STATUS "[INFO] AGILITY_SDK_DIR:                      ${AGILITY_SDK_DIR}")
    endif()
    MESSAGE(STATUS "[INFO] DOXYGEN_DIR:                          ${DOXYGEN_DIR}")
    MESSAGE(STATUS "[INFO] GRAPHVIZ_DIR:                         ${GRAPHVIZ_DIR}")
endif()

option(MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES "Links the MSCV dynamic runtime (\\MD) instead of static (\\MT)." ON)

# check for dependencies
# pre-declare all options that are used
# in order to show them in CMake-Gui, even the script stops because of an error.
option(MDL_ENABLE_CUDA_EXAMPLES "Enable examples that require CUDA." ${NOT_MACOSX})
option(MDL_ENABLE_OPENGL_EXAMPLES "Enable examples that require OpenGL." ON)
option(MDL_ENABLE_QT_EXAMPLES "Enable examples that require Qt." ${ARCH_X64})
option(MDL_ENABLE_VULKAN_EXAMPLES "Enable examples that require Vulkan." ${ARCH_X64})
option(MDL_ENABLE_AXF_EXAMPLES "Enable examples that require AXF." OFF)
option(MDL_ENABLE_D3D12_EXAMPLES "Enable examples that require Direct3D and DirectX 12." ${WINDOWS})
option(MDL_ENABLE_OPTIX7_EXAMPLES "Enable examples that require OptiX 7." OFF)
option(MDL_ENABLE_MATERIALX "Enable MaterialX in examples that support it." OFF)
option(MDL_ENABLE_SLANG "Enable Slang in examples that support it." OFF)
option(MDL_ENABLE_PYTHON_BINDINGS "Enable the generation of python bindings." OFF)
option(MDL_ENABLE_PYTHON_UNIT_TEST_COVERAGE "Generates a coverage report when running python unit test. Requires the `coverage` module installed." OFF)

if(EXISTS ${MDL_SRC_FOLDER}/api)
    option(MDL_BUILD_DOCUMENTATION "Enable the build of the API reference documentation." ON)
else()
    set(MDL_BUILD_DOCUMENTATION OFF CACHE INTERNAL "Enable the build of the API reference documentation." FORCE)
endif()

if(EXISTS ${MDL_SRC_FOLDER}/api)
    option(MDL_ENABLE_UNIT_TESTS "Generates unit tests." ON)
else()
    set(MDL_ENABLE_UNIT_TESTS OFF CACHE INTERNAL "Generates unit tests." FORCE)
endif()

# list of tests that can be defined only after all other targets are setup (clear that list here)
set(MDL_TEST_LIST_POST "" CACHE INTERNAL "list of test directories to add after regular targets are defined")

macro(EVALUATE var)
     if(${ARGN})
         set(${var} ON)
     else()
         set(${var} OFF)
     endif()
endmacro()

evaluate(NEED_BOOST      TRUE)
evaluate(NEED_OIIO       ((MDL_BUILD_SDK AND MDL_BUILD_OPENIMAGEIO_PLUGIN) OR MDL_BUILD_CORE_EXAMPLES))

evaluate(ANY_EXAMPLE     ((MDL_BUILD_SDK AND MDL_BUILD_SDK_EXAMPLES) OR MDL_BUILD_CORE_EXAMPLES))
evaluate(NEED_CUDA       (ANY_EXAMPLE AND MDL_ENABLE_CUDA_EXAMPLES))
evaluate(NEED_OPENGL     (ANY_EXAMPLE AND MDL_ENABLE_OPENGL_EXAMPLES))
evaluate(NEED_VULKAN     (ANY_EXAMPLE AND MDL_ENABLE_VULKAN_EXAMPLES))
evaluate(NEED_GLFW       (NEED_OPENGL OR NEED_VULKAN))
evaluate(NEED_QT         (ANY_EXAMPLE AND MDL_ENABLE_QT_EXAMPLES))
evaluate(NEED_D3D12      (ANY_EXAMPLE AND WINDOWS AND MDL_ENABLE_D3D12_EXAMPLES))
evaluate(NEED_MATERIALX  (ANY_EXAMPLE AND WINDOWS AND MDL_ENABLE_MATERIALX))
evaluate(NEED_SLANG      (ANY_EXAMPLE AND WINDOWS AND MDL_ENABLE_SLANG))
evaluate(NEED_OPTIX7     (ANY_EXAMPLE AND MDL_ENABLE_OPTIX7_EXAMPLES))
evaluate(NEED_AXF        (ANY_EXAMPLE AND ARCH_X64 AND MDL_ENABLE_AXF_EXAMPLES))

if(NEED_OIIO AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_openimageio_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_openimageio_ext.cmake)
    find_openimageio_ext()
endif()

if(NEED_BOOST AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    find_boost_ext()
endif()

if(NEED_CUDA AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_cuda_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_cuda_ext.cmake)
    find_cuda_ext()
endif()
if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER_ID:               ${CMAKE_CUDA_COMPILER_ID}")
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER_VERSION:          ${CMAKE_CUDA_COMPILER_VERSION}")
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER:                  ${CMAKE_CUDA_COMPILER}")
endif()

if(NEED_GLFW AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_glfw_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_glfw_ext.cmake)
    find_glfw_ext()
endif()

if(NEED_OPENGL AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_opengl_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_opengl_ext.cmake)
    find_opengl_ext()
endif()

if(NEED_VULKAN AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_vulkan_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_vulkan_ext.cmake)
    find_vulkan_ext()
endif()

if(NEED_QT AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_qt_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_qt_ext.cmake)
    find_qt_ext()
endif()

if(NEED_D3D12 AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_d3d12_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_d3d12_ext.cmake)
    find_d3d12_ext()
endif()

if(NEED_OPTIX7 AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_optix7_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_optix7_ext.cmake)
    find_optix7_ext()
endif()

if(NEED_AXF AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_pantora_axf_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_pantora_axf_ext.cmake)
    find_pantora_axf_ext()
endif()

if(NEED_MATERIALX AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_materialx.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_materialx.cmake)
    find_materialx()
endif()

if(NEED_SLANG AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_slang_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_slang_ext.cmake)
    find_slang_ext()
endif()


# dependencies for MDL Arnold
if(MDL_BUILD_SDK AND MDL_BUILD_ARNOLD_PLUGIN AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_arnoldsdk_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_arnoldsdk_ext.cmake)
    find_arnoldsdk_ext()
endif()

# dependencies for the bindings
if(MDL_ENABLE_PYTHON_BINDINGS AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_python_dev_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_python_dev_ext.cmake)
    find_python_dev_ext()
endif()

# dependencies for documentation
if(MDL_BUILD_DOCUMENTATION AND EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_doxygen_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_doxygen_ext.cmake)
    find_doxygen_ext()
endif()

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_ENABLE_OPENGL_EXAMPLES:           ${MDL_ENABLE_OPENGL_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_CUDA_EXAMPLES:             ${MDL_ENABLE_CUDA_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_VULKAN_EXAMPLES:           ${MDL_ENABLE_VULKAN_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_D3D12_EXAMPLES:            ${MDL_ENABLE_D3D12_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_OPTIX7_EXAMPLES:           ${MDL_ENABLE_OPTIX7_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_QT_EXAMPLES:               ${MDL_ENABLE_QT_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_AXF_EXAMPLES:              ${MDL_ENABLE_AXF_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_MATERIALX:                 ${MDL_ENABLE_MATERIALX}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_SLANG:                     ${MDL_ENABLE_SLANG}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_PYTHON_BINDINGS:           ${MDL_ENABLE_PYTHON_BINDINGS}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_UNIT_TESTS:                ${MDL_ENABLE_UNIT_TESTS}")
    MESSAGE(STATUS "[INFO] MDL_BUILD_DOCUMENTATION:              ${MDL_BUILD_DOCUMENTATION}")
endif()
