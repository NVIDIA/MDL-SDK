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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# policies
if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()
if(POLICY CMP0102)
    cmake_policy(SET CMP0102 NEW)
endif()


# MDL SDK build directories
set(MDL_BASE_FOLDER ${CMAKE_SOURCE_DIR} CACHE PATH "The folder that contains the targets to build and the cmake utility scripts.")
set(MDL_INCLUDE_FOLDER ${CMAKE_SOURCE_DIR}/include CACHE PATH "The folder that interface headers, i.e., mi/base, mi/neuray, ... directories.")
set(MDL_SRC_FOLDER ${CMAKE_SOURCE_DIR}/src CACHE PATH "The folder that contains the sources, i.e., the base, io, mdl, ... directories.")
set(MDL_EXAMPLES_FOLDER ${CMAKE_SOURCE_DIR}/examples CACHE PATH "The folder that contains the mdl examples.")
set(MDL_DOC_FOLDER ${CMAKE_SOURCE_DIR}/doc CACHE PATH "The folder that contains the documentation.")

# default values for internal builds, before project()
if(EXISTS ${MDL_BASE_FOLDER}/CMakeInternalBuildConfig_Toolchain_File.cmake AND NOT MDL_USE_LOCAL_DEPENDENCIES)
   include(${MDL_BASE_FOLDER}/CMakeInternalBuildConfig_Toolchain_File.cmake)
# otherwise, the toolchain has to be provided manually (unless system packages are used on Linux)
elseif(NOT CMAKE_TOOLCHAIN_FILE AND NOT UNIX)
    set(CMAKE_TOOLCHAIN_FILE "" CACHE FILEPATH "Path to toolchain file supplied to cmake.")
    message(FATAL_ERROR "No toolchain set. Please set CMAKE_TOOLCHAIN_FILE to <vcpkg_root>/scripts/buildsystems/vcpkg.cmake")
endif()

# before project() due to its use in CMAKE_TOOLCHAIN_FILE
if(WIN32 AND NOT UNIX AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET x64-windows-static)
endif()

project(MDL LANGUAGES C CXX)

# default values for internal builds, after project()
if(EXISTS ${MDL_BASE_FOLDER}/CMakeInternalBuildConfig.cmake AND NOT MDL_USE_LOCAL_DEPENDENCIES)
   include(${MDL_BASE_FOLDER}/CMakeInternalBuildConfig.cmake)
endif()


#--------------------------------------------------------------------------------------------------
# configuration options
option(MDL_BUILD_SDK "Build the MDL SDK, otherwise only Core." ON)
option(MDL_BUILD_SDK_EXAMPLES "Adds MDL SDK examples to the build." ON)
option(MDL_BUILD_CORE_EXAMPLES "Adds MDL Core examples to the build." ON)
option(MDL_BUILD_ARNOLD_PLUGIN "Enable the build of the MDL Arnold plugin." OFF)
option(MDL_BUILD_DDS_PLUGIN "Enable the build of the MDL DDS image plugin." ON)
option(MDL_BUILD_OPENIMAGEIO_PLUGIN "Enable the build of the MDL OpenImageIO image plugin." ON)
option(MDL_LOG_PLATFORM_INFOS "Prints some infos about the current build system (relevant for error reports)." ON)
option(MDL_LOG_DEPENDENCIES "Prints the list of dependencies during the generation step." ON)
option(MDL_LOG_FILE_DEPENDENCIES "Prints the list of files that is copied after a successful build." OFF)
option(MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS "Treat runtime dependencies as build dependencies." ON)
set(MDL_ADDITIONAL_COMPILER_DEFINES "MDL_SOURCE_RELEASE" CACHE STRING "Additional compile defines that are passed to each of the projects")
set(MDL_ADDITIONAL_COMPILER_OPTIONS "" CACHE STRING "Additional compile options that are passed to each of the projects")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# set the default installation path
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
endif()

# -------------------------------------------------------------------------------------------------
# general setup
include(${MDL_BASE_FOLDER}/cmake/setup.cmake)

# if the SDK is not build, also disable...
if(NOT MDL_BUILD_SDK)
    set(MDL_BUILD_SDK_EXAMPLES OFF CACHE BOOL "" FORCE) # the SDK examples
    set(MDL_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE) # the SDK Bindings
    set(MDL_BUILD_ARNOLD_PLUGIN OFF CACHE BOOL "" FORCE) # the Arnold Plugin
endif()

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_BUILD_SDK:                        ${MDL_BUILD_SDK}")
    MESSAGE(STATUS "[INFO] MDL_BUILD_SDK_EXAMPLES:               ${MDL_BUILD_SDK_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_BUILD_CORE_EXAMPLES:              ${MDL_BUILD_CORE_EXAMPLES}")
    MESSAGE(STATUS "[INFO] MDL_BUILD_ARNOLD_PLUGIN:              ${MDL_BUILD_ARNOLD_PLUGIN}")
    MESSAGE(STATUS "[INFO] MDL_BUILD_OPENIMAGEIO_PLUGIN:         ${MDL_BUILD_OPENIMAGEIO_PLUGIN}")
    MESSAGE(STATUS "[INFO] MDL_BUILD_DDS_PLUGIN:                 ${MDL_BUILD_DDS_PLUGIN}")
    MESSAGE(STATUS "[INFO] MDL_ENABLE_PYTHON_BINDINGS:           ${MDL_ENABLE_PYTHON_BINDINGS}")
endif()

# enable CTest if requested
if(MDL_ENABLE_UNIT_TESTS)
    enable_testing()
endif()

# -------------------------------------------------------------------------------------------------
# presets and utility functions used in all CMakeLists
include(${MDL_BASE_FOLDER}/cmake/utilities.cmake)

#--------------------------------------------------------------------------------------------------
# list of modules in defined order

# PUBLIC HEADERS (for convenience in IDEs)
#--------------------------------------------------------------------------------------------------
add_subdirectory(${MDL_INCLUDE_FOLDER})

# MDL SDK, MDL CORE, libs/tools, and plugins
#--------------------------------------------------------------------------------------------------

add_subdirectory(${MDL_SRC_FOLDER}/api)
add_subdirectory(${MDL_SRC_FOLDER}/base)
add_subdirectory(${MDL_SRC_FOLDER}/mdl)
add_subdirectory(${MDL_SRC_FOLDER}/io)
add_subdirectory(${MDL_SRC_FOLDER}/render)
add_subdirectory(${MDL_SRC_FOLDER}/prod)
add_subdirectory(${MDL_SRC_FOLDER}/shaders)

# EXAMPLES
#--------------------------------------------------------------------------------------------------

# third party
if (MDL_BUILD_SDK_EXAMPLES OR MDL_BUILD_CORE_EXAMPLES)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/thirdparty/imgui)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/thirdparty/tinyxml2)
endif()

# Shared code used by MDL SDK examples, Python bindings, and other tools
if(MDL_BUILD_SDK)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/shared)
endif()

# MDL SDK
if(MDL_BUILD_SDK_EXAMPLES)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/archives)
    if(MDL_ENABLE_AXF_EXAMPLES AND ARCH_X64)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/axf_to_mdl)
    endif()
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/calls)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/code_gen)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/compilation)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/create_module)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/discovery)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/distilling)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/distilling_target)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_native)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/generate_mdl_identifier)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/instantiation)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/mdle)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/modules)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/start_shutdown)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/traversal)

    if(MDL_ENABLE_OPENGL_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/df_native)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/distilling_glsl)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_glsl)
    endif()

    if(MDL_ENABLE_VULKAN_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/df_vulkan)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_glsl_vk)
    endif()

    if(MDL_ENABLE_CUDA_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/distilling_unity)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_cuda)
        if(MDL_ENABLE_OPENGL_EXAMPLES)
            add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/df_cuda)
            if(MDL_ENABLE_OPTIX7_EXAMPLES)
                add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/optix7)
            endif()
        endif()
    endif()

    if(MDL_ENABLE_QT_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/mdl_browser)
    endif()

    if(WINDOWS AND MDL_ENABLE_D3D12_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/dxr)
    endif()
endif()

# MDL CORE
if(MDL_BUILD_CORE_EXAMPLES)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_core/shared)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_core/calls)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_core/code_gen)
    if(MDL_ENABLE_CUDA_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_core/execution_cuda)
        if(MDL_ENABLE_OPENGL_EXAMPLES)
            add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_core/df_cuda)
        endif()
    endif()
endif()

# Example Content
if(MDL_BUILD_SDK_EXAMPLES OR MDL_BUILD_CORE_EXAMPLES)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl)
endif()

# MDL SDK libs/products and plugins that depend on the MDL SDK and mdl_sdk::shared
#--------------------------------------------------------------------------------------------------
# MDL Arnold
if(MDL_BUILD_ARNOLD_PLUGIN)
    add_subdirectory(${MDL_SRC_FOLDER}/prod/lib/mdl_arnold)
endif()
# Language Bindings
if(MDL_ENABLE_PYTHON_BINDINGS)
    add_subdirectory(${MDL_SRC_FOLDER}/prod/bindings/mdl_python)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_python)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_python/create_expression_graph)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_python/modules)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_python/pymdl_inspection)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_python/start_shutdown)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_python/tests)
endif()

# Documentation
#--------------------------------------------------------------------------------------------------
if(MDL_BUILD_DOCUMENTATION)
    add_subdirectory(${MDL_DOC_FOLDER}/mdl_coreapi ${CMAKE_BINARY_DIR}/doc/mdl_coreapi)
    add_subdirectory(${MDL_DOC_FOLDER}/mdl_sdkapi ${CMAKE_BINARY_DIR}/doc/mdl_sdkapi)
    add_custom_target(doc ALL DEPENDS doc-mdl_coreapi doc-mdl_sdkapi)
    set_target_properties(doc PROPERTIES PROJECT_LABEL "ALL" FOLDER "doc")
    install(DIRECTORY ${MDL_DOC_FOLDER}/mdl_coreapi/html DESTINATION doc/mdl_coreapi)
    install(DIRECTORY ${MDL_DOC_FOLDER}/mdl_sdkapi/html DESTINATION doc/mdl_sdkapi)
endif()

install(FILES ${MDL_DOC_FOLDER}/index.html DESTINATION doc)
foreach(_DIRECTORY base_module core_definitions css images js specification)
    install(DIRECTORY ${MDL_DOC_FOLDER}/${_DIRECTORY} DESTINATION doc)
endforeach()


# add tests with the POST option; there require all regular targets to be defined already
foreach(_TEST_POST ${MDL_TEST_LIST_POST})
    add_subdirectory(${_TEST_POST})
endforeach()

# Config
#--------------------------------------------------------------------------------------------------
set(mdl_config_cmake_in_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(_find_neuray_version 1)


set(PATH_EXPORT_TARGETS "${CMAKE_INSTALL_DATADIR}/mdl/mdl-targets.cmake")
configure_package_config_file(
    "${mdl_config_cmake_in_dir}/mdl-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mdl-config.cmake"
    PATH_VARS
        PATH_EXPORT_TARGETS
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/mdl
)

if(_find_neuray_version)
   file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/mi/neuraylib/version.h" neuraylib_product_version_line REGEX "^#define MI_NEURAYLIB_PRODUCT_VERSION_STRING ")

   string(REGEX MATCHALL [[[0-9]+(\.[0-9]+)+|trunk]] neuraylib_product_version "${neuraylib_product_version_line}")

   if(NOT neuraylib_product_version)
       message(FATAL_ERROR "Unable to extract MDL-SDK product version from header.")
   endif()
endif()

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/mdl-config-version.cmake"
    VERSION ${neuraylib_product_version}
    COMPATIBILITY SameMinorVersion
)

# Install
#--------------------------------------------------------------------------------------------------

install(
    EXPORT       mdl-targets
    NAMESPACE    "mdl::"
    DESTINATION  "${CMAKE_INSTALL_DATADIR}/mdl"
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/mdl-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/mdl-config-version.cmake
    DESTINATION  "${CMAKE_INSTALL_DATADIR}/mdl"
)
