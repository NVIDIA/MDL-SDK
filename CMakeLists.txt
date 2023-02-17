#*****************************************************************************
# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

project(MDL LANGUAGES C CXX)


# MDL SDK build directories
set(MDL_BASE_FOLDER ${CMAKE_SOURCE_DIR} CACHE PATH "The folder that contains the targets to build and the cmake utility scripts.")
set(MDL_INCLUDE_FOLDER ${CMAKE_SOURCE_DIR}/include CACHE PATH "The folder that interface headers, i.e., mi/base, mi/neuray, ... directories.")
set(MDL_SRC_FOLDER ${CMAKE_SOURCE_DIR}/src CACHE PATH "The folder that contains the sources, i.e., the base, io, mdl, ... directories.")
set(MDL_EXAMPLES_FOLDER ${CMAKE_SOURCE_DIR}/examples CACHE PATH "The folder that contains the mdl examples")

#--------------------------------------------------------------------------------------------------
# configuration options
option(MDL_BUILD_SDK_EXAMPLES "Adds MDL SDK examples to the build." ON)
option(MDL_BUILD_CORE_EXAMPLES "Adds MDL Core examples to the build." ON)
option(MDL_BUILD_ARNOLD_PLUGIN "Enable the build of the MDL Arnold plugin." OFF)
option(MDL_BUILD_DDS_PLUGIN "Enable the build of the MDL DDS image plugin." ON)
option(MDL_BUILD_FREEIMAGE_PLUGIN "Enable the build of the MDL Freeimage image plugin." ON)
option(MDL_BUILD_OPENIMAGEIO_PLUGIN "Enable the build of the MDL OpenImageIO image plugin." OFF)
option(MDL_LOG_PLATFORM_INFOS "Prints some infos about the current build system (relevant for error reports)." ON)
option(MDL_LOG_DEPENDENCIES "Prints the list of dependencies during the generation step." OFF)
option(MDL_LOG_FILE_DEPENDENCIES "Prints the list of files that is copied after a successful build." OFF)
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

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_BUILD_SDK_EXAMPLES:             " ${MDL_BUILD_SDK_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_BUILD_CORE_EXAMPLES:            " ${MDL_BUILD_CORE_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_BUILD_ARNOLD_PLUGIN:            " ${MDL_BUILD_ARNOLD_PLUGIN})
endif()

# enable tests if available
if(MDL_ENABLE_TESTS)
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

# MDL SDK and MDL CORE
#--------------------------------------------------------------------------------------------------

# first thing to build
add_subdirectory(${MDL_SRC_FOLDER}/base/system/main)
add_subdirectory(${MDL_SRC_FOLDER}/base/system/stlext)
add_subdirectory(${MDL_SRC_FOLDER}/base/system/version)

# third party
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/libzip)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/zlib)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/tinyxml2)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/coco)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/llvm)

# libs
add_subdirectory(${MDL_SRC_FOLDER}/base/util/string_utils)
add_subdirectory(${MDL_SRC_FOLDER}/base/util/registry)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/disk)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/hal)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/link)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/time)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/config)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/cont)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/log)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/mem)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/path)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/plug)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/attr)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/db)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/dblight)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/serial)
add_subdirectory(${MDL_SRC_FOLDER}/io/image)
add_subdirectory(${MDL_SRC_FOLDER}/io/scene)
add_subdirectory(${MDL_SRC_FOLDER}/api/api/mdl)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/codegenerators/generator_code)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/codegenerators/generator_dag)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/compilercore)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/compiler_glsl)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/compiler/compiler_hlsl)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/libbsdf)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/libmdlrt)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/devlib)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/jit/generator_jit)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/no_jit/generator_stub)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/runtime)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/integration/i18n)
add_subdirectory(${MDL_SRC_FOLDER}/mdl/integration/mdlnr)
add_subdirectory(${MDL_SRC_FOLDER}/render/mdl/backends)
add_subdirectory(${MDL_SRC_FOLDER}/render/mdl/runtime)

# libs/products
add_subdirectory(${MDL_SRC_FOLDER}/prod/lib/mdl_core)
add_subdirectory(${MDL_SRC_FOLDER}/prod/lib/mdl_sdk)
add_subdirectory(${MDL_SRC_FOLDER}/prod/bin/i18n)
add_subdirectory(${MDL_SRC_FOLDER}/prod/bin/mdlc)
add_subdirectory(${MDL_SRC_FOLDER}/prod/bin/mdlm)

# PLUGINS
#--------------------------------------------------------------------------------------------------

if(MDL_BUILD_DDS_PLUGIN)
    add_subdirectory(${MDL_SRC_FOLDER}/shaders/plugin/dds)
endif()
if(MDL_BUILD_FREEIMAGE_PLUGIN)
    add_subdirectory(${MDL_SRC_FOLDER}/shaders/plugin/freeimage)
endif()
if(MDL_BUILD_OPENIMAGEIO_PLUGIN)
    add_subdirectory(${MDL_SRC_FOLDER}/shaders/plugin/openimageio)
endif()

# EXAMPLES
#--------------------------------------------------------------------------------------------------

# third party
add_subdirectory(${MDL_EXAMPLES_FOLDER}/thirdparty/imgui)
add_subdirectory(${MDL_EXAMPLES_FOLDER}/thirdparty/tinyxml2)

# Shared code used by MDL SDK examples, Python bindings, and other tools
add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/shared)

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
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_native)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/generate_mdl_identifier)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/instantiation)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/mdle)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/modules)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/start_shutdown)
    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/traversal)

    if(MDL_ENABLE_OPENGL_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/df_native)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_glsl)
    endif()

    if(MDL_ENABLE_VULKAN_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/df_vulkan)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_glsl_vk)
    endif()

    if(MDL_ENABLE_CUDA_EXAMPLES)
        add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/execution_cuda)
        if(MDL_ENABLE_OPENGL_EXAMPLES)
            add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/df_cuda)
            if(MDL_ENABLE_OPTIX7_EXAMPLES)
                if(WINDOWS)
                    add_subdirectory(${MDL_EXAMPLES_FOLDER}/mdl_sdk/optix7)
                else()
                    message(WARNING "The OptiX 7 example is currently only available for Windows.")
                endif()
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

# MDL SDK libs/products and
# PLUGINS FOR EXTERNAL APPLICATIONS that use the MDL SDK and mdl_sdk::shared
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
endif()

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
    COMPATIBILITY SameMajorVersion
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
