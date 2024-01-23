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

# name of the target and the resulting library
set(PROJECT_NAME prod-lib-mdl_sdk)

# collect sources
set(PROJECT_SOURCES
    "mdl_sdk_factory.cpp"
    "mdl_sdk_required_modules.cpp"
    "mdl_sdk_stubs.cpp"
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    SOURCES ${PROJECT_SOURCES}
    EMBED_RC "mdl_sdk.rc"
    OUTPUT_NAME "mdl_sdk"
    )

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
)

# customize name
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "lib")

if(MACOSX)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so") # corresponding to the binary release
endif()

# target alias for the custom name
add_library(mdl_sdk ALIAS ${PROJECT_NAME})
add_library(mdl::mdl_sdk ALIAS ${PROJECT_NAME})

# add mdl and other dependencies
#set(_LINKER_DEPENDENCIES_BASE_LIST ${_LINKER_DEPENDENCIES_BASE})
#separate_arguments(_LINKER_DEPENDENCIES_BASE_LIST)
#list(TRANSFORM _LINKER_DEPENDENCIES_BASE_LIST STRIP)

target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        boost
        ${LINKER_WHOLE_ARCHIVE}
        ${LINKER_START_GROUP}
        mdl::include-mi
        ${LINKER_DEPENDENCIES_API}
        ${LINKER_DEPENDENCIES_BASE}
        ${LINKER_DEPENDENCIES_IO}
        ${LINKER_DEPENDENCIES_MDL_JIT}
        ${LINKER_DEPENDENCIES_RENDER}
        ${LINKER_END_GROUP}
        llvm
        ${LINKER_NO_WHOLE_ARCHIVE}
    )

# on Linux/MacOS set rpath for plugins
if(NOT WINDOWS)

        target_add_rpath(TARGET ${PROJECT_NAME}
            RPATHS
                ${CMAKE_BINARY_DIR}/src/shaders/plugin/dds/${CMAKE_BUILD_TYPE}
                ${CMAKE_BINARY_DIR}/src/shaders/plugin/openimageio/${CMAKE_BUILD_TYPE}
                ${CMAKE_BINARY_DIR}/src/shaders/plugin/mdl_distiller/${CMAKE_BUILD_TYPE}
                ${CMAKE_BINARY_DIR}/examples/mdl_sdk/distilling_target_plugin/${CMAKE_BUILD_TYPE}
            )

endif()

# -------------------------------------------------------------------------------------------------
# Create installation rules to copy the build directory
# -------------------------------------------------------------------------------------------------
add_target_install(
    TARGET ${PROJECT_NAME}
    )

# add unit tests
add_unit_tests(POST)
