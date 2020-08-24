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

# name of the target and the resulting example
set(PROJECT_NAME examples-mdl_sdk-mdl_browser)

# collect resources
set(PROJECT_RESOURCES
    "resources/Main.qml"

    "resources/embedded_resources.qrc"
    )

# collect sources
set(PROJECT_SOURCES
    "mdl_browser_command_line_options.h"
    "mdl_browser.cpp"
    "mdl_sdk_helper.h"

    ${PROJECT_RESOURCES}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk
    OUTPUT_NAME "mdl_browser"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
)

# additional definitions
target_compile_definitions(${PROJECT_NAME} 
    PRIVATE 
        "QT_QML_DEBUG"
    )

# add plugin dependencies
target_add_qt_plugin_dependency(TARGET ${PROJECT_NAME}
    PLUGIN mdl_qtplugin
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
        tinyxml2
    )

target_add_dependencies(TARGET ${PROJECT_NAME} DEPENDS qt 
    COMPONENTS 
        Core Gui Quick Svg Xml Qml QuickControls2 Network Widgets
    )

# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})


# -------------------------------------------------------------------------------------------------
# Generate Qt moc and embedded resources
# -------------------------------------------------------------------------------------------------

# Files to generate
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)


# embed resources into a source file
qt_gen_embedded_resources(
    INPUT       "resources/embedded_resources.qrc"
    DEPENDS     ${PROJECT_RESOURCES}
    OUTPUT      EMBEDDED_RESOUCES
    DESTINATION ${GENERATED_DIR}
    )

# add generated sources to the project
target_sources(${PROJECT_NAME}
    PRIVATE
        ${EMBEDDED_RESOUCES}
    )
