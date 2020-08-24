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
set(PROJECT_NAME examples-mdl_sdk-mdl_qt_plugin)
set(PLUGIN_NAME "MdlQtPlugin")


# generate qt's moc files manually to selected files 
set(PROJECT_SOURCES_TO_MOC
    "mdl_browser_settings.h"
    "plugin.h"
    "view_model/navigation/vm_nav_package.h"
    "view_model/navigation/vm_nav_stack.h"
    "view_model/navigation/vm_nav_stack_level_model.h"
    "view_model/navigation/vm_nav_stack_level_proxy_model.h"
    "view_model/selection/vm_sel_element.h"
    "view_model/selection/vm_sel_model.h"
    "view_model/selection/vm_sel_proxy_model.h"
    "view_model/view_model.h"
    )

# collect resources
set(PROJECT_RESOURCES
    "resources/dummydata/nav_breadcrumbs_mockup.qml"
    "resources/dummydata/nav_node_mockup.qml"
    "resources/dummydata/nav_stack_level_mockup.qml"
    "resources/dummydata/nav_stack_mockup.qml"
    "resources/dummydata/sel_element_mockup.qml"
    "resources/dummydata/sel_mockup.qml"
    "resources/graphics/clear_icon.svg"
    "resources/graphics/default_thumbnail_material.svg"
    "resources/graphics/grid_icon.svg"
    "resources/graphics/home_icon.svg"
    "resources/graphics/info_icon.svg"
    "resources/graphics/list_icon.svg"
    "resources/graphics/mdl_icon.svg"
    "resources/graphics/module_icon.svg"
    "resources/graphics/package_icon.svg"
    "resources/navigation/NavBreadcrumbs.qml"
    "resources/navigation/NavPackage.qml"
    "resources/navigation/NavStack.qml"
    "resources/navigation/NavStackLevelSeperator.qml"
    "resources/search/SearchBar.qml"
    "resources/search/SelectionViewOption.qml"
    "resources/search/SortingCriterion.qml"
    "resources/search/StatusBar.qml"
    "resources/selection/SelGrid.qml"
    "resources/selection/SelGridItem.qml"
    "resources/selection/SelItemBase.qml"
    "resources/selection/SelList.qml"
    "resources/selection/SelListItem.qml"
    "resources/side_panels/SidePanel.qml"
    "resources/side_panels/SidePanelTab.qml"
    "resources/side_panels/SidePanelTabDelegate.qml"
    "resources/side_panels/SidePanelView.qml"
    "resources/utils/Icon.qml"
    "resources/utils/ToolTipExtended.qml"

    "resources/BrowserApp.qml"
    "resources/BrowserDialog.qml"
    "resources/BrowserMain.qml"

    "resources/embedded_resources.qrc"
    "resources/qmldir"
    "metadata.json"
)

# public interface
set(PROJECT_PUBLIC_HEADERS
    "include/mdl_qt_plugin.h"
    )

# collect sources
set(PROJECT_SOURCES
    "cache/imdl_cache.h"
    "cache/mdl_cache.cpp"
    "cache/mdl_cache.h"
    "cache/mdl_cache_function.cpp"
    "cache/mdl_cache_function.h"
    "cache/mdl_cache_impl.cpp"
    "cache/mdl_cache_impl.h"
    "cache/mdl_cache_material.cpp"
    "cache/mdl_cache_material.h"
    "cache/mdl_cache_module.cpp"
    "cache/mdl_cache_module.h"
    "cache/mdl_cache_package.cpp"
    "cache/mdl_cache_package.h"
    "cache/mdl_cache_serializer_xml_impl.cpp"
    "cache/mdl_cache_serializer_xml_impl.h"
    "index/index.h"
    "index/index_cache_elements.cpp"
    "index/index_cache_elements.h"
    "index/index_document.cpp"
    "index/index_document.h"
    "index/index_document_cache_element.cpp"
    "index/index_document_cache_element.h"
    "index/tokenizer.cpp"
    "index/tokenizer.h"
    "mdl_browser_node.cpp"
    "mdl_browser_node.h"
    "mdl_browser_settings.cpp"
    "plugin.cpp"
    "utilities/application_settings.cpp"
    "utilities/application_settings.h"
    "utilities/application_settings_serializer_xml.cpp"
    "utilities/application_settings_serializer_xml.h"
    "utilities/platform_helper.cpp"
    "utilities/platform_helper.h"
    "utilities/qt/mdl_archive_image_provider.cpp"
    "utilities/qt/mdl_archive_image_provider.h"
    "utilities/string_helper.cpp"
    "utilities/string_helper.h"
    "utilities/xml_helper.cpp"
    "utilities/xml_helper.h"
    "view_model/navigation/vm_nav_package.cpp"
    "view_model/navigation/vm_nav_stack.cpp"
    "view_model/navigation/vm_nav_stack_level_model.cpp"
    "view_model/navigation/vm_nav_stack_level_proxy_model.cpp"
    "view_model/selection/selection_filter.cpp"
    "view_model/selection/selection_filter.h"
    "view_model/selection/selection_sorter.cpp"
    "view_model/selection/selection_sorter.h"
    "view_model/selection/vm_sel_element.cpp"
    "view_model/selection/vm_sel_model.cpp"
    "view_model/selection/vm_sel_proxy_model.cpp"
    "view_model/view_model.cpp"

    ${PROJECT_PUBLIC_HEADERS}
    ${PROJECT_SOURCES_TO_MOC}
    ${PROJECT_RESOURCES}
)

# mark all non cpp files as header so that they are not compiled
set_source_files_properties(
        ${PROJECT_PUBLIC_HEADERS}
        ${PROJECT_SOURCES_TO_MOC}
        ${PROJECT_RESOURCES}
    PROPERTIES 
        HEADER_FILE_ONLY ON
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE SHARED
    NAMESPACE mdl_sdk
    OUTPUT_NAME "mdl_qt_plugin"
    SOURCES ${PROJECT_SOURCES}
    EXAMPLE
    )
add_library(mdl_qtplugin ALIAS ${PROJECT_NAME})
add_library(mdl_sdk::mdl_qtplugin ALIAS ${PROJECT_NAME})

# create an output folder that looks like a qt conform module 
set_target_properties(${PROJECT_NAME} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/Debug/${PLUGIN_NAME}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/Release/${PLUGIN_NAME}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_BINARY_DIR}/MinSizeRel/${PLUGIN_NAME}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/${PLUGIN_NAME}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/Debug/${PLUGIN_NAME}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/Release/${PLUGIN_NAME}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_BINARY_DIR}/MinSizeRel/${PLUGIN_NAME}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/${PLUGIN_NAME}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/Debug/${PLUGIN_NAME}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/Release/${PLUGIN_NAME}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_BINARY_DIR}/MinSizeRel/${PLUGIN_NAME}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/${PLUGIN_NAME}
    )

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} DEPENDS qt 
    COMPONENTS 
        Core Gui Quick Svg Xml Qml QuickControls2 Network Widgets
    )

target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk::shared
        tinyxml2
        )

# define interface
target_include_directories(${PROJECT_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )


# -------------------------------------------------------------------------------------------------
# Generate Qt moc and embedded resources
# -------------------------------------------------------------------------------------------------

# Files to generate
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

# generate moc files
qt_gen_moc(
    INPUT       ${PROJECT_SOURCES_TO_MOC}
    OUTPUT      GENERATED_MOC_FILES
    DESTINATION ${GENERATED_DIR}
    )

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
        ${GENERATED_MOC_FILES}
        ${EMBEDDED_RESOUCES}
    )

# copy the module definition into the output dir
target_copy_to_output_dir(TARGET ${PROJECT_NAME}
    FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/qmldir"
    )
