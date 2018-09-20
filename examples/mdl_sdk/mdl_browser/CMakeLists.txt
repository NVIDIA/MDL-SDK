# name of the target and the resulting example
set(PROJECT_NAME mdl_sdk_example-mdl_browser)

# -------------------------------------------------------------------------------------------------
# Pre-build Step: generate Qt moc
# -------------------------------------------------------------------------------------------------

# generate qt's moc files manually to selected files 
set(PROJECT_SOURCES_TO_MOC
    "mdl_browser_settings.h"
    "view_model/navigation/vm_nav_package.h"
    "view_model/navigation/vm_nav_stack.h"
    "view_model/navigation/vm_nav_stack_level_model.h"
    "view_model/navigation/vm_nav_stack_level_proxy_model.h"
    "view_model/selection/vm_sel_element.h"
    "view_model/selection/vm_sel_model.h"
    "view_model/selection/vm_sel_proxy_model.h"
    "view_model/view_model.h"
    )

# files to generate
if(NOT CMAKE_AUTOMOC)
    set(_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
    foreach(_SOURCE_TO_MOC ${PROJECT_SOURCES_TO_MOC})
        get_filename_component(_FILE_NAME ${_SOURCE_TO_MOC} NAME_WE)
        list(APPEND GENERATED_MOC_FILES ${_GENERATED_DIR}/moc_${_FILE_NAME}.cpp)
    endforeach()
endif()

# mark files as generated to disable the check for existence during configure
set_source_files_properties(${GENERATED_MOC_FILES} PROPERTIES GENERATED TRUE)

# -------------------------------------------------------------------------------------------------
# Main Target 
# -------------------------------------------------------------------------------------------------

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
    "resources/graphics/module_icon.svg"
    "resources/graphics/nvidia_icon.svg"
    "resources/graphics/package_icon.svg"
    "resources/Main.qml"
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
    "mdl_browser_main.cpp"
    "mdl_browser_node.cpp"
    "mdl_browser_node.h"
    "mdl_browser_settings.cpp"
    "mdl_sdk_wrapper.cpp"
    "mdl_sdk_wrapper.h"
    "utilities/application_settings.cpp"
    "utilities/application_settings.h"
    "utilities/application_settings_serializer_xml.cpp"
    "utilities/application_settings_serializer_xml.h"
    "utilities/mdl_helper.cpp"
    "utilities/mdl_helper.h"
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
    ${PROJECT_SOURCES_TO_MOC}
    ${GENERATED_MOC_FILES}
    ${PROJECT_RESOURCES}
)


# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    NAMESPACE mdl_sdk_examples
    SOURCES ${PROJECT_SOURCES}
)

# add dependencies
target_add_dependencies(TARGET ${PROJECT_NAME} 
    DEPENDS
        mdl::mdl_sdk
        mdl_sdk_examples::mdl_sdk_shared
        tinyxml2
        )

target_add_dependencies(TARGET ${PROJECT_NAME} DEPENDS qt 
    COMPONENTS 
        Core Gui Quick Svg Xml Qml QuickControls2 Network Widgets
    )

# create moc files manually using the moc tool
if(NOT CMAKE_AUTOMOC)
    find_program(qtmoc_PATH 
        NAMES
            moc
            moc-qt5
        HINTS
            ${Qt5_BASE_DIR}/bin
        )
    if(NOT EXISTS ${qtmoc_PATH})
        message(FATAL_ERROR "The Qt moc-tool required for target \"${PROJECT_NAME}\" could not be resolved.")
    endif()

    list(LENGTH PROJECT_SOURCES_TO_MOC _LENGTH)
    math(EXPR _LENGTH ${_LENGTH}-1)
    foreach(_INDEX RANGE ${_LENGTH})
        list(GET PROJECT_SOURCES_TO_MOC ${_INDEX} _SOURCE_FILE)
        list(GET GENERATED_MOC_FILES ${_INDEX} _MOC_FILE)
        list(APPEND _MOC_COMMANDS COMMAND ${qtmoc_PATH} -o ${_MOC_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_FILE})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- generate moc:   ${_MOC_FILE}")
        endif()
    endforeach()

    add_custom_command(
        OUTPUT ${GENERATED_MOC_FILES}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_GENERATED_DIR}
        ${_MOC_COMMANDS}
        DEPENDS 
            ${PROJECT_SOURCES_TO_MOC}
        VERBATIM
        )
endif()

# copy additional resources
target_copy_to_output_dir(TARGET ${PROJECT_NAME}
    FILES
        #${PROJECT_RESOURCES} # results in a too long command line for the ninja generator
        "${CMAKE_CURRENT_SOURCE_DIR}/resources"
    )
    
# creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
target_create_vs_user_settings(TARGET ${PROJECT_NAME})
