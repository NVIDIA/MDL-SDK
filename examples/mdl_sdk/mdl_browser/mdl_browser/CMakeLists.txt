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
