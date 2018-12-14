
# -------------------------------------------------------------------------------------------------
# function that takes a qt qrc file and creates a cpp file that can be compiled and added to a project.
#
function(TARGET_ADD_QT_PLUGIN_DEPENDENCY)
    set(options)
    set(oneValueArgs TARGET PLUGIN)
    set(multiValueArgs)
    cmake_parse_arguments(TARGET_ADD_QT_PLUGIN_DEPENDENCY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # build order
    add_dependencies(${TARGET_ADD_QT_PLUGIN_DEPENDENCY_TARGET} ${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN})

    # dependency file
    set(_DEP ${CMAKE_CURRENT_BINARY_DIR}/depends/${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN}.d)
    source_group("generated" FILES ${_DEP})

    # add interface headers
    target_include_directories(${TARGET_ADD_QT_PLUGIN_DEPENDENCY_TARGET}
        PRIVATE
            $<TARGET_PROPERTY:${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN},INTERFACE_INCLUDE_DIRECTORIES>
        )

    # copy command
    get_filename_component(MODULE_NAME ${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN} NAME)
    add_custom_command(
        OUTPUT ${_DEP}
        COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN}>/.. $<TARGET_FILE_DIR:${TARGET_ADD_QT_PLUGIN_DEPENDENCY_TARGET}>
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/depends
        COMMAND ${CMAKE_COMMAND} -E touch ${_DEP}
        COMMAND ${CMAKE_COMMAND} -E echo "copy $<TARGET_FILE_DIR:${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN}> to $<TARGET_FILE_DIR:${TARGET_ADD_QT_PLUGIN_DEPENDENCY_TARGET}>"
        DEPENDS $<TARGET_FILE:${TARGET_ADD_QT_PLUGIN_DEPENDENCY_PLUGIN}>
        VERBATIM
        )

    # add dependency file to target
    target_sources(${TARGET_ADD_QT_PLUGIN_DEPENDENCY_TARGET}
        PRIVATE ${_DEP}
        )

endfunction()

# -------------------------------------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------------------------------------

# qt plugin
add_subdirectory(mdl_qt_plugin)

# applications
add_subdirectory(mdl_browser)

# add tests if available
add_tests()
