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
