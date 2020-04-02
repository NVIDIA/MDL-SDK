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

function(FIND_QT_EXT)

    set(Qt5_DIR "" CACHE PATH "Directory that contains Qt for the selected compiler, e.g., ../Qt/5.10.1/msvc2017_64")

    if(NOT MDL_ENABLE_QT_EXAMPLES)
        message(WARNING "Examples that require Qt are disabled. Enable the option 'MDL_ENABLE_QT_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    # check glibc version on linux
    #-----------------------------------------------------------------------------------------------

    if(LINUX)
        execute_process(COMMAND "ldd" "--version" 
            OUTPUT_VARIABLE 
                _LIBC_VERSION_STRING 
            ERROR_VARIABLE 
                _LIBC_VERSION_STRING
            )

        # parse version number
        STRING(REGEX MATCH "[0-9]+\\.[0-9]+" _LIBC_VERSION_STRING ${_LIBC_VERSION_STRING})

        # check version
        if( ${_LIBC_VERSION_STRING} VERSION_LESS "2.14.0")
            message(WARNING "At least LIBC 2.14 is required but LIBC version ${_LIBC_VERSION_STRING} was found instead. 'MDL_ENABLE_QT_EXAMPLES' will be disabled as the required Qt version will not run on the current system.")
            set(MDL_ENABLE_QT_EXAMPLES OFF CACHE BOOL "Enable examples that require Qt." FORCE)
            return()
        endif()
    endif()

    # probe the core packages
    #-----------------------------------------------------------------------------------------------
    
    # if found, the Qt5_DIR is set to <qt root dir>/lib/cmake/qt5
    find_package(Qt5 COMPONENTS Core HINTS ${Qt5_DIR})

    if(NOT ${Qt5_FOUND})
        message(FATAL_ERROR "The dependency \"qt\" could not be resolved. Install Qt on your system or specify the 'Qt5_DIR' variable.")
    endif()

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] Qt5_DIR:                            ${Qt5_DIR}")
    endif()

endfunction()


# -------------------------------------------------------------------------------------------------
# create moc files manually using the moc tool
#
function(QT_GEN_MOC)
    set(options)
    set(oneValueArgs DESTINATION)
    set(multiValueArgs INPUT OUTPUT)
    cmake_parse_arguments(QT_GEN_MOC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # no-op if automoc is used
    if(CMAKE_AUTOMOC)
        return()
    endif()

    # find the qt moc tool
    find_program(qtmoc_PATH 
    NAMES
        moc
        moc-qt5
    HINTS
        ${Qt5_BASE_DIR}/bin
    )
    if(NOT EXISTS ${qtmoc_PATH})
        message(FATAL_ERROR "The Qt moc-tool required for at lease one target could not be resolved.")
    endif()

    # prepare files
    set(_GENERATED_MOC_FILES "")
    foreach(_SOURCE_FILE ${QT_GEN_MOC_INPUT})
        get_filename_component(_FILE_NAME ${_SOURCE_FILE} NAME_WE)
        set(_MOC_FILE ${QT_GEN_MOC_DESTINATION}/moc_${_FILE_NAME}.cpp)
        list(APPEND _MOC_COMMANDS COMMAND ${qtmoc_PATH} -o ${_MOC_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_FILE})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- generate moc:   moc_${_FILE_NAME}.cpp")
        endif()
        list(APPEND _GENERATED_MOC_FILES ${_MOC_FILE})
    endforeach()

    # absolute path to dependencies
    foreach(_DEP ${QT_GEN_MOC_INPUT})
        list(APPEND _DEPS ${CMAKE_CURRENT_SOURCE_DIR}/${_DEP})
    endforeach()

    # create target dir if not existing
    if(QT_GEN_EMBEDDED_RESOURCES_DESTINATION AND NOT EXISTS ${QT_GEN_EMBEDDED_RESOURCES_DESTINATION})
        file(MAKE_DIRECTORY ${QT_GEN_EMBEDDED_RESOURCES_DESTINATION})
    endif()
 
    # create a custom command to run the tool
    add_custom_command(
        OUTPUT ${_GENERATED_MOC_FILES}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
        ${_MOC_COMMANDS}
        DEPENDS ${_DEPS}
        VERBATIM
        )

    # mark files as generated to disable the check for existence
    set_source_files_properties(${_GENERATED_MOC_FILES} PROPERTIES GENERATED TRUE)
    source_group("generated" FILES ${_GENERATED_MOC_FILES})

    # pass to output variable
    set(${QT_GEN_MOC_OUTPUT} ${_GENERATED_MOC_FILES} PARENT_SCOPE) 

endfunction()

# -------------------------------------------------------------------------------------------------
# function that takes a qt qrc file and creates a cpp file that can be compiled and added to a project.
#
function(QT_GEN_EMBEDDED_RESOURCES)
    set(options)
    set(oneValueArgs INPUT OUTPUT DESTINATION)
    set(multiValueArgs DEPENDS)
    cmake_parse_arguments(QT_GEN_EMBEDDED_RESOURCES "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # no-op if autorcc is used
    if(CMAKE_AUTORCC)
        return()
    endif()

    # find the qt rcc tool
    find_program(qtrcc_PATH 
        NAMES
            rcc
            rcc-qt5
        HINTS
            ${Qt5_BASE_DIR}/bin
        )
    if(NOT EXISTS ${qtrcc_PATH})
        message(FATAL_ERROR "The Qt rcc-tool required for at lease one target could not be resolved.")
    endif()

    # create target dir if not existing
    if(QT_GEN_EMBEDDED_RESOURCES_DESTINATION AND NOT EXISTS ${QT_GEN_EMBEDDED_RESOURCES_DESTINATION})
        file(MAKE_DIRECTORY ${QT_GEN_EMBEDDED_RESOURCES_DESTINATION})
    endif()

    # create a custom command to run the tool
    get_filename_component(_FILE_NAME ${QT_GEN_EMBEDDED_RESOURCES_INPUT} NAME_WE)
    set(_OUTPUT_FILE ${QT_GEN_EMBEDDED_RESOURCES_DESTINATION}/${_FILE_NAME}.cpp)

    # absolute path to dependencies
    list(APPEND _DEPS ${CMAKE_CURRENT_SOURCE_DIR}/${QT_GEN_EMBEDDED_RESOURCES_INPUT})
    foreach(_DEP ${QT_GEN_EMBEDDED_RESOURCES_DEPENDS})
        list(APPEND _DEPS ${CMAKE_CURRENT_SOURCE_DIR}/${_DEP})
        message(STATUS "- depends:        ${CMAKE_CURRENT_SOURCE_DIR}/${_DEP}")
    endforeach()

    add_custom_command(
        OUTPUT ${_OUTPUT_FILE}
        COMMAND ${qtrcc_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/${QT_GEN_EMBEDDED_RESOURCES_INPUT} -o ${_OUTPUT_FILE}
        DEPENDS ${_DEPS}
        VERBATIM
        )

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- embedding:      ${QT_GEN_EMBEDDED_RESOURCES_INPUT}")
    endif()

    # mark files as generated to disable the check for existence
    set_source_files_properties(${_OUTPUT_FILE} PROPERTIES GENERATED TRUE)
    source_group("generated" FILES ${_OUTPUT_FILE})

    # pass filename out
    set(${QT_GEN_EMBEDDED_RESOURCES_OUTPUT} ${_OUTPUT_FILE} PARENT_SCOPE) 

endfunction()

