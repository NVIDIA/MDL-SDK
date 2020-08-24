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
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# assuming the find_qt_ext script was successful
# if not, this is an error case. The corresponding project should not have been selected for build.
if(NOT MDL_ENABLE_QT_EXAMPLES)
    message(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
else()
    # get the base dir variable
    string(LENGTH ${Qt5_DIR} _BASE_DIR_LENGTH)
    math(EXPR _BASE_DIR_LENGTH "${_BASE_DIR_LENGTH}-14") 
    string(SUBSTRING ${Qt5_DIR} 0 ${_BASE_DIR_LENGTH} Qt5_BASE_DIR)
    set(Qt5_BASE_DIR ${Qt5_BASE_DIR} CACHE INTERNAL "qt root directory for the current platform. This directory contains the bin directory for example.") 

    # options depending on the target type
    get_target_property(_TARGET_TYPE ${__TARGET_ADD_DEPENDENCY_TARGET} TYPE)

    # find the required packages
    find_package(Qt5 COMPONENTS ${__TARGET_ADD_DEPENDENCY_COMPONENTS} REQUIRED)

    foreach (qt_component ${__TARGET_ADD_DEPENDENCY_COMPONENTS})

        # add include directories
        target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
            PRIVATE
                $<TARGET_PROPERTY:Qt5::${qt_component},INCLUDE_DIRECTORIES>
            )

        # link dependencies
        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
            PRIVATE
                Qt5::${qt_component}
            )

        # copy runtime dependencies
        if (WINDOWS AND _TARGET_TYPE STREQUAL "EXECUTABLE")

            # we assume that qt is not installed locally but available, e.g., on a network drive
            target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
                RELATIVE  ${Qt5_BASE_DIR}/bin
                FILES     Qt5${qt_component}$<$<CONFIG:DEBUG>:d>.dll)

            # copy opengl es
            target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
                RELATIVE  ${Qt5_BASE_DIR}/bin
                FILES     libEGL$<$<CONFIG:DEBUG>:d>.dll
                          libGLESv2$<$<CONFIG:DEBUG>:d>.dll)

            # collect plugins and other libraries that are required
            if(${qt_component} STREQUAL Svg)
                target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
                    RELATIVE    ${Qt5_BASE_DIR}/plugins
                    FILES       imageformats/qsvg$<$<CONFIG:DEBUG>:d>.dll)

            elseif(${qt_component} STREQUAL QuickControls2)
                target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
                    RELATIVE    ${Qt5_BASE_DIR}/bin
                    FILES       Qt5QuickTemplates2$<$<CONFIG:DEBUG>:d>.dll)
            endif()
        endif()

        #foreach(plugin ${Qt5${qt_component}_PLUGINS})
        #    get_target_property(_loc ${plugin} LOCATION)
        #    message("${qt_component}-Plugin ${plugin} is at location ${_loc}")
        #endforeach()

    endforeach()

    # add platform dependencies
    if(LINUX)

        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
            PRIVATE
                ${LINKER_NO_AS_NEEDED}
                ${Qt5_BASE_DIR}/plugins/platforms/libqxcb${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/plugins/imageformats/libqsvg${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/plugins/xcbglintegrations/libqxcb-egl-integration${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/plugins/xcbglintegrations/libqxcb-glx-integration${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/plugins/egldeviceintegrations/libqeglfs-x11-integration${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/lib/libQt5XcbQpa${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/lib/libQt5DBus${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/lib/libQt5QuickTemplates2${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/lib/libicuuc${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/lib/libicui18n${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${Qt5_BASE_DIR}/lib/libicudata${CMAKE_SHARED_LIBRARY_SUFFIX}
                ${LINKER_AS_NEEDED}
            )

        if (_TARGET_TYPE STREQUAL "EXECUTABLE")
            target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
                FILES
                    "${Qt5_BASE_DIR}/plugins/xcbglintegrations"
                    "${Qt5_BASE_DIR}/plugins/egldeviceintegrations"
                )
        endif()

    elseif(MACOSX)
        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
            PRIVATE
                ${Qt5_BASE_DIR}/plugins/imageformats/libqsvg${CMAKE_SHARED_LIBRARY_SUFFIX}
            )
    endif()

    if (_TARGET_TYPE STREQUAL "EXECUTABLE")
        # copy qml dependencies which are not found if qt is not installed locally
        target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
            FILES
                "${Qt5_BASE_DIR}/qml/QtGraphicalEffects"
                "${Qt5_BASE_DIR}/qml/QtQuick"
                "${Qt5_BASE_DIR}/qml/QtQuick.2"
                "${Qt5_BASE_DIR}/plugins/platforms"
                "${Qt5_BASE_DIR}/plugins/imageformats"
            )
    endif()
endif()
