#*****************************************************************************
# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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

set(LINKER_START_GROUP      "$<$<CXX_COMPILER_ID:GNU>:-Wl,--start-group>")
set(LINKER_END_GROUP        "$<$<CXX_COMPILER_ID:GNU>:-Wl,--end-group>")
set(LINKER_WHOLE_ARCHIVE    "$<$<CXX_COMPILER_ID:GNU>:-Wl,--whole-archive>")
set(LINKER_NO_WHOLE_ARCHIVE "$<$<CXX_COMPILER_ID:GNU>:-Wl,--no-whole-archive>")
set(LINKER_AS_NEEDED        "$<$<CXX_COMPILER_ID:GNU>:-Wl,--as-needed>")
set(LINKER_NO_AS_NEEDED     "$<$<CXX_COMPILER_ID:GNU>:-Wl,--no-as-needed>")

# -------------------------------------------------------------------------------------------------
# setup the compiler options and definitions.
# very simple set of flags depending on the compiler instead of the combination of compiler, OS, ...
# for more complex scenarios, replace this function by tool-chain files for instance
#
#   target_build_setup(TARGET <NAME>)
#
function(TARGET_BUILD_SETUP)
    set(options DYNAMIC_MSVC_RUNTIME WINDOWS_UNICODE)
    set(oneValueArgs TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(TARGET_BUILD_SETUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # options depending on the target type
    get_target_property(_TARGET_TYPE ${TARGET_BUILD_SETUP_TARGET} TYPE)

    # very simple set of flags depending on the compiler instead of the combination of compiler, OS, ...
    # for more complex scenarios, replace that

    if(WINDOWS AND TARGET_BUILD_SETUP_WINDOWS_UNICODE)
        set(_ADDITIONAL_COMPILER_DEFINES
            "_UNICODE"
            "UNICODE"
        )
        if(MDL_LOG_FILE_DEPENDENCIES)
            MESSAGE(STATUS "- add defines:    '_UNICODE' and 'UNICODE'")
        endif()
    endif()

    # GENERAL
    #---------------------------------------------------------------------------------------

    # specify the c++ standard to use
    set_property(TARGET ${TARGET_BUILD_SETUP_TARGET} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${TARGET_BUILD_SETUP_TARGET} PROPERTY C_STANDARD 11)

    target_compile_definitions(${TARGET_BUILD_SETUP_TARGET}
        PRIVATE
            "$<$<CONFIG:DEBUG>:DEBUG>"
            "$<$<CONFIG:DEBUG>:_DEBUG>"
            "BIT64=1"
            "X86=1"
            ${_ADDITIONAL_COMPILER_DEFINES}      # additional build defines
            ${MDL_ADDITIONAL_COMPILER_DEFINES}   # additional user defines
        )


    target_compile_options(${TARGET_BUILD_SETUP_TARGET}
        PRIVATE
            ${MDL_ADDITIONAL_COMPILER_OPTIONS}   # additional user options
        )

    # WINDOWS
    #---------------------------------------------------------------------------------------
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")

        target_compile_options(${TARGET_BUILD_SETUP_TARGET}
        PRIVATE
            "/Zc:__cplusplus"                    # make visual studio report the correct __cplusplus
        )

        target_compile_definitions(${TARGET_BUILD_SETUP_TARGET}
            PUBLIC
                "MI_PLATFORM_WINDOWS"
                "WIN_NT"
            PRIVATE
                "_MSC_VER=${MSVC_VERSION}"
                "_CRT_SECURE_NO_WARNINGS"
                "_SCL_SECURE_NO_WARNINGS"
            )

        # set static or dynamic runtime
        # cmake 3.15 has an option for that. MSVC_RUNTIME_LIBRARY
        if(TARGET_BUILD_SETUP_DYNAMIC_MSVC_RUNTIME)
            # examples can use the dynamic runtime
            target_compile_options(${TARGET_BUILD_SETUP_TARGET} PRIVATE "/MD$<$<CONFIG:Debug>:d>")
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- msvc runtime:   dynamic (/MD or /MDd)")
            endif()
        else()
            # a libraries 'in' the sdk use the static runtime
            target_compile_options(${TARGET_BUILD_SETUP_TARGET} PRIVATE "/MT$<$<CONFIG:Debug>:d>")
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- msvc runtime:   static (/MT or /MTd)")
            endif()
        endif()

        target_compile_options(${TARGET_BUILD_SETUP_TARGET}
            PRIVATE
                "/MP"
                "/wd4267"   # Suppress Warning C4267 'argument': conversion from 'size_t' to 'int', possible loss of data
            )

        if(MSVC_VERSION GREATER 1900)
            target_compile_options(${TARGET_BUILD_SETUP_TARGET}
                PRIVATE
                    "/permissive-"  # show more errors
                )
        endif()
    endif()

    # LINUX
    #---------------------------------------------------------------------------------------
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_definitions(${TARGET_BUILD_SETUP_TARGET}
            PUBLIC
                "MI_PLATFORM_UNIX"
                "$<$<STREQUAL:${MI_PLATFORM_NAME},linux-aarch64>:AARCH64>"
                "$<$<STREQUAL:${MI_PLATFORM_NAME},linux-x86-64>:HAS_SSE>"
                "LINUX"
            )

        target_compile_options(${TARGET_BUILD_SETUP_TARGET}
            PRIVATE
                "-fPIC"   # position independent code since we will build a shared object
                "-fno-strict-aliasing"
                "$<$<STREQUAL:${MI_PLATFORM_NAME},linux-x86-64>:-march=nocona>"

                # enable additional warnings
                "-Wall"
                "-Wvla"

                "$<$<COMPILE_LANGUAGE:CXX>:-Wno-init-list-lifetime>"
                "$<$<COMPILE_LANGUAGE:CXX>:-Wno-placement-new>"
                "-Wno-parentheses"
                "-Wno-sign-compare"
                "-Wno-narrowing"
                "-Wno-unused-but-set-variable"
                "-Wno-unused-local-typedefs"
                "-Wno-deprecated-declarations"
                "-Wno-unknown-pragmas"
            )
        target_link_libraries(${TARGET_BUILD_SETUP_TARGET}
            PRIVATE
                "-static-libstdc++"
                "-static-libgcc"
            )
    endif()

    # MACOSX
    #---------------------------------------------------------------------------------------
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_definitions(${TARGET_BUILD_SETUP_TARGET}
            PUBLIC
                "MI_PLATFORM_MACOSX"
                "$<$<STREQUAL:${MI_PLATFORM_NAME},macosx-aarch64>:AARCH64>"
                "MACOSX"
            )

        target_compile_options(${TARGET_BUILD_SETUP_TARGET}
            PRIVATE
                "-fPIC"
                "-m64"
                "-stdlib=libc++"
                "$<$<CONFIG:DEBUG>:-gdwarf-2>"
                "-fvisibility-inlines-hidden"
                "-fdiagnostics-fixit-info"
                "-fdiagnostics-parseable-fixits"
                "-Wno-unused-parameter"
                "-Wno-inconsistent-missing-override"
                "-Wno-unnamed-type-template-args"
                "-Wno-invalid-offsetof"
                "-Wno-long-long"
                "-Wwrite-strings"
                "-Wmissing-field-initializers"
                "-Wno-covered-switch-default"
                "-Wno-non-virtual-dtor"
                "-Wno-unusable-partial-specialization"
            )

        # set the library paths
        set(CMAKE_SKIP_BUILD_RPATH FALSE)
        set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

        set_target_properties(${TARGET_BUILD_SETUP_TARGET} PROPERTIES
            MACOSX_RPATH TRUE
            )
    endif()

    # setup specific to shared libraries
    if (_TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR _TARGET_TYPE STREQUAL "MODULE_LIBRARY")
        target_compile_definitions(${TARGET_BUILD_SETUP_TARGET}
            PRIVATE
                "MI_DLL_BUILD"            # export/import macro
            )
    endif()
endfunction()

# -------------------------------------------------------------------------------------------------
# setup IDE specific stuff

function(SETUP_IDE_FOLDERS)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs FILES)
    cmake_parse_arguments(SETUP_IDE_FOLDERS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - SETUP_IDE_FOLDERS_FILES

    # keep the folder structure in visual studio
    foreach(_SOURCE ${SETUP_IDE_FOLDERS_FILES})
        string(FIND ${_SOURCE} "/" _POS REVERSE)

        # file in project root
        if(${_POS} EQUAL -1)
            source_group("" FILES ${_SOURCE})
            continue()
        endif()

        # generated files
        math(EXPR _START ${_POS}-9)
        if(${_START} GREATER 0)
            string(SUBSTRING ${_SOURCE} ${_START} 9 FOLDER_PATH)
            if(FOLDER_PATH STREQUAL "generated")
                source_group("generated" FILES ${_SOURCE})
                continue()
            endif()
        endif()

        # relative files outside the current target
        string(SUBSTRING ${_SOURCE} 0 3 _SOURCE_FIRST_3_CHARS)
        if(_SOURCE_FIRST_3_CHARS STREQUAL "../")
            get_filename_component(_FOLDER ${_SOURCE} DIRECTORY)
            string(REPLACE "/" "\\" _FOLDER ${_FOLDER})
            string(REPLACE "..\\" "" _FOLDER ${_FOLDER}) # drop leading points (could be replaced by "../", too)
            source_group(${_FOLDER} FILES ${_SOURCE})
            continue()
        endif()

        # absolute files (probably outside the current target)
        if(IS_ABSOLUTE ${_SOURCE})
            source_group("" FILES ${_SOURCE})
            continue()
        endif()

        # files in folders
        source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${_SOURCE})
    endforeach()
endfunction()


function(SETUP_IDE)
    set(options)
    set(oneValueArgs TARGET VS_PROJECT_NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(SETUP_IDE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - SETUP_IDE_TARGET
    # - SETUP_IDE_SOURCES
    # - SETUP_IDE_VS_PROJECT_NAME

    # not required without visual studio or xcode
    if(NOT MSVC AND NOT MSVC_IDE)
        return()
    endif()

    # compute the folder relative to the top level and use it as project folder hierarchy in the IDEs
    get_filename_component(FOLDER_PREFIX ${CMAKE_SOURCE_DIR} REALPATH)
    get_filename_component(FOLDER_PATH ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)
    # remove potential trailing slash, e.g., if CMAKE_SOURCE_DIR refers to a drive letter
    string(REGEX REPLACE "/$" "" FOLDER_PREFIX ${FOLDER_PREFIX})
    string(LENGTH ${FOLDER_PREFIX} OFFSET)
    string(LENGTH ${FOLDER_PATH} TOTAL_LENGTH)
    math(EXPR OFFSET ${OFFSET}+1)
    math(EXPR REMAINING ${TOTAL_LENGTH}-${OFFSET})
    string(SUBSTRING ${FOLDER_PATH} ${OFFSET} ${REMAINING} FOLDER_PATH)

    get_filename_component(FOLDER_NAME ${FOLDER_PATH} NAME)         # last folder is used as IDE project label
    get_filename_component(FOLDER_PATH ${FOLDER_PATH} PATH)         # drop the last folder (equals the IDE project label)

    if (SETUP_IDE_VS_PROJECT_NAME AND NOT (SETUP_IDE_VS_PROJECT_NAME STREQUAL ""))
        set(FOLDER_NAME ${SETUP_IDE_VS_PROJECT_NAME})
    endif()

    # Move projects with tests one level down such that the project and the tests end up in the
    # same folder.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        set(FOLDER_PATH ${FOLDER_PATH}/${FOLDER_NAME})
    endif()

    set_target_properties(${SETUP_IDE_TARGET} PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY           ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)
        PROJECT_LABEL                           ${FOLDER_NAME}
        MAP_IMPORTED_CONFIG_DEBUG               Debug
        MAP_IMPORTED_CONFIG_RELEASE             Release
        MAP_IMPORTED_CONFIG_MINSIZEREL          Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO      Release
        )

    if(NOT ${FOLDER_PATH}) # if not, fall back to root level
        set_target_properties(${SETUP_IDE_TARGET} PROPERTIES
            FOLDER                              ${FOLDER_PATH}      # hierarchy
        )
    endif()

    # keep the folder structure in visual studio
    setup_ide_folders(FILES ${SETUP_IDE_SOURCES})
endfunction()

# -------------------------------------------------------------------------------------------------
# prints the name and type of the target
#
function(TARGET_PRINT_LOG_HEADER)
    set(options)
    set(oneValueArgs TARGET VERSION TYPE)
    set(multiValueArgs)
    cmake_parse_arguments(TARGET_PRINT_LOG_HEADER "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(NOT TARGET_PRINT_LOG_HEADER_TYPE)
        get_target_property(TARGET_PRINT_LOG_HEADER_TYPE ${TARGET_PRINT_LOG_HEADER_TARGET} TYPE)
    endif()
    if(MDL_LOG_DEPENDENCIES)
        MESSAGE(STATUS "")
        MESSAGE(STATUS "---------------------------------------------------------------------------------")
    endif()
    MESSAGE(STATUS "[PROJECT] ${TARGET_PRINT_LOG_HEADER_TARGET}   (${TARGET_PRINT_LOG_HEADER_TYPE})")
    if(MDL_LOG_DEPENDENCIES)
        if(TARGET_PRINT_LOG_HEADER_VERSION)
            MESSAGE(STATUS "VERSION:          ${TARGET_PRINT_LOG_HEADER_VERSION}")
        endif()
        MESSAGE(STATUS "SOURCE DIR:       ${CMAKE_CURRENT_SOURCE_DIR}")
        MESSAGE(STATUS "BINARY DIR:       ${CMAKE_CURRENT_BINARY_DIR}")
    endif()

endfunction()

# -------------------------------------------------------------------------------------------------
# function that copies a list of files into the target directory
#
#   target_copy_to_output_dir(TARGET foo
#       [RELATIVE <path_prefix>]                                # allows to keep the folder structure starting from this level
#       FILES <absolute_file_path> [<absolute_file_path>]
#       )
#
function(TARGET_COPY_TO_OUTPUT_DIR)
    set(options)
    set(oneValueArgs TARGET RELATIVE DEST_SUBFOLDER)
    set(multiValueArgs FILES)
    cmake_parse_arguments(TARGET_COPY_TO_OUTPUT_DIR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    foreach(_ELEMENT ${TARGET_COPY_TO_OUTPUT_DIR_FILES} )

        # handle absolute and relative paths
        if(TARGET_COPY_TO_OUTPUT_DIR_RELATIVE)
            set(_SOURCE_FILE ${TARGET_COPY_TO_OUTPUT_DIR_RELATIVE}/${_ELEMENT})
            set(_FOLDER_PATH ${_ELEMENT})
        else()
            set(_SOURCE_FILE ${_ELEMENT})
            get_filename_component(_FOLDER_PATH ${_ELEMENT} NAME)
            set (_ELEMENT "")
        endif()

        if(TARGET_COPY_TO_OUTPUT_DIR_DEST_SUBFOLDER)
            set(_SUBFOLDER /${TARGET_COPY_TO_OUTPUT_DIR_DEST_SUBFOLDER})
            set(_SUBFOLDERLOG " in subfolder ${TARGET_COPY_TO_OUTPUT_DIR_DEST_SUBFOLDER}")
        else()
            set(_SUBFOLDER "")
            set(_SUBFOLDERLOG "")
        endif()

        # handle directories and files slightly different
        if(IS_DIRECTORY ${_SOURCE_FILE})
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- folder to copy: ${_SOURCE_FILE}")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "copy to output dir: ${_SOURCE_FILE}${_SUBFOLDERLOG}"
                COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>${_SUBFOLDER}
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${_SOURCE_FILE} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>${_SUBFOLDER}/${_FOLDER_PATH}
            )
        elseif (TARGET ${_SOURCE_FILE})
            GET_TARGET_PROPERTY(__libname ${_SOURCE_FILE} LOCATION)

            if(MDL_LOG_FILE_DEPENDENCIES)
                    MESSAGE(STATUS "- libname to copy: ${__libname} (for target: ${_SOURCE_FILE}) ")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "copy to output dir: ${__libname}${_SUBFOLDERLOG}"
                COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>${_SUBFOLDER}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${__libname} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>${_SUBFOLDER}
            )
        else()
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- file to copy:   ${_SOURCE_FILE}")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E echo "copy to output dir: ${_SOURCE_FILE}${_SUBFOLDERLOG}"
                COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>${_SUBFOLDER}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_SOURCE_FILE} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>${_SUBFOLDER}
            )
        endif()
    endforeach()


endfunction()

# -------------------------------------------------------------------------------------------------
# Adds a dependency to a target, meant as shortcut for several more or less similar examples
# Meant for internal use by the function below
function(__TARGET_ADD_DEPENDENCY)
    # TODO Why is NO_RUNTIME not handled here?
    set(options NO_RUNTIME_COPY NO_LINKING NO_INCLUDE)
    set(oneValueArgs TARGET DEPENDS)
    set(multiValueArgs COMPONENTS)
    cmake_parse_arguments(__TARGET_ADD_DEPENDENCY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - __TARGET_ADD_DEPENDENCY_TARGET
    # - __TARGET_ADD_DEPENDENCY_DEPENDS
    # - __TARGET_ADD_DEPENDENCY_COMPONENTS
    # - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
    # - __TARGET_ADD_DEPENDENCY_NO_LINKING
    # - __TARGET_ADD_DEPENDENCY_NO_INCLUDE

    # handle some special symbols
    if(__TARGET_ADD_DEPENDENCY_DEPENDS STREQUAL LINKER_START_GROUP OR
       __TARGET_ADD_DEPENDENCY_DEPENDS STREQUAL LINKER_END_GROUP OR
       __TARGET_ADD_DEPENDENCY_DEPENDS STREQUAL LINKER_WHOLE_ARCHIVE OR
       __TARGET_ADD_DEPENDENCY_DEPENDS STREQUAL LINKER_NO_WHOLE_ARCHIVE OR
       __TARGET_ADD_DEPENDENCY_DEPENDS STREQUAL LINKER_AS_NEEDED OR
       __TARGET_ADD_DEPENDENCY_DEPENDS STREQUAL LINKER_NO_AS_NEEDED)
        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
            PRIVATE
                ${__TARGET_ADD_DEPENDENCY_DEPENDS}
            )
        return()
    endif()

    # split the dependency into namespace and target name, separated by "::"
    string(REGEX MATCHALL "[^:]+" _RESULTS ${__TARGET_ADD_DEPENDENCY_DEPENDS})
    list(LENGTH _RESULTS _RESULTS_LENGTH)
    if(_RESULTS_LENGTH EQUAL 2)
        list(GET _RESULTS 0 __TARGET_ADD_DEPENDENCY_DEPENDS_NS)
        list(GET _RESULTS 1 __TARGET_ADD_DEPENDENCY_DEPENDS_MODULE)
    else()
        set(__TARGET_ADD_DEPENDENCY_DEPENDS_MODULE ${__TARGET_ADD_DEPENDENCY_DEPENDS})
    endif()

    # log dependency
    if(MDL_LOG_DEPENDENCIES)
        set(_FLAGS "")
        if(__TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY)
            list(APPEND _FLAGS "NO_RUNTIME_COPY")
        endif()
        if(__TARGET_ADD_DEPENDENCY_NO_LINKING)
            list(APPEND _FLAGS "NO_LINKING")
        endif()
        if(__TARGET_ADD_DEPENDENCY_NO_INCLUDE)
            list(APPEND _FLAGS "NO_INCLUDE")
        endif()
        string(JOIN ", " _FLAGS ${_FLAGS})
        if(_FLAGS)
            set(_FLAGS " (${_FLAGS})")
        endif()
        message(STATUS "- depends on:     " ${__TARGET_ADD_DEPENDENCY_DEPENDS}${_FLAGS})
    endif()

    # customized dependency scripts have highest priority
    # to use it, define a variable like this: OVERRIDE_DEPENDENCY_SCRIPT_<upper case dependency name>
    string(TOUPPER ${__TARGET_ADD_DEPENDENCY_DEPENDS_MODULE} __TARGET_ADD_DEPENDENCY_DEPENDS_MODULE_UPPER)
    if(OVERRIDE_DEPENDENCY_SCRIPT_${__TARGET_ADD_DEPENDENCY_DEPENDS_MODULE_UPPER})
        set(_FILE_TO_INCLUDE ${OVERRIDE_DEPENDENCY_SCRIPT_${__TARGET_ADD_DEPENDENCY_DEPENDS_MODULE_UPPER}})
    # if no custom script is defined, we check if there is a default one
    else()
        set(_FILE_TO_INCLUDE "${MDL_BASE_FOLDER}/cmake/dependencies/add_${__TARGET_ADD_DEPENDENCY_DEPENDS_MODULE}.cmake")
    endif()

    # check if there is a add_dependency file to include (custom or default)
    if(EXISTS ${_FILE_TO_INCLUDE})
        include(${_FILE_TO_INCLUDE})
    # if not, we try to interpret the dependency as a target contained in the top level project
    else()

        # if this is no internal dependency we use the default find mechanism
        if(NOT TARGET ${__TARGET_ADD_DEPENDENCY_DEPENDS})
            # checks if there is such a "sub project"
            find_package(${__TARGET_ADD_DEPENDENCY_DEPENDS})
            # if the target was not found this is a error
            if(NOT ${__TARGET_ADD_DEPENDENCY_DEPENDS}_FOUND)
                MESSAGE(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
            endif()
        endif()

        # check the type
        get_target_property(_TARGET_TYPE ${__TARGET_ADD_DEPENDENCY_DEPENDS} TYPE)
        # libraries
        if (_TARGET_TYPE STREQUAL "STATIC_LIBRARY" OR
            _TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR
            _TARGET_TYPE STREQUAL "INTERFACE_LIBRARY" OR
            _TARGET_TYPE STREQUAL "MODULE_LIBRARY")

            # add the dependency to the target
            if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)
                # include directories and link dependencies (ignore NO_INCLUDE)
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        ${__TARGET_ADD_DEPENDENCY_DEPENDS}
                    )
            elseif(NOT __TARGET_ADD_DEPENDENCY_NO_INCLUDE)
                # if NO_LINKING was specified, but not NO_INCLUDE, we add the include directories only
                target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        $<TARGET_PROPERTY:${__TARGET_ADD_DEPENDENCY_DEPENDS},INTERFACE_INCLUDE_DIRECTORIES>
                    )
            else()
                # add dependency manually
                add_dependencies(${__TARGET_ADD_DEPENDENCY_TARGET} ${__TARGET_ADD_DEPENDENCY_DEPENDS})
            endif()
        # executables, custom targets, ...
        else()
            # add dependency manually
            add_dependencies(${__TARGET_ADD_DEPENDENCY_TARGET} ${__TARGET_ADD_DEPENDENCY_DEPENDS})
        endif()
    endif()
endfunction()

# -------------------------------------------------------------------------------------------------
# adds multiple dependencies. Convenience helper for dependencies without components.
# in case of one dependencies with component, you can add components.
#
# * target_add_dependencies(TARGET foo
#       DEPENDENCIES
#           mdl::base-system
#           mdl::base-hal-disk
#       )
#
#
# * target_add_dependency(TARGET foo
#       DEPENDS
#           qt
#       COMPONENTS
#           core
#           quick
#           gui
#       )
function(TARGET_ADD_DEPENDENCIES)
    set(options NO_RUNTIME_COPY NO_RUNTIME NO_LINKING NO_INCLUDE)
    set(oneValueArgs TARGET)
    set(multiValueArgs DEPENDS COMPONENTS)
    cmake_parse_arguments(TARGET_ADD_DEPENDENCIES "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_DEPENDENCIES_TARGET
    # - TARGET_ADD_DEPENDENCIES_DEPENDS
    # - TARGET_ADD_DEPENDENCIES_COMPONENTS
    # - TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY
    # - TARGET_ADD_DEPENDENCIES_NO_RUNTIME
    # - TARGET_ADD_DEPENDENCIES_NO_LINKING
    # - TARGET_ADD_DEPENDENCIES_NO_INCLUDE

    # make sure components are not used for multiple dependencies
    list(LENGTH TARGET_ADD_DEPENDENCIES_DEPENDS _NUM_DEP)
    if(_NUM_DEP GREATER 1 AND TARGET_ADD_DEPENDENCIES_COMPONENTS)
        message(FATAL_ERROR "COMPONENTs are not allowed when specifying multiple dependencies for target '${TARGET_ADD_DEPENDENCIES_TARGET}'")
    endif()

    # forward options
    if(TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY)
        set(TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY NO_RUNTIME_COPY)
    else()
        set(TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY "")
    endif()

    if(TARGET_ADD_DEPENDENCIES_NO_RUNTIME)
        set(TARGET_ADD_DEPENDENCIES_NO_RUNTIME NO_RUNTIME)
    else()
        set(TARGET_ADD_DEPENDENCIES_NO_RUNTIME "")
    endif()

    if(TARGET_ADD_DEPENDENCIES_NO_LINKING)
        set(TARGET_ADD_DEPENDENCIES_NO_LINKING NO_LINKING)
    else()
        set(TARGET_ADD_DEPENDENCIES_NO_LINKING "")
    endif()

    if(TARGET_ADD_DEPENDENCIES_NO_INCLUDE)
        set(TARGET_ADD_DEPENDENCIES_NO_INCLUDE NO_INCLUDE)
    else()
        set(TARGET_ADD_DEPENDENCIES_NO_INCLUDE "")
    endif()

    # in case we have components we pass them to the single dependency
    if(TARGET_ADD_DEPENDENCIES_COMPONENTS)
        __target_add_dependency(
            TARGET      ${TARGET_ADD_DEPENDENCIES_TARGET}
            DEPENDS     ${TARGET_ADD_DEPENDENCIES_DEPENDS}
            COMPONENTS  ${TARGET_ADD_DEPENDENCIES_COMPONENTS}
            ${TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY}
            ${TARGET_ADD_DEPENDENCIES_NO_RUNTIME}
            ${TARGET_ADD_DEPENDENCIES_NO_LINKING}
            ${TARGET_ADD_DEPENDENCIES_NO_INCLUDE}
            )
    # if not, we iterate over the list of dependencies and pass no components
    else()
        foreach(_DEP ${TARGET_ADD_DEPENDENCIES_DEPENDS})
            __target_add_dependency(
                TARGET      ${TARGET_ADD_DEPENDENCIES_TARGET}
                DEPENDS     ${_DEP}
                ${TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY}
                ${TARGET_ADD_DEPENDENCIES_NO_RUNTIME}
                ${TARGET_ADD_DEPENDENCIES_NO_LINKING}
                ${TARGET_ADD_DEPENDENCIES_NO_INCLUDE}
                )
        endforeach()
    endif()
endfunction()


# -------------------------------------------------------------------------------------------------
# Adds a tool dependency to a target, meant as shortcut for several more or less similar examples.
# This also works for tools that are part of the build, see scripts in the 'cmake/tools' sub folder.
#
# target_add_tool_dependency(TARGET foo
#     TOOL
#         python
#     )
# message(STATUS "python_PATH> ${python_PATH}")
#
function(TARGET_ADD_TOOL_DEPENDENCY)
    set(options)
    set(oneValueArgs TARGET TOOL)
    set(multiValueArgs)
    cmake_parse_arguments(TARGET_ADD_TOOL_DEPENDENCY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_TOOL_DEPENDENCY_TARGET
    # - TARGET_ADD_TOOL_DEPENDENCY_TOOL

    # log dependency
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- depends on:     " ${TARGET_ADD_TOOL_DEPENDENCY_TOOL})
    endif()

    set(_FILE_TO_INCLUDE "${MDL_BASE_FOLDER}/cmake/tools/add_${TARGET_ADD_TOOL_DEPENDENCY_TOOL}.cmake")

    # check if there is a add_dependency file to include
    if(EXISTS ${_FILE_TO_INCLUDE})
        include(${_FILE_TO_INCLUDE})
    else()

        # use a default fallback
        find_program(${TARGET_ADD_TOOL_DEPENDENCY_TOOL}_PATH ${TARGET_ADD_TOOL_DEPENDENCY_TOOL})
        if(NOT ${TARGET_ADD_TOOL_DEPENDENCY_TOOL}_PATH)
            MESSAGE(FATAL_ERROR "The tool dependency \"${TARGET_ADD_TOOL_DEPENDENCY_TOOL}\" for target \"${TARGET_ADD_TOOL_DEPENDENCY_TARGET}\" could not be resolved.")
        endif()

    endif()

endfunction()

# -------------------------------------------------------------------------------------------------
# the reduce the redundant code in the base library projects, we can bundle several repeated tasks
#
function(CREATE_FROM_BASE_PRESET)
    # the options DYNAMIC_MSVC_RUNTIME and STATIC_MSVC_RUNTIME override the
    # general option MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES
    set(options WIN32 WINDOWS_UNICODE EXAMPLE DYNAMIC_MSVC_RUNTIME STATIC_MSVC_RUNTIME SKIP_UNDEFINED_SYMBOL_CHECK)
    set(oneValueArgs TARGET VERSION TYPE NAMESPACE EXPORT_NAME OUTPUT_NAME VS_PROJECT_NAME EMBED_RC)
    set(multiValueArgs SOURCES ADDITIONAL_INCLUDE_DIRS EXPORTED_SYMBOLS)
    cmake_parse_arguments(CREATE_FROM_BASE_PRESET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # sanity check
    if(CREATE_FROM_BASE_PRESET_DYNAMIC_MSVC_RUNTIME AND CREATE_FROM_BASE_PRESET_STATIC_MSVC_RUNTIME)
        message(FATAL_ERROR "The options DYNAMIC_MSVC_RUNTIME and STATIC_MSVC_RUNTIME are mutually exclusive.")
    endif()

    # the option EXAMPLE bundles a few things
    if(CREATE_FROM_BASE_PRESET_EXAMPLE)
        set(CREATE_FROM_BASE_PRESET_WINDOWS_UNICODE ON)           # enable unicode
        if(MDL_MSVC_DYNAMIC_RUNTIME_EXAMPLES AND NOT CREATE_FROM_BASE_PRESET_STATIC_MSVC_RUNTIME)
            set(CREATE_FROM_BASE_PRESET_DYNAMIC_MSVC_RUNTIME ON)  # runtime linking
        endif()
    endif()

    if(CREATE_FROM_BASE_PRESET_DYNAMIC_MSVC_RUNTIME)
        list(APPEND BUILD_SETUP_OPTIONS DYNAMIC_MSVC_RUNTIME) # runtime linking
    endif()

    if(CREATE_FROM_BASE_PRESET_WINDOWS_UNICODE)
        list(APPEND BUILD_SETUP_OPTIONS WINDOWS_UNICODE) # unicode
    endif()

    # create the project
    if(CREATE_FROM_BASE_PRESET_VERSION)
        project(${CREATE_FROM_BASE_PRESET_TARGET} VERSION ${CREATE_FROM_BASE_PRESET_VERSION})
    else()
        project(${CREATE_FROM_BASE_PRESET_TARGET})
    endif()

    # default type is STATIC library
    if(NOT CREATE_FROM_BASE_PRESET_TYPE)
        set(CREATE_FROM_BASE_PRESET_TYPE STATIC)
    endif()

    # default export name is the same as output name if provided, CMake default otherwise (i.e. target name)
    if(NOT CREATE_FROM_BASE_PRESET_EXPORT_NAME AND CREATE_FROM_BASE_PRESET_OUTPUT_NAME)
        set(CREATE_FROM_BASE_PRESET_EXPORT_NAME ${CREATE_FROM_BASE_PRESET_OUTPUT_NAME})
    endif()

    # default namespace is mdl
    if(NOT CREATE_FROM_BASE_PRESET_NAMESPACE)
        set(CREATE_FROM_BASE_PRESET_NAMESPACE mdl)
    endif()

    # add empty pch
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/pch.h)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pch.h "")
    endif()
    # list(APPEND CREATE_FROM_BASE_PRESET_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/pch.h)

    # special executable
    if(CREATE_FROM_BASE_PRESET_WIN32)
        set(EXEC_TYPE WIN32)
    endif()

    # create target and alias
    if(CREATE_FROM_BASE_PRESET_TYPE STREQUAL "STATIC" OR CREATE_FROM_BASE_PRESET_TYPE STREQUAL "SHARED")
        add_library(${CREATE_FROM_BASE_PRESET_TARGET} ${CREATE_FROM_BASE_PRESET_TYPE} ${CREATE_FROM_BASE_PRESET_SOURCES})
        add_library(${CREATE_FROM_BASE_PRESET_NAMESPACE}::${CREATE_FROM_BASE_PRESET_TARGET} ALIAS ${CREATE_FROM_BASE_PRESET_TARGET})
    elseif(CREATE_FROM_BASE_PRESET_TYPE STREQUAL "EXECUTABLE" OR CREATE_FROM_BASE_PRESET_TYPE STREQUAL "WIN_EXECUTABLE")
        add_executable(${CREATE_FROM_BASE_PRESET_TARGET} ${EXEC_TYPE} ${CREATE_FROM_BASE_PRESET_SOURCES})
        #set_target_properties(${CREATE_FROM_BASE_PRESET_TARGET} PROPERTIES WIN32_EXECUTABLE ON)
        # inject compile constant that contains the binary name
        target_compile_definitions(${PROJECT_NAME}
            PUBLIC
                "BINARY_NAME=\"${CREATE_FROM_BASE_PRESET_OUTPUT_NAME}\""
            )
        if(CREATE_FROM_BASE_PRESET_TYPE STREQUAL "WIN_EXECUTABLE")
            set_target_properties(${CREATE_FROM_BASE_PRESET_TARGET} PROPERTIES WIN32_EXECUTABLE ON)
        endif()
    else()
        message(FATAL_ERROR "Unexpected Type for target '${CREATE_FROM_BASE_PRESET_TARGET}': ${CREATE_FROM_BASE_PRESET_TYPE}.")
    endif()

    # change the output directory for non-multi-config-generators as well
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT _GENERATOR_IS_MULTI_CONFIG)
        set_target_properties(${CREATE_FROM_BASE_PRESET_TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
            PDB_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}
        )
    endif()

    # adjust output file name if requested
    if(CREATE_FROM_BASE_PRESET_OUTPUT_NAME)
        set_target_properties(${CREATE_FROM_BASE_PRESET_TARGET} PROPERTIES OUTPUT_NAME ${CREATE_FROM_BASE_PRESET_OUTPUT_NAME})
    endif()

    # adjust export target name if requested
    if(CREATE_FROM_BASE_PRESET_EXPORT_NAME)
        set_target_properties(${CREATE_FROM_BASE_PRESET_TARGET} PROPERTIES EXPORT_NAME ${CREATE_FROM_BASE_PRESET_EXPORT_NAME})
    endif()

    # log message
    target_print_log_header(TARGET ${CREATE_FROM_BASE_PRESET_TARGET} VERSION ${CREATE_FROM_BASE_PRESET_VERSION})

    # add include directories
    target_include_directories(${CREATE_FROM_BASE_PRESET_TARGET}
        PUBLIC
            $<BUILD_INTERFACE:${MDL_INCLUDE_FOLDER}>
        PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${MDL_SRC_FOLDER}>
            ${CREATE_FROM_BASE_PRESET_ADDITIONAL_INCLUDE_DIRS}
        )

    # add system dependencies
    if(CREATE_FROM_BASE_PRESET_TYPE STREQUAL "SHARED" OR CREATE_FROM_BASE_PRESET_TYPE STREQUAL "EXECUTABLE" OR CREATE_FROM_BASE_PRESET_TYPE STREQUAL "WIN_EXECUTABLE")
        target_add_dependencies(TARGET ${CREATE_FROM_BASE_PRESET_TARGET}
            DEPENDS
                system
            )
    endif()

    # embed .rc files
    if(CREATE_FROM_BASE_PRESET_EMBED_RC AND WINDOWS AND CREATE_FROM_BASE_PRESET_TYPE STREQUAL "SHARED")
        if(MDL_LOG_FILE_DEPENDENCIES)
            message(STATUS "- embedding:      ${CREATE_FROM_BASE_PRESET_EMBED_RC}")
        endif()
        target_sources(${CREATE_FROM_BASE_PRESET_TARGET}
            PRIVATE
                ${CREATE_FROM_BASE_PRESET_EMBED_RC}
            )
        target_include_directories(${CREATE_FROM_BASE_PRESET_TARGET}
            PRIVATE
                ${MDL_SRC_FOLDER}/base/system/version # for the version.h
            )
        # TODO: _NAME => short name?
        set_source_files_properties(${CREATE_FROM_BASE_PRESET_EMBED_RC}
            PROPERTIES COMPILE_DEFINITIONS
                "MI_ARCH_LITTLE_ENDIAN;TARGET_FILENAME=\"$<TARGET_FILE_NAME:${CREATE_FROM_BASE_PRESET_TARGET}>\""
            )
    endif()

    # compiler flags and defines
    target_build_setup(TARGET ${CREATE_FROM_BASE_PRESET_TARGET} ${BUILD_SETUP_OPTIONS})

    # undefined symbol check
    if(NOT CREATE_FROM_BASE_PRESET_SKIP_UNDEFINED_SYMBOL_CHECK)
        if(LINUX AND NOT (${CREATE_FROM_BASE_PRESET_TYPE} STREQUAL "STATIC"))
            add_custom_command(
                TARGET ${CREATE_FROM_BASE_PRESET_TARGET}
                POST_BUILD
                COMMAND ! ldd -r $<TARGET_FILE:${CREATE_FROM_BASE_PRESET_TARGET}> | grep "undefined symbol:"
                )
        endif()
    endif()

    # exported symbols
    if(CREATE_FROM_BASE_PRESET_EXPORTED_SYMBOLS)
        if(LINUX)
            set(_GLOBAL "")
            foreach(_SYMBOL ${CREATE_FROM_BASE_PRESET_EXPORTED_SYMBOLS})
                set(_GLOBAL "${_GLOBAL}${_SYMBOL};\n")
            endforeach()
            configure_file(
                "${MDL_BASE_FOLDER}/cmake/linkerscript.txt.in"
                "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/linkerscript.txt"
                @ONLY
            )
            target_link_libraries(${CREATE_FROM_BASE_PRESET_TARGET}
                PRIVATE
                    -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/linkerscript.txt
            )
        endif()
    endif()

    # configure visual studio and maybe other IDEs
    setup_ide(TARGET ${CREATE_FROM_BASE_PRESET_TARGET}
        SOURCES ${CREATE_FROM_BASE_PRESET_SOURCES}
        VS_PROJECT_NAME ${CREATE_FROM_BASE_PRESET_VS_PROJECT_NAME})
endfunction()

# -------------------------------------------------------------------------------------------------
# Creates an object library to compile cuda sources to ptx and adds a rule to copy the ptx to
# the related projects binary directory.
#
# target_add_cuda_ptx_rule(TARGET foo
#     DEPENDS
#       mdl::mdl_sdk
#       mdl_sdk_examples::mdl_sdk_shared
#     CUDA_SOURCES
#       "example.cu"
#     FAST_MATH
#     )
#
function(TARGET_ADD_CUDA_PTX_RULE)
    set(options FAST_MATH)
    set(oneValueArgs TARGET ARCH)
    set(multiValueArgs CUDA_SOURCES DEPENDS)
    cmake_parse_arguments(TARGET_ADD_CUDA_PTX_RULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_CUDA_PTX_RULE_TARGET
    # - TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES
    # - TARGET_ADD_CUDA_PTX_RULE_DEPENDS

    # options
    if(NOT TARGET_ADD_CUDA_PTX_RULE_CUDA_ARCH)
        set(TARGET_ADD_CUDA_PTX_RULE_CUDA_ARCH "sm_50")
    endif()

    # policy CMP0104, we use the '-arch' flag to pass the sm version
    # string(SUBSTRING ${TARGET_ADD_CUDA_PTX_RULE_CUDA_ARCH} 3 -1 CUDA_ARCH_VERSION) # potential alternative
    set(CUDA_ARCH_VERSION OFF)

    # create PTX target
    add_library(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX OBJECT ${TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES})
    set_target_properties(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX PROPERTIES
        CUDA_PTX_COMPILATION ON
        CUDA_ARCHITECTURES ${CUDA_ARCH_VERSION}
        )

    # flags set per project
    target_compile_options(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX
        PRIVATE
            "-rdc=true"
            "-arch=${TARGET_ADD_CUDA_PTX_RULE_CUDA_ARCH}"
    )

    if(TARGET_ADD_CUDA_PTX_RULE_FAST_MATH)
        # workaround for CMake filtering out --use_fast_math option from NVCC command line
        target_compile_options(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX
            PRIVATE
                --options-file=${MDL_BASE_FOLDER}/cmake/nvcc-option-use_fast_math.lst
        )
    endif()

    # add dependencies (no linking no post builds since this creates a ptx only)
    target_add_dependencies(TARGET ${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX
        DEPENDS
            ${TARGET_ADD_CUDA_PTX_RULE_DEPENDS}
            cuda
        NO_LINKING
        NO_RUNTIME_COPY
        )

    # configure visual studio and maybe other IDEs
    setup_ide(TARGET ${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX
        SOURCES
            ${TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES}
        )

    # extend to project names
    get_target_property(_PROJECT_LABEL ${TARGET_ADD_CUDA_PTX_RULE_TARGET} PROJECT_LABEL)
    set_target_properties(${TARGET_ADD_CUDA_PTX_RULE_TARGET} PROPERTIES
        PROJECT_LABEL   "${_PROJECT_LABEL} (main)"      # project name
        )
    set_target_properties(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX PROPERTIES
        PROJECT_LABEL   "${_PROJECT_LABEL} (ptx)"       # project name
        )

    # post build
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    foreach(_SRC ${TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES})

        # copy ptx to example binary folder
        get_filename_component(_SRC_NAME ${_SRC} NAME_WE)
        list(APPEND PTX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_SRC_NAME}.ptx)

        if(MDL_LOG_FILE_DEPENDENCIES)
            MESSAGE(STATUS "- file to copy:   ${_SRC_NAME}.ptx")
        endif()

        if(_GENERATOR_IS_MULTI_CONFIG)
            set(_CONFIG_FOLDER /$<CONFIG>)
            set(_PTX_CONFIG_FOLDER /$<CONFIG>)
            set(_CMAKEFILES_FOLDER "")
        else()
            set(_CONFIG_FOLDER /${CMAKE_BUILD_TYPE})
            set(_PTX_CONFIG_FOLDER "")
            set(_CMAKEFILES_FOLDER /CMakeFiles)
        endif()

        list(APPEND MOVE_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E echo "Copy ${_SRC_NAME}.ptx to binary dir..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_BINARY_DIR}${_CMAKEFILES_FOLDER}/${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX.dir${_PTX_CONFIG_FOLDER}/${_SRC_NAME}.ptx   # resulting ptx file
                ${CMAKE_CURRENT_BINARY_DIR}${_CONFIG_FOLDER}                                                                                        # to binary dir
        )

    endforeach()

    # due to a bug visual studio 2017 does not detect changes in cu files, so for now we compile ptx files every time
    # https://devtalk.nvidia.com/default/topic/1029759/visual-studio-2017-not-detecting-changes-in-cuda-cu-files/
    if(CMAKE_GENERATOR STREQUAL "Visual Studio 15 2017 Win64")
        list(APPEND MOVE_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E echo "Delete ${_SRC_NAME}.ptx to force next rebuild..."
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}${_CMAKEFILES_FOLDER}/${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX.dir
        )
    endif()

    add_custom_command(
        OUTPUT ${PTX_OUTPUT}   # note, not correct for multi config generators (like VS)
                               # this will cause copying with every build (when using these generators)
        DEPENDS $<TARGET_OBJECTS:${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX>
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}${_CONFIG_FOLDER}
        ${MOVE_COMMANDS}
        )

    # make sure the copying is repeated when only the ptx changed and the main project did not
    add_custom_target(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX_COPY DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_SRC_NAME}.ptx)
    add_dependencies(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX_COPY ${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX)
    add_dependencies(${TARGET_ADD_CUDA_PTX_RULE_TARGET} ${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX_COPY)
    set_target_properties(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX_COPY PROPERTIES FOLDER "_cmake/ptx_copy")

endfunction()


# -------------------------------------------------------------------------------------------------
# function that adds content files to the project that are copied to the output dir
#
function(TARGET_ADD_CONTENT)
    set(options)
    set(oneValueArgs TARGET FILE_BASE TARGET_BASE DEP_NAME)
    set(multiValueArgs FILES)
    cmake_parse_arguments(TARGET_ADD_CONTENT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # dependency file
    if(NOT TARGET_ADD_CONTENT_DEP_NAME)
        set(TARGET_ADD_CONTENT_DEP_NAME "content")
    endif()
    if(WINDOWS)
        set(_DEP ${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/${TARGET_ADD_CONTENT_DEP_NAME}.d)
    else()
        set(_DEP ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_ADD_CONTENT_DEP_NAME}.d)
    endif()

    # make path absolute
    if (NOT IS_ABSOLUTE ${TARGET_ADD_CONTENT_FILE_BASE})
        set (TARGET_ADD_CONTENT_FILE_BASE ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_ADD_CONTENT_FILE_BASE})
    endif()

    # add to project
    foreach(_FILE ${TARGET_ADD_CONTENT_FILES})
        list(APPEND _ADD_TO_SOURCES ${TARGET_ADD_CONTENT_FILE_BASE}${_FILE})
    endforeach()
    target_sources(${TARGET_ADD_CONTENT_TARGET}
        PRIVATE
            ${_ADD_TO_SOURCES}
            ${_DEP}
        )

    # do not compile and set group (visual studio)
    set_source_files_properties(${_ADD_TO_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${_DEP} PROPERTIES GENERATED TRUE)
    source_group("dependencies" FILES ${_DEP})

    # collect files
    foreach(_FILE ${TARGET_ADD_CONTENT_FILES})
        # adjust source groups
        get_filename_component(_GROUP ${TARGET_ADD_CONTENT_TARGET_BASE}${_FILE} DIRECTORY)
        if(_GROUP)
            string(REPLACE "/" "\\" _GROUP ${_GROUP})
        endif()
        source_group(${_GROUP} FILES ${TARGET_ADD_CONTENT_FILE_BASE}${_FILE})

        # copy commands
        if(MDL_LOG_FILE_DEPENDENCIES)
            MESSAGE(STATUS "- content to copy:   ${TARGET_ADD_CONTENT_FILE_BASE}${_FILE}")
        endif()
        list(APPEND _COMMAND_LIST COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TARGET_ADD_CONTENT_FILE_BASE}${_FILE} $<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/${TARGET_ADD_CONTENT_TARGET_BASE}${_FILE})
        list(APPEND _COMMAND_LIST COMMAND ${CMAKE_COMMAND} -E echo "update content file: ${TARGET_ADD_CONTENT_TARGET_BASE}${_FILE}")
        list(APPEND _COPY_DEPENDS ${TARGET_ADD_CONTENT_FILE_BASE}${_FILE})
    endforeach()

    # delete dependency files of other configurations to enforce an update after the build type changed
    if(WINDOWS)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Debug)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Release)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo)

        list(APPEND _COMMAND_LIST COMMAND ${CMAKE_COMMAND} -E echo "removing: "
            $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Release/${TARGET_ADD_CONTENT_DEP_NAME}.d>
            $<$<CONFIG:Release>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Debug/${TARGET_ADD_CONTENT_DEP_NAME}.d>
            $<$<CONFIG:RelWithDebInfo>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Debug/${TARGET_ADD_CONTENT_DEP_NAME}.d>
            )
        list(APPEND _COMMAND_LIST COMMAND ${CMAKE_COMMAND} -E remove -f $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Release/${TARGET_ADD_CONTENT_DEP_NAME}.d>$<$<CONFIG:Release>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Debug/${TARGET_ADD_CONTENT_DEP_NAME}.d>$<$<CONFIG:RelWithDebInfo>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Debug/${TARGET_ADD_CONTENT_DEP_NAME}.d>)

        list(APPEND _COMMAND_LIST COMMAND ${CMAKE_COMMAND} -E echo "removing: "
            $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../RelWithDebInfo/${TARGET_ADD_CONTENT_DEP_NAME}.d>
            $<$<CONFIG:Release>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../RelWithDebInfo/${TARGET_ADD_CONTENT_DEP_NAME}.d>
            $<$<CONFIG:RelWithDebInfo>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Release/${TARGET_ADD_CONTENT_DEP_NAME}.d>
            )
        list(APPEND _COMMAND_LIST COMMAND ${CMAKE_COMMAND} -E remove -f $<$<CONFIG:Debug>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../RelWithDebInfo/${TARGET_ADD_CONTENT_DEP_NAME}.d>$<$<CONFIG:Release>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../RelWithDebInfo/${TARGET_ADD_CONTENT_DEP_NAME}.d>$<$<CONFIG:RelWithDebInfo>:$<TARGET_FILE_DIR:${TARGET_ADD_CONTENT_TARGET}>/../Release/${TARGET_ADD_CONTENT_DEP_NAME}.d>)
    endif()

    # copy on files changed
    add_custom_command(
        OUTPUT
            ${_DEP}
        DEPENDS
            ${_COPY_DEPENDS}

        ${_COMMAND_LIST}
        COMMAND ${CMAKE_COMMAND} -E echo "touching: " ${_DEP}
        COMMAND ${CMAKE_COMMAND} -E touch ${_DEP}
        )
endfunction()

# -------------------------------------------------------------------------------------------------
# Add a path to the visual studio environment variables for the debugger.
# requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' to actually create the user settings file.
function(TARGET_ADD_VS_DEBUGGER_ENV_PATH)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs PATHS PATHS_DEBUG PATHS_RELEASE PATHS_RELWITHDEBINFO)
    cmake_parse_arguments(TARGET_ADD_VS_DEBUGGER_ENV_PATH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET
    # - TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS
    # - TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS_DEBUG
    # - TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS_RELEASE
    # - TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS_RELWITHDEBINFO

    if(NOT WINDOWS)
        return()
    endif()

    # read current property value
    get_property(_ENV_PATHS TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS)
    get_property(_ENV_PATHS_DEBUG TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS_DEBUG)
    get_property(_ENV_PATHS_RELEASE TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS_RELEASE)
    get_property(_ENV_PATHS_RELWITHDEBINFO TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS_RELWITHDEBINFO)

    foreach(_PATH ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- add property:   Visual Studio Debugger Environment path: ${_PATH}")
        endif()
        list(APPEND _ENV_PATHS ${_PATH})
    endforeach()

    foreach(_PATH ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS_DEBUG})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- add property:   Visual Studio Debugger Environment path (Debug only): ${_PATH}")
        endif()
        list(APPEND _ENV_PATHS_DEBUG ${_PATH})
    endforeach()

    foreach(_PATH ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS_RELEASE})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- add property:   Visual Studio Debugger Environment path (Release only): ${_PATH}")
        endif()
        list(APPEND _ENV_PATHS_RELEASE ${_PATH})
    endforeach()

    foreach(_PATH ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_PATHS_RELWITHDEBINFO})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- add property:   Visual Studio Debugger Environment path (Release with debug info only): ${_PATH}")
        endif()
        list(APPEND _ENV_PATHS_RELWITHDEBINFO ${_PATH})
    endforeach()

    # update property value
    set_property(TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS ${_ENV_PATHS})
    set_property(TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS_DEBUG ${_ENV_PATHS_DEBUG})
    set_property(TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS_RELEASE ${_ENV_PATHS_RELEASE})
    set_property(TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_PATH_TARGET} PROPERTY VS_DEBUGGER_PATHS_RELWITHDEBINFO ${_ENV_PATHS_RELWITHDEBINFO})
endfunction()


# -------------------------------------------------------------------------------------------------
# Add a variable to the visual studio environment for the debugger.
# requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' to actually create the user settings file.
function(TARGET_ADD_VS_DEBUGGER_ENV_VAR)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs VARS)
    cmake_parse_arguments(TARGET_ADD_VS_DEBUGGER_ENV_VAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_VS_DEBUGGER_ENV_VAR_TARGET
    # - TARGET_ADD_VS_DEBUGGER_ENV_VAR_VARS

    if(NOT WINDOWS)
        return()
    endif()

    # read current property value
    get_property(_ENV_VARS TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_VAR_TARGET} PROPERTY VS_DEBUGGER_ENV_VARS)

    foreach(_VAR ${TARGET_ADD_VS_DEBUGGER_ENV_VAR_VARS})
        string(REPLACE "%3B" ";" _VAR ${_VAR})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- add property:   Visual Studio Debugger Environment Variable: ${_VAR}")
        endif()
        if(_ENV_VARS)
            set(_ENV_VARS ${_ENV_VARS}\n${_VAR})
        else()
            set(_ENV_VARS ${_VAR})
        endif()
    endforeach()

    # update property value
    set_property(TARGET ${TARGET_ADD_VS_DEBUGGER_ENV_VAR_TARGET} PROPERTY VS_DEBUGGER_ENV_VARS ${_ENV_VARS})
endfunction()

# -------------------------------------------------------------------------------------------------
# Creates a visual studio user settings file to set environment variables for the debugger.
# This should only be called after the dependencies of an executable target are added.
function(TARGET_CREATE_VS_USER_SETTINGS)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(TARGET_CREATE_VS_USER_SETTINGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_CREATE_VS_USER_SETTINGS_TARGET

    if(NOT WINDOWS)
        return()
    endif()

    set(SETTINGS_FILE "${TARGET_CREATE_VS_USER_SETTINGS_TARGET}.vcxproj.user")

    if(MDL_LOG_FILE_DEPENDENCIES)
        message(STATUS "- writing file:   Visual Studio user settings: ${SETTINGS_FILE}")
    endif()

    get_property(_ENV_PATHS TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_PATHS)
    get_property(_ENV_PATHS_DEBUG TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_PATHS_DEBUG)
    get_property(_ENV_PATHS_RELEASE TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_PATHS_RELEASE)
    get_property(_ENV_PATHS_RELWITHDEBINFO TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_PATHS_RELWITHDEBINFO)
    get_property(_ENV_VARS TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_ENV_VARS)
    get_property(_COMMAND_ARGUMENTS TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_COMMAND_ARGUMENTS)
    get_property(_COMMAND TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_COMMAND)
    get_property(_COMMAND_CWD TARGET ${TARGET_CREATE_VS_USER_SETTINGS_TARGET} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY)
    if(_COMMAND)
    else()
        set(_COMMAND "$(TargetPath)")
    endif()

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${SETTINGS_FILE}
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<Project ToolsVersion=\"4.0\" xmlns=\"http://schemas.microsoft.com/developer/msbuild/2003\">\n"
        "   <PropertyGroup Condition=\"'$(Configuration)'=='Debug'\">\n"
        "       <LocalDebuggerCommand>${_COMMAND}</LocalDebuggerCommand>\n"
        "       <LocalDebuggerWorkingDirectory>${_COMMAND_CWD}</LocalDebuggerWorkingDirectory>\n"
        "       <LocalDebuggerCommandArguments>${_COMMAND_ARGUMENTS}</LocalDebuggerCommandArguments>\n"
        "       <LocalDebuggerEnvironment>\nPATH=${_ENV_PATHS_DEBUG};${_ENV_PATHS};%PATH%\n${_ENV_VARS}\n"
        "       </LocalDebuggerEnvironment>\n"
        "       <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>\n"
        "   </PropertyGroup>\n"
        "   <PropertyGroup Condition=\"'$(Configuration)'=='Release'\">\n"
        "       <LocalDebuggerCommand>${_COMMAND}</LocalDebuggerCommand>\n"
        "       <LocalDebuggerWorkingDirectory>${_COMMAND_CWD}</LocalDebuggerWorkingDirectory>\n"
        "       <LocalDebuggerCommandArguments>${_COMMAND_ARGUMENTS}</LocalDebuggerCommandArguments>\n"
        "       <LocalDebuggerEnvironment>\nPATH=${_ENV_PATHS_RELEASE};${_ENV_PATHS};%PATH%\n${_ENV_VARS}\n"
        "       </LocalDebuggerEnvironment>\n"
        "       <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>\n"
        "   </PropertyGroup>\n"
        "   <PropertyGroup Condition=\"'$(Configuration)'=='RelWithDebInfo'\">\n"
        "       <LocalDebuggerCommand>${_COMMAND}</LocalDebuggerCommand>\n"
        "       <LocalDebuggerWorkingDirectory>${_COMMAND_CWD}</LocalDebuggerWorkingDirectory>\n"
        "       <LocalDebuggerCommandArguments>${_COMMAND_ARGUMENTS}</LocalDebuggerCommandArguments>\n"
        "       <LocalDebuggerEnvironment>\nPATH=${_ENV_PATHS_RELWITHDEBINFO};${_ENV_PATHS};%PATH%\n${_ENV_VARS}\n"
        "       </LocalDebuggerEnvironment>\n"
        "       <DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>\n"
        "   </PropertyGroup>\n"
        "</Project>\n"
        )
endfunction()

# -------------------------------------------------------------------------------------------------
# add RPATH to locate references libraries

function(TARGET_ADD_RPATH)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs RPATHS)
    cmake_parse_arguments(TARGET_ADD_RPATH "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_RPATH_TARGET
    # - TARGET_ADD_RPATH_RPATHS

    # no RPATHs on windows
    if(WINDOWS)
        return()
    endif()

    foreach(_PATH ${TARGET_ADD_RPATH_RPATHS})
        if(_PATHS)
            string(APPEND _PATHS ";")
            string(APPEND _PATHS "${_PATH}")
        else()
            set(_PATHS "${_PATH}")
        endif()
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- rpath added:    " ${_PATH})
        endif()
    endforeach()

    # append the rpaths to the build rpath
    get_target_property(CURRENT_BUILD_RPATH ${TARGET_ADD_RPATH_TARGET} BUILD_RPATH)
    if(CURRENT_BUILD_RPATH)
        string(PREPEND _PATHS ";")
        string(PREPEND _PATHS "${CURRENT_BUILD_RPATH}")
    endif()
    set_target_properties(${TARGET_ADD_RPATH_TARGET} PROPERTIES
        BUILD_RPATH "${_PATHS}"
    )

    # check
    # get_target_property(CURRENT_BUILD_RPATH ${TARGET_ADD_RPATH_TARGET} BUILD_RPATH)
    # foreach(_PATH ${CURRENT_BUILD_RPATH})
    #     message(STATUS " - RPATH: ${_PATH}")
    # endforeach()

    # TODO set the install rpath

endfunction()

# -------------------------------------------------------------------------------------------------
# basic install logic to copy the entire output folder

function(ADD_TARGET_INSTALL)
    set(options)
    set(oneValueArgs TARGET DESTINATION)
    set(multiValueArgs)
    cmake_parse_arguments(ADD_TARGET_INSTALL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - ADD_TARGET_INSTALL_TARGET
    # - ADD_TARGET_INSTALL_DESTINATION

    if(ADD_TARGET_INSTALL_DESTINATION)
        install(DIRECTORY $<TARGET_FILE_DIR:${ADD_TARGET_INSTALL_TARGET}>/
            DESTINATION ${ADD_TARGET_INSTALL_DESTINATION}
            USE_SOURCE_PERMISSIONS
            FILES_MATCHING
            PATTERN "*"
            PATTERN "*.d" EXCLUDE
        )
    else()
        install(
            TARGETS ${ADD_TARGET_INSTALL_TARGET}
            EXPORT  mdl-targets
        )
    endif()
endfunction()

# -------------------------------------------------------------------------------------------------
# python binding examples

function(CREATE_FROM_PYTHON_PRESET)
    set(options CREATE_COVERAGE_REPORT)
    set(oneValueArgs TARGET MAIN)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(CREATE_FROM_PYTHON_PRESET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - CREATE_FROM_PYTHON_PRESET_TARGET
    # - CREATE_FROM_PYTHON_PRESET_MAIN
    # - CREATE_FROM_PYTHON_PRESET_SOURCES
    # - CREATE_FROM_PYTHON_PRESET_CREATE_COVERAGE_REPORT

    # add dummy-source to create a dummy-lib
    set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
    if(NOT EXISTS ${DUMMY_CPP})
        file(WRITE ${DUMMY_CPP} "")
    endif()

    # create target from template
    create_from_base_preset(
        TARGET ${CREATE_FROM_PYTHON_PRESET_TARGET}
        SOURCES
            ${CREATE_FROM_PYTHON_PRESET_SOURCES}
            ${DUMMY_CPP}
        )

    # add dependencies
    target_add_dependencies(TARGET ${CREATE_FROM_PYTHON_PRESET_TARGET}
    DEPENDS
        mdl::mdl_python
        mdl::mdl_sdk
    )

    # create working directories for each configuration
    # and also a shell script for running the script (usually in unix environments)
    if(${CREATE_FROM_PYTHON_PRESET_CREATE_COVERAGE_REPORT} AND ${MDL_ENABLE_PYTHON_UNIT_TEST_COVERAGE})
        set(CREATE_COVERAGE_REPORT TRUE)
    else()
        set(CREATE_COVERAGE_REPORT FALSE)
    endif()
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_GENERATOR_IS_MULTI_CONFIG)
        foreach(_CONFIG ${CMAKE_CONFIGURATION_TYPES})
            file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG})
            configure_file(
                ${MDL_EXAMPLES_FOLDER}/mdl_python/cmake_templates/run_example.sh
                ${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG}/run_example.sh
                @ONLY)
            configure_file(
                ${MDL_EXAMPLES_FOLDER}/mdl_python/cmake_templates/run_example.bat
                ${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG}/run_example.bat
                @ONLY)
        endforeach()
        set(_CONFIG "$(Configuration)")
    else()
        set(_CONFIG ${CMAKE_BUILD_TYPE})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG})
        configure_file(
            ${MDL_EXAMPLES_FOLDER}/mdl_python/cmake_templates/run_example.sh
            ${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG}/run_example.sh
            @ONLY)
        configure_file(
            ${MDL_EXAMPLES_FOLDER}/mdl_python/cmake_templates/run_example.bat
            ${CMAKE_CURRENT_BINARY_DIR}/${_CONFIG}/run_example.bat
            @ONLY)
    endif()

    # customize name
    set_target_properties(${CREATE_FROM_PYTHON_PRESET_TARGET} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpp.debugging
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpp.debugging
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpp.debugging
        PREFIX ""
        OUTPUT_NAME "cpp.debugging"
        SUFFIX ".d"
    )

    # creates a user settings file to setup the debugger (visual studio only, otherwise this is a no-op)
    set_target_properties(${CREATE_FROM_PYTHON_PRESET_TARGET}
        PROPERTIES
            VS_DEBUGGER_COMMAND             ${MDL_DEPENDENCY_PYTHON_DEV_EXE}
            VS_DEBUGGER_COMMAND_ARGUMENTS   "${CMAKE_CURRENT_SOURCE_DIR}/${CREATE_FROM_PYTHON_PRESET_MAIN}"
        )

    get_property(BINDING_MODULE_PATH TARGET "mdl::mdl_python" PROPERTY BINARY_DIR)
    target_add_vs_debugger_env_var(TARGET ${CREATE_FROM_PYTHON_PRESET_TARGET}
        VARS
            "PYTHONPATH=${BINDING_MODULE_PATH}/$(Configuration)%3B%PYTHONPATH%"
            "MDL_SAMPLES_ROOT=${MDL_EXAMPLES_FOLDER}"
        )

    target_create_vs_user_settings(TARGET ${CREATE_FROM_PYTHON_PRESET_TARGET})

endfunction()

# -------------------------------------------------------------------------------------------------
# build API reference documentation using Doxygen and optionally Dot

function(CREATE_FROM_DOC_PRESET)
    set(oneValueArgs TARGET COMMENT PROJECT_LABEL)
    set(multiValueArgs )
    cmake_parse_arguments(CREATE_FROM_DOC_PRESET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    # provides the following variables:
    # - CREATE_FROM_DOC_PRESET_TARGET
    # - CREATE_FROM_DOC_PRESET_COMMENT
    # - CREATE_FROM_DOC_PRESET_PROJECT_LABEL

    # Set up arguments for doxygen_configure_doxyfile.py.
    if(MDL_DEPENDENCY_DOT_FOUND)
        set(_HAVE_DOT_ARG "YES")
        # Doxygen requires the directory containing the executable.
        get_filename_component(_DOT_PATH_ARG "${MDL_DEPENDENCY_DOT_PATH}" DIRECTORY)
        if(WINDOWS)
            file(TO_NATIVE_PATH "${_DOT_PATH_ARG}" _DOT_PATH_ARG)
        endif()
        set(_IMAGE_PATH_ARG ".")
    else()
        set(_HAVE_DOT_ARG "NO")
        # An empty string would be natural, but causes problems for argument passing below. Handle that
        # in the script itself.
        set(_DOT_PATH_ARG ".")
        set(_IMAGE_PATH_ARG ".")
    endif()

    # Set up the dot wrapper (Linux and MacOS only). This relies on the shebang line in the wrapper
    # since doxygen insists on invoking dot as "dot" (or "dot.exe") and does not accept an explicit
    # command (only the path).
    if(MDL_DEPENDENCY_DOT_FOUND AND (LINUX OR MACOSX))
        # Set up variables for configure_file() below.
        set(REAL_DOT "${_DOT_PATH_ARG}/dot")
        if(LINUX)
            set(SED_I "-i.bak")
        else()
           set(SED_I "-i .bak")
        endif()
        # Configure the dot wrapper.
        configure_file(
            ${MDL_BASE_FOLDER}/doc/build/doxygen_dot_wrapper.sh
            ${CMAKE_CURRENT_BINARY_DIR}/dot
            @ONLY)
        # User dot wrapper.
        set(_DOT_PATH_ARG ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    add_custom_target(
        ${CREATE_FROM_DOC_PRESET_TARGET}
        COMMAND
            ${python_PATH}
            ${MDL_BASE_FOLDER}/doc/build/doxygen_configure_doxyfile.py
            ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            ${_HAVE_DOT_ARG}
            ${_DOT_PATH_ARG}
            ${_IMAGE_PATH_ARG}
        COMMAND
            ${MDL_DEPENDENCY_DOXYGEN_PATH}
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        COMMAND
            ${python_PATH}
            ${MDL_BASE_FOLDER}/doc/build/doxygen_postprocess_css.py
        COMMENT
            ${CREATE_FROM_DOC_PRESET_COMMENT}
        VERBATIM
        WORKING_DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR})

    set_target_properties(
        ${CREATE_FROM_DOC_PRESET_TARGET}
        PROPERTIES
            ADDITIONAL_CLEAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/html
            PROJECT_LABEL ${CREATE_FROM_DOC_PRESET_PROJECT_LABEL}
            FOLDER "doc")

    target_print_log_header(
        TARGET ${CREATE_FROM_DOC_PRESET_TARGET}
        TYPE DOCUMENTATION)

endfunction()

# -------------------------------------------------------------------------------------------------
# unit tests

function(CREATE_UNIT_TEST)
    set(options USES_IDIFF)
    set(oneValueArgs NAME)
    set(multiValueArgs HEADERS SOURCES DEPENDS LINK_LIBRARIES COMPILE_DEFINITIONS RUNTIME_DEPENDS LIBRARY_PATHS VS_PROJECT_NAME)
    cmake_parse_arguments(CREATE_UNIT_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT MDL_ENABLE_UNIT_TESTS)
        return()
    endif()

    # Defaults
    if(NOT CREATE_UNIT_TEST_NAME)
        set(CREATE_UNIT_TEST_NAME test)
    endif()
    if(NOT CREATE_UNIT_TEST_SOURCES)
        set(CREATE_UNIT_TEST_SOURCES test.cpp)
    endif()
    if(NOT CREATE_UNIT_TEST_VS_PROJECT_NAME)
        set(CREATE_UNIT_TEST_VS_PROJECT_NAME ${CREATE_UNIT_TEST_NAME})
    endif()

    set(CREATE_UNIT_TEST_TARGET ${PROJECT_NAME}-${CREATE_UNIT_TEST_NAME})

    create_from_base_preset(
        TARGET ${CREATE_UNIT_TEST_TARGET}
        TYPE EXECUTABLE
        OUTPUT_NAME ${CREATE_UNIT_TEST_NAME}
        SOURCES ${CREATE_UNIT_TEST_SOURCES}
        VS_PROJECT_NAME ${CREATE_UNIT_TEST_VS_PROJECT_NAME}
    )

    # Add build dependencies.
    if(CREATE_UNIT_TEST_DEPENDS)
        target_add_dependencies(
            TARGET ${CREATE_UNIT_TEST_TARGET}
            DEPENDS ${CREATE_UNIT_TEST_DEPENDS}
        )
    endif()

    # Support link libraries like Boost::filesystem that have not yet an equivalence in
    # target_add_dependencies().
    if(CREATE_UNIT_TEST_LINK_LIBRARIES)
        target_link_libraries(
            ${CREATE_UNIT_TEST_TARGET} PRIVATE ${CREATE_UNIT_TEST_LINK_LIBRARIES})
    endif()

    # Add compile definitions.
    if(CREATE_UNIT_TEST_COMPILE_DEFINITIONS)
        target_compile_definitions(
            ${CREATE_UNIT_TEST_TARGET}
            PRIVATE ${CREATE_UNIT_TEST_COMPILE_DEFINITIONS}
        )
    endif()

    # On Windows, always add /bigobj (could be turned into an option for selected tests).
    if(WINDOWS)
        target_compile_options(
            ${CREATE_UNIT_TEST_TARGET}
            PRIVATE "/bigobj"
        )
    endif()

    # Add runtime dependencies as build dependencies if requested.
    if(MDL_TREAT_RUNTIME_DEPS_AS_BUILD_DEPS AND CREATE_UNIT_TEST_RUNTIME_DEPENDS)
        target_add_dependencies(
            TARGET ${CREATE_UNIT_TEST_TARGET}
            DEPENDS ${CREATE_UNIT_TEST_RUNTIME_DEPENDS}
            NO_RUNTIME_COPY
            NO_LINKING
            NO_INCLUDE
        )
    endif()

    # Add the CTest test for this target.
    add_test(
        NAME ${CREATE_UNIT_TEST_TARGET}
        COMMAND ${CREATE_UNIT_TEST_NAME}
    )

    # Common label for all unit tests.
    set_property(
        TEST ${CREATE_UNIT_TEST_TARGET}
        PROPERTY LABELS "unit_test"
    )

    # Obtain environment variable to modify for LIBRARY_PATHS option.
    if(LINUX)
        set(_PREFIX "LD_LIBRARY_PATH")
    elseif(MACOSX)
        set(_PREFIX "DYLD_LIBRARY_PATH")
    elseif(WINDOWS)
        set(_PREFIX "PATH")
    endif()

    # Obtain last directory component to add to LIBRARY_PATHS values.
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_GENERATOR_IS_MULTI_CONFIG)
        set(_SUFFIX "$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Release>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>")
    else()
        set(_SUFFIX ${CMAKE_BUILD_TYPE})
    endif()

    # Construct sequence of environment modifications for each LIBRARY_PATHS value. Repeat for
    # each argument due to ambiguity of semicolon on Windows (separator of values inside PATH
    # or between different environment variables.)
    set(_PATHS "")
    set(_SH_PATHS "")
    set(_VS_PATHS "")
    foreach(_PATH ${CREATE_UNIT_TEST_LIBRARY_PATHS})
        set(_VALUE ${_PREFIX}=path_list_prepend:${_PATH}/${_SUFFIX})
        set(_SH_VALUE "export ${_PREFIX}=${_PATH}/${_SUFFIX}:$${_PREFIX}")
        set(_VS_VALUE "${_PATH}/$(Configuration)")
        list(APPEND _PATHS ${_VALUE})
        list(APPEND _SH_PATHS ${_SH_VALUE})
        list(APPEND _VS_PATHS ${_VS_VALUE})
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "- ctest env:      " ${_VALUE})
        endif()
    endforeach()

    # Set PATH for idiff from OpenImageIO.
    if(CREATE_UNIT_TEST_USES_IDIFF)
        if(NOT MDL_DEPENDENCY_OPENIMAGEIO_IDIFF)
            message(WARNING "The idiff tool from OpenImageIO is required for this test, but was not found.")
        else()
            get_filename_component(_IDIFF_PATH ${MDL_DEPENDENCY_OPENIMAGEIO_IDIFF} DIRECTORY)
            set(_VALUE PATH=path_list_prepend:${_IDIFF_PATH})
            set(_SH_VALUE "export PATH=${_IDIFF_PATH}:$PATH")
            set(_VS_VALUE "${_IDIFF_PATH}")
            list(APPEND _PATHS ${_VALUE})
            list(APPEND _SH_PATHS ${_SH_VALUE})
            list(APPEND _VS_PATHS ${_VS_VALUE})
            if(MDL_LOG_DEPENDENCIES)
                message(STATUS "- ctest env:      " ${_VALUE})
            endif()
        endif()
    endif()

    # Flatten paths list (not necessary for _VS_PATHS).
    string(JOIN ";" _PATHS ${_PATHS})
    string(JOIN "\n" _SH_PATHS ${_SH_PATHS})

    set(_MI_VARS "MI_SRC=set:${MDL_SRC_FOLDER};MI_DATA=unset:;MI_LARGE_DATA=unset:")

    # Set environment variables for LIBRARY_PATHS option and MI variables.
    set_property(
        TEST ${CREATE_UNIT_TEST_TARGET}
        PROPERTY ENVIRONMENT_MODIFICATION "${_MI_VARS};${_PATHS}"
    )

    # On Unix, add a script to run the test manually.
    # Primarily for running in a debugger, otherwise launching via ctest is preferred.
    if(NOT WINDOWS)
        configure_file(
            "${MDL_BASE_FOLDER}/cmake/run_test.sh.in"
            "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/run_${CREATE_UNIT_TEST_NAME}.sh"
            @ONLY
            FILE_PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        )
    endif()

    # Set up environment for VS debugger.
    target_add_vs_debugger_env_path(
        TARGET ${CREATE_UNIT_TEST_TARGET}
        PATHS ${_VS_PATHS}
    )
    target_add_vs_debugger_env_var(
        TARGET ${CREATE_UNIT_TEST_TARGET}
        VARS
            "MI_SRC=${MDL_SRC_FOLDER}"
            "MI_DATA="
            "MI_LARGE_DATA="
    )
    target_create_vs_user_settings(TARGET ${CREATE_UNIT_TEST_TARGET})
endfunction()

function(ADD_UNIT_TESTS)
    set(options POST)
    set(oneValueArgs)
    set(multiValueArgs)
    cmake_parse_arguments(ADD_UNIT_TESTS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if(MDL_ENABLE_UNIT_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        # postpone the test until all regular targets are created
        if(ADD_UNIT_TESTS_POST)
            if(MDL_TEST_LIST_POST STREQUAL "") # first element
                set(MDL_TEST_LIST_POST "${CMAKE_CURRENT_SOURCE_DIR}/tests" CACHE INTERNAL "list of test directories to add after regular targets are defined")
            else()
                set(MDL_TEST_LIST_POST "${MDL_TEST_LIST_POST};${CMAKE_CURRENT_SOURCE_DIR}/tests" CACHE INTERNAL "list of test directories to add after regular targets are defined")
            endif()
        else()
            add_subdirectory(tests)
        endif()
    endif()
endfunction()
