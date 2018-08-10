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
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs)
    cmake_parse_arguments(TARGET_BUILD_SETUP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # options depending on the target type
    get_target_property(_TARGET_TYPE ${TARGET_BUILD_SETUP_TARGET} TYPE)

    # very simple set of flags depending on the compiler instead of the combination of compiler, OS, ...
    # for more complex scenarios, replace that 
    target_compile_options(${TARGET_BUILD_SETUP_TARGET} 
        PUBLIC
            # general 
            "$<$<CONFIG:DEBUG>:-DDEBUG>"
            "$<$<CONFIG:DEBUG>:-D_DEBUG>"
            "$<$<CONFIG:RELEASE>:-DNDEBUG>"
            "-DMISTD=std"                       # Use system std lib rather than a custom one
            "-DBIT64=1"
            "-DX86=1"

            # platform specific
            "$<$<CXX_COMPILER_ID:MSVC>:-DMI_PLATFORM=\"nt-x86-64-vc14\">"
            "$<$<CXX_COMPILER_ID:MSVC>:-DMI_PLATFORM_WINDOWS>"
            "$<$<CXX_COMPILER_ID:MSVC>:-DWIN_NT>"
            "$<$<CXX_COMPILER_ID:MSVC>:-D_MSC_VER=${MSVC_VERSION}>"
            "$<$<CXX_COMPILER_ID:MSVC>:/MT$<$<CONFIG:Debug>:d>>"
            "$<$<CXX_COMPILER_ID:MSVC>:/MP>"
            
            "$<$<CXX_COMPILER_ID:GNU>:-DMI_PLATFORM=\"linux-x86-64-gcc\">" #todo add major version number
            "$<$<CXX_COMPILER_ID:GNU>:-DMI_PLATFORM_UNIX>"
            "$<$<CXX_COMPILER_ID:GNU>:-fPIC>"   # position independent code since we will build a shared object
            "$<$<CXX_COMPILER_ID:GNU>:-m64>"    # sets int to 32 bits and long and pointer to 64 bits and generates code for x86-64 architecture
            "$<$<CXX_COMPILER_ID:GNU>:-fno-strict-aliasing>"

            "$<$<CXX_COMPILER_ID:GNU>:-march=nocona>"
            "$<$<PLATFORM_ID:Linux>:-DHAS_SSE>"

            # debug symbols
            "$<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:GNU>>:-gdwarf-3>"
            "$<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:GNU>>:-gstrict-dwarf>"

            # additional user defined options
            ${MDL_ADDITIONAL_COMPILER_OPTIONS}

        PRIVATE 
            # enable additional warnings
            "$<$<CXX_COMPILER_ID:GNU>:-Wall>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wvla>"

            # temporary ignored warnings
            "$<$<CXX_COMPILER_ID:MSVC>:-D_CRT_SECURE_NO_WARNINGS>"
            "$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS>"
            "$<$<CXX_COMPILER_ID:MSVC>:/wd4267>" # Suppress Warning C4267	'argument': conversion from 'size_t' to 'int', possible loss of data

            "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CXX>>:-Wno-placement-new>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-parentheses>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-unused-but-set-variable>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-unused-local-typedefs>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>"
            "$<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>"
        )

    # setup specific to shared libraries
    if (_TARGET_TYPE STREQUAL "SHARED_LIBRARY" OR _TARGET_TYPE STREQUAL "MODULE_LIBRARY")
        target_compile_options(${PROJECT_NAME} 
            PRIVATE
                "-DMI_DLL_BUILD"            # export/import macro
                "-DMI_ARCH_LITTLE_ENDIAN"   # used in the .rc files
                "-DTARGET_FILENAME=\"$<TARGET_FILE_NAME:${PROJECT_NAME}>\""     # used in .rc
            )
    endif()
endfunction()

# -------------------------------------------------------------------------------------------------
# setup IDE specific stuff
function(SETUP_IDE)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(SETUP_IDE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    # not required without visual studio or xcode
    if(NOT MSVC AND NOT MSVC_IDE)
        return()
    endif()

    # compute the folder relative to the top level
    get_filename_component(FOLDER_PREFIX ${CMAKE_SOURCE_DIR} REALPATH)
    get_filename_component(FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} REALPATH)

    string(LENGTH ${FOLDER_PREFIX} OFFSET)
    string(LENGTH ${FOLDER_NAME} TOTAL_LENGTH)
    math(EXPR OFFSET ${OFFSET}+1)
    math(EXPR REMAINING ${TOTAL_LENGTH}-${OFFSET})
    string(SUBSTRING ${FOLDER_NAME} ${OFFSET} ${REMAINING} FOLDER_NAME)
    
    set_target_properties(${SETUP_IDE_TARGET} PROPERTIES 
        VS_DEBUGGER_WORKING_DIRECTORY           "$(OutDir)"         # working directory
        FOLDER                                  ${FOLDER_NAME}      # folder
        MAP_IMPORTED_CONFIG_DEBUG               Debug
        MAP_IMPORTED_CONFIG_RELEASE             Release
        MAP_IMPORTED_CONFIG_MINSIZEREL          Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO      Release
        )

        # keep the folder structure in visual studio
        foreach(_SOURCE ${SETUP_IDE_SOURCES})
            string(FIND ${_SOURCE} "/" _POS REVERSE)

            # file in project root
            if(${_POS} EQUAL -1)
                source_group("" FILES ${_SOURCE})
                continue()
            endif()

            # generated files
            math(EXPR _START ${_POS}-9)
            if(${_START} GREATER 0)
                string(SUBSTRING ${_SOURCE} ${_START} 9 FOLDER_NAME)
                if(FOLDER_NAME STREQUAL "generated")
                    source_group("generated" FILES ${_SOURCE})
                    continue()
                endif()
            endif()

            # relative files outside the current target
            if(${_SOURCE} MATCHES "^../.*")
                source_group("" FILES ${_SOURCE})
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
    MESSAGE(STATUS "---------------------------------------------------------------------------------")
    MESSAGE(STATUS "PROJECT_NAME:     ${TARGET_PRINT_LOG_HEADER_TARGET}   (${TARGET_PRINT_LOG_HEADER_TYPE})")

    if(TARGET_PRINT_LOG_HEADER_VERSION)
        MESSAGE(STATUS "VERSION:          ${TARGET_PRINT_LOG_HEADER_VERSION}")
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
    set(oneValueArgs TARGET RELATIVE)
    set(multiValueArgs FILES)
    cmake_parse_arguments(TARGET_COPY_TO_OUTPUT_DIR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    foreach(_ELEMENT ${TARGET_COPY_TO_OUTPUT_DIR_FILES} )

        # handle absolute and relative paths
        if(TARGET_COPY_TO_OUTPUT_DIR_RELATIVE)
            set(_SOURCE_FILE ${TARGET_COPY_TO_OUTPUT_DIR_RELATIVE}/${_ELEMENT})
            set(_FOLDER_NAME ${_ELEMENT})
        else()
            set(_SOURCE_FILE ${_ELEMENT})
            get_filename_component(_FOLDER_NAME ${_ELEMENT} NAME)
            set (_ELEMENT "")
        endif()

        # handle directories and files slightly different
        if(IS_DIRECTORY ${_SOURCE_FILE})
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- folder to copy: ${_SOURCE_FILE}")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${_SOURCE_FILE} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>/${_FOLDER_NAME}
            )
        else()   
            if(MDL_LOG_FILE_DEPENDENCIES)
                MESSAGE(STATUS "- file to copy:   ${_SOURCE_FILE}")
            endif()
            add_custom_command(
                TARGET ${TARGET_COPY_TO_OUTPUT_DIR_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_SOURCE_FILE} $<TARGET_FILE_DIR:${TARGET_COPY_TO_OUTPUT_DIR_TARGET}>/${_ELEMENT}
            )
        endif()
    endforeach()


endfunction()

# -------------------------------------------------------------------------------------------------
# Adds a dependency to a target, meant as shortcut for several more or less similar examples
# Meant for internal use by the function below
function(__TARGET_ADD_DEPENDENCY)
    set(options NO_RUNTIME_COPY NO_LINKING)
    set(oneValueArgs TARGET DEPENDS)
    set(multiValueArgs COMPONENTS)
    cmake_parse_arguments(__TARGET_ADD_DEPENDENCY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - __TARGET_ADD_DEPENDENCY_TARGET
    # - __TARGET_ADD_DEPENDENCY_DEPENDS
    # - __TARGET_ADD_DEPENDENCY_COMPONENTS
    # - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
    # - __TARGET_ADD_DEPENDENCY_NO_LINKING

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
        list(GET _RESULTS 1 __TARGET_ADD_DEPENDENCY_DEPENDS)
    endif()

    # log dependency
     if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- depends on:     " ${__TARGET_ADD_DEPENDENCY_DEPENDS})
    endif()

    # customized depencency scripts have highest priority
    # to use it, define a variable like this: OVERRIDE_DEPENDENCY_SCRIPT_<upper case depencency name>
    string(TOUPPER ${__TARGET_ADD_DEPENDENCY_DEPENDS} __TARGET_ADD_DEPENDENCY_DEPENDS_UPPER)
    if(OVERRIDE_DEPENDENCY_SCRIPT_${__TARGET_ADD_DEPENDENCY_DEPENDS_UPPER})
        set(_FILE_TO_INCLUDE ${OVERRIDE_DEPENDENCY_SCRIPT_${__TARGET_ADD_DEPENDENCY_DEPENDS_UPPER}})
    # if no custom script is defined, we check if there is a default one
    else()
        set(_FILE_TO_INCLUDE "${MDL_BASE_FOLDER}/cmake/dependencies/add_${__TARGET_ADD_DEPENDENCY_DEPENDS}.cmake")
    endif()

    # check if there is a add_dependency file to include (custom or default)
    if(EXISTS ${_FILE_TO_INCLUDE})
        include(${_FILE_TO_INCLUDE})
    # if not, we try to interpret the dependency as a target contained in the top level project
    else()
        
        # if this is no internal dependency we use the default find mechanism
        if(NOT TARGET ${__TARGET_ADD_DEPENDENCY_DEPENDS})
            # checks if there is such a "subproject"
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
            _TARGET_TYPE STREQUAL "MODULE_LIBRARY")

            # add the dependency to the target
            if(__TARGET_ADD_DEPENDENCY_NO_LINKING)
                # if NO_LINKING was specified, we add the include directories only
                target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
                    PRIVATE
                        $<TARGET_PROPERTY:${__TARGET_ADD_DEPENDENCY_DEPENDS},INTERFACE_INCLUDE_DIRECTORIES>
                    )
            else()
                # include directories and link dependencies
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        ${__TARGET_ADD_DEPENDENCY_DEPENDS}
                    )
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
    set(options NO_RUNTIME_COPY NO_LINKING)
    set(oneValueArgs TARGET)
    set(multiValueArgs DEPENDS COMPONENTS)
    cmake_parse_arguments(TARGET_ADD_DEPENDENCIES "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_DEPENDENCIES_TARGET
    # - TARGET_ADD_DEPENDENCIES_DEPENDS
    # - TARGET_ADD_DEPENDENCIES_COMPONENTS
    # - TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY
    # - TARGET_ADD_DEPENDENCIES_NO_LINKING

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

    if(TARGET_ADD_DEPENDENCIES_NO_LINKING)
        set(TARGET_ADD_DEPENDENCIES_NO_LINKING NO_LINKING)
    else()
        set(TARGET_ADD_DEPENDENCIES_NO_LINKING "")
    endif()


    # in case we have components we pass them to the single dependency
    if(TARGET_ADD_DEPENDENCIES_COMPONENTS)
        __target_add_dependency(
            TARGET      ${TARGET_ADD_DEPENDENCIES_TARGET}
            DEPENDS     ${TARGET_ADD_DEPENDENCIES_DEPENDS}
            COMPONENTS  ${TARGET_ADD_DEPENDENCIES_COMPONENTS}
            ${TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY}
            ${TARGET_ADD_DEPENDENCIES_NO_LINKING}
            )
    # if not, we iterate over the list of dependencies and pass no components
    else()
        foreach(_DEP ${TARGET_ADD_DEPENDENCIES_DEPENDS})
            __target_add_dependency(
                TARGET      ${TARGET_ADD_DEPENDENCIES_TARGET}
                DEPENDS     ${_DEP}
                ${TARGET_ADD_DEPENDENCIES_NO_RUNTIME_COPY}
                ${TARGET_ADD_DEPENDENCIES_NO_LINKING}
                )
        endforeach()
    endif()
endfunction()


# -------------------------------------------------------------------------------------------------
# Adds a tool dependency to a target, meant as shortcut for several more or less similar examples.
# This also works for tools that are part of the build, see scripts in the 'cmake/tools' subfolder.
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
    set(options)
    set(oneValueArgs TARGET VERSION TYPE NAMESPACE)
    set(multiValueArgs SOURCES ADDITIONAL_INCLUDE_DIRS)
    cmake_parse_arguments(CREATE_FROM_BASE_PRESET "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

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

    # default namespace is mdl_sdk
    if(NOT CREATE_FROM_BASE_PRESET_NAMESPACE)
        set( CREATE_FROM_BASE_PRESET_NAMESPACE mdl_sdk)
    endif()

    # add empty pch
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/pch.h)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pch.h "")
    endif()
    # list(APPEND CREATE_FROM_BASE_PRESET_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/pch.h)

    # create target and alias
    if(CREATE_FROM_BASE_PRESET_TYPE STREQUAL "STATIC" OR CREATE_FROM_BASE_PRESET_TYPE STREQUAL "SHARED")
        add_library(${CREATE_FROM_BASE_PRESET_TARGET} ${CREATE_FROM_BASE_PRESET_TYPE} ${CREATE_FROM_BASE_PRESET_SOURCES})
        add_library(${CREATE_FROM_BASE_PRESET_NAMESPACE}::${CREATE_FROM_BASE_PRESET_TARGET} ALIAS ${CREATE_FROM_BASE_PRESET_TARGET})
    elseif(CREATE_FROM_BASE_PRESET_TYPE STREQUAL "EXECUTABLE")
        add_executable(${CREATE_FROM_BASE_PRESET_TARGET} ${CREATE_FROM_BASE_PRESET_SOURCES})
    else()
        message(FATAL_ERROR "Unexpected Type for target '${CREATE_FROM_BASE_PRESET_TARGET}': ${CREATE_FROM_BASE_PRESET_TYPE}.")
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

    # includes used .rc
    if(WINDOWS OR CREATE_FROM_BASE_PRESET_TYPE STREQUAL "SHARED")
        target_include_directories(${CREATE_FROM_BASE_PRESET_TARGET} 
            PRIVATE
                ${MDL_SRC_FOLDER}/base/system/version    
            )
    endif()

    # compiler flags and defines
    target_build_setup(TARGET ${CREATE_FROM_BASE_PRESET_TARGET})

    # configure visual studio and maybe other IDEs
    setup_ide(TARGET ${CREATE_FROM_BASE_PRESET_TARGET} 
        SOURCES ${CREATE_FROM_BASE_PRESET_SOURCES})

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
#     )
#
function(TARGET_ADD_CUDA_PTX_RULE)
    set(options)
    set(oneValueArgs TARGET)
    set(multiValueArgs CUDA_SOURCES DEPENDS)
    cmake_parse_arguments(TARGET_ADD_CUDA_PTX_RULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    # provides the following variables:
    # - TARGET_ADD_CUDA_PTX_RULE_TARGET
    # - TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES
    # - TARGET_ADD_CUDA_PTX_RULE_DEPENDS

    # create PTX target
    add_library(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX OBJECT ${TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES})
    set_target_properties(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX PROPERTIES 
        CUDA_PTX_COMPILATION ON
        )

    # options
    target_compile_options(${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    )

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

    # build ptx when building the project
    add_dependencies(${TARGET_ADD_CUDA_PTX_RULE_TARGET} ${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX)

    # post build
    foreach(_SRC ${TARGET_ADD_CUDA_PTX_RULE_CUDA_SOURCES})

        get_filename_component(_SRC_NAME ${_SRC} NAME_WE)

        # mark files as generated to disable the check for existence during configure
        set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${_SRC_NAME}.ptx PROPERTIES GENERATED TRUE)
        # add to generated group
        source_group("generated" FILES ${CMAKE_CURRENT_BINARY_DIR}/${_SRC_NAME}.ptx)

        if(MDL_LOG_FILE_DEPENDENCIES)
            MESSAGE(STATUS "- file to copy:   ${_SRC_NAME}.ptx")
        endif()

        # copy ptx to example binary folder
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_SRC_NAME}.ptx
            DEPENDS $<TARGET_OBJECTS:${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX>
            COMMAND ${CMAKE_COMMAND} -E echo "Copy ${_SRC_NAME}.ptx to example dir..."
            COMMAND ${CMAKE_COMMAND} -E copy_if_different           
                $<TARGET_OBJECTS:${TARGET_ADD_CUDA_PTX_RULE_TARGET}_PTX>    # resulting ptx file
                $<TARGET_FILE_DIR:${TARGET_ADD_CUDA_PTX_RULE_TARGET}>       # to example binary dir
            )
    endforeach()
    
endfunction()