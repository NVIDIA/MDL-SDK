#if nothing is specified, we assume a debug build
set(_DEFAULT_BUILD_TYPE "Debug")

# no command line argument and no multi-config generator
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting default build type: '${_DEFAULT_BUILD_TYPE}'")
    set(CMAKE_BUILD_TYPE ${_DEFAULT_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)

    # define possible build types for cmake-gui and to be able to iterate over it during configuration
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo") # "MinSizeRel" "RelWithDebInfo"
else()
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Build types available to IDEs." FORCE)
endif()

# IDE Setup
set_property(GLOBAL PROPERTY USE_FOLDERS ON)  # Generate folders for IDE targets
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "_cmake")
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run not moc automatically (Qt)
# We select the files manually and add the generated files to the project and the IDEs. 
# set(CMAKE_AUTOMOC ON)

# custom properties
define_property(TARGET PROPERTY VS_DEBUGGER_PATHS
    BRIEF_DOCS "List of paths that are added to the Visual Studio debugger environment PATH."
    FULL_DOCS "List of paths that are added to the Visual Studio debugger environment PATH. Usually added by dependency scripts. Requires a call to 'TARGET_CREATE_VS_USER_SETTINGS' after adding all dependencies."
    )

# set platform variable
set(WINDOWS FALSE)
set(LINUX FALSE)
set(MACOSX FALSE)

if(WIN32 AND NOT UNIX)
    set(WINDOWS TRUE)
elseif(UNIX AND NOT WIN32 AND NOT APPLE)
    set(LINUX TRUE)
elseif(APPLE AND UNIX)
    set(MACOSX TRUE)
else()
    MESSAGE(AUTHOR_WARNING "System is currently not supported explicitly. Assuming Linux.")
    set(LINUX TRUE)
endif()

# remove the /MD flag cmake sets by default
set(CompilerFlags CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE)
foreach(CompilerFlag ${CompilerFlags})
  string(REPLACE "/MDd" "" ${CompilerFlag} "${${CompilerFlag}}")
  string(REPLACE "/MD" "" ${CompilerFlag} "${${CompilerFlag}}")
endforeach()

# mark internally used variables as advanced (since the are not supposed to be changed in CMake-Gui)
mark_as_advanced(MDL_BASE_FOLDER)
mark_as_advanced(MDL_INCLUDE_FOLDER)
mark_as_advanced(MDL_SRC_FOLDER)
mark_as_advanced(MDL_EXAMPLES_FOLDER)

# print system/build information for error reporting
if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_BASE_FOLDER:                    " ${MDL_BASE_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_INCLUDE_FOLDER:                 " ${MDL_INCLUDE_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_SRC_FOLDER:                     " ${MDL_SRC_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_EXAMPLES_FOLDER:                " ${MDL_EXAMPLES_FOLDER})
    MESSAGE(STATUS "[INFO] MDL_ADDITIONAL_COMPILER_OPTIONS:    " ${MDL_ADDITIONAL_COMPILER_OPTIONS})
    MESSAGE(STATUS "[INFO] CMAKE_VERSION:                      " ${CMAKE_VERSION})
    MESSAGE(STATUS "[INFO] CMAKE_SYSTEM_NAME:                  " ${CMAKE_SYSTEM_NAME})
    MESSAGE(STATUS "[INFO] WINDOWS:                            " ${WINDOWS})
    MESSAGE(STATUS "[INFO] LINUX:                              " ${LINUX})
    MESSAGE(STATUS "[INFO] MACOSX:                             " ${MACOSX})
    MESSAGE(STATUS "[INFO] CMAKE_BUILD_TYPE:                   " ${CMAKE_BUILD_TYPE})
    MESSAGE(STATUS "[INFO] CMAKE_GENERATOR:                    " ${CMAKE_GENERATOR})
    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    MESSAGE(STATUS "[INFO] GENERATOR_IS_MULTI_CONFIG:          " ${_GENERATOR_IS_MULTI_CONFIG})
    
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_ID:              " ${CMAKE_CXX_COMPILER_ID})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER_VERSION:         " ${CMAKE_CXX_COMPILER_VERSION})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_COMPILER:                 " ${CMAKE_CXX_COMPILER})
    if(MSVC)
        MESSAGE(STATUS "[INFO] MSVC_VERSION:                       " ${MSVC_VERSION})
        MESSAGE(STATUS "[INFO] MSVC_IDE:                           " ${MSVC_IDE})
    endif()
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS:                    " ${CMAKE_CXX_FLAGS})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_DEBUG:              " ${CMAKE_CXX_FLAGS_DEBUG})
    MESSAGE(STATUS "[INFO] CMAKE_CXX_FLAGS_RELEASE:            " ${CMAKE_CXX_FLAGS_RELEASE})
endif()

# check for dependencies
# pre-declare all options that are used
# in order to show them in CMake-Gui, even the script stops because of an error.
option(MDL_ENABLE_CUDA_EXAMPLES "Enable examples that require CUDA." ON)
option(MDL_ENABLE_OPENGL_EXAMPLES "Enable examples that require OpenGL." ON)
option(MDL_ENABLE_QT_EXAMPLES "Enable examples that require Qt." ON) 

include(${MDL_BASE_FOLDER}/cmake/find/find_cuda_ext.cmake)
find_cuda_ext()

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER_ID:             " ${CMAKE_CUDA_COMPILER_ID})
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER_VERSION:        " ${CMAKE_CUDA_COMPILER_VERSION})
    MESSAGE(STATUS "[INFO] CMAKE_CUDA_COMPILER:                " ${CMAKE_CUDA_COMPILER})
endif()

include(${MDL_BASE_FOLDER}/cmake/find/find_opengl_ext.cmake)
find_opengl_ext()

include(${MDL_BASE_FOLDER}/cmake/find/find_qt_ext.cmake)
find_qt_ext()

# examples could potentially use FreeImage directly
if(EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_freeimage_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_freeimage_ext.cmake)
    find_freeimage_ext()
endif()

if(EXISTS ${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    include(${MDL_BASE_FOLDER}/cmake/find/find_boost_ext.cmake)
    find_boost_ext()
endif()

if(MDL_LOG_PLATFORM_INFOS)
    MESSAGE(STATUS "[INFO] MDL_ENABLE_OPENGL_EXAMPLES:         " ${MDL_ENABLE_OPENGL_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_CUDA_EXAMPLES:           " ${MDL_ENABLE_CUDA_EXAMPLES})
    MESSAGE(STATUS "[INFO] MDL_ENABLE_QT_EXAMPLES:             " ${MDL_ENABLE_QT_EXAMPLES})
endif()