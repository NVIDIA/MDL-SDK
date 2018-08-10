#if nothing is specified, we assume a debug build
set(_DEFAULT_BUILD_TYPE "Debug")

# no command line argument and no multi-config generator
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting default build type: '${_DEFAULT_BUILD_TYPE}'")
    set(CMAKE_BUILD_TYPE ${_DEFAULT_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)

    # define possible build types for cmake-gui and to be able to iterate over it during configuration
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    
endif()
