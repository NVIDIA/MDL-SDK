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
