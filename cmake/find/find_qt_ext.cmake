function(FIND_QT_EXT)

    set(Qt5_DIR "" CACHE PATH "Directory that contains Qt for the selected compiler, e.g., ../Qt/5.10.1/msvc2017_64")

    if(NOT MDL_ENABLE_QT_EXAMPLES)
        message(WARNING "Examples that require Qt are disabled. Enable the option 'MDL_ENABLE_QT_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    #-----------------------------------------------------------------------------------------------

    # probe the core packages
    # if found, the Qt5_DIR is set to <qt root dir>/lib/cmake/qt5
    find_package(Qt5 COMPONENTS Core HINTS ${Qt5_DIR})

    if(NOT ${Qt5_FOUND})
        message(FATAL_ERROR "The dependency \"qt\" could not be resolved. Install Qt on your system or specify the 'Qt5_DIR' variable.")
    endif()

endfunction()
