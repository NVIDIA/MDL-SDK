function(FIND_BOOST_EXT)

    set(BOOST_INCLUDEDIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the boost headers.")

    if(EXISTS ${BOOST_INCLUDEDIR})
        # remove trailing slashes/backslashes
        string(REGEX REPLACE "[/\\\\]$" "" _BOOST_INCLUDEDIR "${BOOST_INCLUDEDIR}")
        set(BOOST_INCLUDEDIR ${_BOOST_INCLUDEDIR} CACHE PATH "Directory that contains the boost headers." FORCE)
        set(Boost_NO_SYSTEM_PATHS ON CACHE INTERNAL "")
        set(Boost_NO_BOOST_CMAKE ON CACHE INTERNAL "")
        #set(Boost_DEBUG ON)
    endif()

    # headers only
    find_package(Boost QUIET)
    mark_as_advanced(CLEAR BOOST_INCLUDEDIR)
    set(Boost_FOUND ${Boost_FOUND} CACHE INTERNAL "Dependency boost has been resolved.")

    if(NOT Boost_FOUND)
        message(STATUS "BOOST_INCLUDEDIR: ${BOOST_INCLUDEDIR}")
        message(FATAL_ERROR "The dependency \"boost\" could not be resolved. Please specify 'BOOST_INCLUDEDIR'.")
    endif()
    
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] BOOST_INCLUDEDIR:                   ${BOOST_INCLUDEDIR}")
    endif()

endfunction()
