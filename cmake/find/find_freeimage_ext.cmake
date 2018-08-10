function(FIND_FREEIMAGE_EXT)

    set(FREEIMAGE_DIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the freeimage library and the corresponding headers.")
    #-----------------------------------------------------------------------------------------------

    # try to find FreeImage manually
    set(_FREEIMAGE_INCLUDE "NOTFOUND")
    set(_FREEIMAGE_LIB "NOTFOUND")
    set(_FREEIMAGE_SHARED "NOTFOUND")

    find_file(_FREEIMAGE_HEADER_FILE "FreeImage.h" 
        HINTS 
            ${FREEIMAGE_DIR}
            ${FREEIMAGE_DIR}/include
            ${FREEIMAGE_DIR}/Dist/x64
            /usr/include
        )
    mark_as_advanced(_FREEIMAGE_HEADER_FILE)
    mark_as_advanced(_FREEIMAGE_SHARED)
    mark_as_advanced(_FREEIMAGE_LIB)

    if(EXISTS ${_FREEIMAGE_HEADER_FILE})
        get_filename_component(_FREEIMAGE_INCLUDE ${_FREEIMAGE_HEADER_FILE} PATH)

        if(WINDOWS)
            # assuming that the windows (x64) binaries from http://freeimage.sourceforge.net/download.html are used
            find_file(_FREEIMAGE_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}freeimage${CMAKE_STATIC_LIBRARY_SUFFIX}" 
                HINTS 
                    ${FREEIMAGE_DIR}
                    ${FREEIMAGE_DIR}/Dist/x64
                )

            find_file(_FREEIMAGE_SHARED "${CMAKE_SHARED_LIBRARY_PREFIX}freeimage${CMAKE_SHARED_LIBRARY_SUFFIX}" 
                HINTS 
                    ${FREEIMAGE_DIR}
                    ${FREEIMAGE_DIR}/Dist/x64
                )

        elseif(LINUX)
            # assuming the 'freeimage-dev' package is installed
            # or freeimage was build manually and follows a common folder structure
            set(_FREEIMAGE_LIB "") # not used
            find_file(_FREEIMAGE_SHARED
                NAMES
                    "${CMAKE_SHARED_LIBRARY_PREFIX}freeimage${CMAKE_SHARED_LIBRARY_SUFFIX}"
                    "libfreeimage.so"
                HINTS 
                    ${FREEIMAGE_DIR}
                    ${FREEIMAGE_DIR}/lib64
                    ${FREEIMAGE_DIR}/lib
                    /usr/lib64
                    /usr/lib/x86_64-linux-gnu
                    /usr/lib
                )

            if(NOT EXISTS ${_FREEIMAGE_SHARED})
                set(_OS_MESSAGE " install the 'libfreeimage-dev' package or")
            endif()
        endif()
    endif()

    # error if dependencies can not be resolved
    if(NOT EXISTS ${_FREEIMAGE_INCLUDE} OR (WINDOWS AND NOT EXISTS ${_FREEIMAGE_LIB}) OR NOT EXISTS ${_FREEIMAGE_SHARED})
        message(STATUS "FREEIMAGE_DIR: ${FREEIMAGE_DIR}")
        message(STATUS "_FREEIMAGE_HEADER_FILE: ${_FREEIMAGE_HEADER_FILE}")
        message(STATUS "_FREEIMAGE_INCLUDE: ${_FREEIMAGE_INCLUDE}")
        message(STATUS "_FREEIMAGE_LIB: ${_FREEIMAGE_LIB}")
        message(STATUS "_FREEIMAGE_SHARED: ${_FREEIMAGE_SHARED}")
        message(FATAL_ERROR "The dependency \"freeimage\" could not be resolved. Please${_OS_MESSAGE} specify 'FREEIMAGE_DIR'.")
    endif()

    # store path that are later used in the add_freeimage.cmake
    set(MDL_DEPENDENCY_FREEIMAGE_INCLUDE ${_FREEIMAGE_INCLUDE} CACHE INTERNAL "freeimage headers")
    set(MDL_DEPENDENCY_FREEIMAGE_LIBS ${_FREEIMAGE_LIB} CACHE INTERNAL "freeimage libs")
    set(MDL_DEPENDENCY_FREEIMAGE_SHARED ${_FREEIMAGE_SHARED} CACHE INTERNAL "freeimage shared libs")
    set(MDL_FREEIMAGE_FOUND ON CACHE INTERNAL "")

endfunction()
