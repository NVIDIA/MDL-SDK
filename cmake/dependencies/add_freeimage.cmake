# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# assuming the find_freeimage_ext script was successful
# if not, this is an error case. The corresponding project should not have been selected for build.
if(NOT MDL_FREEIMAGE_FOUND)
    message(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
else()

    # add the include directory
    target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${MDL_DEPENDENCY_FREEIMAGE_INCLUDE}
        )

    # link static/shared object
    if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)
        if(WINDOWS)
            # static library (part)
            target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                PRIVATE
                    ${LINKER_WHOLE_ARCHIVE}
                    ${MDL_DEPENDENCY_FREEIMAGE_LIBS}
                    ${LINKER_NO_WHOLE_ARCHIVE}
                )
        else()
            # shared library
            target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                PRIVATE
                    ${LINKER_NO_AS_NEEDED}
                    ${MDL_DEPENDENCY_FREEIMAGE_SHARED}
                    ${LINKER_AS_NEEDED}
                )
        endif()
    endif()

    # copy runtime dependencies
    # copy system libraries only on windows, we assume the libraries are installed in a unix environment
    if(NOT __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY AND WINDOWS)
        target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
            FILES
                ${MDL_DEPENDENCY_FREEIMAGE_SHARED}
            )
    endif()
endif()
