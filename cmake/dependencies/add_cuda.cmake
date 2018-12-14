# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# assuming the find_cuda_ext script was successful
# if not, this is an error case. The corresponding project should not have been selected for build.
if(NOT MDL_ENABLE_CUDA_EXAMPLES)
    message(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
else()

    # headers
    target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${MDL_DEPENDENCY_CUDA_INCLUDE}
        )

    if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)
        if(WINDOWS)
            target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                PRIVATE
                    ${MDL_DEPENDENCY_CUDA_LIBS} # static library (part)
                )
        else()
            # shared library
            target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                PRIVATE
                    ${LINKER_NO_AS_NEEDED}
                    ${MDL_DEPENDENCY_CUDA_SHARED}
                    ${LINKER_AS_NEEDED}
                )

            if(MACOSX)
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                    PRIVATE
                        ${MDL_DEPENDENCY_CUDA_LIBS} # cuda framework
                    )
            endif()
        endif()
    endif()
endif()