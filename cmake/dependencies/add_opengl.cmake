# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# the opengl dependency consists of a set of packages

# assuming the find_opengl_ext script was successful
# if not, this is an error case. The corresponding project should not have been selected for build.
if(NOT MDL_ENABLE_OPENGL_EXAMPLES)
    message(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
else()

    # headers
    target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${MDL_DEPENDENCY_GL_INCLUDE}
            ${MDL_DEPENDENCY_GLEW_INCLUDE}
            ${MDL_DEPENDENCY_GLFW_INCLUDE}
        )

    if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)
        if(WINDOWS)
            target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                PRIVATE
                    ${MDL_DEPENDENCY_GL_SHARED} # imported projects
                    ${MDL_DEPENDENCY_GLEW_LIBS} # static library (part)
                    ${MDL_DEPENDENCY_GLFW_LIBS} # static library (part)
                )
        else()
            # shared library
            target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
                PRIVATE
                    ${MDL_DEPENDENCY_GL_SHARED}
                    ${MDL_DEPENDENCY_GLEW_SHARED}
                    ${MDL_DEPENDENCY_GLFW_SHARED}
                )
        endif()
    endif()

    # copy runtime dependencies
    # copy system libraries only on windows, we assume the libraries are installed in a unix environment
    if(NOT __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY AND WINDOWS)
        target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
            FILES
                ${MDL_DEPENDENCY_GLEW_SHARED}
                ${MDL_DEPENDENCY_GLFW_SHARED}
            )
    endif()
endif()
