# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# if available on the current platform, 
# we add these libraries/options for every shared object and executable

# dl
if(CMAKE_DL_LIBS)
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- depends on:     * ${CMAKE_DL_LIBS}")
    endif()
    target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${CMAKE_DL_LIBS}
        )
endif()

# threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED QUITE)
if(CMAKE_THREAD_LIBS_INIT)
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- depends on:     * ${CMAKE_THREAD_LIBS_INIT}")
    endif()
    target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${CMAKE_THREAD_LIBS_INIT}
        )
endif()
