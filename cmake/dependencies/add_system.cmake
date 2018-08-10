# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# handling of common system dependencies
foreach (system_component ${__TARGET_ADD_DEPENDENCY_COMPONENTS})

    # ld
    if(${system_component} STREQUAL ld AND LINUX)
        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
            PRIVATE
                 ${CMAKE_DL_LIBS}
            )

    # thread
    elseif(${system_component} STREQUAL threads)
        find_package(Threads REQUIRED QUITE)
        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
            PRIVATE
                  ${CMAKE_THREAD_LIBS_INIT}
            )

    endif()
endforeach()
