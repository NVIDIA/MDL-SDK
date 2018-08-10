# -------------------------------------------------------------------------------------------------
# script expects the following variables:
# - __TARGET_ADD_DEPENDENCY_TARGET
# - __TARGET_ADD_DEPENDENCY_DEPENDS
# - __TARGET_ADD_DEPENDENCY_COMPONENTS
# - __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY
# - __TARGET_ADD_DEPENDENCY_NO_LINKING
# -------------------------------------------------------------------------------------------------

# we don't use find cuda here, we assume we can find all our dependencies relative to nvcc

#find the required packages
#find_package(Cuda REQUIRED)

if(EXISTS ${CMAKE_CUDA_COMPILER})
    get_filename_component(_CUDA_BIN_DIR ${CMAKE_CUDA_COMPILER} PATH)
    set(_CUDA_SDK_DIR ${_CUDA_BIN_DIR}/..)
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "                  (found CUDA using the compiler)")
    endif()
else()
    find_file(_CUDA_HEADER "include/cuda.h")
    if(_CUDA_HEADER)
        get_filename_component(_CUDA_INCLUDE_DIR ${_CUDA_HEADER} PATH)
        set(_CUDA_SDK_DIR ${_CUDA_INCLUDE_DIR}/..)
        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "                  (found CUDA using 'cuda.h')")
        endif()
    endif()
endif()

if(_CUDA_SDK_DIR)
    # add include directories
    target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${_CUDA_SDK_DIR}/include
            ${_CUDA_SDK_DIR}/curand_dev/include
        )
    if(WIN32)
        set(_CUDA_LIB_DIRECTORY ${_CUDA_SDK_DIR}/lib/x64)
        list(APPEND _CUDA_LIBRARIES "${_CUDA_LIB_DIRECTORY}/cuda.lib")
        list(APPEND _CUDA_LIBRARIES "${_CUDA_LIB_DIRECTORY}/cudart.lib")
    else()
        set(_CUDA_LIB_DIRECTORY ${_CUDA_SDK_DIR}/lib64)
        list(APPEND _CUDA_LIBRARIES "${_CUDA_LIB_DIRECTORY}/stubs/libcuda.so")
        list(APPEND _CUDA_LIBRARIES "${_CUDA_LIB_DIRECTORY}/libcudart.so")
    endif()

    # link dependencies
    if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)
        target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
            PRIVATE
                ${LINKER_NO_AS_NEEDED}
                ${_CUDA_LIBRARIES}
                ${LINKER_AS_NEEDED}
            )
    endif()
endif()
