# check if CUDA is available
# Note, this enables CUDA for all projects (only of concern for Visual Studio)
if(NOT MDL_ENABLE_CUDA_EXAMPLES)
    message(WARNING "Examples that require CUDA are disabled. Enable the option 'MDL_ENABLE_CUDA_EXAMPLES' to re-enable them.")
else()
    # use the c++ compiler as host compiler
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
    # try to enable cuda, can fail even though the compiler is found (see error message of second call)
    message(STATUS "If you don't want to use CUDA based examples, please disable 'MDL_ENABLE_CUDA_EXAMPLES'.")
    enable_language(CUDA OPTIONAL)
    if(NOT EXISTS ${CMAKE_CUDA_COMPILER})
        set(MDL_ENABLE_CUDA_EXAMPLES OFF CACHE BOOL "Enable examples that require CUDA." FORCE)
        message(STATUS "Failed enable CUDA. Please install the CUDA SDK and then enable 'MDL_ENABLE_CUDA_EXAMPLES'.")
        enable_language(CUDA) # call again to get an error
    endif()
endif()
