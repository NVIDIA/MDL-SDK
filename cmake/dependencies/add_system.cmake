#*****************************************************************************
# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#*****************************************************************************

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
find_package(Threads REQUIRED QUIET)
if(CMAKE_THREAD_LIBS_INIT)
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- depends on:     * ${CMAKE_THREAD_LIBS_INIT}")
    endif()
    target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${CMAKE_THREAD_LIBS_INIT}
        )
endif()

# mini dumps
if(WINDOWS)
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "- depends on:     * dbghelp.lib")
    endif()
    target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
    PRIVATE
        "dbghelp.lib"
    )
endif()

