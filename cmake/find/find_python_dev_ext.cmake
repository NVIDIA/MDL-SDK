#*****************************************************************************
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_PYTHON_DEV_EXT)
    if(NOT PYTHON_DIR)
        set(PYTHON_DIR "PYTHON_DIR-NOTFOUND" CACHE PATH "Directory that contains the python dev library and the corresponding headers.")
    endif()
    #-----------------------------------------------------------------------------------------------

    # use the cmake build-in find script (requires 3.12)
    if(EXISTS ${PYTHON_DIR})
        # message(STATUS " Try forcing PYTHON_DIR: ${PYTHON_DIR}")
        set(Python3_ROOT_DIR ${PYTHON_DIR})
    endif()
    find_package (Python3 COMPONENTS Interpreter Development)

    # use the found interpreter as the default python interpreter tool
    if(TARGET Python3::Interpreter)
        # message(STATUS " Found Python3::Interpreter")
        # message(STATUS " Python_EXECUTABLE: ${Python3_EXECUTABLE}")
        # message(STATUS " Python_VERSION: ${Python3_VERSION}")
        if (NOT python_PATH)
            set(python_PATH ${Python3_EXECUTABLE} CACHE FILEPATH "Path of the Python 3.8+ binary." FORCE)
        endif()
    else()
        message(WARNING " Failed to find Python3::Interpreter")
    endif()

    # python dev
    if(TARGET Python3::Python)
        # message(STATUS " Found Python3::Python")
        # message(STATUS " Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
        # message(STATUS " Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
        # message(STATUS " Python3_LIBRARIES: ${Python3_LIBRARIES}")
        # message(STATUS " Python3_LIBRARY_RELEASE: ${Python3_LIBRARY_RELEASE}")
        # message(STATUS " Python3_RUNTIME_LIBRARY_DIRS: ${Python3_RUNTIME_LIBRARY_DIRS}")
        if(NOT PYTHON_DIR)
            get_filename_component(_PYTHON_DIR ${Python3_LIBRARY_DIRS} DIRECTORY)
            set(PYTHON_DIR ${_PYTHON_DIR} CACHE PATH "Directory that contains the python dev library and the corresponding headers." FORCE)
        endif()    
    else()
        if(LINUX OR MACOSX)
            set(_OS_MESSAGE " install the 'python3-dev' package or")
        endif()
        message(FATAL_ERROR "The dependency \"python\" could not be resolved. Please${_OS_MESSAGE} specify 'PYTHON_DIR'.")
    endif()

    # store paths that are later used in the add_python.cmake
    set(MDL_DEPENDENCY_PYTHON_DEV_INCLUDE ${Python3_INCLUDE_DIRS} CACHE INTERNAL "python headers")
    set(MDL_DEPENDENCY_PYTHON_DEV_LIBS ${Python3_LIBRARY_RELEASE} CACHE INTERNAL "python libs")
    set(MDL_DEPENDENCY_PYTHON_DEV_EXE ${Python3_EXECUTABLE} CACHE INTERNAL "python interpreter")
    #set(MDL_DEPENDENCY_PYTHON_DEV_SHARED ${Python3_SHARED} CACHE INTERNAL "python shared libs")
    set(MDL_PYTHON_DEV_FOUND ON CACHE INTERNAL "")

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] MDL_DEPENDENCY_PYTHON_DEV_INCLUDE:  ${MDL_DEPENDENCY_PYTHON_DEV_INCLUDE}")
        message(STATUS "[INFO] MDL_DEPENDENCY_PYTHON_DEV_LIBS:     ${MDL_DEPENDENCY_PYTHON_DEV_LIBS}")
        message(STATUS "[INFO] MDL_DEPENDENCY_PYTHON_DEV_SHARED:   ${MDL_DEPENDENCY_PYTHON_DEV_SHARED}")
        message(STATUS "[INFO] MDL_DEPENDENCY_PYTHON_DEV_EXE:      ${MDL_DEPENDENCY_PYTHON_DEV_EXE}")
    endif()
endfunction()
