#*****************************************************************************
# Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_BOOST_EXT)

    if(NOT CMAKE_TOOLCHAIN_FILE)
        set(Boost_NO_SYSTEM_PATHS OFF CACHE INTERNAL "")
    else()
        set(Boost_NO_SYSTEM_PATHS ON CACHE INTERNAL "")
    endif()

    #set(Boost_DEBUG ON)

    find_package(Boost)

    set(Boost_FOUND ${Boost_FOUND} CACHE INTERNAL "Dependency boost has been resolved.")

    if(NOT Boost_FOUND)
        message(FATAL_ERROR "The dependency \"boost\" could not be resolved. "
            "Please specify 'CMAKE_TOOLCHAIN_FILE'.")
    else()
        set(MDL_DEPENDENCY_BOOST_INCLUDE ${Boost_INCLUDE_DIRS} CACHE INTERNAL
            "Boost header directory")
        set(MDL_BOOST_FOUND ON CACHE INTERNAL "")

        if(MDL_LOG_DEPENDENCIES)
            message(STATUS "[INFO] MDL_DEPENDENCY_BOOST_INCLUDE:             ${MDL_DEPENDENCY_BOOST_INCLUDE}")
        endif()
    endif()

endfunction()
