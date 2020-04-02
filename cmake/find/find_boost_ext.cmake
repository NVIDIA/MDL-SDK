#*****************************************************************************
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

    set(BOOST_INCLUDEDIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the boost headers.")

    if(EXISTS ${BOOST_INCLUDEDIR})
        # remove trailing slashes/backslashes
        string(REGEX REPLACE "[/\\\\]$" "" _BOOST_INCLUDEDIR "${BOOST_INCLUDEDIR}")
        set(BOOST_INCLUDEDIR ${_BOOST_INCLUDEDIR} CACHE PATH "Directory that contains the boost headers." FORCE)
        set(Boost_NO_SYSTEM_PATHS ON CACHE INTERNAL "")
        set(Boost_NO_BOOST_CMAKE ON CACHE INTERNAL "")
        #set(Boost_DEBUG ON)
    endif()

    # headers only
    find_package(Boost QUIET)
    mark_as_advanced(CLEAR BOOST_INCLUDEDIR)
    set(Boost_FOUND ${Boost_FOUND} CACHE INTERNAL "Dependency boost has been resolved.")

    if(NOT Boost_FOUND)
        message(STATUS "BOOST_INCLUDEDIR: ${BOOST_INCLUDEDIR}")
        message(FATAL_ERROR "The dependency \"boost\" could not be resolved. Please specify 'BOOST_INCLUDEDIR'.")
    endif()
    
    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] BOOST_INCLUDEDIR:                   ${BOOST_INCLUDEDIR}")
    endif()

endfunction()
