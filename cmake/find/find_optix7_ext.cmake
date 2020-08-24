#*****************************************************************************
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

function(FIND_OPTIX7_EXT)

    if(NOT MDL_ENABLE_OPTIX7_EXAMPLES)
        message(WARNING "Examples that require OPTIX7 are disabled. Enable the option 'MDL_ENABLE_OPTIX7_EXAMPLES' and resolve the required dependencies to re-enable them.")
        return()
    endif()

    set(OPTIX7_DIR "NOT-SPECIFIED" CACHE PATH "Directory that contains the OptiX 7 include folder.")

    if(EXISTS ${OPTIX7_DIR}/include/optix.h)
        set(MDL_OPTIX7_FOUND ON CACHE INTERNAL "")
    else()
        message(STATUS "OPTIX7_DIR: ${OPTIX7_DIR}")
        message(FATAL_ERROR "The dependency \"OptiX7\" could not be resolved. Please specify 'OPTIX7_DIR'.")
    endif()

    if(MDL_LOG_DEPENDENCIES)
        message(STATUS "[INFO] OPTIX7_DIR:                         ${OPTIX7_DIR}")
    endif()

endfunction()
