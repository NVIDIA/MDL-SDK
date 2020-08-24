#*****************************************************************************
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

# the opengl dependency consists of a set of packages

# assuming the find_opengl_ext script was successful
# if not, this is an error case. The corresponding project should not have been selected for build.
if(NOT MDL_ENABLE_D3D12_EXAMPLES)
    message(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
else()

    # headers
    target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${MDL_DEPENDENCY_D3D12_INCLUDE}
            ${MDL_DEPENDENCY_DXGI_INCLUDE}
        )

    # static library
    target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET} 
        PRIVATE
            ${MDL_DEPENDENCY_D3D12_LIBS}
        )

    # copy runtime dependencies
    # copy system libraries only on windows, we assume the libraries are installed in a unix environment
    if(NOT __TARGET_ADD_DEPENDENCY_NO_RUNTIME_COPY AND WINDOWS)
        target_copy_to_output_dir(TARGET ${__TARGET_ADD_DEPENDENCY_TARGET}
            FILES
                ${MDL_DEPENDENCY_D3D12_SHARED}
            )
    endif()
endif()
