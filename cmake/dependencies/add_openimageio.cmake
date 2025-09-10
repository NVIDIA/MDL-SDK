#*****************************************************************************
# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

# assuming the find_openiamgeio_ext script was successful
# if not, this is an error case. The corresponding project should not have been selected for build.
if(NOT MDL_OPENIMAGEIO_FOUND)
    message(FATAL_ERROR "The dependency \"${__TARGET_ADD_DEPENDENCY_DEPENDS}\" for target \"${__TARGET_ADD_DEPENDENCY_TARGET}\" could not be resolved.")
else()

    # add the include directory
    target_include_directories(${__TARGET_ADD_DEPENDENCY_TARGET}
        PRIVATE
            ${MDL_DEPENDENCY_OPENIMAGEIO_INCLUDE}
        )
    target_compile_options(${__TARGET_ADD_DEPENDENCY_TARGET}
        PRIVATE
           "$<$<STREQUAL:${MI_PLATFORM_NAME},linux-aarch64>:-flax-vector-conversions>"
           # Workaround for https://github.com/microsoft/vcpkg/issues/44128
           "$<$<STREQUAL:${MI_PLATFORM_NAME},nt-x86-64>:/utf-8>"
        )

    # link static/shared object
    if(NOT __TARGET_ADD_DEPENDENCY_NO_LINKING)
        if(WINDOWS)
            # Misuse OpenImageIO version to distinguish older vcpkg versions (e.g. 42f74e3db
            # plus patch) from newer vcpkg versions (e.g. 3640e7cb1).
            if(MDL_DEPENDENCY_OPENIMAGEIO_VERSION VERSION_LESS "2.4.5.0")
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        ${LINKER_WHOLE_ARCHIVE}
                        OpenImageIO::OpenImageIO
                        TIFF::TIFF
                        liblzma::liblzma
                        ${LINKER_NO_WHOLE_ARCHIVE}
                    )
            else()
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        ${LINKER_WHOLE_ARCHIVE}
                        OpenImageIO::OpenImageIO
                        ${LINKER_NO_WHOLE_ARCHIVE}
                    )
            endif()
        else()
            # Misuse OpenImageIO version to distinguish older vcpkg versions (e.g. 42f74e3db
            # plus patch) from newer vcpkg versions (e.g. 3640e7cb1).
            if(MDL_DEPENDENCY_OPENIMAGEIO_VERSION VERSION_LESS "2.4.5.0")
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        ${LINKER_NO_AS_NEEDED}
                        OpenImageIO::OpenImageIO
                        # Explicitly add TIFF library here to avoid that it first
                        # appears (as dependency of OIIO) after LZMA below.
                        TIFF::TIFF
                        # Necessary to avoid undefined symbols in TIFF library.
                        liblzma::liblzma
                        ${LINKER_AS_NEEDED}
                    )
            else()
                target_link_libraries(${__TARGET_ADD_DEPENDENCY_TARGET}
                    PRIVATE
                        ${LINKER_NO_AS_NEEDED}
                        OpenImageIO::OpenImageIO
                        ${LINKER_AS_NEEDED}
                    )
            endif()
        endif()
    endif()
endif()
