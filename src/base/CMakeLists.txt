#*****************************************************************************
# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
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

# modules required by the MDL Core API and MDLC
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/libzip)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/zlib)
add_subdirectory(${MDL_SRC_FOLDER}/base/system/version)

if(NOT MDL_BUILD_SDK)
    return()
endif()

# modules for the MDL SDK
add_subdirectory(${MDL_SRC_FOLDER}/base/data/attr)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/db)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/dblight)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/idata)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/serial)
add_subdirectory(${MDL_SRC_FOLDER}/base/data/thread_pool)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/disk)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/hal)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/host)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/link)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/thread)
add_subdirectory(${MDL_SRC_FOLDER}/base/hal/time)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/config)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/cont)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/log)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/mem)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/path)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/plug)
add_subdirectory(${MDL_SRC_FOLDER}/base/lib/tinyxml2)
add_subdirectory(${MDL_SRC_FOLDER}/base/util/registry)
add_subdirectory(${MDL_SRC_FOLDER}/base/util/string_utils)
add_subdirectory(${MDL_SRC_FOLDER}/base/system/main)
add_subdirectory(${MDL_SRC_FOLDER}/base/system/stlext)
add_subdirectory(${MDL_SRC_FOLDER}/base/system/test)

set(LINKER_DEPENDENCIES_BASE
    mdl::base-data-attr
    mdl::base-data-db
    mdl::base-data-dblight
    mdl::base-data-idata
    mdl::base-data-serial
    mdl::base-data-thread_pool
    mdl::base-hal-disk
    mdl::base-hal-hal
    mdl::base-hal-host
    mdl::base-hal-link
    mdl::base-hal-thread
    mdl::base-hal-time
    mdl::base-lib-config
    mdl::base-lib-cont
    mdl::base-lib-libzip
    mdl::base-lib-log
    mdl::base-lib-mem
    mdl::base-lib-path
    mdl::base-lib-plug
    mdl::base-lib-tinyxml2
    mdl::base-lib-zlib
    mdl::base-util-string_utils
    mdl::base-util-registry
    mdl::base-system-main
    mdl::base-system-stlext
    mdl::base-system-version
    CACHE INTERNAL "" FORCE)

set(_LINKER_DEPENDENCIES_BASE_WITHOUT_BASE_DATA ${LINKER_DEPENDENCIES_BASE})
list(FILTER _LINKER_DEPENDENCIES_BASE_WITHOUT_BASE_DATA EXCLUDE REGEX mdl::base-data-)

set(LINKER_DEPENDENCIES_BASE_WITHOUT_BASE_DATA
    ${_LINKER_DEPENDENCIES_BASE_WITHOUT_BASE_DATA}
    CACHE INTERNAL "" FORCE)
