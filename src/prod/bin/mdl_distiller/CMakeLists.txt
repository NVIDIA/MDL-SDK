#*****************************************************************************
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

# name of the target and the resulting library
set(PROJECT_NAME prod-bin-mdl_distiller)

# collect sources
set(PROJECT_HEADERS
    "condition_checker.h"
    "framework.h"
    "logger.h"
    "mdl_assert.h"
    "mdl_distiller_utils.h"
    "mdl_printer.h"
    "neuray_factory.h"
    "options.h"
    "user_timer.h"
    )

set(PROJECT_SOURCES
    "framework.cpp"
    "condition_checker.cpp"
    "logger.cpp"
    "mdl_assert.cpp"
    "mdl_distiller.cpp"
    "mdl_distiller_utils.cpp"
    "mdl_printer.cpp"
    "user_timer.cpp"
    ${PROJECT_HEADERS}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    TYPE EXECUTABLE
    OUTPUT_NAME "mdl_distiller"
    SOURCES ${PROJECT_SOURCES}
    WINDOWS_UNICODE
)

# add mdl and other dependencies
target_add_dependencies(TARGET ${PROJECT_NAME}
    DEPENDS
        ${LINKER_START_GROUP}
        mdl::mdl_sdk
        ${LINKER_END_GROUP}
    )

# -------------------------------------------------------------------------------------------------
# Create installation rules to copy the build directory
# -------------------------------------------------------------------------------------------------
add_target_install(
    TARGET ${PROJECT_NAME}
    )
