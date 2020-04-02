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

# name of the target and the resulting library
set(PROJECT_NAME base-lib-cont)

# add dummy-source to create a dummy-lib
set(DUMMY_CPP ${CMAKE_CURRENT_BINARY_DIR}/generated/empty.cpp)
if(NOT EXISTS ${DUMMY_CPP})
    file(WRITE ${DUMMY_CPP} "")
endif()

# collect sources
set(PROJECT_HEADERS
    "cont_allocators.h"
    "cont_allocators_inline.h"
    "cont_array_inline.h"
    "cont_bitset_inline.h"
    "cont_bitvector_inline.h"
    "cont_dlist_inline.h"
    "cont_rle_array_inline.h"
    "cont_set_inline.h"
    "cont_tree.h"
    "i_cont_array.h"
    "i_cont_bitset.h"
    "i_cont_bitvector.h"
    "i_cont_dlist.h"
    "i_cont_rle_array.h"
    "i_cont_set.h"
    "i_cont_traits.h"
    )

set(PROJECT_SOURCES
     ${PROJECT_HEADERS}
     ${DUMMY_CPP}
    )

# create target from template
create_from_base_preset(
    TARGET ${PROJECT_NAME}
    SOURCES ${PROJECT_SOURCES}
    )
