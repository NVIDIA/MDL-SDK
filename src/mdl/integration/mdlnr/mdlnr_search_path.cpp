/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include "pch.h"

#include "mdlnr_search_path.h"

#include <mi/mdl/mdl_mdl.h>
#include "base/system/main/access_module.h"
#include "base/lib/path/i_path.h"

namespace MI {
namespace MDLC {

MDL_search_path::MDL_search_path(mi::mdl::IAllocator *alloc)
: Base(alloc)
, tmp()
{
}

// Get the number of search paths.
size_t MDL_search_path::get_search_path_count(Path_set set) const
{
    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    PATH::Kind kind = set == mi::mdl::IMDL_search_path::MDL_SEARCH_PATH ?
        PATH::MDL : PATH::RESOURCE;

    return m_path_module->get_path_count( kind);
}

// Get the i'th search path.
char const *MDL_search_path::get_search_path(Path_set set, size_t i) const
{
    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    PATH::Kind kind = set == mi::mdl::IMDL_search_path::MDL_SEARCH_PATH ?
        PATH::MDL : PATH::RESOURCE;

    tmp = m_path_module->get_path( kind, i);
    return tmp.c_str();
}

MDL_search_path *MDL_search_path::create(mi::mdl::IAllocator *alloc)
{
    mi::mdl::Allocator_builder builder(alloc);

    return builder.create<MDL_search_path>(alloc);
}

}  // MDLC
}  // MI
