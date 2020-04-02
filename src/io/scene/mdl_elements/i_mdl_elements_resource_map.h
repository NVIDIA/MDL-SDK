/***************************************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_RESOURCE_MAP_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_RESOURCE_MAP_H

#include <string>
#include <vector>

#include <mi/mdl/mdl_generated_code.h>

namespace MI {
namespace MDL {

/// An entry in the resource tag map, mapping accessible resources to (tag, version) pair.
struct Resource_tag_tuple {
    mi::mdl::Resource_tag_tuple::Kind m_kind;    ///< The resource kind.
    std::string                       m_url;     ///< The resource URL.
    int                               m_tag;     ///< The assigned tag.

    /// Default constructor.
    Resource_tag_tuple()
    : m_kind(mi::mdl::Resource_tag_tuple::RK_BAD)
    , m_url("")
    , m_tag(0)
    {}

    /// Constructor.
    Resource_tag_tuple(
        mi::mdl::Resource_tag_tuple::Kind kind,
        std::string const                 &url,
        int                               tag)
    : m_kind(kind)
    , m_url(url)
    , m_tag(tag)
    {
    }
};

typedef std::vector<Resource_tag_tuple> Resource_tag_map;

} // namespace MDL
} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_RESOURCE_MAP_H
