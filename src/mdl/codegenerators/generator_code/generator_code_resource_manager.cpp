/******************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

#include "generator_code_resource_manager.h"

namespace mi {
namespace mdl {

// Constructor.
Source_res_manag::Source_res_manag(
    IAllocator              *alloc,
    Resource_attr_map const *resource_attr_map)
: m_alloc(alloc)
, m_resource_attr_map(alloc)
, m_res_indexes(
    0, Tag_index_map::hasher(), Tag_index_map::key_equal(), alloc)
, m_string_indexes(
    0, String_index_map::hasher(), String_index_map::key_equal(), alloc)
, m_curr_res_idx(0)
, m_curr_string_idx(0)
{
    if (resource_attr_map != NULL) {
        // import
        m_resource_attr_map.insert(resource_attr_map->begin(), resource_attr_map->end());
    }
}

// Register the given resource value and return its 1-based index in the resource table.
// Index 0 represents an invalid resource reference.
size_t Source_res_manag::get_resource_index(
    Resource_tag_tuple::Kind   kind,
    char const                 *url,
    int                        tag,
    IType_texture::Shape,
    IValue_texture::gamma_mode,
    char const                 *selector)
{
    if (!m_resource_attr_map.empty()) {
        // If the tag is not known, attempt a lookup without taking the tag into account.
        Resource_tag_tuple key(kind, url, selector, tag == 0 ? Resource_equal_to::IGNORE_TAG : tag);

        Resource_attr_map::const_iterator it(m_resource_attr_map.find(key));
        if (it != m_resource_attr_map.end()) {
            mi::mdl::Resource_attr_entry const &e = it->second;
            return e.index;
        }
        // Bad: we have a resource map, but could not find the requested resource.
        // This means the integration was not able to retrieve it from the material
        // and has not loaded it. Return 0 (invalid) here, the resource *is* missing.
        return 0;
    }

    switch (kind) {
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT:
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:
    case Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:
    case Resource_tag_tuple::RK_LIGHT_PROFILE:
    case Resource_tag_tuple::RK_BSDF_MEASUREMENT:
    case Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
    case Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
    case Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
    case Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
    case Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
    case Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
    case Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
    case Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
    case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL:
    case Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER:
        // we support textures, light profiles, bsdf_measurements, and bsdf_data textures
        {
            Tag_index_map::const_iterator it = m_res_indexes.find(tag);
            if (it != m_res_indexes.end())
                return it->second;

            size_t idx = ++m_curr_res_idx;
            m_res_indexes[tag] = idx;
            return idx;
        }

    default:
        // those should never occur in functions
        MDL_ASSERT(!"Unexpected resource type");
        return tag;
    }
}

// Register a string constant and return its 1 based index in the string table.
size_t Source_res_manag::get_string_index(IValue_string const *s)
{
    string str(s->get_value(), m_alloc);

    String_index_map::const_iterator it = m_string_indexes.find(str);
    if (it != m_string_indexes.end()) {
        return it->second;
    }

    if (m_curr_res_idx == 0) {
        // zero is reserved for "Not-a-known-String"
        m_string_indexes[string("<NULL>", m_alloc)] = 0;
    }

    size_t idx = ++m_curr_string_idx;
    m_string_indexes[str] = idx;
    return idx;
}

// Imports a new resource attribute map.
void Source_res_manag::import_resource_attribute_map(
    Resource_attr_map const *resource_attr_map)
{
    if (resource_attr_map != NULL) {
        m_resource_attr_map.insert(resource_attr_map->begin(), resource_attr_map->end());
    }
}

} // mdl
} // mi
