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

#ifndef MDL_GENERATOR_CODE_RESOURCE_MANAGER_H
#define MDL_GENERATOR_CODE_RESOURCE_MANAGER_H 1

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_generated_code.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_cstring_hash.h"
#include "mdl/compiler/compilercore/compilercore_string.h"

namespace mi {
namespace mdl {

/// An Interface to handle resources in JIT compiled code.
class IResource_manager {
public:
    /// Returns the resource index for the given resource usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// \param kind        the resource kind
    /// \param url         the resource url (might be NULL)
    /// \param tag         the resource tag (if assigned)
    /// \param shape       if the resource is a texture: its shape
    /// \param gamma_mode  if the resource is a texture: its gamma mode
    /// \param selector    if the resource is a texture: its selector
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    virtual size_t get_resource_index(
        Resource_tag_tuple::Kind   kind,
        char const                 *url,
        int                        tag,
        IType_texture::Shape       shape = IType_texture::TS_2D,
        IValue_texture::gamma_mode gamma_mode = IValue_texture::gamma_default,
        char const                 *selector = "") = 0;

    /// Register a string constant and return its 1 based index in the string table.
    ///
    /// \param string  the MDL string value to register
    virtual size_t get_string_index(IValue_string const *string) = 0;
};

/// A value entry in the resource attribute map.
struct Resource_attr_entry {
    size_t index;               ///< The "index" value of this resource.
    bool valid;                 ///< True if this resource is valid.
    union {
        struct {
            unsigned width;              ///< texture width
            unsigned height;             ///< texture height
            unsigned depth;              ///< texture depth
            IType_texture::Shape shape;  ///< texture shape
        } tex;
        struct {
            float power;        ///< light profile power
            float maximum;      ///< light profile maximum
        } lp;
    } u;
};

/// A hash functor for Resource_tag_tuple.
struct Resource_hasher {
    size_t operator()(Resource_tag_tuple const &p) const {
        cstring_hash cstring_hasher;

        /// Avoid hashing the tag to support lookups without knowing the tag. But include it if no
        /// URL is known to avoid degeneration to a linear list (in such a case a lookup without
        /// knowing the tag does not make much sense, either).
        size_t hash = size_t(p.m_kind) ^ cstring_hasher(p.m_url) ^ cstring_hasher(p.m_selector);
        if (!p.m_url[0]) {
            hash ^= p.m_tag;
        }
        return hash;
    }
};

/// A equal_to functor for Resource_tag_tuple.
struct Resource_equal_to {
    /// Sentinel value to indicate that the tag should not be considered for comparisons.
    static const int IGNORE_TAG = -1;

    bool operator()(Resource_tag_tuple const &a, Resource_tag_tuple const &b) const {
        if (a.m_kind != b.m_kind) {
            return false;
        }
        if (a.m_tag != IGNORE_TAG && b.m_tag != IGNORE_TAG && a.m_tag != b.m_tag) {
            return false;
        }

        cstring_equal_to cstring_cmp;

        if (!cstring_cmp(a.m_url, b.m_url)) {
            return false;
        }
        return cstring_cmp(a.m_selector, b.m_selector);
    }
};

/// The resource attribute map: maps (Resource, tag) tuple to its attributes.
typedef hash_map<
    Resource_tag_tuple,
    Resource_attr_entry,
    Resource_hasher,
    Resource_equal_to>::Type Resource_attr_map;

/// A generic resource manager for generated source code.
class Source_res_manag MDL_FINAL : public IResource_manager {
    typedef hash_map<unsigned, size_t>::Type                      Tag_index_map;
    typedef hash_map<string, size_t, string_hash<string> >::Type  String_index_map;
public:
    /// Constructor.
    ///
    /// \param alloc              The allocator.
    /// \param resource_attr_map  If non-NULL, import this map to resolve resources
    Source_res_manag(
        IAllocator              *alloc,
        Resource_attr_map const *resource_attr_map);

    /// Returns the resource index for the given resource usable by the target code resource
    /// handler for the corresponding resource type.
    ///
    /// \param kind        the resource kind
    /// \param url         the resource url (might be NULL)
    /// \param tag         the resource tag (if assigned)
    /// \param shape       if the resource is a texture: its shape
    /// \param gamma_mode  if the resource is a texture: its gamma mode
    /// \param selector    if the resource is a texture: its selector
    ///
    /// \returns a resource index or 0 if no resource index can be returned
    size_t get_resource_index(
        Resource_tag_tuple::Kind   kind,
        char const *url,
        int                        tag,
        IType_texture::Shape       shape,
        IValue_texture::gamma_mode gamma_mode,
        char const *selector) MDL_FINAL;

    /// Register a string constant and return its 1 based index in the string table.
    ///
    /// \param string  the MDL string value to register
    size_t get_string_index(IValue_string const *string) MDL_FINAL;

    /// Imports a new resource attribute map.
    ///
    /// \param resource_attr_map  if non-NULL, the map to be imported
    void import_resource_attribute_map(Resource_attr_map const *resource_attr_map);

private:
    /// The current allocator.
    IAllocator *m_alloc;

    /// The accumulated resource-attribute-map.
    Resource_attr_map       m_resource_attr_map;

    /// Lookup-table for resource indexes.
    Tag_index_map           m_res_indexes;

    /// Lookup-table for string indexes;
    String_index_map        m_string_indexes;

    /// The current resource index.
    size_t m_curr_res_idx;

    /// The current string index.
    size_t m_curr_string_idx;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_CODE_TREAD_CONTEXT_H
