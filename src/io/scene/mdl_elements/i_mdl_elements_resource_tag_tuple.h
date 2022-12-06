/***************************************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_RESOURCE_TAG_TUPLE_H
#define IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_RESOURCE_TAG_TUPLE_H

#include <string>
#include <vector>

#include <mi/mdl/mdl_generated_code.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/serial/i_serializer.h>

#include "i_mdl_elements_type.h"

namespace MI {

namespace SERIAL { class Serializer; class Deserializer; }

namespace MDL {

/// An entry in the resource vector, mapping accessible resources to a tag.
class Resource_tag_tuple
{
public:
    /// Default constructor.
    Resource_tag_tuple() : m_kind( mi::mdl::Resource_tag_tuple::RK_BAD) { }

    /// Constructor.
    Resource_tag_tuple(
        mi::mdl::Resource_tag_tuple::Kind kind,
        const std::string& mdl_file_path,
        const std::string& selector,
        DB::Tag tag)
      : m_kind( kind),
        m_mdl_file_path( mdl_file_path),
        m_selector( selector),
        m_tag(tag)
    {
    }

    /// Constructor from mi::mdl::Resource_tag_tuple.
    Resource_tag_tuple( const mi::mdl::Resource_tag_tuple& t)
      : m_kind( t.m_kind),
        m_mdl_file_path( t.m_url),
        m_selector( t.m_selector),
        m_tag( t.m_tag)
     {
     }

    mi::mdl::Resource_tag_tuple::Kind m_kind;           ///< The resource kind.
    std::string                       m_mdl_file_path;  ///< The MDL file path.
    std::string                       m_selector;       ///< The selector.
    DB::Tag                           m_tag;            ///< The assigned tag.
};

/// An entry in the resource vector, mapping accessible resources to a tag.
///
/// Also includes IType::Kind and IType_texture::Shape.
class Resource_tag_tuple_ext : public Resource_tag_tuple
{
public:
    /// Default constructor.
    Resource_tag_tuple_ext()
      : m_type_kind( IType::TK_TEXTURE),
        m_texture_shape( IType_texture::TS_2D)
    { }

    /// Constructor.
    Resource_tag_tuple_ext(
        const mi::mdl::Resource_tag_tuple& t,
        IType::Kind type_kind,
        IType_texture::Shape texture_shape)
      : Resource_tag_tuple( t),
        m_type_kind( type_kind),
        m_texture_shape( texture_shape)
    {
    }

    // The type kind.
    IType::Kind m_type_kind;
    /// The texture shape. Invalid if m_type_kind != TK_TEXTURE.
    IType_texture::Shape m_texture_shape;
};

inline void write( SERIAL::Serializer* serializer, const Resource_tag_tuple& r)
{
   SERIAL::write_enum( serializer, r.m_kind);
   SERIAL::write( serializer, r.m_mdl_file_path);
   SERIAL::write( serializer, r.m_selector);
   SERIAL::write( serializer, r.m_tag);
}

inline void read( SERIAL::Deserializer* deserializer, Resource_tag_tuple* r)
{
   SERIAL::read_enum( deserializer, &r->m_kind);
   SERIAL::read( deserializer, &r->m_mdl_file_path);
   SERIAL::read( deserializer, &r->m_selector);
   SERIAL::read( deserializer, &r->m_tag);
}

inline void write( SERIAL::Serializer* serializer, const Resource_tag_tuple_ext& r)
{
   write( serializer, *static_cast<const Resource_tag_tuple*>( &r));
   SERIAL::write_enum( serializer, r.m_type_kind);
   SERIAL::write_enum( serializer, r.m_texture_shape);
}

inline void read( SERIAL::Deserializer* deserializer, Resource_tag_tuple_ext* r)
{
   read( deserializer, static_cast<Resource_tag_tuple*>( r));
   SERIAL::read_enum( deserializer, &r->m_type_kind);
   SERIAL::read_enum( deserializer, &r->m_texture_shape);
}

// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption( const Resource_tag_tuple&) { return true; }
inline size_t dynamic_memory_consumption( const Resource_tag_tuple& r)
{
    return dynamic_memory_consumption( r.m_mdl_file_path)
         + dynamic_memory_consumption( r.m_selector);
}

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_I_MDL_ELEMENTS_RESOURCE_TAG_TUPLE_H
