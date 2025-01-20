/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the class factory.
 **/

#include "pch.h"

#include "i_idata_factory.h"

#include <iomanip>

#include <boost/core/ignore_unused.hpp>

#include <base/system/main/i_assert.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>

#include <mi/neuraylib/ifactory.h> // mi::neuraylib::IFactory::NO_CONVERSION etc.
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>

#include "idata_array_impl.h"
#include "idata_compound_impl.h"
#include "idata_interfaces.h"
#include "idata_map_impl.h"
#include "idata_simple_impl.h"
#include "idata_structure_impl.h"

namespace MI {

namespace IDATA {

Factory::Factory( ITag_handler* tag_handler)
  : m_tag_handler( tag_handler, mi::base::DUP_INTERFACE)
{
    mi::Sint32 result = 0;
    boost::ignore_unused( result);

#define REG2( a, b)    result = register_class( a, b);    MI_ASSERT( result == 0)
#define REG3( a, b, c) result = register_class( a, b, c); MI_ASSERT( result == 0)

    // IData_simple
    REG3( "Boolean",         Number_impl<mi::IBoolean,    bool>::create_instance);
    REG3( "Sint8",           Number_impl<mi::ISint8,      mi::Sint8>::create_instance);
    REG3( "Sint16",          Number_impl<mi::ISint16,     mi::Sint16>::create_instance);
    REG3( "Sint32",          Number_impl<mi::ISint32,     mi::Sint32>::create_instance);
    REG3( "Sint64",          Number_impl<mi::ISint64,     mi::Sint64>::create_instance);
    REG3( "Uint8",           Number_impl<mi::IUint8,      mi::Uint8>::create_instance);
    REG3( "Uint16",          Number_impl<mi::IUint16,     mi::Uint16>::create_instance);
    REG3( "Uint32",          Number_impl<mi::IUint32,     mi::Uint32>::create_instance);
    REG3( "Uint64",          Number_impl<mi::IUint64,     mi::Uint64>::create_instance);
    REG3( "Float32",         Number_impl<mi::IFloat32,    mi::Float32>::create_instance);
    REG3( "Float64",         Number_impl<mi::IFloat64,    mi::Float64>::create_instance);
    REG3( "Size",            Number_impl<mi::ISize,       mi::Size>::create_instance);
    REG3( "Difference",      Number_impl<mi::IDifference, mi::Difference>::create_instance);
    REG2( "String",          String_impl::create_instance);
    REG2( "Uuid",            Uuid_impl::create_instance);
    REG2( "Void",            Void_impl::create_instance);
    REG2( "__Pointer",       Pointer_impl::create_instance);
    REG2( "__Const_pointer", Const_pointer_impl::create_instance);
    REG2( "__Enum",          Enum_impl::create_instance);
    REG2( "Enum_decl",       Enum_decl_impl::create_instance);

    // vector variants of ICompound interface
    REG2( "Boolean<2>", Boolean_2_impl::create_instance);
    REG2( "Boolean<3>", Boolean_3_impl::create_instance);
    REG2( "Boolean<4>", Boolean_4_impl::create_instance);
    REG2( "Sint32<2>",  Sint32_2_impl::create_instance );
    REG2( "Sint32<3>",  Sint32_3_impl::create_instance );
    REG2( "Sint32<4>",  Sint32_4_impl::create_instance );
    REG2( "Uint32<2>",  Uint32_2_impl::create_instance );
    REG2( "Uint32<3>",  Uint32_3_impl::create_instance );
    REG2( "Uint32<4>",  Uint32_4_impl::create_instance );
    REG2( "Float32<2>", Float32_2_impl::create_instance);
    REG2( "Float32<3>", Float32_3_impl::create_instance);
    REG2( "Float32<4>", Float32_4_impl::create_instance);
    REG2( "Float64<2>", Float64_2_impl::create_instance);
    REG2( "Float64<3>", Float64_3_impl::create_instance);
    REG2( "Float64<4>", Float64_4_impl::create_instance);

    // matrix variants of ICompound interface
    REG2( "Boolean<2,2>", Boolean_2_2_impl::create_instance);
    REG2( "Boolean<2,3>", Boolean_2_3_impl::create_instance);
    REG2( "Boolean<2,4>", Boolean_2_4_impl::create_instance);
    REG2( "Boolean<3,2>", Boolean_3_2_impl::create_instance);
    REG2( "Boolean<3,3>", Boolean_3_3_impl::create_instance);
    REG2( "Boolean<3,4>", Boolean_3_4_impl::create_instance);
    REG2( "Boolean<4,2>", Boolean_4_2_impl::create_instance);
    REG2( "Boolean<4,3>", Boolean_4_3_impl::create_instance);
    REG2( "Boolean<4,4>", Boolean_4_4_impl::create_instance);

    REG2( "Sint32<2,2>",  Sint32_2_2_impl::create_instance );
    REG2( "Sint32<2,3>",  Sint32_2_3_impl::create_instance );
    REG2( "Sint32<2,4>",  Sint32_2_4_impl::create_instance );
    REG2( "Sint32<3,2>",  Sint32_3_2_impl::create_instance );
    REG2( "Sint32<3,3>",  Sint32_3_3_impl::create_instance );
    REG2( "Sint32<3,4>",  Sint32_3_4_impl::create_instance );
    REG2( "Sint32<4,2>",  Sint32_4_2_impl::create_instance );
    REG2( "Sint32<4,3>",  Sint32_4_3_impl::create_instance );
    REG2( "Sint32<4,4>",  Sint32_4_4_impl::create_instance );

    REG2( "Uint32<2,2>",  Uint32_2_2_impl::create_instance );
    REG2( "Uint32<2,3>",  Uint32_2_3_impl::create_instance );
    REG2( "Uint32<2,4>",  Uint32_2_4_impl::create_instance );
    REG2( "Uint32<3,2>",  Uint32_3_2_impl::create_instance );
    REG2( "Uint32<3,3>",  Uint32_3_3_impl::create_instance );
    REG2( "Uint32<3,4>",  Uint32_3_4_impl::create_instance );
    REG2( "Uint32<4,2>",  Uint32_4_2_impl::create_instance );
    REG2( "Uint32<4,3>",  Uint32_4_3_impl::create_instance );
    REG2( "Uint32<4,4>",  Uint32_4_4_impl::create_instance );

    REG2( "Float32<2,2>", Float32_2_2_impl::create_instance);
    REG2( "Float32<2,3>", Float32_2_3_impl::create_instance);
    REG2( "Float32<2,4>", Float32_2_4_impl::create_instance);
    REG2( "Float32<3,2>", Float32_3_2_impl::create_instance);
    REG2( "Float32<3,3>", Float32_3_3_impl::create_instance);
    REG2( "Float32<3,4>", Float32_3_4_impl::create_instance);
    REG2( "Float32<4,2>", Float32_4_2_impl::create_instance);
    REG2( "Float32<4,3>", Float32_4_3_impl::create_instance);
    REG2( "Float32<4,4>", Float32_4_4_impl::create_instance);

    REG2( "Float64<2,2>", Float64_2_2_impl::create_instance);
    REG2( "Float64<2,3>", Float64_2_3_impl::create_instance);
    REG2( "Float64<2,4>", Float64_2_4_impl::create_instance);
    REG2( "Float64<3,2>", Float64_3_2_impl::create_instance);
    REG2( "Float64<3,3>", Float64_3_3_impl::create_instance);
    REG2( "Float64<3,4>", Float64_3_4_impl::create_instance);
    REG2( "Float64<4,2>", Float64_4_2_impl::create_instance);
    REG2( "Float64<4,3>", Float64_4_3_impl::create_instance);
    REG2( "Float64<4,4>", Float64_4_4_impl::create_instance);

    // other variants of the ICompound interface
    REG2( "Color",    Color_impl::create_instance);
    REG2( "Color3",   Color3_impl::create_instance);
    REG2( "Spectrum", Spectrum_impl::create_instance);
    REG2( "Bbox3",    Bbox3_impl::create_instance);

    // IData_collection
    REG2( "__Array",         Array_impl::create_instance);
    REG2( "__Dynamic_array", Dynamic_array_impl::create_instance);
    REG2( "__Map",           Map_impl::create_instance);
    REG2( "__Structure",     Structure_impl::create_instance);
    REG2( "Structure_decl",  Structure_decl_impl::create_instance);

    // proxies (for attributes and elements of compounds)
    REG3( "__Boolean_proxy", Number_impl_proxy<mi::IBoolean, bool>::create_instance);
    REG3( "__Sint32_proxy",  Number_impl_proxy<mi::ISint32,  mi::Sint32>::create_instance);
    REG3( "__Uint32_proxy",  Number_impl_proxy<mi::IUint32,  mi::Uint32>::create_instance);
    REG3( "__Float32_proxy", Number_impl_proxy<mi::IFloat32, mi::Float32>::create_instance);
    REG3( "__Float64_proxy", Number_impl_proxy<mi::IFloat64, mi::Float64>::create_instance);

    // remaining proxies (for attributes)
    REG3( "__Sint8_proxy",  Number_impl_proxy<mi::ISint8,   mi::Sint8>::create_instance);
    REG3( "__Sint16_proxy", Number_impl_proxy<mi::ISint16,  mi::Sint16>::create_instance);
    REG3( "__Sint64_proxy", Number_impl_proxy<mi::ISint64,  mi::Sint64>::create_instance);
    REG3( "__Uint8_proxy",  Number_impl_proxy<mi::IUint8,   mi::Uint8>::create_instance);
    REG3( "__Uint16_proxy", Number_impl_proxy<mi::IUint16,  mi::Uint16>::create_instance);
    REG3( "__Uint64_proxy", Number_impl_proxy<mi::IUint64,  mi::Uint64>::create_instance);
    REG2( "__String_proxy", String_impl_proxy::create_instance);
    REG2( "__Enum_proxy",   Enum_impl_proxy::create_instance);
    REG2( "__Array_proxy",  Array_impl_proxy::create_instance);
    REG2( "__Dynamic_array_proxy", Dynamic_array_impl_proxy::create_instance);
    REG2( "__Structure_proxy", Structure_impl_proxy::create_instance);

    // IRef is only supported with a tag handler
    if( tag_handler) {
        REG2( "Ref",         Ref_impl::create_instance);
        REG2( "__Ref_proxy", Ref_impl_proxy::create_instance);
    }

#undef REG2
#undef REG3
}

Factory::~Factory()
{
    MI_ASSERT( m_structure_decls.empty());
    MI_ASSERT( m_enum_decls.empty());
}

mi::base::IInterface* Factory::create(
    DB::Transaction* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    if( !type_name)
        return nullptr;

    std::string type_name_string( type_name);
    mi::Size length = type_name_string.size();

    if( (argc == 0) && !argv) {

        // handle arrays
        if( type_name[length-1] == ']')
            return create_array( transaction, type_name);

        // handle maps
        if( type_name_string.substr( 0, 4) == "Map<")
            return create_map( transaction, type_name);

        // handle pointers
        if(    (type_name_string.substr( 0,  8) == "Pointer<")
            || (type_name_string.substr( 0, 14) == "Const_pointer<"))
            return create_pointer( transaction, type_name);

        // handle structures
        mi::base::Handle<const mi::IStructure_decl> structure_decl( get_structure_decl( type_name));
        if( structure_decl)
            return create_structure( transaction, type_name, structure_decl.get());

        // handle enums
        mi::base::Handle<const mi::IEnum_decl> enum_decl( get_enum_decl( type_name));
        if( enum_decl)
            return create_enum( transaction, type_name, enum_decl.get());
    }

    // handle simple types
    return create_registered( transaction, type_name, argc, argv);
}

mi::Uint32 Factory::assign_from_to(
    const mi::IData* source, mi::IData* target, mi::Uint32 options) const
{
    if( !source || !target)
        return mi::neuraylib::IFactory::NULL_POINTER;

    bool adjust_target_keys = (options & 2 /* ADJUST_LENGTH_OF_DYNAMIC_ARRAYS */) != 0;
    bool fix_target_keys    = (options & mi::neuraylib::IFactory::FIX_SET_OF_TARGET_KEYS) != 0;
    if( adjust_target_keys && fix_target_keys)
        return mi::neuraylib::IFactory::INCOMPATIBLE_OPTIONS;

    // handle IData_simple
    mi::base::Handle source_simple( source->get_interface<mi::IData_simple>());
    mi::base::Handle target_simple( target->get_interface<mi::IData_simple>());
    if( source_simple && target_simple)
        return assign_from_to( source_simple.get(), target_simple.get(), options);

    // handle IData_collection
    mi::base::Handle source_collection( source->get_interface<mi::IData_collection>());
    mi::base::Handle target_collection( target->get_interface<mi::IData_collection>());
    if( source_collection && target_collection)
        return assign_from_to( source_collection.get(), target_collection.get(), options);

    return mi::neuraylib::IFactory::STRUCTURAL_MISMATCH;
}

mi::IData* Factory::clone( const mi::IData* source, mi::Uint32 options)
{
    if( !source)
        return nullptr;

    // handle IData_simple
    mi::base::Handle source_simple( source->get_interface<mi::IData_simple>());
    if( source_simple)
        return clone( source_simple.get(), options);

    // handle IData_collection
    mi::base::Handle source_collection( source->get_interface<mi::IData_collection>());
    if( source_collection)
        return clone( source_collection.get(), options);

    MI_ASSERT( false);
    return nullptr;
}

mi::Sint32 Factory::compare( const mi::IData* lhs, const mi::IData* rhs)
{
    if( !lhs && !rhs) return 0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    MI_ASSERT( lhs && rhs);

    const char* lhs_type = lhs->get_type_name();
    const char* rhs_type = rhs->get_type_name();
    int type_cmp = strcmp( lhs_type, rhs_type);
    if( type_cmp != 0)
        return type_cmp;

    // handle IData_simple
    mi::base::Handle lhs_simple( lhs->get_interface<mi::IData_simple>());
    mi::base::Handle rhs_simple( rhs->get_interface<mi::IData_simple>());
    if( lhs_simple && rhs_simple)
        return compare( lhs_simple.get(), rhs_simple.get());

    // handle IData_collection
    mi::base::Handle lhs_collection( lhs->get_interface<mi::IData_collection>());
    mi::base::Handle rhs_collection( rhs->get_interface<mi::IData_collection>());
    if( lhs_collection && rhs_collection)
        return compare( lhs_collection.get(), rhs_collection.get());

    MI_ASSERT( false);
    return 0;
}

mi::Sint32 Factory::register_enum_decl( const char* enum_name, const mi::IEnum_decl* decl)
{
    MI_ASSERT( enum_name);
    MI_ASSERT( decl);

    mi::base::Lock::Block block2( &m_enum_decls_lock);
    mi::base::Lock::Block block3( &m_structure_decls_lock);

    if( m_factory_functions.find( enum_name) != m_factory_functions.end())
        return -1;
    if( m_enum_decls.find( enum_name)        != m_enum_decls.end())
        return -1;
    if( m_structure_decls.find( enum_name)   != m_structure_decls.end())
        return -1;

    mi::Size n = decl->get_length();
    if( n == 0)
        return -6;

    // Clone the declaration such that modifications after registration have no effect.
    mi::base::Handle<mi::IEnum_decl> copy(
        create_registered<mi::IEnum_decl>( nullptr, "Enum_decl"));
    for( mi::Size i = 0; i < n; ++i) {
        mi::Sint32 result = copy->add_enumerator( decl->get_name( i), decl->get_value( i));
        MI_ASSERT( result == 0);
        boost::ignore_unused( result);
    }

    // Set the type name
    auto* copy_impl = static_cast<Enum_decl_impl*>( copy.get());
    copy_impl->set_enum_type_name( enum_name);

    m_enum_decls[enum_name] = copy;

    return 0;
}

mi::Sint32 Factory::unregister_enum_decl( const char* enum_name)
{
    MI_ASSERT( enum_name);

    mi::base::Lock::Block block( &m_enum_decls_lock);

    auto it = m_enum_decls.find( enum_name);
    if( it == m_enum_decls.end())
        return -1;

    m_enum_decls.erase( it);

    return 0;
}

void Factory::unregister_enum_decls()
{
    mi::base::Lock::Block block( &m_enum_decls_lock);
    m_enum_decls.clear();
}

const mi::IEnum_decl* Factory::get_enum_decl( const char* enum_name) const
{
    MI_ASSERT( enum_name);

    mi::base::Lock::Block block( &m_enum_decls_lock);

    auto it = m_enum_decls.find( enum_name);
    if( it == m_enum_decls.end())
        return nullptr;

    it->second->retain();
    return it->second.get();
}

mi::Sint32 Factory::register_structure_decl(
    const char* structure_name, const mi::IStructure_decl* decl)
{
    MI_ASSERT( structure_name);
    MI_ASSERT( decl);

    mi::base::Lock::Block block2( &m_enum_decls_lock);
    mi::base::Lock::Block block3( &m_structure_decls_lock);

    if( m_factory_functions.find( structure_name) != m_factory_functions.end())
        return -1;
    if( m_enum_decls.find( structure_name)        != m_enum_decls.end())
        return -1;
    if( m_structure_decls.find( structure_name)   != m_structure_decls.end())
        return -1;

    // Maybe we should check in addition that all member type names are valid. But there are two
    // problems:
    // - Some types require a transaction for instantiation, which we do not have here (could
    //   implement a check for valid type names without creating instances of it).
    // - Type names might become invalid since there is no reference counting between structure
    //   and enum declarations.

    std::vector<std::string> blacklist;
    blacklist.emplace_back( structure_name);
    if( contains_blacklisted_type_names( decl, blacklist))
        return -5;

    // Clone the declaration such that modifications after registration have no effect.
    mi::base::Handle<mi::IStructure_decl> copy(
        create_registered<mi::IStructure_decl>( nullptr, "Structure_decl"));
    mi::Size n = decl->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::Sint32 result
            = copy->add_member( decl->get_member_type_name( i), decl->get_member_name( i));
        MI_ASSERT( result == 0);
        boost::ignore_unused( result);
    }

    // Set the type name
    auto* copy_impl = static_cast<Structure_decl_impl*>( copy.get());
    copy_impl->set_structure_type_name( structure_name);

    m_structure_decls[structure_name] = copy;

    return 0;
}

mi::Sint32 Factory::unregister_structure_decl( const char* structure_name)
{
    MI_ASSERT( structure_name);

    mi::base::Lock::Block block( &m_structure_decls_lock);

    auto it = m_structure_decls.find( structure_name);
    if( it == m_structure_decls.end())
        return -1;

    m_structure_decls.erase( it);

    return 0;
}

void Factory::unregister_structure_decls()
{
    mi::base::Lock::Block block( &m_structure_decls_lock);
    m_structure_decls.clear();
}

const mi::IStructure_decl* Factory::get_structure_decl( const char* structure_name) const
{
    MI_ASSERT( structure_name);

    mi::base::Lock::Block block( &m_structure_decls_lock);

    auto it = m_structure_decls.find( structure_name);
    if( it == m_structure_decls.end())
        return nullptr;

    it->second->retain();
    return it->second.get();
}

std::string Factory::uuid_to_string( const mi::base::Uuid& uuid)
{
    std::ostringstream s;
    s.fill( '0');
    s << std::setiosflags( std::ios::right) << std::hex << '('
        << "0x" << std::setw( 8) <<    uuid.m_id1                << ','
        << "0x" << std::setw( 4) <<   (uuid.m_id2      & 0xffff) << ','
        << "0x" << std::setw( 4) <<   (uuid.m_id2 >> 16)         << ','
        << "0x" << std::setw( 2) <<  ((uuid.m_id3      ) & 0xff) << ','
        << "0x" << std::setw( 2) <<  ((uuid.m_id3 >>  8) & 0xff) << ','
        << "0x" << std::setw( 2) <<  ((uuid.m_id3 >> 16) & 0xff) << ','
        << "0x" << std::setw( 2) <<   (uuid.m_id3 >> 24)         << ','
        << "0x" << std::setw( 2) <<  ((uuid.m_id4      ) & 0xff) << ','
        << "0x" << std::setw( 2) <<  ((uuid.m_id4 >>  8) & 0xff) << ','
        << "0x" << std::setw( 2) <<  ((uuid.m_id4 >> 16) & 0xff) << ','
        << "0x" << std::setw( 2) <<   (uuid.m_id4 >> 24)         << ')';
    return s.str();
}

std::string Factory::strip_array( const std::string& type_name, mi::Size& length)
{
    // extract delimiters
    mi::Size left_bracket = type_name.rfind( '[');
    if( left_bracket == std::string::npos)
        return {};
    mi::Size right_bracket = type_name.rfind( ']');
    if( right_bracket != type_name.length() - 1)
        return {};

    // extract length
    std::string length_str = type_name.substr( left_bracket+1, right_bracket-left_bracket-1);
    if( !length_str.empty()) {
        std::optional<mi::Size> length_optional
            = STRING::lexicographic_cast_s<mi::Size>( length_str);
        if( !length_optional.has_value())
            return {};
        if( length_optional.value() == 0)
            return {};
        length = length_optional.value();
    } else
        length = 0;

    // extract element type name
    return type_name.substr( 0, left_bracket);
}

std::string Factory::strip_map( const std::string& type_name)
{
    if( type_name.substr( 0, 4) != "Map<")
        return {};
    if( type_name[type_name.size() - 1] != '>')
        return {};
    return type_name.substr( 4, type_name.size() - 5);
}

std::string Factory::strip_pointer( const std::string& type_name)
{
    if( type_name.substr( 0, 8) != "Pointer<")
        return {};
    if( type_name[type_name.size()-1] != '>')
        return {};
    return type_name.substr( 8, type_name.size() - 9);
}

std::string Factory::strip_const_pointer( const std::string& type_name)
{
    if( type_name.substr( 0, 14) != "Const_pointer<")
        return {};
    if( type_name[type_name.size()-1] != '>')
        return {};
    return type_name.substr( 14, type_name.size() - 15);
}

DB::Transaction* Factory::get_transaction( const mi::IData* data)
{
    MI_ASSERT( data);

    // extract transaction from IRef
    mi::base::Handle ref( data->get_interface<mi::IRef>());
    if( ref) {
        mi::base::Handle proxy( data->get_interface<IProxy>());
        if( proxy) {
            const auto* impl_proxy = static_cast<const Ref_impl_proxy*>( data);
            return impl_proxy->get_transaction();
        } else {
            const auto* impl = static_cast<const Ref_impl*>( data);
           return impl->get_transaction();
        }
    }

    // extract transaction from IStructure
    mi::base::Handle structure( data->get_interface<mi::IStructure>());
    if( structure) {
        mi::base::Handle proxy( data->get_interface<IProxy>());
        if( proxy) {
            const auto* impl_proxy = static_cast<const Structure_impl_proxy*>( data);
            return impl_proxy->get_transaction();
        } else {
            const auto* impl = static_cast<const Structure_impl*>( data);
            return impl->get_transaction();
        }
    }

    // extract transaction from IDynamic_array
    mi::base::Handle dynamic_array( data->get_interface<mi::IDynamic_array>());
    if( dynamic_array) {
        mi::base::Handle proxy( data->get_interface<IProxy>());
        if( proxy) {
            const auto* impl_proxy = static_cast<const Dynamic_array_impl_proxy*>( data);
            return impl_proxy->get_transaction();
        } else {
            const auto* impl = static_cast<const Dynamic_array_impl*>( data);
            return impl->get_transaction();
        }
    }

    // extract transaction from IArray
    mi::base::Handle array( data->get_interface<mi::IArray>());
    if( array) {
        mi::base::Handle proxy( data->get_interface<IProxy>());
        if( proxy) {
            const auto* impl_proxy = static_cast<const Array_impl_proxy*>( data);
            return impl_proxy->get_transaction();
        } else {
            const auto* impl = static_cast<const Array_impl*>( data);
            return impl->get_transaction();
        }
    }

    // extract transaction from IMap
    mi::base::Handle map( data->get_interface<mi::IMap>());
    if( map) {
        const auto* impl = static_cast<const Map_impl*>( data);
        return impl->get_transaction();
    }

    // extract transaction from ICompound
    if( mi::ICompound::compare_iid( data->get_iid()))
        return nullptr;

    // all interfaces derived from IData_collection should be handled now
    MI_ASSERT( !mi::IData_collection::compare_iid( data->get_iid()));

    // extract transaction from IPointer
    mi::base::Handle pointer( data->get_interface<mi::IPointer>());
    if( pointer) {
        const auto* impl = static_cast<const Pointer_impl*>( data);
        return impl->get_transaction();
    }

    // extract transaction from IConst_pointer
    mi::base::Handle const_pointer( data->get_interface<mi::IConst_pointer>());
    if( const_pointer) {
        const auto* impl = static_cast<const Const_pointer_impl*>( data);
        return impl->get_transaction();
    }

    return nullptr;
}

ITag_handler* Factory::get_tag_handler() const
{
    if( !m_tag_handler)
        return nullptr;

    m_tag_handler->retain();
    return m_tag_handler.get();
}

mi::Sint32 Factory::register_class( const char* class_name, Factory_function factory)
{
    MI_ASSERT( class_name);
    MI_ASSERT( factory);

    mi::base::Lock::Block block2( &m_enum_decls_lock);
    mi::base::Lock::Block block3( &m_structure_decls_lock);

    if( m_factory_functions.find( class_name) != m_factory_functions.end())
        return -1;
    if( m_enum_decls.find( class_name)        != m_enum_decls.end())
        return -1;
    if( m_structure_decls.find( class_name)   != m_structure_decls.end())
        return -1;

    m_factory_functions[class_name] = factory;
    return 0;
}

bool Factory::contains_blacklisted_type_names(
    const std::string& type_name, std::vector<std::string>& blacklist) const
{
    // check if type_name is blacklisted
    auto it = find( blacklist.begin(), blacklist.end(), type_name);
    if( it != blacklist.end())
        return true;

    // descend into structures
    auto it_structure_decl = m_structure_decls.find( type_name);
    if( it_structure_decl != m_structure_decls.end()) {
        blacklist.push_back( type_name);
        bool result = contains_blacklisted_type_names( it_structure_decl->second.get(), blacklist);
        blacklist.pop_back();
        return result;
    }

    // descend into arrays
    mi::Size length;
    const std::string& array_element = strip_array( type_name, length);
    if( !array_element.empty())
        return contains_blacklisted_type_names( array_element, blacklist);

    // descend into maps
    const std::string& map_value = strip_map( type_name);
    if( !map_value.empty())
        return contains_blacklisted_type_names( map_value, blacklist);

    // descend into pointers
    const std::string& pointer_nested = strip_pointer( type_name);
    if( !pointer_nested.empty())
        return contains_blacklisted_type_names( pointer_nested, blacklist);

    // descend into const pointers
    const std::string& const_pointer_nested = strip_const_pointer( type_name);
    if( !const_pointer_nested.empty())
        return contains_blacklisted_type_names( const_pointer_nested, blacklist);

    return false;

}

bool Factory::contains_blacklisted_type_names(
    const mi::IStructure_decl* decl, std::vector<std::string>& blacklist) const
{
    mi::Size n = decl->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        std::string member_type_name = decl->get_member_type_name( i);
        if( contains_blacklisted_type_names( member_type_name, blacklist))
            return true;
    }
    return false;
}

mi::base::IInterface* Factory::create_registered(
    DB::Transaction* transaction,
    const char* class_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    auto it = m_factory_functions.find( class_name);
    if( it == m_factory_functions.end())
        return nullptr;

    Factory_function factory = it->second;
    return factory( this, transaction, argc, argv);
}

mi::base::IInterface* Factory::create_array(
    DB::Transaction* transaction, const char* type_name) const
{
    MI_ASSERT( type_name);
    std::string type_name_str( type_name);
    MI_ASSERT( type_name_str[type_name_str.size()-1] == ']');

    const mi::base::IInterface* new_argv[2];

    // extract delimiters
    mi::Size left_bracket = type_name_str.rfind( '[');
    if( left_bracket == std::string::npos)
        return nullptr;
    mi::Size right_bracket = type_name_str.rfind( ']');
    if( right_bracket != type_name_str.length() - 1)
        return nullptr;

    // extract element type name
    std::string element_type_name = type_name_str.substr( 0, left_bracket);
    mi::base::Handle<mi::IString> element_type_name_string(
        create_registered<mi::IString>( transaction, "String"));
    element_type_name_string->set_c_str( element_type_name.c_str());
    new_argv[0] = element_type_name_string.get();

    // extract length
    std::string length_str = type_name_str.substr( left_bracket+1, right_bracket-left_bracket-1);
    if( length_str.empty()) {
        // create dynamic array instance
        return create_registered( transaction, "__Dynamic_array", 1, new_argv);
    }

    // extract length (continued)
    std::optional<mi::Size> length_optional
        = STRING::lexicographic_cast_s<mi::Size>( length_str);
    if( !length_optional.has_value())
        return nullptr;
    mi::Size length = length_optional.value();
    mi::base::Handle<mi::ISize> length_value(
        create_registered<mi::ISize>( transaction, "Size"));
    length_value->set_value( length);

    // create array instance
    new_argv[1] = length_value.get();
    return create_registered( transaction, "__Array", 2, new_argv);
}

mi::base::IInterface* Factory::create_map(
    DB::Transaction* transaction, const char* type_name) const
{
    MI_ASSERT( type_name);
    std::string type_name_str( type_name);
    MI_ASSERT( type_name_str.substr( 0, 4) == "Map<");

    const mi::base::IInterface* new_argv[1];

    // extract delimiters
    mi::Size left_angle_bracket = type_name_str.find( '<');
    mi::Size right_angle_bracket = type_name_str.rfind( '>');
    if( left_angle_bracket != 3)
        return nullptr;
    if( right_angle_bracket != type_name_str.length() - 1)
        return nullptr;

    // extract value type name
    std::string value_type_name
        = type_name_str.substr( left_angle_bracket+1, right_angle_bracket-left_angle_bracket-1);
    mi::base::Handle<mi::IString> value_type_name_string(
        create_registered<mi::IString>( transaction, "String"));
    value_type_name_string->set_c_str( value_type_name.c_str());
    new_argv[0] = value_type_name_string.get();

    // create map instance
    return create_registered( transaction, "__Map", 1, new_argv);
}

mi::base::IInterface* Factory::create_pointer(
    DB::Transaction* transaction, const char* type_name) const
{
    MI_ASSERT( type_name);
    std::string type_name_str( type_name);
    MI_ASSERT(    (type_name_str.substr( 0,  8) == "Pointer<")
               || (type_name_str.substr( 0, 14) == "Const_pointer<"));
    bool is_const = type_name_str.substr( 0, 14) == "Const_pointer<";

    const mi::base::IInterface* new_argv[1];

    // extract delimiters
    mi::Size left_angle_bracket = type_name_str.find( '<');
    if( !is_const && left_angle_bracket != 7)
        return nullptr;
    mi::Size right_angle_bracket = type_name_str.rfind( '>');
    if( right_angle_bracket != type_name_str.length() - 1)
        return nullptr;

    // extract value type name
    std::string value_type_name
        = type_name_str.substr( left_angle_bracket+1, right_angle_bracket-left_angle_bracket-1);
    mi::base::Handle<mi::IString> value_type_name_string(
        create_registered<mi::IString>( transaction, "String"));
    value_type_name_string->set_c_str( value_type_name.c_str());
    new_argv[0] = value_type_name_string.get();

    // create pointer instance
    return create_registered(
        transaction, is_const ? "__Const_pointer" : "__Pointer", 1, new_argv);
}

mi::base::IInterface* Factory::create_structure(
    DB::Transaction* transaction, const char* type_name, const mi::IStructure_decl* decl) const
{
    MI_ASSERT( type_name);

    const mi::base::IInterface* new_argv[2];
    new_argv[0] = decl;

    mi::base::Handle<mi::IString> type_name_string(
        create_registered<mi::IString>( transaction, "String"));
    type_name_string->set_c_str( type_name);
    new_argv[1] = type_name_string.get();

    // create structure instance
    mi::base::IInterface* result = create_registered( transaction, "__Structure", 2, new_argv);
    return result;
}

mi::base::IInterface* Factory::create_enum(
    DB::Transaction* transaction, const char* type_name, const mi::IEnum_decl* decl) const
{
    MI_ASSERT( type_name);

    const mi::base::IInterface* new_argv[2];
    new_argv[0] = decl;

    mi::base::Handle<mi::IString> type_name_string(
        create_registered<mi::IString>( transaction, "String"));
    type_name_string->set_c_str( type_name);
    new_argv[1] = type_name_string.get();

    // create enum instance
    mi::base::IInterface* result = create_registered( transaction, "__Enum", 2, new_argv);
    return result;
}

mi::base::IInterface* Factory::create_with_transaction(
    const char* type_name, const mi::IData* prototype)
{
    mi::base::IInterface* result = create( /*transaction*/ nullptr, type_name);
    if( result)
        return result;

    // The first create() call might have failed for IRef's, non-IData's or no longer registered
    // type names. Extract transaction from prototype for fallback.
    DB::Transaction* transaction = get_transaction( prototype);
    if( !transaction)
        return nullptr;

    // This create() call might fail for non-IData's (not supported via the general factory) or no
    // longer registered type names.
    return create( transaction, type_name);
}

mi::Uint32 Factory::assign_from_to(
    const mi::IData_simple* source, mi::IData_simple* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    // handle INumber
    mi::base::Handle source_value( source->get_interface<mi::INumber>());
    mi::base::Handle target_value( target->get_interface<mi::INumber>());
    if( source_value && target_value)
        return assign_from_to( source_value.get(), target_value.get());

    // handle IString
    mi::base::Handle source_string( source->get_interface<mi::IString>());
    mi::base::Handle target_string( target->get_interface<mi::IString>());
    if( source_string && target_string)
        return assign_from_to( source_string.get(), target_string.get());

    // handle IRef
    mi::base::Handle source_ref( source->get_interface<mi::IRef>());
    mi::base::Handle target_ref( target->get_interface<mi::IRef>());
    if( source_ref && target_ref)
        return assign_from_to( source_ref.get(), target_ref.get());

    // handle IEnum/IEnum
    mi::base::Handle source_enum( source->get_interface<mi::IEnum>());
    mi::base::Handle target_enum( target->get_interface<mi::IEnum>());
    if( source_enum && target_enum)
        return assign_from_to( source_enum.get(), target_enum.get(), options);

    // handle IEnum/ISint32
    mi::base::Handle target_sint32( target->get_interface<mi::ISint32>());
    if( source_enum && target_sint32)
        return assign_from_to( source_enum.get(), target_sint32.get());

    // handle IUuid
    mi::base::Handle source_uuid( source->get_interface<mi::IUuid>());
    mi::base::Handle target_uuid( target->get_interface<mi::IUuid>());
    if( source_uuid && target_uuid)
        return assign_from_to( source_uuid.get(), target_uuid.get());

    // handle IVoid
    mi::base::Handle source_void( source->get_interface<mi::IVoid>());
    mi::base::Handle target_void( target->get_interface<mi::IVoid>());
    if( source_void && target_void)
        return 0; // nothing to do

    // handle IPointer/IPointer
    mi::base::Handle source_pointer( source->get_interface<mi::IPointer>());
    mi::base::Handle target_pointer( target->get_interface<mi::IPointer>());
    if( source_pointer && target_pointer)
        return assign_from_to( source_pointer.get(), target_pointer.get(), options);

    // handle IConst_pointer/IConst_pointer
    mi::base::Handle source_const_pointer( source->get_interface<mi::IConst_pointer>());
    mi::base::Handle target_const_pointer( target->get_interface<mi::IConst_pointer>());
    if( source_const_pointer && target_const_pointer)
        return assign_from_to( source_const_pointer.get(), target_const_pointer.get(), options);

    // handle IConst_pointer/IPointer
    if( source_const_pointer && target_pointer)
        return assign_from_to( source_const_pointer.get(), target_pointer.get(), options);

    // handle IPointer/IConst_pointer
    if( source_pointer && target_const_pointer)
        return assign_from_to( source_pointer.get(), target_const_pointer.get(), options);

    return mi::neuraylib::IFactory::NO_CONVERSION;
}

mi::Uint32 Factory::assign_from_to(
    const mi::IData_collection* source, mi::IData_collection* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    mi::Uint32 result = 0;

    // check that both or none arguments are of type ICompound
    mi::base::Handle source_compound( source->get_interface<mi::ICompound>());
    mi::base::Handle target_compound( target->get_interface<mi::ICompound>());
    if( !!source_compound ^ !!target_compound)
        result |= mi::neuraylib::IFactory::DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IDynamic_array
    mi::base::Handle source_dynamic_array( source->get_interface<mi::IDynamic_array>());
    mi::base::Handle target_dynamic_array( target->get_interface<mi::IDynamic_array>());
    if( !!source_dynamic_array ^ !!target_dynamic_array)
        result |= mi::neuraylib::IFactory::DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IArray
    mi::base::Handle source_array( source->get_interface<mi::IArray>());
    mi::base::Handle target_array( target->get_interface<mi::IArray>());
    if( !!source_array ^ !!target_array)
        result |= mi::neuraylib::IFactory::DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IStructure
    mi::base::Handle source_structure( source->get_interface<mi::IStructure>());
    mi::base::Handle target_structure( target->get_interface<mi::IStructure>());
    if( !!source_structure ^ !!target_structure)
        result |= mi::neuraylib::IFactory::DIFFERENT_COLLECTIONS;

    // check that both or none arguments are of type IMap
    mi::base::Handle source_map( source->get_interface<mi::IMap>());
    mi::base::Handle target_map( target->get_interface<mi::IMap>());
    if( !!source_map ^ !!target_map)
        result |= mi::neuraylib::IFactory::DIFFERENT_COLLECTIONS;

    // adjust length of target if it is a dynamic array or map (unless disabled by option)
    if( !(options & mi::neuraylib::IFactory::FIX_SET_OF_TARGET_KEYS)) {

        if( target_dynamic_array) {

            if( source_array) {
                mi::Size new_length = source_array->get_length();
                target_dynamic_array->set_length( new_length);
            } else {
                // find highest index-like key in source
                mi::Size max_index_plus_one = 0;
                mi::Size n = source->get_length();
                for( mi::Size i = 0; i < n; ++i) {
                    const char* key = source->get_key( i);
                    std::optional<mi::Size> index_optional
                        = STRING::lexicographic_cast_s<mi::Size>( key);
                    if( !index_optional.has_value())
                        continue;
                    mi::Size index = index_optional.value();
                    if( index+1 > max_index_plus_one)
                        max_index_plus_one = index+1;
                }
                target_dynamic_array->set_length( max_index_plus_one);
            }
        }

        if( target_map) {

            // remove target keys not in source
            mi::Size n = target_map->get_length();
            for( mi::Size i = 0; i < n; ) {
                const char* key = target_map->get_key( i);
                if( source->has_key( key))
                    ++i;
                else {
                    target_map->erase( key);
                    --n;
                }
            }

            std::string target_type_name = target->get_type_name();
            std::string target_element_name = strip_map( target_type_name);
            DB::Transaction* transaction = get_transaction( target);

            // insert source keys not in target
            n = source->get_length();
            for( mi::Size i = 0; i < n; ++i) {

                const char* key = source->get_key( i);
                if( target_map->has_key( key))
                    continue;

                // for untyped maps use the type name of the source element
                std::string element_name = target_element_name;
                if( target_element_name == "Interface") {
                    mi::base::Handle<const mi::IData> source_value_data(
                        source->get_value<mi::IData>( key));
                    if( source_value_data)
                        element_name = source_value_data->get_type_name();
                }

                mi::base::Handle<mi::base::IInterface> target_value(
                    create( transaction, element_name.c_str()));
                if( !target_value)
                    continue;

                mi::Sint32 result2 = target_map->insert( key, target_value.get());
                MI_ASSERT( result2 == 0);
                boost::ignore_unused( result2);
            }
        }
    }

    // iterate over keys in source, and try to assign to the corresponding key in target
    mi::Size keys_found_in_target   = 0;
    mi::Size keys_missing_in_target = 0;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th key from source and check whether target has that key
        const char* key = source->get_key( i);
        if( !target->has_key( key)) {
            ++keys_missing_in_target;
            continue;
        }
        ++keys_found_in_target;

        // check if source value is of type IData
        mi::base::Handle source_element( source->get_value<mi::IData>( key));
        if( !source_element) {
            result |= mi::neuraylib::IFactory::NON_IDATA_VALUES;
            continue;
        }

        // check if target value is of type IData
        mi::base::Handle target_element( target->get_value<mi::IData>( key));
        if( !target_element){
            result |= mi::neuraylib::IFactory::NON_IDATA_VALUES;
            continue;
        }

        // invoke assign_from_to() for this key, and set the target value for this key again
        result |= assign_from_to( source_element.get(), target_element.get(), options);
        mi::Uint32 result2 = target->set_value( key, target_element.get());
        MI_ASSERT( result2 == 0);
        boost::ignore_unused( result2);
    }

    if( keys_missing_in_target > 0)
        result |= mi::neuraylib::IFactory::TARGET_KEY_MISSING;
    if( target->get_length() > keys_found_in_target)
        result |= mi::neuraylib::IFactory::SOURCE_KEY_MISSING;

    return result;
}

mi::Uint32 Factory::assign_from_to( const mi::INumber* source, mi::INumber* target) const
{
    MI_ASSERT( source && target);

    const std::string target_type_name = target->get_type_name();

    if( target_type_name == "Boolean")
        target->set_value( source->get_value<bool>());
    else if( target_type_name == "Sint8")
        target->set_value( source->get_value<mi::Sint8>());
    else if( target_type_name == "Sint16")
        target->set_value( source->get_value<mi::Sint16>());
    else if( target_type_name == "Sint32")
        target->set_value( source->get_value<mi::Sint32>());
    else if( target_type_name == "Sint64")
        target->set_value( source->get_value<mi::Sint64>());
    else if( target_type_name == "Uint8")
        target->set_value( source->get_value<mi::Uint8>());
    else if( target_type_name == "Uint16")
        target->set_value( source->get_value<mi::Uint16>());
    else if( target_type_name == "Uint32")
        target->set_value( source->get_value<mi::Uint32>());
    else if( target_type_name == "Uint64")
        target->set_value( source->get_value<mi::Uint64>());
    else if( target_type_name == "Float32")
        target->set_value( source->get_value<mi::Float32>());
    else if( target_type_name == "Float64")
        target->set_value( source->get_value<mi::Float64>());
    else if( target_type_name == "Size")
        target->set_value( source->get_value<mi::Size>());
    else if( target_type_name == "Difference")
        target->set_value( source->get_value<mi::Difference>());
    else {
        MI_ASSERT( false);
    }

    return 0;
}

mi::Uint32 Factory::assign_from_to( const mi::IString* source, mi::IString* target) const
{
    MI_ASSERT( source && target);

    const char* s = source->get_c_str();
    target->set_c_str( s);
    return 0;
}

mi::Uint32 Factory::assign_from_to( const mi::IRef* source, mi::IRef* target) const
{
    MI_ASSERT( source && target);

    mi::base::Handle<const mi::base::IInterface> reference( source->get_reference());
    mi::Sint32 result = target->set_reference( reference.get());
    if( result == -4)
        return mi::neuraylib::IFactory::INCOMPATIBLE_PRIVACY_LEVELS;

    return 0;
}

mi::Uint32 Factory::assign_from_to(
    const mi::IEnum* source, mi::IEnum* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    if( strcmp( source->get_type_name(), target->get_type_name()) != 0)
        return mi::neuraylib::IFactory::INCOMPATIBLE_ENUM_TYPES;

    const char* name = source->get_value_by_name();
    mi::Uint32 result = target->set_value_by_name( name);
    MI_ASSERT( result == 0);
    boost::ignore_unused( result);
    return 0;
}

mi::Uint32 Factory::assign_from_to( const mi::IEnum* source, mi::ISint32* target) const
{
    MI_ASSERT( source && target);

    mi::Sint32 value = source->get_value();
    target->set_value( value);
    return 0;
}

mi::Uint32 Factory::assign_from_to( const mi::IUuid* source, mi::IUuid* target) const
{
    MI_ASSERT( source && target);

    mi::base::Uuid uuid = source->get_uuid();
    target->set_uuid( uuid);
    return 0;
}

mi::Uint32 Factory::assign_from_to(
    const mi::IPointer* source, mi::IPointer* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    // shallow assignment
    if( (options & mi::neuraylib::IFactory::DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::base::Handle<mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        if( result != 0)
            return mi::neuraylib::IFactory::INCOMPATIBLE_POINTER_TYPES;
        return 0;
    }

    // deep assignment, source has nullptr
    mi::base::Handle<mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer) {
        target->set_pointer( nullptr);
        return 0;
    }

    // deep assignment, source has non-IData pointer
    mi::base::Handle source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data)
        return mi::neuraylib::IFactory::NON_IDATA_VALUES;

    // deep assignment, target has nullptr
    mi::base::Handle<mi::base::IInterface> target_pointer( target->get_pointer());
    if( !target_pointer)
        return mi::neuraylib::IFactory::NULL_POINTER;

    // deep assignment, source has non-IData pointer
    mi::base::Handle target_pointer_data( target->get_pointer<mi::IData>());
    if( !target_pointer_data)
        return mi::neuraylib::IFactory::NON_IDATA_VALUES;

    // deep assignment
    return assign_from_to( source_pointer_data.get(), target_pointer_data.get(), options);
}

mi::Uint32 Factory::assign_from_to(
    const mi::IConst_pointer* source, mi::IPointer* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    // shallow assignment
    if( (options & mi::neuraylib::IFactory::DEEP_ASSIGNMENT_OR_CLONE) == 0)
        return mi::neuraylib::IFactory::INCOMPATIBLE_POINTER_TYPES;

    // deep assignment, source has nullptr
    mi::base::Handle<const mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer) {
        target->set_pointer( nullptr);
        return 0;
    }

    // deep assignment, source has non-IData pointer
    mi::base::Handle source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data)
        return mi::neuraylib::IFactory::NON_IDATA_VALUES;

    // deep assignment, target has nullptr
    mi::base::Handle<mi::base::IInterface> target_pointer( target->get_pointer());
    if( !target_pointer)
        return mi::neuraylib::IFactory::NULL_POINTER;

    // deep assignment, source has non-IData pointer
    mi::base::Handle target_pointer_data( target->get_pointer<mi::IData>());
    if( !target_pointer_data)
        return mi::neuraylib::IFactory::NON_IDATA_VALUES;

    // deep assignment
    return assign_from_to( source_pointer_data.get(), target_pointer_data.get(), options);
}

mi::Uint32 Factory::assign_from_to(
    const mi::IConst_pointer* source, mi::IConst_pointer* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    // shallow assignment
    if( (options & mi::neuraylib::IFactory::DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::base::Handle<const mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        if( result != 0)
            return mi::neuraylib::IFactory::INCOMPATIBLE_POINTER_TYPES;
        return 0;
    }

    // deep assignment
    return mi::neuraylib::IFactory::DEEP_ASSIGNMENT_TO_CONST_POINTER;
}

mi::Uint32 Factory::assign_from_to(
    const mi::IPointer* source, mi::IConst_pointer* target, mi::Uint32 options) const
{
    MI_ASSERT( source && target);

    // shallow assignment
    if( (options & mi::neuraylib::IFactory::DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        mi::base::Handle<mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        if( result != 0)
            return mi::neuraylib::IFactory::INCOMPATIBLE_POINTER_TYPES;
        return 0;
    }

    // deep assignment
    return mi::neuraylib::IFactory::DEEP_ASSIGNMENT_TO_CONST_POINTER;
}

mi::IData_simple* Factory::clone( const mi::IData_simple* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    // handle IRef
    mi::base::Handle source_ref( source->get_interface<mi::IRef>());
    if( source_ref)
        return clone( source_ref.get(), options);

    // handle IPointer
    mi::base::Handle source_pointer( source->get_interface<mi::IPointer>());
    if( source_pointer)
        return clone( source_pointer.get(), options);

    // handle IConst_pointer
    mi::base::Handle source_const_pointer( source->get_interface<mi::IConst_pointer>());
    if( source_const_pointer)
        return clone( source_const_pointer.get(), options);

    // handle other subtypes of IData_simple
    auto* target = create<mi::IData_simple>( /*transaction*/ nullptr, source->get_type_name());
    mi::Uint32 result = assign_from_to( source, target, options);
    MI_ASSERT( result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::IData_collection* Factory::clone( const mi::IData_collection* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    // handle ICompound
    mi::base::Handle source_compound( source->get_interface<mi::ICompound>());
    if( source_compound)
        return clone( source_compound.get(), options);

    // handle IDynamic_array
    mi::base::Handle source_dynamic_array( source->get_interface<mi::IDynamic_array>());
    if( source_dynamic_array)
        return clone( source_dynamic_array.get(), options);

    // handle IArray
    mi::base::Handle source_array( source->get_interface<mi::IArray>());
    if( source_array)
        return clone( source_array.get(), options);

    // handle IStructure
    mi::base::Handle source_structure( source->get_interface<mi::IStructure>());
    if( source_structure)
        return clone( source_structure.get(), options);

    // handle IMap
    mi::base::Handle source_map( source->get_interface<mi::IMap>());
    if( source_map)
        return clone( source_map.get(), options);

    MI_ASSERT( false);
    return nullptr;
}

mi::IRef* Factory::clone( const mi::IRef* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    DB::Transaction* transaction = get_transaction( source);
    auto* target = create<mi::IRef>( transaction, source->get_type_name());
    mi::Uint32 result = assign_from_to( source, target, options);
    MI_ASSERT( result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::IPointer* Factory::clone( const mi::IPointer* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    const char* source_type_name = source->get_type_name();

    // shallow clone
    if( (options & mi::neuraylib::IFactory::DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        auto* target = create_with_transaction<mi::IPointer>( source_type_name, source);
        if( !target)
            return nullptr;
        mi::base::Handle<mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        MI_ASSERT( result == 0);
        boost::ignore_unused( result);
        return target;
    }

    // deep clone, nullptr
    mi::base::Handle<mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer)
        return create<mi::IPointer>( /*transaction*/ nullptr, source_type_name);

    // deep clone, non-nullptr
    mi::base::Handle source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data)
        return nullptr;
    mi::base::Handle<mi::IData> target_pointer_data(
        clone( source_pointer_data.get(), options));
    if( !target_pointer_data)
        return nullptr;

    auto* target = create_with_transaction<mi::IPointer>( source_type_name, source);
    if( !target)
        return nullptr;
    mi::Uint32 result = target->set_pointer( target_pointer_data.get());
    MI_ASSERT( result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::IConst_pointer* Factory::clone( const mi::IConst_pointer* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    const char* source_type_name = source->get_type_name();

    // shallow clone
    if( (options & mi::neuraylib::IFactory::DEEP_ASSIGNMENT_OR_CLONE) == 0) {
        auto* target = create_with_transaction<mi::IConst_pointer>( source_type_name, source);
        if( !target)
            return nullptr;
        mi::base::Handle<const mi::base::IInterface> pointer( source->get_pointer());
        mi::Uint32 result = target->set_pointer( pointer.get());
        MI_ASSERT( result == 0);
        boost::ignore_unused( result);
        return target;
    }

    // deep clone, nullptr
    mi::base::Handle<const mi::base::IInterface> source_pointer( source->get_pointer());
    if( !source_pointer)
        return create<mi::IConst_pointer>( /*transaction*/ nullptr, source_type_name);

    // deep clone, non-nullptr
    mi::base::Handle source_pointer_data( source->get_pointer<mi::IData>());
    if( !source_pointer_data)
        return nullptr;
    mi::base::Handle<const mi::IData> target_pointer_data(
        clone( source_pointer_data.get(), options));
    if( !target_pointer_data)
        return nullptr;

    auto* target = create_with_transaction<mi::IConst_pointer>( source_type_name, source);
    if( !target)
        return nullptr;
    mi::Uint32 result = target->set_pointer( target_pointer_data.get());
    MI_ASSERT( result == 0);
    boost::ignore_unused( result);
    return target;
}

mi::ICompound* Factory::clone( const mi::ICompound* source, mi::Uint32 options)
{
     MI_ASSERT( source);

     auto* target = create<mi::ICompound>( /*transaction*/ nullptr, source->get_type_name());
     mi::Uint32 result = assign_from_to( source, target, options);
     MI_ASSERT( result == 0);
     boost::ignore_unused( result);
     return target;
}

mi::IDynamic_array* Factory::clone( const mi::IDynamic_array* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    auto* target = create_with_transaction<mi::IDynamic_array>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th element
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_element( i));

        // check if source value is of type IData
        mi::base::Handle source_value_data( source_value_interface->get_interface<mi::IData>());
        if( !source_value_data) {
            target->release();
            return nullptr;
        }

        // clone source value
        mi::base::Handle<mi::IData> target_value_data( clone( source_value_data.get(), options));
        if( !target_value_data) {
            // might happen for non-IData's or no longer registered type names
            target->release();
            return nullptr;
        }

        // and set clone in target
        target->push_back( target_value_data.get());
    }

    return target;
}

mi::IArray* Factory::clone( const mi::IArray* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    auto* target = create_with_transaction<mi::IArray>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th element
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_element( i));

        // check if source value is of type IData
        mi::base::Handle source_value_data( source_value_interface->get_interface<mi::IData>());
        if( !source_value_data) {
            target->release();
            return nullptr;
        }

        // clone source value
        mi::base::Handle<mi::IData> target_value_data( clone( source_value_data.get(), options));
        if( !target_value_data) {
            // might happen for non-IData's or no longer registered type names
            target->release();
            return nullptr;
        }

        // and set clone in target
        target->set_value( i, target_value_data.get());
    }

    return target;
}

mi::IStructure* Factory::clone( const mi::IStructure* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    auto* target = create_with_transaction<mi::IStructure>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th key from source and its value
        const char* key = source->get_key( i);
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_value( key));

        // check if source value is of type IData
        mi::base::Handle source_value_data( source_value_interface->get_interface<mi::IData>());
        if( !source_value_data) {
            target->release();
            return nullptr;
        }

        // clone source value
        mi::base::Handle<mi::IData> target_value_data( clone( source_value_data.get(), options));
        if( !target_value_data) {
            // might happen for non-IData's or no longer registered type names
            target->release();
            return nullptr;
        }

        // and set clone in target
        target->set_value( key, target_value_data.get());
    }

    return target;
}

mi::IMap* Factory::clone( const mi::IMap* source, mi::Uint32 options)
{
    MI_ASSERT( source);

    auto* target = create_with_transaction<mi::IMap>( source->get_type_name(), source);
    if( !target)
        return nullptr;

    mi::Size n = source->get_length();
    for( mi::Size i = 0; i < n; ++i) {

        // get i-th key from source and its value
        const char* key = source->get_key( i);
        mi::base::Handle<const mi::base::IInterface> source_value_interface(
            source->get_value( key));

        // check if source value is of type IData
        mi::base::Handle source_value_data( source_value_interface->get_interface<mi::IData>());
        if( !source_value_data) {
            target->release();
            return nullptr;
        }

        // clone source value
        mi::base::Handle<mi::IData> target_value_data( clone( source_value_data.get(), options));
        if( !target_value_data) {
            // might happen for non-IData's or no longer registered type names
            target->release();
            return nullptr;
        }

        // and insert clone into map
        target->insert( key, target_value_data.get());
    }

    return target;
}

mi::Sint32 Factory::compare( const mi::IData_simple* lhs, const mi::IData_simple* rhs)
{
    MI_ASSERT( lhs && rhs);

    // handle INumber
    mi::base::Handle lhs_value( lhs->get_interface<mi::INumber>());
    mi::base::Handle rhs_value( rhs->get_interface<mi::INumber>());
    if( lhs_value && rhs_value)
        return compare( lhs_value.get(), rhs_value.get());

    // handle IString
    mi::base::Handle lhs_string( lhs->get_interface<mi::IString>());
    mi::base::Handle rhs_string( rhs->get_interface<mi::IString>());
    if( lhs_string && rhs_string)
        return compare( lhs_string.get(), rhs_string.get());

    // handle IRef
    mi::base::Handle lhs_ref( lhs->get_interface<mi::IRef>());
    mi::base::Handle rhs_ref( rhs->get_interface<mi::IRef>());
    if( lhs_ref && rhs_ref)
        return compare( lhs_ref.get(), rhs_ref.get());

    // handle IEnum
    mi::base::Handle lhs_enum( lhs->get_interface<mi::IEnum>());
    mi::base::Handle rhs_enum( rhs->get_interface<mi::IEnum>());
    if( lhs_enum && rhs_enum)
        return compare( lhs_enum.get(), rhs_enum.get());

    // handle IUuid
    mi::base::Handle lhs_uuid( lhs->get_interface<mi::IUuid>());
    mi::base::Handle rhs_uuid( rhs->get_interface<mi::IUuid>());
    if( lhs_uuid && rhs_uuid)
        return compare( lhs_uuid.get(), rhs_uuid.get());

    // handle IVoid
    mi::base::Handle lhs_void( lhs->get_interface<mi::IVoid>());
    mi::base::Handle rhs_void( rhs->get_interface<mi::IVoid>());
    if( lhs_void && rhs_void)
        return 0; // nothing to do

    // handle IPointer/IPointer
    mi::base::Handle lhs_pointer( lhs->get_interface<mi::IPointer>());
    mi::base::Handle rhs_pointer( rhs->get_interface<mi::IPointer>());
    if( lhs_pointer && rhs_pointer)
        return compare( lhs_pointer.get(), rhs_pointer.get());

    // handle IConst_pointer/IConst_pointer
    mi::base::Handle lhs_const_pointer( lhs->get_interface<mi::IConst_pointer>());
    mi::base::Handle rhs_const_pointer( rhs->get_interface<mi::IConst_pointer>());
    if( lhs_const_pointer && rhs_const_pointer)
        return compare( lhs_const_pointer.get(), rhs_const_pointer.get());

    MI_ASSERT( false);
    return 0;
}

mi::Sint32 Factory::compare( const mi::IData_collection* lhs, const mi::IData_collection* rhs)
{
    // compare length
    mi::Size lhs_n = lhs->get_length();
    mi::Size rhs_n = rhs->get_length();
    if( lhs_n < rhs_n) return -1;
    if( lhs_n > rhs_n) return +1;

    MI_ASSERT( lhs_n == rhs_n);

    for( mi::Size i = 0; i < lhs_n; ++i) {

        // compare keys for index i
        const char* lhs_key = lhs->get_key( i);
        const char* rhs_key = rhs->get_key( i);
        int key_cmp = strcmp( lhs_key, rhs_key);
        if( key_cmp != 0)
            return key_cmp;

        // get value for index i from lhs and rhs
        mi::base::Handle<const mi::base::IInterface> lhs_value_interface(
            lhs->get_value( i));
        MI_ASSERT( lhs_value_interface);
        mi::base::Handle<const mi::base::IInterface> rhs_value_interface(
            rhs->get_value( i));
        MI_ASSERT( rhs_value_interface);

        // check if lhs and rhs value is of type IData
        mi::base::Handle lhs_value_data( lhs_value_interface->get_interface<mi::IData>());
        mi::base::Handle rhs_value_data( rhs_value_interface->get_interface<mi::IData>());

        // if one of the values is not of type IData compare the interface pointers
        if( !lhs_value_data || !rhs_value_data) {
            if( lhs_value_interface.get() < rhs_value_interface.get()) return -1;
            if( lhs_value_interface.get() > rhs_value_interface.get()) return +1;
            continue;
        }

        // of both values are of type IData invoke compare() on them
        mi::Sint32 value_cmp = compare( lhs_value_data.get(), rhs_value_data.get());
        if( value_cmp != 0)
            return value_cmp;
    }

    return 0;
}

mi::Sint32 Factory::compare( const mi::INumber* lhs, const mi::INumber* rhs)
{
    const char* lhs_type_name = lhs->get_type_name();

    // bool
    if( strcmp( lhs_type_name, "Boolean") == 0) {
        bool lhs_value = lhs->get_value<bool>();
        bool rhs_value = rhs->get_value<bool>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    // signed integral types
    if( strncmp( lhs_type_name, "Sint", 4) == 0 || strcmp( lhs_type_name, "Difference") == 0) {
        auto lhs_value = lhs->get_value<mi::Sint64>();
        auto rhs_value = rhs->get_value<mi::Sint64>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    // unsigned integral types
    if( strncmp( lhs_type_name, "Uint", 4) == 0 || strcmp( lhs_type_name, "Size") == 0) {
        auto lhs_value = lhs->get_value<mi::Uint64>();
        auto rhs_value = rhs->get_value<mi::Uint64>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    // floating-point types
    if( strncmp( lhs_type_name, "Float", 5) == 0) {
        auto lhs_value = lhs->get_value<mi::Float64>();
        auto rhs_value = rhs->get_value<mi::Float64>();
        if( lhs_value < rhs_value) return -1;
        if( lhs_value > rhs_value) return +1;
        return 0;
    }

    MI_ASSERT( false);
    return 0;
}

mi::Sint32 Factory::compare( const mi::IString* lhs, const mi::IString* rhs)
{
    int result = strcmp( lhs->get_c_str(), rhs->get_c_str());
    return result < 0 ? -1 : (result == 0 ? 0 : +1);
}

mi::Sint32 Factory::compare( const mi::IRef* lhs, const mi::IRef* rhs)
{
    const char* lhs_name = lhs->get_reference_name();
    const char* rhs_name = rhs->get_reference_name();

    if( !lhs_name &&  rhs_name) return -1;
    if(  lhs_name && !rhs_name) return +1;
    if( !lhs_name && !rhs_name) return 0;

    MI_ASSERT( lhs_name && rhs_name);
    int result = strcmp( lhs_name, rhs_name);
    return result < 0 ? -1 : (result == 0 ? 0 : +1);
}

mi::Sint32 Factory::compare( const mi::IEnum* lhs, const mi::IEnum* rhs)
{
    mi::Sint32 lhs_value = lhs->get_value();
    mi::Sint32 rhs_value = rhs->get_value();
    if( lhs_value < rhs_value) return -1;
    if( lhs_value > rhs_value) return +1;
    return 0;
}

mi::Sint32 Factory::compare( const mi::IUuid* lhs, const mi::IUuid* rhs)
{
    mi::base::Uuid lhs_uuid = lhs->get_uuid();
    mi::base::Uuid rhs_uuid = rhs->get_uuid();
    if( lhs_uuid < rhs_uuid) return -1;
    if( lhs_uuid > rhs_uuid) return +1;
    return 0;
}

mi::Sint32 Factory::compare( const mi::IPointer* lhs, const mi::IPointer* rhs)
{
    // get pointer for lhs and rhs as IInterface
    mi::base::Handle<const mi::base::IInterface> lhs_interface( lhs->get_pointer());
    mi::base::Handle<const mi::base::IInterface> rhs_interface( rhs->get_pointer());

    // if at least one of the pointers is \c nullptr compare the interface pointers
    if( !lhs_interface || !rhs_interface) {
        if( !lhs_interface ||  rhs_interface) return -1;
        if(  lhs_interface || !rhs_interface) return +1;
        return 0;
    }

    // get pointer for lhs and rhs as IData (if possible)
    mi::base::Handle lhs_data( lhs_interface->get_interface<mi::IData>());
    mi::base::Handle rhs_data( rhs_interface->get_interface<mi::IData>());

    // if at least one of the values is not of type IData compare the interface pointers
    if( !lhs_data || !rhs_data) {
        if( lhs_interface.get() < rhs_interface.get()) return -1;
        if( lhs_interface.get() > rhs_interface.get()) return +1;
        return 0;
    }

    // if both values are of type IData invoke compare() on them
    return compare( lhs_data.get(), rhs_data.get());
}

mi::Sint32 Factory::compare( const mi::IConst_pointer* lhs, const mi::IConst_pointer* rhs)
{
    // get pointer for lhs and rhs as IInterface
    mi::base::Handle<const mi::base::IInterface> lhs_interface( lhs->get_pointer());
    mi::base::Handle<const mi::base::IInterface> rhs_interface( rhs->get_pointer());

    // if at least one of the pointers is \c nullptr compare the interface pointers
    if( !lhs_interface || !rhs_interface) {
        if( !lhs_interface ||  rhs_interface) return -1;
        if(  lhs_interface || !rhs_interface) return +1;
        return 0;
    }

    // get pointer for lhs and rhs as IData (if possible)
    mi::base::Handle lhs_data( lhs_interface->get_interface<mi::IData>());
    mi::base::Handle rhs_data( rhs_interface->get_interface<mi::IData>());

    // if at least one of the values is not of type IData compare the interface pointers
    if( !lhs_data || !rhs_data) {
        if( lhs_interface.get() < rhs_interface.get()) return -1;
        if( lhs_interface.get() > rhs_interface.get()) return +1;
        return 0;
    }

    // if both values are of type IData invoke compare() on them
    return compare( lhs_data.get(), rhs_data.get());
}

namespace {

std::string get_prefix( mi::Size depth)
{
    std::string prefix;
    prefix.reserve( 4*depth);
    for( mi::Size i = 0; i < depth; i++)
        prefix += "    ";
    return prefix;
}

} // namespace

void Factory::dump(
    const char* name, const mi::IData* data, mi::Size depth, std::ostringstream& s)
{
    if( name) {
        const char* type_name = data->get_type_name();
        // For instances of IStructure and IEnum we skip artificial type names starting with '{' and
        // print just "(struct)" or "(enum)".
        mi::base::Handle<const mi::IStructure> structure( data->get_interface<mi::IStructure>());
        mi::base::Handle<const mi::IEnum> enum_( data->get_interface<mi::IEnum>());
        if( structure && (type_name[0] == '{'))
            s << "(struct) ";
        else if( enum_ && (type_name[0] == '{'))
            s << "(enum) ";
        else
            s << type_name << ' ';
        s << name << " = ";
    }

    switch( uuid_hash32( data->get_iid())) {

        case mi::IBoolean::IID::hash32: {
            mi::base::Handle boolean( data->get_interface<mi::IBoolean>());
             s << (boolean->get_value<bool>() ? "true" : "false");
            return;
        }

        case mi::IUint8::IID::hash32: {
            mi::base::Handle uint8( data->get_interface<mi::IUint8>());
            s << uint8->get_value<mi::Uint8>();
            return;
        }

        case mi::IUint16::IID::hash32: {
            mi::base::Handle uint16( data->get_interface<mi::IUint16>());
            s << uint16->get_value<mi::Uint16>();
            return;
        }

        case mi::IUint32::IID::hash32: {
            mi::base::Handle uint32( data->get_interface<mi::IUint32>());
            s << uint32->get_value<mi::Uint32>();
            return;
        }

        case mi::IUint64::IID::hash32: {
            mi::base::Handle uint64( data->get_interface<mi::IUint64>());
            s << uint64->get_value<mi::Uint64>();
            return;
        }

        case mi::ISint8::IID::hash32: {
            mi::base::Handle sint8( data->get_interface<mi::ISint8>());
            s << sint8->get_value<mi::Sint8>();
            return;
        }

        case mi::ISint16::IID::hash32: {
            mi::base::Handle sint16( data->get_interface<mi::ISint16>());
            s << sint16->get_value<mi::Sint16>();
            return;
        }

        case mi::ISint32::IID::hash32: {
            mi::base::Handle sint32( data->get_interface<mi::ISint32>());
            s << sint32->get_value<mi::Sint32>();
            return;
        }

        case mi::ISint64::IID::hash32: {
            mi::base::Handle sint64( data->get_interface<mi::ISint64>());
            s << sint64->get_value<mi::Sint64>();
            return;
        }

        case mi::IFloat32::IID::hash32: {
            mi::base::Handle float32( data->get_interface<mi::IFloat32>());
            s << float32->get_value<mi::Float32>();
            return;
        }

        case mi::IFloat64::IID::hash32: {
            mi::base::Handle float64( data->get_interface<mi::IFloat64>());
            s << float64->get_value<mi::Float64>();
            return;
        }

        case mi::ISize::IID::hash32: {
            mi::base::Handle size( data->get_interface<mi::ISize>());
            s << size->get_value<mi::Size>();
            return;
        }

        case mi::IDifference::IID::hash32: {
            mi::base::Handle diff( data->get_interface<mi::IDifference>());
            s << diff->get_value<mi::Difference>();
            return;
        }

        case mi::IUuid::IID::hash32: {
            mi::base::Handle uuid( data->get_interface<mi::IUuid>());
            mi::base::Uuid u = uuid->get_uuid();
            s << uuid_to_string( u);
            return;
        }

        case mi::IPointer::IID::hash32: {
            mi::base::Handle pointer(
                data->get_interface<mi::IPointer>());
            mi::base::Handle p( pointer->get_pointer());
            s << p.get();
            return;
        }

        case mi::IConst_pointer::IID::hash32: {
            mi::base::Handle pointer(
                data->get_interface<mi::IConst_pointer>());
            mi::base::Handle p( pointer->get_pointer());
            s << p.get();
            return;
        }

        case mi::IString::IID::hash32: {
            mi::base::Handle string( data->get_interface<mi::IString>());
            s << '\"' << string->get_c_str() << '\"';
            return;
        }

        case mi::IRef::IID::hash32: {
            mi::base::Handle ref( data->get_interface<mi::IRef>());
            const char* reference_name = ref->get_reference_name();
            if( reference_name)
                s << "points to \"" << reference_name << '\"';
            else
                s << "(unset)";
            return;
        }

        case mi::IEnum::IID::hash32: {
            mi::base::Handle e( data->get_interface<mi::IEnum>());
            s << e->get_value_by_name() << '(' << e->get_value() << ')';
            return;
        }

        case mi::IVoid::IID::hash32: {
            s << "(void)";
            return;
        }

        case mi::IColor::IID::hash32: {
            mi::base::Handle color( data->get_interface<mi::IColor>());
            mi::math::Color c = color->get_value();
            s << '(' << c.r << ", " << c.g << ", " << c.b << ')';
            return;
        }

        case mi::IColor3::IID::hash32: {
            mi::base::Handle color( data->get_interface<mi::IColor3>());
            mi::math::Color c = color->get_value();
            s << '(' << c.r << ", " << c.g << ", " << c.b << ')';
            return;
        }

        case mi::ISpectrum::IID::hash32: {
            mi::base::Handle spectrum( data->get_interface<mi::ISpectrum>());
            mi::math::Spectrum sp = spectrum->get_value();
            s << '(' << sp.get( 0) << ", " << sp.get( 1) << ", " << sp.get( 2) << ')';
            return;
        }

        case mi::IBbox3::IID::hash32: {
            mi::base::Handle bbox( data->get_interface<mi::IBbox3>());
            mi::Bbox3 b = bbox->get_value();
            s << '(' << b.min[0] << ", " << b.min[1] << ", " << b.min[2] << ") - "
              << '(' << b.max[0] << ", " << b.max[1] << ", " << b.max[2] << ')';
            return;
        }

        case mi::IBoolean_2::IID::hash32:
        case mi::IBoolean_3::IID::hash32:
        case mi::IBoolean_4::IID::hash32:
        case mi::IBoolean_2_2::IID::hash32:
        case mi::IBoolean_2_3::IID::hash32:
        case mi::IBoolean_2_4::IID::hash32:
        case mi::IBoolean_3_2::IID::hash32:
        case mi::IBoolean_3_3::IID::hash32:
        case mi::IBoolean_3_4::IID::hash32:
        case mi::IBoolean_4_2::IID::hash32:
        case mi::IBoolean_4_3::IID::hash32:
        case mi::IBoolean_4_4::IID::hash32: {
            mi::base::Handle collection( data->get_interface<mi::IData_collection>());
            s << '(';
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle field(
                    collection->get_value<mi::IBoolean>( i));
                s << (i > 0 ? ", " : "") << field->get_value<bool>();
            }
            s << ')';
            return;
        }

        case mi::ISint32_2::IID::hash32:
        case mi::ISint32_3::IID::hash32:
        case mi::ISint32_4::IID::hash32:
        case mi::ISint32_2_2::IID::hash32:
        case mi::ISint32_2_3::IID::hash32:
        case mi::ISint32_2_4::IID::hash32:
        case mi::ISint32_3_2::IID::hash32:
        case mi::ISint32_3_3::IID::hash32:
        case mi::ISint32_3_4::IID::hash32:
        case mi::ISint32_4_2::IID::hash32:
        case mi::ISint32_4_3::IID::hash32:
        case mi::ISint32_4_4::IID::hash32: {
            mi::base::Handle collection( data->get_interface<mi::IData_collection>());
            s << '(';
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle field(
                    collection->get_value<mi::ISint32>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Sint32>();
            }
            s << ')';
            return;
        }

        case mi::IUint32_2::IID::hash32:
        case mi::IUint32_3::IID::hash32:
        case mi::IUint32_4::IID::hash32:
        case mi::IUint32_2_2::IID::hash32:
        case mi::IUint32_2_3::IID::hash32:
        case mi::IUint32_2_4::IID::hash32:
        case mi::IUint32_3_2::IID::hash32:
        case mi::IUint32_3_3::IID::hash32:
        case mi::IUint32_3_4::IID::hash32:
        case mi::IUint32_4_2::IID::hash32:
        case mi::IUint32_4_3::IID::hash32:
        case mi::IUint32_4_4::IID::hash32: {
            mi::base::Handle collection( data->get_interface<mi::IData_collection>());
            s << '(';
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle field(
                    collection->get_value<mi::IUint32>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Uint32>();
            }
            s << ')';
            return;
        }

        case mi::IFloat32_2::IID::hash32:
        case mi::IFloat32_3::IID::hash32:
        case mi::IFloat32_4::IID::hash32:
        case mi::IFloat32_2_2::IID::hash32:
        case mi::IFloat32_2_3::IID::hash32:
        case mi::IFloat32_2_4::IID::hash32:
        case mi::IFloat32_3_2::IID::hash32:
        case mi::IFloat32_3_3::IID::hash32:
        case mi::IFloat32_3_4::IID::hash32:
        case mi::IFloat32_4_2::IID::hash32:
        case mi::IFloat32_4_3::IID::hash32:
        case mi::IFloat32_4_4::IID::hash32: {
            mi::base::Handle collection( data->get_interface<mi::IData_collection>());
            s << '(';
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle field(
                    collection->get_value<mi::IFloat32>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Float32>();
            }
            s << ')';
            return;
        }

        case mi::IFloat64_2::IID::hash32:
        case mi::IFloat64_3::IID::hash32:
        case mi::IFloat64_4::IID::hash32:
        case mi::IFloat64_2_2::IID::hash32:
        case mi::IFloat64_2_3::IID::hash32:
        case mi::IFloat64_2_4::IID::hash32:
        case mi::IFloat64_3_2::IID::hash32:
        case mi::IFloat64_3_3::IID::hash32:
        case mi::IFloat64_3_4::IID::hash32:
        case mi::IFloat64_4_2::IID::hash32:
        case mi::IFloat64_4_3::IID::hash32:
        case mi::IFloat64_4_4::IID::hash32: {
            mi::base::Handle collection( data->get_interface<mi::IData_collection>());
            s << '(';
            for( mi::Size i = 0; i < collection->get_length(); i++) {
                mi::base::Handle field(
                    collection->get_value<mi::IFloat64>( i));
                s << (i > 0 ? ", " : "") << field->get_value<mi::Float64>();
            }
            s << ')';
            return;
        }

        case mi::IArray::IID::hash32:
        case mi::IDynamic_array::IID::hash32:
        case mi::IMap::IID::hash32:
        case mi::IStructure::IID::hash32: {
            mi::base::Handle collection( data->get_interface<mi::IData_collection>());
            mi::base::Handle array( data->get_interface<mi::IArray>());
            s << '{';
            mi::Size length = collection->get_length();
            if( length > 0)
                s << std::endl;
            else
                s << ' ';
            for( mi::Size i = 0; i < length; i++) {
                s << get_prefix( depth+1);
                if( array)
                    s << '[' << i << "] = ";
                const char* key = collection->get_key( i);
                mi::base::Handle field( collection->get_value<mi::IData>( i));
                if( field)
                    dump( key, field.get(), depth+1, s);
                else
                    s << "(dumping of this type not supported)";
                s << ';' << std::endl;
            }
            if( length > 0)
                s << get_prefix( depth);
            s << '}';
            return;
        }

        default:
            s << "(dumper for this type missing)";
            return;
    }
}

} // namespace IDATA

} // namespace MI
