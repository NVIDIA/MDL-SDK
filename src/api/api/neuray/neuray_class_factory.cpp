/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "neuray_class_factory.h"

#include <mi/base/handle.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/iuser_class.h>

#include "i_neuray_db_element.h"
#include "neuray_enum_decl_impl.h"
#include "neuray_structure_decl_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_utilities.h"

#include "neuray_expression_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"


#include <iomanip>
#include <sstream>
#include <base/data/db/i_db_transaction.h>
#include <base/system/stlext/i_stlext_likely.h>
#include <boost/core/ignore_unused.hpp>
#include <base/lib/log/i_log_logger.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace NEURAY {

Class_factory* s_class_factory;

Class_factory::Class_factory()
{
}

Class_factory::~Class_factory()
{
    ASSERT( M_NEURAY_API, m_map_name_user_class_factory.size() == 0);
    ASSERT( M_NEURAY_API, m_map_uuid_user_class_factory.size() == 0);
    ASSERT( M_NEURAY_API, m_map_name_structure_decl.size() == 0);
    ASSERT( M_NEURAY_API, m_map_name_enum_decl.size() == 0);
}

mi::Sint32 Class_factory::register_class(
    const char* class_name,
    SERIAL::Class_id class_id,
    Api_class_factory api_class_factory,
    Db_element_factory db_element_factory)
{
    ASSERT( M_NEURAY_API, class_name);
    ASSERT( M_NEURAY_API, api_class_factory);
    ASSERT( M_NEURAY_API, db_element_factory);

    mi::base::Lock::Block block2( &m_map_name_enum_decl_lock);
    mi::base::Lock::Block block3( &m_map_name_structure_decl_lock);

    if( m_map_name_id.find( class_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_id_api_class_factory.find( class_id)      != m_map_id_api_class_factory.end())
        return -1;
    if( m_map_name_api_class_factory.find( class_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_db_element_factory.find( class_name) != m_map_name_db_element_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( class_name) != m_map_name_user_class_factory.end())
        return -1;
    if( m_map_name_enum_decl.find( class_name)          != m_map_name_enum_decl.end())
        return -1;
    if( m_map_name_structure_decl.find( class_name)     != m_map_name_structure_decl.end())
        return -1;

    m_map_name_id[class_name]                 = class_id;
    m_map_id_api_class_factory[class_id]      = api_class_factory;
    m_map_name_api_class_factory[class_name]  = api_class_factory;
    m_map_name_db_element_factory[class_name] = db_element_factory;
    return 0;
}

mi::Sint32 Class_factory::register_class(
    const char* class_name,
    Api_class_factory api_class_factory)
{
    ASSERT( M_NEURAY_API, class_name);
    ASSERT( M_NEURAY_API, api_class_factory);

    mi::base::Lock::Block block2( &m_map_name_enum_decl_lock);
    mi::base::Lock::Block block3( &m_map_name_structure_decl_lock);

    if( m_map_name_id.find( class_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( class_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( class_name) != m_map_name_user_class_factory.end())
        return -1;
    if( m_map_name_enum_decl.find( class_name)          != m_map_name_enum_decl.end())
        return -1;
    if( m_map_name_structure_decl.find( class_name)     != m_map_name_structure_decl.end())
        return -1;

    m_map_name_api_class_factory[class_name] = api_class_factory;
    return 0;
}

mi::Sint32 Class_factory::register_class(
    const char* class_name,
    const mi::base::Uuid& uuid,
    mi::neuraylib::IUser_class_factory* factory)
{
    ASSERT( M_NEURAY_API, class_name);
    ASSERT( M_NEURAY_API, factory);

    mi::base::Lock::Block block2( &m_map_name_enum_decl_lock);
    mi::base::Lock::Block block3( &m_map_name_structure_decl_lock);

    if( m_map_name_id.find( class_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( class_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( class_name) != m_map_name_user_class_factory.end())
        return -1;
    if( m_map_uuid_user_class_factory.find( uuid)       != m_map_uuid_user_class_factory.end())
        return -1;
    if( m_map_name_enum_decl.find( class_name)          != m_map_name_enum_decl.end())
        return -1;
    if( m_map_name_structure_decl.find( class_name)     != m_map_name_structure_decl.end())
        return -1;

    if( strcmp( class_name, "__DiCE_class_without_name") != 0) {
        m_map_name_user_class_factory[class_name] = factory;
        factory->retain();
    }

    m_map_uuid_user_class_factory[uuid] = factory;
    factory->retain();

    return 0;
}

bool Class_factory::is_class_registered( SERIAL::Class_id class_id) const
{
    return m_map_id_api_class_factory.find( class_id) != m_map_id_api_class_factory.end();
}

bool Class_factory::is_class_registered( const mi::neuraylib::ISerializable* serializable) const
{
    ASSERT( M_NEURAY_API, serializable);

    mi::base::Uuid uuid = serializable->get_class_id();
    return is_class_registered( uuid);
}

bool Class_factory::is_class_registered( const mi::base::Uuid& uuid) const
{
    return m_map_uuid_user_class_factory.find( uuid) != m_map_uuid_user_class_factory.end();
}

mi::Sint32 Class_factory::register_structure_decl(
    const char* structure_name, const mi::IStructure_decl* decl)
{
    ASSERT( M_NEURAY_API, structure_name);
    ASSERT( M_NEURAY_API, decl);

    mi::base::Lock::Block block2( &m_map_name_enum_decl_lock);
    mi::base::Lock::Block block3( &m_map_name_structure_decl_lock);

    if( m_map_name_id.find( structure_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( structure_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( structure_name) != m_map_name_user_class_factory.end())
        return -1;
    if( m_map_name_enum_decl.find( structure_name)          != m_map_name_enum_decl.end())
        return -1;
    if( m_map_name_structure_decl.find( structure_name)     != m_map_name_structure_decl.end())
        return -1;

    // Maybe we should check in addition that all member type names are valid. But there are two
    // problems:
    // - Some types require a transaction for instantiation, which we do not have here (could
    //   implement a check for valid type names without creating instances of it).
    // - Type names might become invalid since there is no reference counting between structure
    //   and enum declarations.

    std::vector<std::string> blacklist;
    blacklist.push_back( structure_name);
    if( contains_blacklisted_type_names( decl, blacklist))
        return -5;

    // Clone the declaration such that modifications after registration have no effect.
    mi::IStructure_decl* copy = create_class_instance<mi::IStructure_decl>( nullptr, "Structure_decl");
    mi::Size n = decl->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        mi::Sint32 result
            = copy->add_member( decl->get_member_type_name( i), decl->get_member_name( i));
        ASSERT( M_NEURAY_API, result == 0);
        boost::ignore_unused( result);
    }

    // Set the type name
    Structure_decl_impl* copy_impl = static_cast<Structure_decl_impl*>( copy);
    copy_impl->set_structure_type_name( structure_name);

    m_map_name_structure_decl[structure_name] = copy;

    // The cppcheck error about a leak of copy_impl here is wrong (owned by the map above).
    return 0;
}

mi::Sint32 Class_factory::unregister_structure_decl( const char* structure_name)
{
    ASSERT( M_NEURAY_API, structure_name);

    mi::base::Lock::Block block( &m_map_name_structure_decl_lock);

    std::map<std::string, const mi::IStructure_decl*>::iterator it
        = m_map_name_structure_decl.find( structure_name);
    if( it == m_map_name_structure_decl.end())
        return -1;

    it->second->release();
    m_map_name_structure_decl.erase( it);

    return 0;
}

const mi::IStructure_decl* Class_factory::get_structure_decl( const char* structure_name) const
{
    mi::base::Lock::Block block( &m_map_name_structure_decl_lock);

    std::map<std::string, const mi::IStructure_decl*>::const_iterator it
        = m_map_name_structure_decl.find( structure_name);
    if( it == m_map_name_structure_decl.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

mi::Sint32 Class_factory::register_enum_decl(
    const char* enum_name, const mi::IEnum_decl* decl)
{
    ASSERT( M_NEURAY_API, enum_name);
    ASSERT( M_NEURAY_API, decl);

    mi::base::Lock::Block block2( &m_map_name_enum_decl_lock);
    mi::base::Lock::Block block3( &m_map_name_structure_decl_lock);

    if( m_map_name_id.find( enum_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( enum_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( enum_name) != m_map_name_user_class_factory.end())
        return -1;
    if( m_map_name_enum_decl.find( enum_name)          != m_map_name_enum_decl.end())
        return -1;
    if( m_map_name_structure_decl.find( enum_name)     != m_map_name_structure_decl.end())
        return -1;

    mi::Size n = decl->get_length();
    if( n == 0)
        return -6;

    // Clone the declaration such that modifications after registration have no effect.
    mi::IEnum_decl* copy = create_class_instance<mi::IEnum_decl>( nullptr, "Enum_decl");
    for( mi::Size i = 0; i < n; ++i) {
        mi::Sint32 result = copy->add_enumerator( decl->get_name( i), decl->get_value( i));
        ASSERT( M_NEURAY_API, result == 0);
        boost::ignore_unused( result);
    }

    // Set the type name
    Enum_decl_impl* copy_impl = static_cast<Enum_decl_impl*>( copy);
    copy_impl->set_enum_type_name( enum_name);

    m_map_name_enum_decl[enum_name] = copy;

    // The cppcheck error about a leak of copy_impl here is wrong (owned by the map above).
    return 0;
}

mi::Sint32 Class_factory::unregister_enum_decl( const char* enum_name)
{
    ASSERT( M_NEURAY_API, enum_name);

    mi::base::Lock::Block block( &m_map_name_enum_decl_lock);

    std::map<std::string, const mi::IEnum_decl*>::iterator it
        = m_map_name_enum_decl.find( enum_name);
    if( it == m_map_name_enum_decl.end())
        return -1;

    it->second->release();
    m_map_name_enum_decl.erase( it);

    return 0;
}

const mi::IEnum_decl* Class_factory::get_enum_decl( const char* enum_name) const
{
    mi::base::Lock::Block block( &m_map_name_enum_decl_lock);

    std::map<std::string, const mi::IEnum_decl*>::const_iterator it
        = m_map_name_enum_decl.find( enum_name);
    if( it == m_map_name_enum_decl.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

void Class_factory::unregister_user_defined_classes()
{
    for( std::map<std::string, mi::neuraylib::IUser_class_factory*>::iterator it
        = m_map_name_user_class_factory.begin(); it != m_map_name_user_class_factory.end(); ++it) {
        it->second->release();
        it->second = 0;
    }
    m_map_name_user_class_factory.clear();
    for( std::map<mi::base::Uuid, mi::neuraylib::IUser_class_factory*>::iterator it
        = m_map_uuid_user_class_factory.begin(); it != m_map_uuid_user_class_factory.end(); ++it) {
        it->second->release();
        it->second = 0;
    }
    m_map_uuid_user_class_factory.clear();
}

void Class_factory::unregister_structure_decls()
{
    mi::base::Lock::Block block( &m_map_name_structure_decl_lock);

    for( std::map<std::string, const mi::IStructure_decl*>::iterator it
        = m_map_name_structure_decl.begin(); it != m_map_name_structure_decl.end(); ++it) {
        it->second->release();
        it->second = 0;
    }
    m_map_name_structure_decl.clear();
}

void Class_factory::unregister_enum_decls()
{
    mi::base::Lock::Block block( &m_map_name_enum_decl_lock);

    for( std::map<std::string, const mi::IEnum_decl*>::iterator it
        = m_map_name_enum_decl.begin(); it != m_map_name_enum_decl.end(); ++it) {
        it->second->release();
        it->second = 0;
    }
    m_map_name_enum_decl.clear();
}

SERIAL::Class_id Class_factory::get_class_id( const char* class_name) const
{
    std::map<std::string, SERIAL::Class_id>::const_iterator it
        = m_map_name_id.find( class_name);
    if( it == m_map_name_id.end())
        return 0;

    return it->second;
}

SERIAL::Class_id Class_factory::get_class_id(
    const Transaction_impl* transaction, DB::Tag tag) const
{
    ASSERT( M_NEURAY_API, transaction);
    if( !tag)
        return 0;

    SERIAL::Class_id class_id = transaction->get_db_transaction()->get_class_id( tag);
    return class_id;
}

mi::base::IInterface* Class_factory::create_class_instance(
    Transaction_impl* transaction,
    DB::Tag tag,
    bool is_edit) const
{
    ASSERT( M_NEURAY_API, transaction);
    if( !tag)
        return nullptr;

    SERIAL::Class_id class_id = get_class_id( transaction, tag);

    // create API class instance
    mi::base::Handle<mi::base::IInterface> interface(
        invoke_api_class_factory( transaction, class_id));
    if( !interface.is_valid_interface())
        return nullptr;

    // connect DB element and API class
    IDb_element* idb_element = interface->get_interface<IDb_element>();
    ASSERT( M_NEURAY_API, idb_element);
    if( is_edit)
        idb_element->set_state_edit( transaction, tag);
    else
        idb_element->set_state_access( transaction, tag);

    return idb_element;
}

mi::base::IInterface* Class_factory::create_type_instance(
    Transaction_impl* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    if( !type_name)
        return nullptr;
    std::string type_name_string( type_name);
    mi::Size length = type_name_string.size();

    // handle arrays
    if( type_name[length-1] == ']')
        return create_array_instance( transaction, type_name, argc, argv);

    // handle maps
    if( type_name_string.substr( 0, 4) == "Map<")
        return create_map_instance( transaction, type_name, argc, argv);

    // handle pointers
    if(( type_name_string.substr( 0,  8) == "Pointer<")
         || ( type_name_string.substr( 0, 14) == "Const_pointer<"))
        return create_pointer_instance( transaction, type_name, argc, argv);

    // handle structures
    mi::base::Handle<const mi::IStructure_decl> structure_decl( get_structure_decl( type_name));
    if( structure_decl.is_valid_interface())
        return create_structure_instance( transaction, type_name, argc, argv, structure_decl.get());

    // handle enums
    mi::base::Handle<const mi::IEnum_decl> enum_decl( get_enum_decl( type_name));
    if( enum_decl.is_valid_interface())
        return create_enum_instance( transaction, type_name, argc, argv, enum_decl.get());

    // handle simple types
    return create_class_instance( transaction, type_name, argc, argv);
}

mi::base::IInterface* Class_factory::create_class_instance( const mi::base::Uuid& uuid) const
{
    return invoke_user_class_factory( uuid);
}

std::string Class_factory::uuid_to_string( const mi::base::Uuid& uuid)
{
    std::ostringstream s;
    s.fill( '0');
    s << std::setiosflags( std::ios::right) << std::hex << "("
        << "0x" << std::setw( 8) <<    uuid.m_id1                << ","
        << "0x" << std::setw( 4) <<   (uuid.m_id2      & 0xffff) << ","
        << "0x" << std::setw( 4) <<   (uuid.m_id2 >> 16)         << ","
        << "0x" << std::setw( 2) <<  ((uuid.m_id3      ) & 0xff) << ","
        << "0x" << std::setw( 2) <<  ((uuid.m_id3 >>  8) & 0xff) << ","
        << "0x" << std::setw( 2) <<  ((uuid.m_id3 >> 16) & 0xff) << ","
        << "0x" << std::setw( 2) <<   (uuid.m_id3 >> 24)         << ","
        << "0x" << std::setw( 2) <<  ((uuid.m_id4      ) & 0xff) << ","
        << "0x" << std::setw( 2) <<  ((uuid.m_id4 >>  8) & 0xff) << ","
        << "0x" << std::setw( 2) <<  ((uuid.m_id4 >> 16) & 0xff) << ","
        << "0x" << std::setw( 2) <<   (uuid.m_id4 >> 24)         << ")";
    return s.str();
}

Type_factory* Class_factory::create_type_factory(
    mi::neuraylib::ITransaction* transaction) const
{
    return new Type_factory( transaction);
}

Value_factory* Class_factory::create_value_factory(
    mi::neuraylib::ITransaction* transaction) const
{
    mi::base::Handle<Type_factory> tf( create_type_factory( transaction));
    return new Value_factory( transaction, tf.get());
}

Expression_factory* Class_factory::create_expression_factory(
    mi::neuraylib::ITransaction* transaction) const
{
    mi::base::Handle<Value_factory> vf( create_value_factory( transaction));
    return new Expression_factory( transaction, vf.get());
}

mi::base::IInterface* Class_factory::create_class_instance(
    Transaction_impl* transaction,
    const char* class_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    // create API/user class instance
    mi::base::Handle<mi::base::IInterface> interface(
        invoke_api_or_user_class_factory( transaction, class_name, argc, argv));
    if( !interface.is_valid_interface())
        return nullptr;

    // if it is a user class instance, we are done
    mi::base::Handle<mi::neuraylib::IUser_class> user_class(
        interface.get_interface<mi::neuraylib::IUser_class>());
    if( user_class.is_valid_interface()) {
        user_class->retain();
        return user_class.get();
    }

    std::map<std::string, Db_element_factory>::const_iterator it
        = m_map_name_db_element_factory.find( class_name);

    if( it == m_map_name_db_element_factory.end()) {

        // there is no DB element factory registered for this class name
        interface->retain();
        return interface.get();

    } else {

        // create DB element instance
        DB::Element_base* db_element
            = invoke_db_element_factory( transaction, class_name, argc, argv);
        if( !db_element)
            return nullptr;

        // connect DB element and API class
        IDb_element* idb_element = interface->get_interface<IDb_element>();
        ASSERT( M_NEURAY_API, idb_element);
        idb_element->set_state_pointer( transaction, db_element);

        return idb_element;
    }
}

mi::base::IInterface* Class_factory::create_array_instance(
    Transaction_impl* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    ASSERT( M_NEURAY_API, type_name);
    std::string type_name_str( type_name);
    ASSERT( M_NEURAY_API, type_name_str[type_name_str.size()-1] == ']');

    if( argc != 0)
        return nullptr;

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
        create_class_instance<mi::IString>( transaction, "String"));
    element_type_name_string->set_c_str( element_type_name.c_str());
    new_argv[0] = element_type_name_string.get();

    // extract length
    std::string length_str = type_name_str.substr( left_bracket+1, right_bracket-left_bracket-1);
    if( length_str.empty()) {
        // create dynamic array instance
        return create_class_instance( transaction, "__Dynamic_array", 1, new_argv);
    }

    // extract length (continued)
    STLEXT::Likely<mi::Size> length_likely
        = STRING::lexicographic_cast_s<mi::Size>( length_str);
    if( !length_likely.get_status())
        return nullptr;
    mi::Size length = *length_likely.get_ptr(); //-V522 PVS
    mi::base::Handle<mi::ISize> length_value(
        create_class_instance<mi::ISize>( transaction, "Size"));
    length_value->set_value( length);

    // create array instance
    new_argv[1] = length_value.get();
    return create_class_instance( transaction, "__Array", 2, new_argv);
}

mi::base::IInterface* Class_factory::create_map_instance(
    Transaction_impl* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    ASSERT( M_NEURAY_API, type_name);
    std::string type_name_str( type_name);
    ASSERT( M_NEURAY_API, type_name_str.substr( 0, 4) == "Map<");

    if( argc != 0)
        return nullptr;

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
        create_class_instance<mi::IString>( transaction, "String"));
    value_type_name_string->set_c_str( value_type_name.c_str());
    new_argv[0] = value_type_name_string.get();

    // create map instance
    return create_class_instance( transaction, "__Map", 1, new_argv);
}

mi::base::IInterface* Class_factory::create_pointer_instance(
    Transaction_impl* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    ASSERT( M_NEURAY_API, type_name);
    std::string type_name_str( type_name);
    ASSERT( M_NEURAY_API, ( type_name_str.substr( 0,  8) == "Pointer<")
                       || ( type_name_str.substr( 0, 14) == "Const_pointer<"));
    bool is_const = type_name_str.substr( 0, 14) == "Const_pointer<";

    if( argc != 0)
        return nullptr;

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
        create_class_instance<mi::IString>( transaction, "String"));
    value_type_name_string->set_c_str( value_type_name.c_str());
    new_argv[0] = value_type_name_string.get();

    // create pointer instance
    return create_class_instance(
        transaction, is_const ? "__Const_pointer" : "__Pointer", 1, new_argv);
}

mi::base::IInterface* Class_factory::create_structure_instance(
    Transaction_impl* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[],
    const mi::IStructure_decl* decl) const
{
    ASSERT( M_NEURAY_API, type_name);

    if( argc != 0)
        return nullptr;

    if( strcmp( type_name, "Preset_data") == 0)
        LOG::mod_log->warning( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "The type name \"Preset_data\" is deprecated. Please use \"Variant_data\" instead (and"
            " use the member \"variant_name\" instead of \"preset_name\").");

    const mi::base::IInterface* new_argv[2];
    new_argv[0] = decl;

    mi::base::Handle<mi::IString> type_name_string(
        create_class_instance<mi::IString>( transaction, "String"));
    type_name_string->set_c_str( type_name);
    new_argv[1] = type_name_string.get();

    // create structure instance
    mi::base::IInterface* result = create_class_instance( transaction, "__Structure", 2, new_argv);
    return result;
}

mi::base::IInterface* Class_factory::create_enum_instance(
    Transaction_impl* transaction,
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[],
    const mi::IEnum_decl* decl) const
{
    ASSERT( M_NEURAY_API, type_name);

    if( argc != 0)
        return nullptr;

    const mi::base::IInterface* new_argv[2];
    new_argv[0] = decl;

    mi::base::Handle<mi::IString> type_name_string(
        create_class_instance<mi::IString>( transaction, "String"));
    type_name_string->set_c_str( type_name);
    new_argv[1] = type_name_string.get();

    // create enum instance
    mi::base::IInterface* result = create_class_instance( transaction, "__Enum", 2, new_argv);
    return result;
}

mi::base::IInterface* Class_factory::extract_user_class(
    IDb_element* idb_element,
    bool is_edit) const
{
    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

mi::base::IInterface* Class_factory::extract_element(
    IDb_element* idb_element,
    bool is_edit) const
{
    ASSERT( M_NEURAY_API, false);
    return 0;
}

mi::base::IInterface* Class_factory::invoke_api_class_factory(
    Transaction_impl* transaction,
    SERIAL::Class_id class_id) const
{
    // lookup API class factory by class ID
    std::map<SERIAL::Class_id, Api_class_factory>::const_iterator it
        = m_map_id_api_class_factory.find( class_id);
    if( it == m_map_id_api_class_factory.end())
        return nullptr;

    // create API class instance
    Api_class_factory api_class_factory = it->second;
    return api_class_factory( transaction, 0, nullptr);
}

mi::base::IInterface* Class_factory::invoke_api_or_user_class_factory(
    Transaction_impl* transaction,
    const char* class_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    // lookup API class factory by class name
    std::map<std::string, Api_class_factory>::const_iterator it_api
        = m_map_name_api_class_factory.find( class_name);
    if( it_api != m_map_name_api_class_factory.end()) {

        // create API class instance
        Api_class_factory api_class_factory = it_api->second;
        return api_class_factory( transaction, argc, argv);
    }

    // lookup user class factory by class name
    std::map<std::string, mi::neuraylib::IUser_class_factory*>::const_iterator it_user
        = m_map_name_user_class_factory.find( class_name);
    if( it_user == m_map_name_user_class_factory.end())
        return nullptr;

    // create user class instance
    mi::neuraylib::IUser_class_factory* user_class_factory = it_user->second;
    return user_class_factory->create( transaction, argc, argv);
}

DB::Element_base* Class_factory::invoke_db_element_factory(
    Transaction_impl* transaction,
    const char* class_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    // lookup DB element factory
    std::map<std::string, Db_element_factory>::const_iterator it
        = m_map_name_db_element_factory.find( class_name);
    if( it == m_map_name_db_element_factory.end())
        return nullptr;

    // create DB element instance
    Db_element_factory db_element_factory = it->second;
    return db_element_factory( transaction, argc, argv);
}

mi::base::IInterface* Class_factory::invoke_user_class_factory( const mi::base::Uuid& uuid) const
{
    // lookup user class factory by class UUID
    std::map<mi::base::Uuid, mi::neuraylib::IUser_class_factory*>::const_iterator it
        = m_map_uuid_user_class_factory.find( uuid);
    if( it == m_map_uuid_user_class_factory.end())
        return nullptr;

    // create user class instance
    mi::neuraylib::IUser_class_factory* user_class_factory = it->second;
    return user_class_factory->create( nullptr, 0, nullptr);
}

bool Class_factory::contains_blacklisted_type_names(
    const std::string& type_name, std::vector<std::string>& blacklist)
{
    // check if type_name is blacklisted
    std::vector<std::string>::iterator it
        = find( blacklist.begin(), blacklist.end(), type_name);
    if( it != blacklist.end())
        return true;

    // descend into structures
    std::map<std::string, const mi::IStructure_decl*>::const_iterator it_structure_decl
        = m_map_name_structure_decl.find( type_name);
    if( it_structure_decl != m_map_name_structure_decl.end()) {
        blacklist.push_back( type_name);
        bool result = contains_blacklisted_type_names( it_structure_decl->second, blacklist);
        blacklist.pop_back();
        return result;
    }

    // descend into arrays
    mi::Size length;
    const std::string& array_element = Type_utilities::strip_array( type_name, length);
    if( !array_element.empty())
        return contains_blacklisted_type_names( array_element, blacklist);

    // descend into maps
    const std::string& map_value = Type_utilities::strip_map( type_name);
    if( !map_value.empty())
        return contains_blacklisted_type_names( map_value, blacklist);

    // descend into pointers
    const std::string& pointer_nested = Type_utilities::strip_pointer( type_name);
    if( !pointer_nested.empty())
        return contains_blacklisted_type_names( pointer_nested, blacklist);

    // descend into const pointers
    const std::string& const_pointer_nested = Type_utilities::strip_const_pointer( type_name);
    if( !const_pointer_nested.empty())
        return contains_blacklisted_type_names( const_pointer_nested, blacklist);

    return false;

}

bool Class_factory::contains_blacklisted_type_names(
    const mi::IStructure_decl* decl, std::vector<std::string>& blacklist)
{
    mi::Size n = decl->get_length();
    for( mi::Size i = 0; i < n; ++i) {
        std::string member_type_name = decl->get_member_type_name( i);
        if( contains_blacklisted_type_names( member_type_name, blacklist))
            return true;
    }
    return false;
}

} // namespace NEURAY

} // namespace MI

