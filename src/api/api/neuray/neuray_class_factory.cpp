/***************************************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure_decl.h>
#include <mi/neuraylib/iuser_class.h>

#include "i_neuray_db_element.h"
#include "neuray_transaction_impl.h"

#include "neuray_expression_impl.h"
#include "neuray_type_impl.h"
#include "neuray_value_impl.h"

#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>


#include <iomanip>
#include <sstream>

#include <boost/core/ignore_unused.hpp>

#include <base/data/db/i_db_transaction.h>
#include <base/data/idata/i_idata_factory.h>
#include <base/data/idata/idata_interfaces.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <api/api/neuray/neuray_transaction_impl.h>

namespace MI {

namespace NEURAY {

class Tag_handler : public mi::base::Interface_implement<IDATA::ITag_handler>
{
public:
    Tag_handler( Class_factory* class_factory) : m_class_factory( class_factory) { }

    const char* tag_to_name( DB::Transaction* transaction, DB::Tag tag) final
    {
        return transaction->tag_to_name( tag);
    }

    DB::Tag name_to_tag( DB::Transaction* transaction, const char* name) final
    {
        return transaction->name_to_tag( name);
    }

    const mi::base::IInterface* access_tag( DB::Transaction* transaction, DB::Tag tag) final
    {
        mi::base::Handle api_transaction( new Transaction_impl(
            transaction, m_class_factory, /*commit_or_abort_warning*/ false));
        return api_transaction->access( tag);
    }

    mi::base::IInterface* edit_tag( DB::Transaction* transaction, DB::Tag tag) final
    {
        mi::base::Handle api_transaction( new Transaction_impl(
            transaction, m_class_factory, /*commit_or_abort_warning*/ false));
        return api_transaction->edit( tag);
    }

    std::pair<DB::Tag,mi::Sint32> get_tag( const mi::base::IInterface* interface) final
    {
        mi::base::Handle<const IDb_element> db_element( interface->get_interface<IDb_element>());
        if( !db_element)
            return {{}, -2};

        DB::Tag tag = db_element->get_tag();
        if( !tag)
            return {{}, -3};

        return {tag, 0};
    }

private:
    Class_factory* m_class_factory;
};

Class_factory* s_class_factory;

Class_factory::Class_factory()
  : m_tag_handler( new Tag_handler( this)),
    m_idata_factory( new IDATA::Factory( m_tag_handler.get()))
{
}

Class_factory::~Class_factory()
{
    ASSERT( M_NEURAY_API, m_map_name_user_class_factory.empty());
    ASSERT( M_NEURAY_API, m_map_uuid_user_class_factory.empty());
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
    if( make_handle( m_idata_factory->get_enum_decl( class_name)))
        return -1;
    if( make_handle( m_idata_factory->get_structure_decl( class_name)))
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

    if( m_map_name_id.find( class_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( class_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( class_name) != m_map_name_user_class_factory.end())
        return -1;
    if( make_handle( m_idata_factory->get_enum_decl( class_name)))
        return -1;
    if( make_handle( m_idata_factory->get_structure_decl( class_name)))
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

    if( m_map_name_id.find( class_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( class_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( class_name) != m_map_name_user_class_factory.end())
        return -1;
    if( m_map_uuid_user_class_factory.find( uuid)       != m_map_uuid_user_class_factory.end())
        return -1;
    if( make_handle( m_idata_factory->get_enum_decl( class_name)))
        return -1;
    if( make_handle( m_idata_factory->get_structure_decl( class_name)))
        return -1;

    if( strcmp( class_name, "__DiCE_class_without_name") != 0)
        m_map_name_user_class_factory[class_name] = make_handle_dup( factory);

    m_map_uuid_user_class_factory[uuid] = make_handle_dup( factory);

    return 0;
}

mi::Sint32 Class_factory::unregister_class(
    const char* class_name,
    const mi::base::Uuid& uuid)
{
    ASSERT( M_NEURAY_API, class_name);

    auto it_name_user_class_factory = m_map_name_user_class_factory.find( class_name);
    auto it_uuid_user_class_factory = m_map_uuid_user_class_factory.find( uuid);

    if( it_uuid_user_class_factory == m_map_uuid_user_class_factory.end())
        return -1;

    if( strcmp( class_name, "__DiCE_class_without_name") != 0) {
        if( it_name_user_class_factory == m_map_name_user_class_factory.end())
            return -1;

        m_map_name_user_class_factory.erase( it_name_user_class_factory);
    }

    m_map_uuid_user_class_factory.erase( it_uuid_user_class_factory);

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

    if( m_map_name_id.find( structure_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( structure_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( structure_name) != m_map_name_user_class_factory.end())
        return -1;

    return m_idata_factory->register_structure_decl( structure_name, decl);
}

mi::Sint32 Class_factory::unregister_structure_decl( const char* structure_name)
{
    ASSERT( M_NEURAY_API, structure_name);

    return m_idata_factory->unregister_structure_decl( structure_name);
}

const mi::IStructure_decl* Class_factory::get_structure_decl( const char* structure_name) const
{
    ASSERT( M_NEURAY_API, structure_name);

    return m_idata_factory->get_structure_decl( structure_name);
}

mi::Sint32 Class_factory::register_enum_decl(
    const char* enum_name, const mi::IEnum_decl* decl)
{
    ASSERT( M_NEURAY_API, enum_name);
    ASSERT( M_NEURAY_API, decl);

    if( m_map_name_id.find( enum_name)                 != m_map_name_id.end())
        return -1;
    if( m_map_name_api_class_factory.find( enum_name)  != m_map_name_api_class_factory.end())
        return -1;
    if( m_map_name_user_class_factory.find( enum_name) != m_map_name_user_class_factory.end())
        return -1;

    return m_idata_factory->register_enum_decl( enum_name, decl);
}

mi::Sint32 Class_factory::unregister_enum_decl( const char* enum_name)
{
    ASSERT( M_NEURAY_API, enum_name);

    return m_idata_factory->unregister_enum_decl( enum_name);
}

const mi::IEnum_decl* Class_factory::get_enum_decl( const char* enum_name) const
{
    ASSERT( M_NEURAY_API, enum_name);

    return m_idata_factory->get_enum_decl( enum_name);
}

void Class_factory::unregister_user_defined_classes()
{
    m_map_name_user_class_factory.clear();
    m_map_uuid_user_class_factory.clear();
}

void Class_factory::unregister_structure_decls()
{
    m_idata_factory->unregister_structure_decls();
}

void Class_factory::unregister_enum_decls()
{
    m_idata_factory->unregister_enum_decls();
}

SERIAL::Class_id Class_factory::get_class_id( const char* class_name) const
{
    auto it = m_map_name_id.find( class_name);
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
    if( !interface)
        return nullptr;

    // connect DB element and API class
    auto* idb_element = interface->get_interface<IDb_element>();
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
    DB::Transaction* db_transaction = transaction ? transaction->get_db_transaction() : nullptr;
    mi::base::IInterface* result = m_idata_factory->create(
        db_transaction, type_name, argc, argv);
    if( result)
        return result;

    return create_class_instance( transaction, type_name, argc, argv);
}

mi::base::IInterface* Class_factory::create_class_instance( const mi::base::Uuid& uuid) const
{
    return invoke_user_class_factory( uuid);
}

std::string Class_factory::uuid_to_string( const mi::base::Uuid& uuid)
{
    return m_idata_factory->uuid_to_string( uuid);
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
    if( !interface)
        return nullptr;

    // if it is a user class instance, we are done
    mi::base::Handle<mi::neuraylib::IUser_class> user_class(
        interface.get_interface<mi::neuraylib::IUser_class>());
    if( user_class)
        return user_class.extract();

    auto it = m_map_name_db_element_factory.find( class_name);

    if( it == m_map_name_db_element_factory.end()) {

        // there is no DB element factory registered for this class name
        return interface.extract();

    } else {

        // create DB element instance
        DB::Element_base* db_element
            = invoke_db_element_factory( transaction, class_name, argc, argv);
        if( !db_element)
            return nullptr;

        // connect DB element and API class
        auto* idb_element = interface->get_interface<IDb_element>();
        ASSERT( M_NEURAY_API, idb_element);
        idb_element->set_state_pointer( transaction, db_element);

        return idb_element;
    }
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
    return nullptr;
}

mi::base::IInterface* Class_factory::invoke_api_class_factory(
    Transaction_impl* transaction,
    SERIAL::Class_id class_id) const
{
    // lookup API class factory by class ID
    auto it = m_map_id_api_class_factory.find( class_id);
    if( it == m_map_id_api_class_factory.end())
        return nullptr;

    // create API class instance: general case
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
    auto it_api = m_map_name_api_class_factory.find( class_name);
    if( it_api != m_map_name_api_class_factory.end()) {

        // create API class instance
        Api_class_factory api_class_factory = it_api->second;
        return api_class_factory( transaction, argc, argv);
    }

    // lookup user class factory by class name
    auto it_user = m_map_name_user_class_factory.find( class_name);
    if( it_user == m_map_name_user_class_factory.end())
        return nullptr;

    // create user class instance
    mi::neuraylib::IUser_class_factory* user_class_factory = it_user->second.get();
    return user_class_factory->create( transaction, argc, argv);
}

DB::Element_base* Class_factory::invoke_db_element_factory(
    Transaction_impl* transaction,
    const char* class_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[]) const
{
    // lookup DB element factory
    auto it = m_map_name_db_element_factory.find( class_name);
    if( it == m_map_name_db_element_factory.end())
        return nullptr;

    // create DB element instance
    Db_element_factory db_element_factory = it->second;
    return db_element_factory( transaction, argc, argv);
}

mi::base::IInterface* Class_factory::invoke_user_class_factory( const mi::base::Uuid& uuid) const
{
    // lookup user class factory by class UUID
    auto it = m_map_uuid_user_class_factory.find( uuid);
    if( it == m_map_uuid_user_class_factory.end())
        return nullptr;

    // create user class instance
    mi::neuraylib::IUser_class_factory* user_class_factory = it->second.get();
    return user_class_factory->create( nullptr, 0, nullptr);
}

} // namespace NEURAY

} // namespace MI

