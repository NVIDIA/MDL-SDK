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
 ** \brief Header for the Attribute_set_impl_helper implementation.
 **/

#include "pch.h"

#include "neuray_attribute_set_impl_helper.h"

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/idata.h>
#include <mi/neuraylib/ienum_decl.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/istructure_decl.h>

#include <sstream>
#include <boost/core/ignore_unused.hpp>
#include <base/util/string_utils/i_string_utils.h>
#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_logger.h>

#include <base/data/attr/attr.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/attr/i_attr_utilities.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/scene/i_scene_attr_resv_id.h>
#include <io/scene/mdl_elements/i_mdl_elements_type.h>

#include "i_neuray_attribute_context.h"
#include "i_neuray_db_element.h"
#include "i_neuray_proxy.h"
#include "neuray_class_factory.h"
#include "neuray_transaction_impl.h"
#include "neuray_type_utilities.h"

// Enable to trace calls of Attribute_set_impl_helper::register_decls()
// #define MI_API_API_NEURAY_TRACE_TYPE_REGISTRATION

namespace MI {

namespace NEURAY {

typedef std::vector<std::pair<int, std::string> > Enum_collection;

class Attribute_context : public mi::base::Interface_implement<IAttribute_context>
{
public:
    Attribute_context(
        const IDb_element* db_element,
        const boost::shared_ptr<ATTR::Attribute>& attribute)
    {
        ASSERT( M_NEURAY_API, db_element);
        ASSERT( M_NEURAY_API, attribute);

        m_db_element = make_handle_dup( db_element);
        m_attribute = attribute;
    }

    const IDb_element* get_db_element() const
    {
        m_db_element->retain();
        return m_db_element.get();
    }

    const ATTR::Type* get_type( const char* attribute_name) const
    {
        ASSERT( M_NEURAY_API, attribute_name);

        const ATTR::Type& attribute_type = m_attribute->get_type();
        char* base_address = m_attribute->set_values();
        char* address;
        return attribute_type.lookup(
            attribute_name, base_address, &address);
    }

    void* get_address( const char* attribute_name) const
    {
        ASSERT( M_NEURAY_API, attribute_name);

        const ATTR::Type& attribute_type = m_attribute->get_type();
        char* base_address = m_attribute->set_values();
        char* address;
        const ATTR::Type* member_type = attribute_type.lookup(
            attribute_name, base_address, &address);
        if( !member_type)
            return nullptr;

        return address;
    }

private:
    mi::base::Handle<const IDb_element> m_db_element;
    boost::shared_ptr<ATTR::Attribute> m_attribute;
};

mi::base::Lock Attribute_set_impl_helper::s_register_decls_lock;

mi::IData* Attribute_set_impl_helper::create_attribute(
    ATTR::Attribute_set* attribute_set,
    IDb_element* db_element,
    const char* name,
    const char* type_name,
    bool skip_type_check)
{
    if( !name || !type_name)
        return nullptr;
    ASSERT( M_NEURAY_API, attribute_set);

    if(    strcmp( type_name, "Ref<Texture>") == 0
        || strcmp( type_name, "Ref<Lightprofile>") == 0
        || strcmp( type_name, "Ref<Bsdf_measurement>") == 0) {
        LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "Using attributes of type \"%s\" is no longer supported. Use type \"Ref\" instead.",
            type_name);
        return nullptr;
    }

    std::string name_str( name);
    std::string type_name_str( type_name);

    // Reject invalid attribute names.
    if( name_str.find_first_of( ".[]") != std::string::npos)
         return nullptr;

    // Check that the attribute does not exist yet.
    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( name);
    ATTR::Attribute* attribute_exists = attribute_set->lookup( attribute_id);
    if( attribute_exists)
        return nullptr;

    // Enforce types for reserved attributes.
    if( !skip_type_check && !is_correct_type_for_attribute( name_str, attribute_id, type_name_str))
        return nullptr;

    // Compute the ATTR::Type for the attribute.
    const ATTR::Type& attribute_type = get_attribute_type( type_name, name);
    if( attribute_type.get_typecode() == ATTR::TYPE_UNDEF)
        return nullptr;

    // Create the attribute.
    ATTR::Attribute_id id = ATTR::Attribute::id_create( name);
    boost::shared_ptr<ATTR::Attribute> attribute = boost::shared_ptr<ATTR::Attribute>(
        new ATTR::Attribute( id, attribute_type, ATTR::PROPAGATION_STANDARD));
    attribute->set_global( true);
    ASSERT( M_NEURAY_API, attribute);
    if( !attribute)
        return nullptr;

    // Attach the attribute.
    bool result = attribute_set->attach( attribute);
    ASSERT( M_NEURAY_API, result);
    boost::ignore_unused( result);

    return edit_attribute( attribute_set, db_element, name);
}

bool Attribute_set_impl_helper::destroy_attribute(
    ATTR::Attribute_set* attribute_set,
    IDb_element* db_element,
    const char* name)
{
    if( !name)
        return false;
    ASSERT( M_NEURAY_API, attribute_set);

    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( name);
    ATTR::Attribute* attribute = attribute_set->lookup( attribute_id);
    if( !attribute)
        return false;

    db_element->add_journal_flag( compute_journal_flags( attribute, attribute_id));
    attribute_set->detach( attribute_id); // invalidates attribute
    return true;
}

const mi::IData* Attribute_set_impl_helper::access_attribute(
    const ATTR::Attribute_set* attribute_set,
    const IDb_element* db_element,
    const char* name)
{
    if( !name)
        return nullptr;
    ASSERT( M_NEURAY_API, attribute_set);

    std::string top_level_name = get_top_level_name( name);
    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( top_level_name.c_str());
    boost::shared_ptr<ATTR::Attribute> attribute = attribute_set->lookup_shared_ptr( attribute_id);
    if( !attribute)
        return nullptr;

    mi::base::Handle<IAttribute_context> owner( new Attribute_context( db_element, attribute));
    return get_attribute( owner.get(), name);
}

mi::IData* Attribute_set_impl_helper::edit_attribute(
    ATTR::Attribute_set* attribute_set,
    IDb_element* db_element,
    const char* name)
{
    if( !name)
        return nullptr;
    ASSERT( M_NEURAY_API, attribute_set);

    std::string top_level_name = get_top_level_name( name);
    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( top_level_name.c_str());
    boost::shared_ptr<ATTR::Attribute> attribute = attribute_set->lookup_shared_ptr( attribute_id);
    if( !attribute)
        return nullptr;

    db_element->add_journal_flag( compute_journal_flags( attribute.get(), attribute_id));

    mi::base::Handle<IAttribute_context> owner( new Attribute_context( db_element, attribute));
    return get_attribute( owner.get(), name);
}

bool Attribute_set_impl_helper::is_attribute(
    const ATTR::Attribute_set* attribute_set,
    const IDb_element* db_element,
    const char* name)
{
    if( !name)
        return false;
    ASSERT( M_NEURAY_API, attribute_set);

    std::string top_level_name = get_top_level_name( name);
    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( top_level_name.c_str());
    boost::shared_ptr<ATTR::Attribute> attribute = attribute_set->lookup_shared_ptr( attribute_id);
    if( !attribute || !has_valid_api_type( attribute.get()))
        return false;

    Attribute_context owner( db_element, attribute);
    const ATTR::Type* attribute_type = owner.get_type( name);
    return attribute_type != nullptr;
}

std::string Attribute_set_impl_helper::get_attribute_type_name(
    const ATTR::Attribute_set* attribute_set,
    const IDb_element* db_element,
    const char* name)
{
    if( !name)
        return "";
    ASSERT( M_NEURAY_API, attribute_set);

    std::string top_level_name = get_top_level_name( name);
    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( top_level_name.c_str());
    boost::shared_ptr<ATTR::Attribute> attribute = attribute_set->lookup_shared_ptr( attribute_id);
    if( !attribute)
        return "";

    Attribute_context owner( db_element, attribute);
    const ATTR::Type* attribute_type = owner.get_type( name);
    if( !attribute_type)
        return "";

    register_decls( *attribute_type);

    bool ignore_array_size = name[strlen(name)-1] == ']';
    return get_attribute_type_name( *attribute_type, ignore_array_size);
}

mi::Sint32 Attribute_set_impl_helper::set_attribute_propagation(
    ATTR::Attribute_set* attribute_set,
    IDb_element* db_element,
    const char* name,
    mi::neuraylib::Propagation_type value)
{
    if( !name)
        return -1;
    ASSERT( M_NEURAY_API, attribute_set);

    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( name);
    ATTR::Attribute* attribute = attribute_set->lookup( attribute_id);
    if( !attribute)
        return -2;

    ATTR::Attribute_propagation new_value;
    switch( value) {
        case( mi::neuraylib::PROPAGATION_STANDARD): new_value = ATTR::PROPAGATION_STANDARD; break;
        case( mi::neuraylib::PROPAGATION_OVERRIDE): new_value = ATTR::PROPAGATION_OVERRIDE; break;
        default:                         ASSERT( M_NEURAY_API, false); return -1;
    }
    ATTR::Attribute_propagation old_value = attribute->get_override();
    if( new_value == old_value)
        return 0;

    attribute->set_override( new_value);
    db_element->add_journal_flag( compute_journal_flags( attribute, attribute_id));
   
    return 0;
}

mi::neuraylib::Propagation_type Attribute_set_impl_helper::get_attribute_propagation(
    const ATTR::Attribute_set* attribute_set,
    const IDb_element* db_element,
    const char* name)
{
    if( !name)
        return mi::neuraylib::PROPAGATION_STANDARD;
    ASSERT( M_NEURAY_API, attribute_set);

    ATTR::Attribute_id attribute_id = ATTR::Attribute::id_lookup( name);
    const ATTR::Attribute* attribute = attribute_set->lookup( attribute_id);
    if( !attribute)
        return mi::neuraylib::PROPAGATION_STANDARD;

    switch( attribute->get_override()) {
        case( ATTR::PROPAGATION_STANDARD): return mi::neuraylib::PROPAGATION_STANDARD;
        case( ATTR::PROPAGATION_OVERRIDE): return mi::neuraylib::PROPAGATION_OVERRIDE;
        default: ASSERT( M_NEURAY_API, false); return mi::neuraylib::PROPAGATION_STANDARD;
    }
}

const char* Attribute_set_impl_helper::enumerate_attributes(
    const ATTR::Attribute_set* attribute_set,
    const IDb_element* db_element,
    mi::Sint32 index)
{
    if( index < 0)
        return nullptr;
    ASSERT( M_NEURAY_API, attribute_set);

    const ATTR::Attributes& attributes = attribute_set->get_attributes();
    ATTR::Attributes::const_iterator it     = attributes.begin();
    ATTR::Attributes::const_iterator it_end = attributes.end();

    while( (it != it_end) && ( (index > 0) || !has_valid_api_type( it->second.get()))) {
        if( has_valid_api_type( it->second.get()))
            --index;
        ++it;
    }

    // failure: index was too large
    if( it == it_end)
        return nullptr;
    // success
    return it->second->get_name();
}

mi::IData* Attribute_set_impl_helper::get_attribute(
    const IAttribute_context* owner, const std::string& attribute_name)
{
    ASSERT( M_NEURAY_API, owner);

    mi::base::Handle<const IDb_element> db_element( owner->get_db_element());
    mi::neuraylib::ITransaction* transaction = db_element->get_transaction();
    const ATTR::Type* attribute_type = owner->get_type( attribute_name.c_str());
    if( !attribute_type)
        return nullptr;

    void* pointer = owner->get_address( attribute_name.c_str());
    return get_attribute( transaction, owner, attribute_name, attribute_type, pointer);
}

mi::IData* Attribute_set_impl_helper::get_attribute(
    mi::neuraylib::ITransaction* transaction,
    const mi::base::IInterface* owner,
    const std::string& attribute_name,
    const ATTR::Type* attribute_type,
    void* pointer)
{
    ASSERT( M_NEURAY_API, owner);

    /// Note that for array elements the returned type is not correct. ATTR::Type::lookup() returns
    /// a type tree where the top-level element has the array size of the array itself (and
    /// not 1 as one would expect for a non-nested array). This is due to the fact that
    /// ATTR::Type::lookup() returns a pointer to a subtree of the type tree of the attribute
    /// itself. (*)
    if( !attribute_type)
        return nullptr;
    bool ignore_array_size = attribute_name[attribute_name.size()-1] == ']';

    register_decls( *attribute_type);

    std::string attribute_type_name
        = get_attribute_type_name( *attribute_type, ignore_array_size);
    if( attribute_type_name.empty())
        return nullptr;

    ATTR::Type_code attribute_type_code = attribute_type->get_typecode();
    mi::base::Handle<IProxy> attribute_proxy;

    // array attribute types
    if(    (attribute_type_code == ATTR::TYPE_ARRAY)
        || ((attribute_type->get_arraysize() != 1) && !ignore_array_size)) {
        mi::base::Handle<mi::IString> type_name( transaction->create<mi::IString>( "String"));
        type_name->set_c_str(
            Type_utilities::get_attribute_array_element_type_name( attribute_type_name).c_str());
        mi::base::Handle<mi::ISize> length( transaction->create<mi::ISize>( "Size"));
        length->set_value( Type_utilities::get_attribute_array_length( attribute_type_name));
        mi::base::Handle<mi::IString> name( transaction->create<mi::IString>( "String"));
        name->set_c_str( attribute_name.c_str());

        if( length->get_value<mi::Size>() == 0) {
            // dynamic arrays
            const mi::base::IInterface* argv[2];
            argv[0] = type_name.get();
            argv[1] = name.get();
            attribute_proxy = transaction->create<IProxy>( "__Dynamic_array_proxy", 2, argv);
        } else {
            // static arrays
            const mi::base::IInterface* argv[3];
            argv[0] = type_name.get();
            argv[1] = length.get();
            argv[2] = name.get();
            attribute_proxy = transaction->create<IProxy>( "__Array_proxy", 3, argv);
        }

    // structure attribute types
    } else if( attribute_type_code == ATTR::TYPE_STRUCT) {
        mi::base::Handle<const mi::IStructure_decl> decl( get_structure_decl( *attribute_type));
        ASSERT( M_NEURAY_API, decl.get());
        const mi::base::IInterface* argv[3];
        mi::base::Handle<mi::IString> type_name_string( transaction->create<mi::IString>("String"));
        type_name_string->set_c_str( attribute_type_name.c_str());
        mi::base::Handle<mi::IString> name( transaction->create<mi::IString>( "String"));
        name->set_c_str( attribute_name.c_str());
        argv[0] = decl.get();
        argv[1] = type_name_string.get();
        argv[2] = name.get();
        attribute_proxy = transaction->create<IProxy>( "__Structure_proxy", 3, argv);

    // enum attribute types
    } else if( attribute_type_code == ATTR::TYPE_ENUM) {
        mi::base::Handle<const mi::IEnum_decl> decl( get_enum_decl( *attribute_type));
        ASSERT( M_NEURAY_API, decl.get());
        const mi::base::IInterface* argv[2];
        mi::base::Handle<mi::IString> type_name_string( transaction->create<mi::IString>("String"));
        type_name_string->set_c_str( attribute_type_name.c_str());
        mi::base::Handle<mi::IString> name( transaction->create<mi::IString>( "String"));
        name->set_c_str( attribute_name.c_str());
        argv[0] = decl.get();
        argv[1] = type_name_string.get();
        attribute_proxy = transaction->create<IProxy>( "__Enum_proxy", 2, argv);

    // simple attribute types
    } else {
        ASSERT( M_NEURAY_API, Type_utilities::is_valid_simple_attribute_type( attribute_type_name));
        std::string proxy_type_name;
        // unsigned integral types should not appear here
        ASSERT( M_NEURAY_API,    attribute_type_name != "Uint8"
                              && attribute_type_name != "Uint16"
                              && attribute_type_name != "Uint32"
                              && attribute_type_name != "Uint64"
                              && attribute_type_name.substr( 0, 7) != "Uint32<");
        // handle ICompound: Compound_impl implements the default and the proxy variant in one class
        if(    (attribute_type_name == "Color")
            || (attribute_type_name == "Color3")
            || (attribute_type_name == "Spectrum")
            || (attribute_type_name == "Bbox3")
            || (attribute_type_name.substr( 0, 8) == "Boolean<")
            || (attribute_type_name.substr( 0, 7) == "Sint32<")
            || (attribute_type_name.substr( 0, 8) == "Float32<")
            || (attribute_type_name.substr( 0, 8) == "Float64<"))
            proxy_type_name = attribute_type_name;
        // handle IRef
        else if( attribute_type_name.substr( 0, 3) == "Ref")
            proxy_type_name = "__Ref_proxy" + attribute_type_name.substr( 3);
        // handle INumber, IString
        else
            proxy_type_name = "__" + attribute_type_name + "_proxy";
        attribute_proxy = transaction->create<IProxy>( proxy_type_name.c_str());
    }

    ASSERT( M_NEURAY_API, attribute_proxy);
    attribute_proxy->set_pointer_and_owner( pointer, owner);

    return attribute_proxy->get_interface<mi::IData>();
}

std::string Attribute_set_impl_helper::get_attribute_type_name(
    const ATTR::Type& type, bool ignore_array_size)
{
    ATTR::Type_code type_code = type.get_typecode();

    std::string attribute_type_name;

    // array attribute types
    if( (type_code == ATTR::TYPE_ARRAY) || ((type.get_arraysize() != 1) && !ignore_array_size)) {
        const ATTR::Type* element_type = (type_code == ATTR::TYPE_ARRAY) ? type.get_child() : &type;
        std::string element_type_name = get_attribute_type_name( *element_type, true);
        if( element_type_name.empty())
            return "";
        attribute_type_name = element_type_name;
        attribute_type_name += "[";
        std::ostringstream str;
        mi::Size length = element_type->get_arraysize();
        if( length > 0)
            str << length;
        attribute_type_name += str.str();
        attribute_type_name += "]";
        return attribute_type_name;
    }

    // structure attribute types
    if( type_code == ATTR::TYPE_STRUCT) {
        // check structure type name provide by ATTR
        const std::string& type_name = type.get_type_name();
        if( !type_name.empty()) {
            mi::base::Handle<const mi::IStructure_decl> decl(
                s_class_factory->get_structure_decl( type_name.c_str()));
            if( decl.is_valid_interface() && type_matches_structure_decl( type, decl.get()))
                return type_name;
        }
        // none provided, not registered, or does not match declaration
        attribute_type_name = "{";
        const ATTR::Type* member_type = type.get_child();
        while( member_type) {
            std::string member_type_name = get_attribute_type_name( *member_type, false);
            if( member_type_name.empty())
                return "";
            attribute_type_name += " ";
            attribute_type_name += member_type_name;
            attribute_type_name += " ";
            attribute_type_name += member_type->get_name(); //-V769 PVS
            attribute_type_name += ";";
            member_type = member_type->get_next();
        }
        attribute_type_name += " }";
        return attribute_type_name;
    }

    // enum attribute types
    if( type_code == ATTR::TYPE_ENUM) {
        // check enum type name provide by ATTR
        const std::string& type_name = type.get_type_name();
        if( !type_name.empty()) {
            mi::base::Handle<const mi::IEnum_decl> decl(
                s_class_factory->get_enum_decl( type_name.c_str()));
            if( decl.is_valid_interface() && type_matches_enum_decl( type, decl.get()))
                return type_name;
        }
        // none provided, not registered, or does not match declaration
        attribute_type_name = "{";
        Enum_collection* enum_collection = type.get_enum();
        mi::Size n = enum_collection->size();
        for( mi::Size i = 0; i < n; ++i) {
            attribute_type_name += " ";
            attribute_type_name += (*enum_collection)[i].second;
            attribute_type_name += " = ";
            attribute_type_name += (*enum_collection)[i].first;
            attribute_type_name += ";";
        }
        attribute_type_name += " }";
        return attribute_type_name;
    }

    // simple attribute types
    const char* result = Type_utilities::convert_type_code_to_attribute_type_name( type_code);
    if( !result)
        return "";

    return result;
}

ATTR::Type Attribute_set_impl_helper::get_attribute_type(
    const std::string& type_name, const std::string& name)
{
    ATTR::Type_code type_code
        = Type_utilities::convert_attribute_type_name_to_type_code( type_name);

    // array attribute types
    if( type_code == ATTR::TYPE_ARRAY) {
        // create ATTR::Type for an array element in element_type
        std::string element_type_name
            = Type_utilities::get_attribute_array_element_type_name( type_name);
        ATTR::Type element_type = get_attribute_type( element_type_name, name); // name important!
        if( element_type.get_typecode() == ATTR::TYPE_UNDEF)
            return ATTR::Type( ATTR::TYPE_UNDEF, nullptr, 1);
        // set the array length
        mi::Size array_length = Type_utilities::get_attribute_array_length( type_name);
        element_type = ATTR::Type( element_type, static_cast<mi::Sint32>( array_length));
        if( array_length == 0)
            return element_type;
        // create ATTR::Type for the static array in array_type
        ATTR::Type array_type( ATTR::TYPE_ARRAY, nullptr, 0);               // name does not matter here
        array_type.set_child( element_type);
        return array_type;

    // structure attribute types
    } else if( type_code == ATTR::TYPE_STRUCT) {
        mi::base::Handle<const mi::IStructure_decl> decl(
            s_class_factory->get_structure_decl( type_name.c_str()));
        if( !decl.is_valid_interface())
            return ATTR::Type( ATTR::TYPE_UNDEF, nullptr, 1);
        // create ATTR::Type
        ATTR::Type structure_type( ATTR::TYPE_STRUCT, name.c_str(), 1);
        structure_type.set_type_name( type_name);
        mi::Size n = decl->get_length();
        for( mi::Size i = 0; i < n; ++i) {
            // create ATTR::Type for the i-th member in member_type
            const char* member_type_name = decl->get_member_type_name( i);
            const char* member_name = decl->get_member_name( i);
            ATTR::Type member_type = get_attribute_type( member_type_name, member_name);
            if( member_type.get_typecode() == ATTR::TYPE_UNDEF)
                return ATTR::Type( ATTR::TYPE_UNDEF, nullptr, 1);
            // attach ATTR::Type for the i-the member to structure_type
            if( i == 0) {
                structure_type.set_child( member_type);
            } else {
                ATTR::Type* last_member = structure_type.get_child();
                for( mi::Size j = 1; j < i; ++j)
                    last_member = last_member->get_next();
                last_member->set_next( member_type);
            }
        }
        return structure_type;

    // enum attribute types
    } else if( type_code == ATTR::TYPE_ENUM) {
        mi::base::Handle<const mi::IEnum_decl> decl(
            s_class_factory->get_enum_decl( type_name.c_str()));
        if( !decl.is_valid_interface())
            return ATTR::Type( ATTR::TYPE_UNDEF, nullptr, 1);
        // create ATTR::Type
        ATTR::Type enum_type( ATTR::TYPE_ENUM, name.c_str(), 1);
        enum_type.set_type_name( type_name);
        mi::Size n = decl->get_length();
        Enum_collection* enum_collection = new Enum_collection();
        for( mi::Size i = 0; i < n; ++i)
            enum_collection->push_back( std::make_pair( decl->get_value( i), decl->get_name( i)));
        *enum_type.set_enum() = enum_collection;
        return enum_type;


    // simple attribute types
    } else {
        return ATTR::Type( type_code, name.c_str(), 1);
    }
}

const mi::IStructure_decl* Attribute_set_impl_helper::get_structure_decl( const ATTR::Type& type)
{
    ASSERT( M_NEURAY_API, type.get_typecode() == ATTR::TYPE_STRUCT);

    std::string type_name = get_attribute_type_name( type, true);
    return s_class_factory->get_structure_decl( type_name.c_str());
}

const mi::IEnum_decl* Attribute_set_impl_helper::get_enum_decl( const ATTR::Type& type)
{
    ASSERT( M_NEURAY_API, type.get_typecode() == ATTR::TYPE_ENUM);

    std::string type_name = get_attribute_type_name( type, true);
    return s_class_factory->get_enum_decl( type_name.c_str());
}

const mi::IStructure_decl* Attribute_set_impl_helper::create_structure_decl( const ATTR::Type& type)
{
    ASSERT( M_NEURAY_API, type.get_typecode() == ATTR::TYPE_STRUCT);

    mi::base::Handle<mi::IStructure_decl> decl(
        s_class_factory->create_type_instance<mi::IStructure_decl>( nullptr, "Structure_decl", 0, nullptr));

    const ATTR::Type* member_type = type.get_child();
    while( member_type) {
        ATTR::Type_code member_type_code = member_type->get_typecode();

        // array attribute types
        if( (member_type_code == ATTR::TYPE_ARRAY) || (member_type->get_arraysize() != 1)) {
            const ATTR::Type* element_type
                = (member_type_code == ATTR::TYPE_ARRAY)
                    ? member_type->get_child() : member_type;
            std::string element_type_name = get_attribute_type_name( *element_type, true);
            if( element_type_name.empty())
                return nullptr;
            std::ostringstream member_type_name;
            member_type_name << element_type_name << "[";
            mi::Size length = element_type->get_arraysize();
            if( length > 0)
                member_type_name << length;
            member_type_name << "]";
            decl->add_member( member_type_name.str().c_str(), member_type->get_name());
            member_type = member_type->get_next();

        // structure attribute types
        } else if( member_type_code == ATTR::TYPE_STRUCT) {
            mi::base::Handle<const mi::IStructure_decl> member_decl(
                get_structure_decl( *member_type));
            if( !member_decl.is_valid_interface())
                return nullptr;
            decl->add_member( member_decl->get_structure_type_name(), member_type->get_name());
            member_type = member_type->get_next();

        // enum attribute types
        } else if( member_type_code == ATTR::TYPE_ENUM) {
            mi::base::Handle<const mi::IEnum_decl> member_decl(
                get_enum_decl( *member_type));
            if( !member_decl.is_valid_interface())
                return nullptr;
            decl->add_member( member_decl->get_enum_type_name(), member_type->get_name());
            member_type = member_type->get_next();

        // simple attribute types
        } else {
            const char* member_type_name
                = Type_utilities::convert_type_code_to_attribute_type_name( member_type_code);
            if( !member_type_name)
                return nullptr;
            decl->add_member( member_type_name, member_type->get_name());
            member_type = member_type->get_next();
        }
    }

    decl->retain();
    return decl.get();
}

const mi::IEnum_decl* Attribute_set_impl_helper::create_enum_decl( const ATTR::Type& type)
{
    ASSERT( M_NEURAY_API, type.get_typecode() == ATTR::TYPE_ENUM);

    mi::base::Handle<mi::IEnum_decl> decl(
        s_class_factory->create_type_instance<mi::IEnum_decl>( nullptr, "Enum_decl", 0, nullptr));

    const Enum_collection* enum_collection = type.get_enum();
    mi::Size n = enum_collection->size();

    for( mi::Size i = 0; i < n; ++i)
        decl->add_enumerator( (*enum_collection)[i].second.c_str(), (*enum_collection)[i].first);

    decl->retain();
    return decl.get();
}

namespace {
#ifdef MI_API_API_NEURAY_TRACE_TYPE_REGISTRATION
void trace( const char* text, const char* name)
{
    LOG::mod_log->info( M_NEURAY_API, LOG::Mod_log::C_DATABASE, text, name);
}
#else // MI_API_API_NEURAY_TRACE_TYPE_REGISTRATION
void trace( const char* text, const char* name) { }
#endif // MI_API_API_NEURAY_TRACE_TYPE_REGISTRATION
} // namespace

void Attribute_set_impl_helper::register_decls( const ATTR::Type& type)
{
    mi::base::Lock::Block block( &s_register_decls_lock);
    return register_decls_locked( type);
}

void Attribute_set_impl_helper::register_decls_locked( const ATTR::Type& type)
{

    ATTR::Type_code type_code = type.get_typecode();

    // array attribute types
    if( (type_code == ATTR::TYPE_ARRAY) || (type.get_arraysize() != 1)) {
        const ATTR::Type* element_type = (type_code == ATTR::TYPE_ARRAY) ? type.get_child() : &type;
        register_decls_locked( ATTR::Type( *element_type, 1));

    // structure attribute types
    } else if( type_code == ATTR::TYPE_STRUCT) {

        // process struct members first
        const ATTR::Type* member_type = type.get_child();
        while( member_type) {
            register_decls_locked( *member_type);
            member_type = member_type->get_next();
        }

        // process the struct itself last (might use type names introduced above)
        const std::string& type_name = type.get_type_name();
        if( !type_name.empty()) {
            // query the structure declaration for the type name
            mi::base::Handle<const mi::IStructure_decl> decl(
                s_class_factory->get_structure_decl( type_name.c_str()));
            if( decl.is_valid_interface()) {
                // declaration exists and matches, nothing to do
                if( type_matches_structure_decl( type, decl.get())) {
                    trace( "Skipping stored type name %s (already registered and matches).",
                        type_name.c_str());
                    return;
                } else {
                    // declaration exists, but does not match, fall back to artificial type name
                    // (most probably we have seen that type already in another version of that
                    // module, e.g., in a different scope)
                    trace( "Ignoring stored type name %s (already registered, but does not "
                            "match).", type_name.c_str());
                    // ASSERT( M_NEURAY_API, false); // just to find these cases
                }
            } else {
                if( type_name.substr( 0, 2) == "::") {
                    // check that new types names created here do not contain characters that
                    // might cause conflicts, in particular no square, angle, or curly braces
                    ASSERT( M_SCENE, std::string::npos == type_name.find_first_not_of(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:"));
                    // create and register declaration for the type name
                    decl = create_structure_decl( type);
                    trace( "Registering stored type name %s.", type_name.c_str());
                    mi::Sint32 result = s_class_factory->register_structure_decl(
                        type_name.c_str(), decl.get());
                    ASSERT( M_NEURAY_API, result == 0);
                    boost::ignore_unused( result);
                    return;
                } else {
                    // ignore unqualified type names (MDL type names are always qualified, user-
                    // defined type names have already been registered upfront)
                    trace( "Ignoring stored type name %s (not fully qualified).",
                        type_name.c_str());
                    ASSERT( M_NEURAY_API, false); // just to find these cases
                }
            }
        }

        // use the artificial name "{...}" as fallback
        const std::string artificial_type_name = get_attribute_type_name( type, false);
        mi::base::Handle<const mi::IStructure_decl> decl( create_structure_decl( type));
        trace( "Registering artificial type name %s.", artificial_type_name.c_str());
        mi::Sint32 result = s_class_factory->register_structure_decl(
            artificial_type_name.c_str(), decl.get());
        // the artificial name might have already been registered
        ASSERT( M_NEURAY_API, result == 0 || result == -1);
        boost::ignore_unused( result);

    // enum types
    } else if( type_code == ATTR::TYPE_ENUM) {

        const std::string& type_name = type.get_type_name();
        if( !type_name.empty()) {
            // query the enum declaration for the type name
            mi::base::Handle<const mi::IEnum_decl> decl(
                s_class_factory->get_enum_decl( type_name.c_str()));
            if( decl.is_valid_interface()) {
                // declaration exists and matches, nothing to do
                if( type_matches_enum_decl( type, decl.get())) {
                    trace( "Skipping stored type name %s (already registered and matches).",
                        type_name.c_str());
                    return;
                } else {
                    // declaration exists, but does not match, fall back to artificial type name
                    // (most probably we have seen that type already in another version of that
                    // module, e.g., in a different scope)
                    trace( "Ignoring stored type name %s (already registered, but does not "
                            "match).", type_name.c_str());
                    // ASSERT( M_NEURAY_API, false); // just to find these cases
                }
            } else {
                if( type_name.substr( 0, 2) == "::") {
                    // check that new types names created here do not contain characters that
                    // might cause conflicts, in particular no square, angle, or curly braces
                    ASSERT( M_SCENE, std::string::npos == type_name.find_first_not_of(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:"));
                    // create and register declaration for the type name
                    decl = create_enum_decl( type);
                    trace( "Registering stored type name %s.", type_name.c_str());
                    mi::Sint32 result = s_class_factory->register_enum_decl(
                        type_name.c_str(), decl.get());
                    ASSERT( M_NEURAY_API, result == 0);
                    boost::ignore_unused( result);
                    return;
                } else {
                    // ignore unqualified type names (MDL type names are always qualified, user-
                    // defined type names have already been registered upfront)
                    trace( "Ignoring stored type name %s (not fully qualified).",
                        type_name.c_str());
                    ASSERT( M_NEURAY_API, false); // just to find these cases
                }
            }
        }

        // use the artificial name "{...}" as fallback
        const std::string artificial_type_name = get_attribute_type_name( type, false);
        mi::base::Handle<const mi::IEnum_decl> decl( create_enum_decl( type));
        trace( "Registering artificial type name %s.", artificial_type_name.c_str());
        mi::Sint32 result = s_class_factory->register_enum_decl(
            artificial_type_name.c_str(), decl.get());
        // the artificial name might have already been registered
        ASSERT( M_NEURAY_API, result == 0 || result == -1);
        boost::ignore_unused( result);

    // simple attribute types
    } else {
        // nothing to do
    }
}

bool Attribute_set_impl_helper::type_matches_structure_decl(
    const ATTR::Type& type, const mi::IStructure_decl* decl)
{
    ASSERT( M_NEURAY_API, type.get_typecode() == ATTR::TYPE_STRUCT);
    ASSERT( M_NEURAY_API, decl);

    mi::Size n = decl->get_length();
    const ATTR::Type* attr_member_type = type.get_child();

    for( mi::Size i = 0; i < n; ++i, attr_member_type = attr_member_type->get_next()) {

        // check that type has as many children as decl
        if( !attr_member_type)
            return false;

        // compare type names
        std::string attr_member_type_name = get_attribute_type_name( *attr_member_type, false);
        std::string decl_member_type_name = decl->get_member_type_name( i);
        if( attr_member_type_name != decl_member_type_name)
            return false;

        // compare member names
        std::string attr_member_name = attr_member_type->get_name(); //-V522 PVS
        if( attr_member_name != decl->get_member_name( i))
            return false;
    }

    // check that type has not more children than decl
    if( attr_member_type)
        return false;

    return true;
}

bool Attribute_set_impl_helper::type_matches_enum_decl(
    const ATTR::Type& type, const mi::IEnum_decl* decl)
{
    ASSERT( M_NEURAY_API, type.get_typecode() == ATTR::TYPE_ENUM);
    ASSERT( M_NEURAY_API, decl);

    mi::Size n = decl->get_length();
    const Enum_collection* enum_collection = type.get_enum();
    if( n != enum_collection->size())
        return false;

    for( mi::Size i = 0; i < n; ++i) {
        if( strcmp( decl->get_name( i), (*enum_collection)[i].second.c_str()) != 0)
           return false;
        if( decl->get_value( i) != (*enum_collection)[i].first)
           return false;
    }

    return true;
}

bool Attribute_set_impl_helper::has_valid_api_type( const ATTR::Attribute* attribute)
{
    ASSERT( M_NEURAY_API, attribute);

    // These attributes are not exposed via the attribute set because:
    // - there exist specialized methods on ITriangle_mesh, etc. that take the special structure
    //   of these attributes into account
    // - the attributes use attribute lists which are not supported by the API
    // - some of them are arrays (SCENE::OBJ_DERIVS) but do not properly set the attribute name
    //   (see 130460 and 130461)
    ATTR::Attribute_id attribute_id = attribute->get_id();
    if(    (attribute_id == SCENE::OBJ_NORMAL)
        || (attribute_id == SCENE::OBJ_MOTION)
        || (attribute_id == SCENE::OBJ_MATERIAL_INDEX)
        || (attribute_id == SCENE::OBJ_MATERIAL_INDEX_QUAD)
        || (attribute_id == SCENE::OBJ_DERIVS)
        || (attribute_id == SCENE::OBJ_PRIM_LABEL)
        || (attribute_id >= SCENE::OBJ_TEXTURE
            && attribute_id < SCENE::OBJ_TEXTURE + SCENE::OBJ_TEXTURE_NUM)
        || (attribute_id >= SCENE::OBJ_USER
            && attribute_id < SCENE::OBJ_USER + SCENE::OBJ_USER_NUM))
        return false;

    return !get_attribute_type_name( attribute->get_type()).empty();
}

bool Attribute_set_impl_helper::is_correct_type_for_attribute(
    const std::string& attribute_name,
    ATTR::Attribute_id attribute_id,
    const std::string& type_name)
{
    // Exceptions for some registered attributes: the internal attribute spec is useless
    // since it stores only a single type code which is not sufficient.
    if( attribute_name == "material")
        return type_name == "Ref" || type_name.substr( 0, 4) == "Ref[";

    if(    attribute_name == "decals" || attribute_name == "enabled_decals"
        || attribute_name == "disabled_decals" || attribute_name == "projectors")
        return type_name.substr( 0, 4) == "Ref[";

    if( attribute_name == "approx" || attribute_name == "approx_curve")
        return Type_utilities::compatible_types( type_name, "Approx", false);

    if( attribute_name == "section_planes") {
        if( !Type_utilities::is_valid_array_attribute_type( type_name))
            return false;
        std::string element_type
            = Type_utilities::get_attribute_array_element_type_name( type_name);
        return Type_utilities::compatible_types( element_type, "Section_plane", false);
    }

    if( attribute_name == "tm_custom_tonemapper")
        return Type_utilities::compatible_types( type_name, "Tm_custom_curve", false);

    // No checks for attributes without attribute spec.
    SYSTEM::Access_module<ATTR::Attr_module> attr_module( false);
    const ATTR::Attribute_spec* attribute_spec
        = attr_module->get_reserved_attr_spec( attribute_id);
    if( !attribute_spec)
        return true;

    // Checks the type code for remaining attributes with attribute spec.
    ATTR::Type_code attribute_type_code
        = Type_utilities::convert_attribute_type_name_to_type_code( type_name);
    if( attribute_type_code == ATTR::TYPE_UNDEF)
        return false;
    return attribute_spec->get_typecode() == attribute_type_code;
}

std::string Attribute_set_impl_helper::get_top_level_name( const char* name)
{
    ASSERT( M_NEURAY_API, name);

    std::string name_str (name);
    mi::Size separator = name_str.find_first_of( ".[");
    if( separator == std::string::npos)
        return name_str;
    return name_str.substr( 0, separator);
}

DB::Journal_type Attribute_set_impl_helper::compute_journal_flags(
    const ATTR::Attribute* attr,
    ATTR::Attribute_id attribute_id)
{
    DB::Journal_type result = DB::JOURNAL_NONE;

    SYSTEM::Access_module<ATTR::Attr_module> attr_module( false);
    const ATTR::Attribute_spec* attribute_spec
        = attr_module->get_reserved_attr_spec( attribute_id);
    if( attribute_spec) {
        result.add_journal( attribute_spec->get_journal_flags());
         return result;
    }

    // also consider userdata attribute types
    const ATTR::Type& attr_type = attr->get_type();
    const ATTR::Type_code tc = attr_type.get_typecode();
    if (tc == ATTR::TYPE_COLOR ||
        tc == ATTR::TYPE_SCALAR ||
        tc == ATTR::TYPE_VECTOR2 ||
        tc == ATTR::TYPE_VECTOR3 ||
        tc == ATTR::TYPE_VECTOR4 ||
        tc == ATTR::TYPE_INT32) {

        // check filter
        std::wstring name_wstr = STRING::utf8_to_wchar( attr->get_name());
        const std::wregex& name_regex = attr_module->get_custom_attr_filter();
        if( !std::regex_search( name_wstr, name_regex)) {
            // no match
            result.add_journal( SCENE::JOURNAL_CHANGE_NON_SHADER_ATTRIBUTE);
        }
    }

    return result;
}

} // namespace NEURAY

} // namespace MI
