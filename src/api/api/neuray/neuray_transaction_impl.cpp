/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the ITransaction implementation.
 **/

#include "pch.h"

//#define VERBOSE_TX // extra verbose info messages to track transactions

#include "neuray_transaction_impl.h"

#include "neuray_class_factory.h"
#include "neuray_db_element_impl.h"
#include "neuray_scope_impl.h"



#include <mi/base/handle.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/iuser_class.h>

#include <sstream>

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_transaction.h>
#include <base/lib/log/i_log_logger.h>
#include <base/util/string_utils/i_string_utils.h>

#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>


namespace MI {

namespace NEURAY {

Transaction_impl::Transaction_impl(
    DB::Transaction* db_transaction,
    const Class_factory* class_factory,
    bool commit_or_abort_warning)
{
    m_db_transaction = db_transaction;
    m_db_transaction->pin();
    m_class_factory = class_factory;
    m_commit_or_abort_warning = commit_or_abort_warning;

    DB::Transaction_id id = m_db_transaction->get_id();
    m_id_as_uint = id();

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u created", m_id_as_uint);
#endif
}

Transaction_impl::~Transaction_impl()
{
    if( m_commit_or_abort_warning) {
        // Abort/commit the transaction here if it was not aborted or committed yet. Since this is
        // not a proper usage, emit a warning/error. The MDL SDK does not support abort(), therefore
        // we have to call commit(). This is unfortunate since it advantages users to omit the
        // commit() call. Hence, it is treated as an error and not just as a warning.
        LOG::mod_log->error( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "Transaction is released without being committed or aborted. Automatically "
            "committing.");
        commit();
    }
    m_db_transaction->unpin();

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u destroyed", m_id_as_uint);
#endif
}

mi::Sint32 Transaction_impl::commit()
{
    if( !is_open())
        return -3;

    check_no_referenced_elements( "committed");

    m_commit_or_abort_warning = false;

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u comitting ...", m_id_as_uint);
#endif

    mi::Sint32 result = m_db_transaction->commit() ? 0 : -1;

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u comitting done.", m_id_as_uint);
#endif

    return result;
}

void Transaction_impl::abort()
{
    LOG::mod_log->error( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "ITransaction::abort() is not supported.");
    return;
}

bool Transaction_impl::is_open() const
{
    return m_db_transaction->is_open();
}

mi::base::IInterface* Transaction_impl::create(
    const char* type_name,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !type_name)
        return nullptr;

    return m_class_factory->create_type_instance( this, type_name, argc, argv);
}

mi::Sint32 Transaction_impl::store(
    mi::base::IInterface* interface,
    const char* name,
    mi::Uint8 privacy)
{
    if( !interface || !name || !name[0])
      return -2;
    if( !is_open())
      return -3;

    mi::Uint8 scope_privacy = m_db_transaction->get_scope()->get_level();
    if( privacy == LOCAL_SCOPE)
        privacy = scope_privacy;
    else if( privacy > scope_privacy)
        return -5;

    // distinguish between built-in classes (IDb_element) and user-defined classes (IUser_class)

    // handle built-in classes
    mi::base::Handle<IDb_element> db_element( interface->get_interface<IDb_element>());
    if( db_element.is_valid_interface()) {
#ifdef VERBOSE_TX
        LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "TX %u storing \"%s\" ...", m_id_as_uint, name);
#endif
        const mi::Sint32 result = db_element->store( this, name, privacy);
#ifdef VERBOSE_TX
        LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "TX %u storing \"%s\" done.", m_id_as_uint, name);
#endif
        return result;
    }


    return -4;
}

const mi::base::IInterface* Transaction_impl::access(
    const char* name)
{
    if( !is_open())
        return nullptr;

    DB::Tag tag = m_db_transaction->name_to_tag( name);
    if( !tag.is_valid())
        return nullptr;

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u accessing \"%s\" tag %u ...", m_id_as_uint, name, tag.get_uint());
#endif

    return access( tag);
}

mi::base::IInterface* Transaction_impl::edit(
    const char* name)
{
    if( !is_open())
        return nullptr;

    DB::Tag tag = m_db_transaction->name_to_tag( name);
    if( !tag.is_valid())
        return nullptr;

    SERIAL::Class_id id = m_db_transaction->get_class_id(tag);
    if (id == MDL::ID_MDL_MATERIAL_DEFINITION || id == MDL::ID_MDL_FUNCTION_DEFINITION)
        return nullptr;

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u editing \"%s\" tag %u ...", m_id_as_uint, name, tag.get_uint());
#endif

    return edit( tag);
}

mi::Sint32 Transaction_impl::copy( const char* source, const char* target, mi::Uint8 privacy)
{
    if( !source || !target || !target[0])
        return -2;

    if ( !is_open())
        return -3;

    DB::Tag source_tag = m_db_transaction->name_to_tag( source);
    if( !source_tag)
        return -4;

    mi::Uint8 scope_privacy = m_db_transaction->get_scope()->get_level();
    if( privacy == LOCAL_SCOPE)
        privacy = scope_privacy;
    else if( privacy > scope_privacy)
        return -5;

    SERIAL::Class_id class_id = m_db_transaction->get_class_id( source_tag);

    // prevent any copies of IMdl_module, IMdl_material_definition, IMdl_function_definition
    if(     (class_id == MDL::ID_MDL_MODULE)
         || (class_id == MDL::ID_MDL_MATERIAL_DEFINITION)
         || (class_id == MDL::ID_MDL_FUNCTION_DEFINITION))
        return -6;

    if (class_id == MDL::ID_MDL_FUNCTION_CALL) {
        DB::Access<MDL::Mdl_function_call> f_call(source_tag, m_db_transaction);
        if (f_call->is_immutable())
            return -6;
    }
    if (class_id == MDL::ID_MDL_MATERIAL_INSTANCE) {
        DB::Access<MDL::Mdl_material_instance> m_inst(source_tag, m_db_transaction);
        if (m_inst->is_immutable())
            return -6;
    }

    if( strcmp( source, target) == 0) {
        // If source and target names are identical, reuse the source tag.
        // Use special DB method that defaults to no journal flags (in contrast to store()).
        mi::Uint8 source_privacy = m_db_transaction->get_tag_privacy_level( source_tag);
        if( privacy == source_privacy)
            return -5;
#ifdef VERBOSE_TX
        LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "TX %u copying \"%s\" to \"%s\" ...", m_id_as_uint, source, target);
#endif
        m_db_transaction->localize( source_tag, privacy);
    } else {
        // If source and target names are different, lookup target tag.
        // Prevent overwriting an existing DB element with one of a different type
        DB::Tag target_tag = m_db_transaction->name_to_tag( target);
        if( target_tag) {
            SERIAL::Class_id target_class_id = m_db_transaction->get_class_id( target_tag);
            if(    (target_class_id == MDL::ID_MDL_MODULE)
                || (target_class_id == MDL::ID_MDL_MATERIAL_DEFINITION)
                || (target_class_id == MDL::ID_MDL_FUNCTION_DEFINITION))
                return -9;

            if (target_class_id == MDL::ID_MDL_FUNCTION_CALL) {
                DB::Access<MDL::Mdl_function_call> f_call(target_tag, m_db_transaction);
                if (f_call->is_immutable())
                    return -9;
            }
            if (target_class_id == MDL::ID_MDL_MATERIAL_INSTANCE) {
                DB::Access<MDL::Mdl_material_instance> m_inst(target_tag, m_db_transaction);
                if (m_inst->is_immutable())
                    return -9;
            }
        }

        DB::Access<DB::Element_base> access(source_tag, m_db_transaction);
        // Create a copy of the DB element.
        DB::Element_base* element = access->copy();
        // And store it.
        target_tag = get_tag_for_store( target_tag);
#ifdef VERBOSE_TX
        LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "TX %u copying \"%s\" to \"%s\" ...", m_id_as_uint, source, target);
#endif
        m_db_transaction->store( target_tag, element, target, privacy);
        // DB took over ownership of element.
    }
#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u copying \"%s\" to \"%s\" done.", m_id_as_uint, source, target);
#endif
    return 0;
}

mi::Sint32 Transaction_impl::remove(
    const char* name,
    bool only_localized)
{
    if( !name)
        return -2;

    if( !is_open())
        return -3;

    DB::Tag tag = m_db_transaction->name_to_tag( name);
    if( !tag)
        return -1;

#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u removing \"%s\" ...", m_id_as_uint, name);
#endif
    const mi::Sint32 result = m_db_transaction->remove( tag, only_localized) ? 0 : -1;
#ifdef VERBOSE_TX
    LOG::mod_log->info( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
        "TX %u removing \"%s\" done.", m_id_as_uint, name);
#endif
    return result;
}

const char* Transaction_impl::name_of(
    const mi::base::IInterface* interface) const
{
    if( !is_open() || !interface)
        return nullptr;

    mi::base::Handle<const mi::base::IInterface> api_class( interface, mi::base::DUP_INTERFACE);



    mi::base::Handle<const IDb_element> db_element( api_class->get_interface<IDb_element>());
    if( !db_element.is_valid_interface())
        return nullptr;

    DB::Tag tag = db_element->get_tag();
    if( !tag)
        return nullptr;

    return m_db_transaction->tag_to_name( tag);
}

const char* Transaction_impl::get_time_stamp() const
{
    DB::Transaction_id transaction_id = m_db_transaction->get_id();

    // Note that get_update_sequence_number() returns the sequence number for the *next* update.
    mi::Uint32 next_sequence_number = m_db_transaction->get_update_sequence_number();
    mi::Sint32 current_sequence_number = static_cast<mi::Sint32>( next_sequence_number) - 1;

    std::ostringstream s;
    s << transaction_id() << "::" << current_sequence_number;
    m_timestamp = s.str();
    return m_timestamp.c_str();
}

const char* Transaction_impl::get_time_stamp( const char* element) const
{
    if( !element || !is_open())
        return nullptr;

    DB::Tag tag = m_db_transaction->name_to_tag( element);
    if( !tag)
        return nullptr;

    return get_time_stamp( tag);
}

bool Transaction_impl::has_changed_since_time_stamp(
    const char* element,
    const char* time_stamp) const
{
    // Unfortunately, we can only return true or false and not signal an error.
    // Let's err on the safe side and return true in case of errors.
    if( !element || !time_stamp || !is_open())
        return true;

    DB::Tag tag = m_db_transaction->name_to_tag( element);
    if( !tag)
        return true;

    return has_changed_since_time_stamp( tag, time_stamp);
}

const char* Transaction_impl::get_id() const
{
    if( m_id_as_string.empty()) {
        std::ostringstream stream;
        stream << m_id_as_uint;
        m_id_as_string = stream.str();
    }
    return m_id_as_string.c_str();
}

mi::neuraylib::IScope* Transaction_impl::get_scope() const
{
    DB::Scope* db_scope = m_db_transaction->get_scope();
    if( !db_scope)
        return nullptr;

    return new Scope_impl( db_scope, m_class_factory);
}

mi::IArray* Transaction_impl::list_elements(
    const char* root_element, const char* name_pattern, const mi::IArray* type_names) const
{
    if( !root_element || !is_open())
        return nullptr;
    DB::Tag root_tag = m_db_transaction->name_to_tag( root_element);
    if( !root_tag)
        return nullptr;

    std::wregex name_regex;
    try {
        if( name_pattern) {
            std::wstring name_pattern_wstr = STRING::utf8_to_wchar( name_pattern);
            name_regex.assign( name_pattern_wstr, std::wregex::extended);
        }
    } catch( const std::regex_error& ) {
        return nullptr;
    }

    LOG::mod_log->vdebug( M_NEURAY_API, LOG::Mod_log::C_MISC, "ITransaction::list_elements()");
    LOG::mod_log->vdebug( M_NEURAY_API, LOG::Mod_log::C_MISC, "  root_element:  %s", root_element);
    LOG::mod_log->vdebug( M_NEURAY_API, LOG::Mod_log::C_MISC, "  name_pattern:  %s", name_pattern);

    // convert type_names to set of SERIAL::Class_id
    std::set<SERIAL::Class_id> class_ids;
    for( size_t i = 0; type_names && i < type_names->get_length(); ++i) {
        mi::base::Handle<const mi::IString> type_name_istring(
            type_names->get_element<mi::IString>( i));
        if( !type_name_istring.is_valid_interface()) {
            const char* s = "  type_names[%" FMT_SIZE_T "]: (invalid interface type)";
            LOG::mod_log->vdebug( M_NEURAY_API, LOG::Mod_log::C_MISC, s, i);
            continue;
        }
        std::string type_name = type_name_istring->get_c_str();
        SERIAL::Class_id class_id = m_class_factory->get_class_id( type_name.c_str());
        if( class_id == 0) {
            // try again with "__" prefix
            std::string internal_type_name = "__";
            internal_type_name += type_name;
            class_id = m_class_factory->get_class_id( internal_type_name.c_str());
            if( class_id == 0) {
                const char* s = "  type_names[%" FMT_SIZE_T "]: (invalid type name)";
                LOG::mod_log->vdebug( M_NEURAY_API, LOG::Mod_log::C_MISC, s, i);
                continue;
            }
        }
        const char* s = "  type_names[%" FMT_SIZE_T "]: %s";
        LOG::mod_log->vdebug( M_NEURAY_API, LOG::Mod_log::C_MISC, s, i, type_name.c_str());
        class_ids.insert( class_id);
    }

    // create result array
    mi::IDynamic_array* result
        = m_class_factory->create_type_instance<mi::IDynamic_array>( nullptr, "String[]", 0, nullptr);

    // start DFS post-order graph traversal at root_tag
    std::set<DB::Tag> tags_seen;
    tags_seen.insert( root_tag); // not really needed if the graph is acyclic
    list_elements_internal(
        root_tag, name_pattern ? &name_regex : nullptr, type_names ? &class_ids : nullptr, result, tags_seen);

    return result;
}

mi::Sint32 Transaction_impl::get_privacy_level( const char* name) const
{
    if( !name)
        return -2;
    if( !is_open())
        return -3;

    DB::Tag tag = m_db_transaction->name_to_tag( name);
    if( !tag.is_valid())
        return -4;

    return m_db_transaction->get_tag_privacy_level( tag);
}

const mi::base::IInterface* Transaction_impl::get_interface(
    const mi::base::Uuid & interface_id) const
{

    return mi::base::Interface_implement<NEURAY::ITransaction>
        ::get_interface( interface_id);
}

mi::base::IInterface* Transaction_impl::get_interface(
    const mi::base::Uuid & interface_id)
{

    return mi::base::Interface_implement<NEURAY::ITransaction>
        ::get_interface( interface_id);
}

DB::Transaction* Transaction_impl::get_db_transaction() const
{
    return m_db_transaction;
}

mi::base::IInterface* Transaction_impl::edit( DB::Tag tag)
{
    if( !is_open())
        return nullptr;

    return m_class_factory->create_class_instance( this, tag, true);
}

const mi::base::IInterface* Transaction_impl::access( DB::Tag tag)
{
    if( !is_open())
        return nullptr;

    return m_class_factory->create_class_instance( this, tag, false);
}

const char* Transaction_impl::get_time_stamp( DB::Tag tag) const
{
    ASSERT( M_NEURAY_API, tag.is_valid());

    DB::Tag_version tag_version = m_db_transaction->get_tag_version( tag);

    std::ostringstream s;
    s << tag_version.m_transaction_id() << "::" << tag_version.m_version;
    m_timestamp = s.str();
    return m_timestamp.c_str();
}

bool Transaction_impl::has_changed_since_time_stamp( DB::Tag tag, const char* time_stamp) const
{
    ASSERT( M_NEURAY_API, tag.is_valid());
    ASSERT( M_NEURAY_API, time_stamp);

    DB::Tag_version tag_version = m_db_transaction->get_tag_version( tag);

    // Parse time_stamp into transaction id and sequence number
    mi::Uint32 time_stamp_pod_transaction_id;
    mi::Sint32 time_stamp_sequence_number;
    if (2 != sscanf (time_stamp, "%10u::%10d",
        &time_stamp_pod_transaction_id, &time_stamp_sequence_number))
        return true;
    DB::Transaction_id time_stamp_transaction_id( time_stamp_pod_transaction_id);

    return    (time_stamp_transaction_id < tag_version.m_transaction_id)
           || (    (time_stamp_transaction_id == tag_version.m_transaction_id)
                && (time_stamp_sequence_number < static_cast<mi::Sint32>( tag_version.m_version)));
}

mi::Sint32 Transaction_impl::get_privacy_level( DB::Tag tag) const
{
    if( !is_open())
        return -3;

    return m_db_transaction->get_tag_privacy_level( tag);
}

const Class_factory* Transaction_impl::get_class_factory() const
{
    return m_class_factory;
}

DB::Tag Transaction_impl::get_tag_for_store( const char* name)
{
    return get_tag_for_store( m_db_transaction->name_to_tag( name));
}

DB::Tag Transaction_impl::get_tag_for_store( DB::Tag tag)
{
    if( tag && (   !m_db_transaction->get_tag_is_removed( tag)
                 || m_db_transaction->get_tag_reference_count( tag) > 0))
        return tag;

    return m_db_transaction->reserve_tag();
}

void Transaction_impl::add_element( const Db_element_impl_base* db_element)
{
    // Note: we store a pointer to a reference-counted object without calling retain() here. This is
    // not a problem since this method will only be called from the set_state_*() methods of that
    // object (and the corresponding method from the destructor of that object).
    //
    // Reference counting as usual is not possible since that would increase the reference count,
    // and the object would never go out of scope.
    mi::base::Lock::Block block( &m_elements_lock);
    m_elements.insert( db_element);
}

void Transaction_impl::remove_element( const Db_element_impl_base* db_element)
{
    // Note: we remove a pointer to reference-counted object without calling release() here. This is
    // not a problem since this method will only be called from the set_state_*() methods of that
    // object (and the corresponding method from the constructor of that object).
    //
    // Reference counting as usual is not possible since that would increase the reference count,
    // and the object would never go out of scope.
    mi::base::Lock::Block block( &m_elements_lock);
    m_elements.erase( db_element);
}

Type_factory* Transaction_impl::get_type_factory()
{
    return m_class_factory->create_type_factory( this);
}

Value_factory* Transaction_impl::get_value_factory()
{
    return m_class_factory->create_value_factory( this);
}

Expression_factory* Transaction_impl::get_expression_factory()
{
    return m_class_factory->create_expression_factory( this);
}

void Transaction_impl::check_no_referenced_elements( const char* committed_or_aborted)
{
    mi::base::Lock::Block block( &m_elements_lock);
    for( Elements::const_iterator it = m_elements.begin(); it != m_elements.end(); ++it) {
        DB::Tag tag = (*it)->get_tag();
        const char* name = m_db_transaction->tag_to_name( tag);
        std::ostringstream s;
        if( name)
            s << "\"" << name << "\"";
        else
            s << "with tag " << tag.get_uint();
        LOG::mod_log->error( SYSTEM::M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "DB element %s is still referenced while transaction %u is %s.",
            s.str().c_str(), m_id_as_uint, committed_or_aborted);
    }
}

void Transaction_impl::list_elements_internal(
    DB::Tag tag,
    const std::wregex* name_regex,
    const std::set<SERIAL::Class_id>* class_ids,
    mi::IDynamic_array* result,
    std::set<DB::Tag>& tags_seen) const
{
    // Skip DB elements that have not been registered with the API's class factory.
    SERIAL::Class_id class_id = m_class_factory->get_class_id( this, tag);
    if( !m_class_factory->is_class_registered( class_id))
        return;

    // Implementation classes of resources should have been skipped. They are an internal
    // implementation detail and not visible in the API.
    ASSERT( M_NEURAY_API, class_id != DBIMAGE::ID_IMAGE_IMPL);
    ASSERT( M_NEURAY_API, class_id != LIGHTPROFILE::ID_LIGHTPROFILE_IMPL);
    ASSERT( M_NEURAY_API, class_id != BSDFM::ID_BSDF_MEASUREMENT_IMPL);

    // get references of tag
    DB::Access<DB::Element_base> element( tag, m_db_transaction);
    DB::Tag_set references;
    element->get_references( &references);

    // call recursively for all references not yet in tags_seen
    for( DB::Tag_set::const_iterator it = references.begin(); it != references.end(); ++it)
        if( tags_seen.find( *it) == tags_seen.end()) {
            tags_seen.insert( *it);
            list_elements_internal( *it, name_regex, class_ids, result, tags_seen);
        }

    // skip tag if it has the wrong class ID
    if( class_ids) {
        if( class_ids->find( class_id) == class_ids->end())
            return;
    }

    // skip tag if it has no name
    const char* name = m_db_transaction->tag_to_name( tag);
    if( !name)
        return;

    // skip tag if its name does not match the regular expression
    if( name_regex) {
        std::wstring name_wstr = STRING::utf8_to_wchar( name);
        if( !std::regex_search( name_wstr, *name_regex))
            return;
    }

    // tag matches criteria, store its name in the result
    mi::base::Handle<mi::IString> s(
        m_class_factory->create_type_instance<mi::IString>( nullptr, "String", 0, nullptr));
    s->set_c_str( name);
    result->push_back( s.get());
}

} // namespace NEURAY

} // namespace MI

