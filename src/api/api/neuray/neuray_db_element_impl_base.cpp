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
 ** \brief Header for the IDb_element implementation.
 **/

#include "pch.h"

#include "neuray_db_element_impl.h"
#include "neuray_db_element_tracker.h"
#include "neuray_transaction_impl.h"

#include <io/scene/mdl_elements/i_mdl_elements_function_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_definition.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/mdl_elements/i_mdl_elements_function_call.h>
#include <io/scene/mdl_elements/i_mdl_elements_material_instance.h>
#include <io/scene/mdl_elements/mdl_elements_annotation_definition_proxy.h>

// If defined, the destructor of Db_element_impl_base dumps all set journal flags for elements in
// state STATE_EDIT.
// #define MI_API_API_NEURAY_DUMP_JOURNAL_FLAGS

#ifdef MI_API_API_NEURAY_DUMP_JOURNAL_FLAGS
#include <sstream>
#include <base/lib/log/i_log_assert.h>
#include <base/data/db/i_db_journal_type.h>
#endif // MI_API_API_NEURAY_DUMP_JOURNAL_FLAGS

namespace MI {

namespace NEURAY {

Db_element_tracker g_db_element_tracker;

Db_element_impl_base::Db_element_impl_base()
{
    m_state = STATE_INVALID;
    m_pointer.m_const = nullptr;
    m_transaction = nullptr;

    g_db_element_tracker.add_element( this);
}

void Db_element_impl_base::set_state_access(
    Transaction_impl* transaction,
    DB::Tag tag)
{
    ASSERT( M_NEURAY_API, m_state == STATE_INVALID);
    ASSERT( M_NEURAY_API, transaction);
    DB::Transaction* db_transaction = transaction->get_db_transaction();
    m_state = STATE_ACCESS;
    m_pointer.m_const
        = m_access_base.set_access( tag, db_transaction, db_transaction->get_class_id( tag));
    m_transaction = transaction;
    m_transaction->retain();
    m_transaction->add_element( this);
}

void Db_element_impl_base::set_state_edit(
    Transaction_impl* transaction,
    DB::Tag tag)
{
    ASSERT( M_NEURAY_API, m_state == STATE_INVALID);
    ASSERT( M_NEURAY_API, transaction);
    DB::Transaction* db_transaction = transaction->get_db_transaction();
    m_state = STATE_EDIT;
    m_pointer.m_mutable
        = m_access_base.set_edit(
            tag, db_transaction, db_transaction->get_class_id( tag), DB::JOURNAL_NONE);
    m_transaction = transaction;
    m_transaction->retain();
    m_transaction->add_element( this);
}

void Db_element_impl_base::set_state_pointer(
    Transaction_impl* transaction, DB::Element_base* element)
{
    ASSERT( M_NEURAY_API, m_state == STATE_INVALID);
    ASSERT( M_NEURAY_API, transaction);
    m_state = STATE_POINTER;
    m_pointer.m_mutable = element;
    m_transaction = transaction;
    m_transaction->retain();
    m_transaction->add_element( this);
}

Db_element_state Db_element_impl_base::get_state() const
{
    return m_state;
}

Db_element_impl_base::~Db_element_impl_base()
{
    if( m_transaction)
        m_transaction->remove_element( this);

    g_db_element_tracker.remove_element( this);

    switch( m_state) {
        case STATE_ACCESS:
            break;
        case STATE_EDIT:
#ifdef MI_API_API_NEURAY_DUMP_JOURNAL_FLAGS
            {
                std::ostringstream s;

                DB::Element_base* element_base = m_access_base.get_base_ptr();
                ASSERT( M_NEURAY_API, element_base);
                s << element_base->get_class_name();

                DB::Tag tag = m_access_base.get_tag();
                s << " " << tag.get_uint();

                ASSERT( M_NEURAY_API, m_transaction);
                DB::Transaction* db_transaction = m_transaction->get_db_transaction();
                ASSERT( M_NEURAY_API, db_transaction);
                const char* element_name = get_db_transaction()->tag_to_name( tag);
                s << " " << (element_name ? element_name : "(unknown)");

                DB::Journal_type flags = m_access_base.get_journal_flags();
                s << " " << flags.get_type() << " " << Db_element_tracker::flags_to_string( flags);

                LOG::mod_log->info( M_NEURAY_API, LOG::Mod_log::C_DATABASE, s.str().c_str());
            }
#endif // MI_API_API_NEURAY_DUMP_JOURNAL_FLAGS
            break;
        case STATE_POINTER:
            delete m_pointer.m_mutable;
            break;
        case STATE_INVALID:
            // ignore
            return; // no transaction to release
    }
    if( m_transaction)
        m_transaction->release();
}

DB::Tag Db_element_impl_base::get_tag() const
{
    switch( m_state) {
        case STATE_ACCESS:
        case STATE_EDIT:
            return m_access_base.get_tag();
            break;
        case STATE_POINTER:
        case STATE_INVALID:
            return DB::Tag();
            break;
    }
    ASSERT( M_NEURAY_API, false);
    return DB::Tag();
}

mi::Sint32 Db_element_impl_base::store(
    Transaction_impl* transaction, const char* name, mi::Uint8 privacy)
{
    mi::Sint32 result = pre_store( transaction, privacy);
    if( result != 0)
        return result;

    DB::Transaction* db_transaction = transaction->get_db_transaction();
    DB::Privacy_level store_level = privacy;

    // prevent overwriting an existing DB element with one of a different type
    DB::Tag tag = db_transaction->name_to_tag( name);
    if( tag) {
        SERIAL::Class_id class_id = db_transaction->get_class_id( tag);
        if(    (class_id == MDL::ID_MDL_MODULE)
            || (class_id == MDL::ID_MDL_MATERIAL_DEFINITION)
            || (class_id == MDL::ID_MDL_FUNCTION_DEFINITION)
            || (class_id == MDL::ID_MDL_ANNOTATION_DEFINITION_PROXY))
            return -9;

        if (class_id == MDL::ID_MDL_FUNCTION_CALL) {
            DB::Access<MDL::Mdl_function_call> f_call(tag, db_transaction);
            if (f_call->is_immutable())
                return -9;
        }
        if (class_id == MDL::ID_MDL_MATERIAL_INSTANCE) {
            DB::Access<MDL::Mdl_material_instance> m_inst(tag, db_transaction);
            if (m_inst->is_immutable())
                return -9;
        }
    }

    tag = transaction->get_tag_for_store( tag);

    // use DB::JOURNAL_ALL instead of journal flags in m_access_base: for intitial stores it does
    // not really matter, but for overwriting existing elements we do not know the journal flags
    // that need to be set
    db_transaction->store( tag, m_pointer.m_mutable, name, privacy, DB::JOURNAL_ALL, store_level);

    post_store();

    return 0;
}

mi::Sint32 Db_element_impl_base::pre_store( Transaction_impl* transaction, mi::Uint8 privacy)
{
    // store() may only be called if in STATE_POINTER state
    if( m_state != STATE_POINTER)
        return -6;
    // check that the element is to be stored in the same transaction that was used to create it
    if( transaction->get_db_transaction() != get_db_transaction())
        return -7;

    return 0;
}

void Db_element_impl_base::post_store()
{
    m_state = STATE_INVALID;
    // DB took over ownership of m_pointer.m_mutable (assuming store() was called before).
    m_pointer.m_mutable = nullptr;
    m_transaction->remove_element( this);
    m_transaction->release();
    m_transaction = nullptr;
}

Transaction_impl* Db_element_impl_base::get_transaction() const
{
    switch( m_state) {
        case STATE_ACCESS:
        case STATE_EDIT:
        case STATE_POINTER:
            return m_transaction;
        case STATE_INVALID:
            return nullptr;
    }
    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

DB::Transaction* Db_element_impl_base::get_db_transaction() const
{
    switch( m_state) {
        case STATE_ACCESS:
        case STATE_EDIT:
        case STATE_POINTER:
            return m_transaction ? m_transaction->get_db_transaction() : nullptr;
        case STATE_INVALID:
            return nullptr;
    }
    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

void Db_element_impl_base::clear_transaction()
{
    if( m_transaction) {
        m_transaction->remove_element( this);
        m_transaction->release();
        m_transaction = nullptr;
    }
    m_access_base.clear_transaction();
}

void Db_element_impl_base::add_journal_flag( DB::Journal_type type)
{
    switch( m_state) {
        case STATE_ACCESS:
        case STATE_INVALID:
            // internal or external programming error (e.g. invalid use of const_cast)
            ASSERT( M_NEURAY_API, false);
            break;
        case STATE_EDIT:
            m_access_base.add_journal_flags( type);
            break;
        case STATE_POINTER:
            // ignore
            break;
    }
}

bool Db_element_impl_base::can_reference_tag( DB::Tag tag) const
{
    switch( m_state) {
        case STATE_INVALID:
            ASSERT( M_NEURAY_API, false);
            return false;
        case STATE_ACCESS:
            ASSERT( M_NEURAY_API, false);
            // fall through is intended
        case STATE_EDIT: {
            DB::Tag this_tag = get_tag();
            DB::Transaction* db_transaction = get_db_transaction();
            return db_transaction->can_reference_tag( this_tag, tag);
        }
        case STATE_POINTER: {
            // We do not know in which scope this DB element will be stored in. To avoid false
            // positives we assume that it will be stored in the local scope.
            DB::Transaction* db_transaction = get_db_transaction();
            DB::Privacy_level this_level = db_transaction->get_scope()->get_level();
            return db_transaction->can_reference_tag( this_level, tag);
        }
    }
    ASSERT( M_NEURAY_API, false);
    return false;
}

const DB::Element_base* Db_element_impl_base::get_db_element_base() const
{
    return m_pointer.m_const;
}

DB::Element_base* Db_element_impl_base::get_db_element_base()
{
    switch( m_state) {
        case STATE_ACCESS:
            // internal or external programming error (e.g. invalid use of const_cast)
            return nullptr; 
        case STATE_INVALID:
        case STATE_EDIT:
        case STATE_POINTER:
            return m_pointer.m_mutable;
    }
    ASSERT( M_NEURAY_API, false);
    return nullptr;
}

} // namespace NEURAY

} // namespace MI

