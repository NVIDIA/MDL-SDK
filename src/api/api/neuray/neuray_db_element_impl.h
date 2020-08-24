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

#ifndef API_API_NEURAY_NEURAY_DB_ELEMENT_IMPL_H
#define API_API_NEURAY_NEURAY_DB_ELEMENT_IMPL_H

#include "i_neuray_db_element.h"

#include <mi/base/iinterface.h>
#include <mi/base/interface_implement.h>
#include <mi/base/interface_merger.h>

#include <boost/core/noncopyable.hpp>
#include <base/lib/log/i_log_module.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>

// see documentation of mi::base::Interface_merger
#include <mi/base/config.h>
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4505 )
#endif

namespace MI {

namespace NEURAY {

class Transaction_impl;
class Db_element_tracker;

/// This class connects API classes with the corresponding DB class.
///
/// API classes are indirectly derived from this class. This class implements the functionality of
/// #IDb_element which can be implemented independently of the actual DB class and/or the public
/// API interface.
///
/// The remaining part (depending on the actual DB class and/or on the public API interface) is
/// implemented in the mixin #Db_element_impl (see below).
class Db_element_impl_base : public IDb_element, public boost::noncopyable
{
public:

    /// Constructor
    ///
    /// Creates an instance in state STATE_INVALID.
    Db_element_impl_base();

    /// Destructor
     ~Db_element_impl_base();

    // methods of IDb_element

    void set_state_access( Transaction_impl* transaction, DB::Tag tag);

    void set_state_edit( Transaction_impl* transaction, DB::Tag tag);

    void set_state_pointer( Transaction_impl* transaction, DB::Element_base* element);

    Db_element_state get_state() const;

    mi::Sint32 store( Transaction_impl* transaction, const char* name, mi::Uint8 privacy);

    DB::Tag get_tag() const;

    void add_journal_flag( DB::Journal_type type);

    /// \note This method does \em not increase the reference count of the return value.
    Transaction_impl* get_transaction() const;

    /// \note This method does \em not increase the reference count of the return value.
    DB::Transaction* get_db_transaction() const;

    void clear_transaction();

    bool can_reference_tag( DB::Tag tag) const;

    /// The tracker needs access to m_access_base and m_parent_element which are not available
    /// from the outside.
    friend class Db_element_tracker;

protected:

    /// Returns a const pointer to the wrapped DB class.
    ///
    /// Provides a unified way to access the wrapped DB class, independent of the state of "this".
    /// Returns \c NULL if the DB element is in state STATE_INVALID.
    ///
    /// \note This method does \em not increase the reference count of the return value.
    const DB::Element_base* get_db_element_base() const;

    /// Returns a mutable pointer to the wrapped DB class.
    ///
    /// Provides a unified way to access the wrapped DB class, independent of the state of "this".
    /// Returns \c NULL if the DB element is in state STATE_INVALID or STATE_ACCESS.
    ///
    /// \note This method does \em not increase the reference count of the return value.
    DB::Element_base* get_db_element_base();

    /// Performs some checks before the DB element is stored in the DB.
    ///
    /// Called by #store() before the DB element is actually store in the DB. Useful if #store()
    /// is re-implemented.
    mi::Sint32 pre_store( Transaction_impl* transaction, mi::Uint8 privacy);

    /// Disconnects this class from the actual DB element.
    ///
    /// Called by #store() after the the DB element has been stored in the DB. Useful if #store()
    /// is re-implemented. Disconnecting the DB element after store is useful to prevent further
    /// editing of the stored object. Upon store the DB gains control of the object and no
    /// external code should further edit it.
    void post_store();

private:

    /// Indicates the state of the DB element
    Db_element_state m_state;

    /// Handle to the DB class if m_state == STATE_ACCESS or == STATE_EDIT, invalid otherwise
    DB::Access_base  m_access_base;

    /// Handle to the DB class if m_state != STATE_INVALID
    ///
    /// If state is STATE_POINTER or STATE_EDIT, use m_mutable. If state is STATE_ACCESS, use
    /// m_const.
    union {
        DB::Element_base*       m_mutable;
        const DB::Element_base* m_const;
    } m_pointer;

    /// The transaction of this element
    Transaction_impl* m_transaction;
};

/// This mixin connects API classes with the corresponding DB class.
///
/// API classes are indirectly derived from this class. The class #Db_element_impl_base (see above)
/// implements the functionality of #IDb_element which can be implemented independently of the
/// actual DB class and/or the public API interface.
///
/// The remaining part (depending on the actual DB class and/or on the public API interface) is
/// implemented in this mixin. The #get_db_element() methods provide access to the DB class
/// itself.
///
/// \tparam I   interface from public/mi/neuraylib, e.g., mi::neuraylib::IOptions
/// \tparam D   DB class from io/scene, e.g., OPTIONS::Options
template <typename I, typename D>
class Db_element_impl
  : public mi::base::Interface_merger<mi::base::Interface_implement<I>, Db_element_impl_base >
{
public:

    /// Returns a const pointer to the wrapped DB class.
    ///
    /// Provides a unified way to access the wrapped class D, independent of the state of "this".
    /// Returns \c NULL if the DB element is in state STATE_INVALID.
    ///
    /// \note This method does \em not increase the reference count of the return value.
    const D* get_db_element() const;

    /// Returns a mutable pointer to the wrapped DB class.
    ///
    /// Provides a unified way to access the wrapped class D, independent of the state of "this".
    /// Returns \c NULL if the DB element is in state STATE_INVALID or STATE_ACCESS.
    ///
    /// \note This method does \em not increase the reference count of the return value.
    D* get_db_element();
};

template <typename I, typename D>
const D* Db_element_impl<I,D>::get_db_element() const
{
    const DB::Element_base* pointer = this->get_db_element_base();
    if( !pointer)
        LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "Invalid use of DB element after ITransaction::store().");
    return static_cast<const D*>( pointer);
}

template <typename I, typename D>
D* Db_element_impl<I,D>::get_db_element()
{
    DB::Element_base* pointer = this->get_db_element_base();
    if( !pointer)
        LOG::mod_log->error( M_NEURAY_API, LOG::Mod_log::C_DATABASE,
            "Invalid use of DB element after ITransaction::store() (or invalid const_cast).");
    return static_cast<D*>( pointer);
}

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_DB_ELEMENT_IMPL_H
