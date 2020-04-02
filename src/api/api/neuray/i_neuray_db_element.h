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
 ** \brief Header for the IDb_element declaration.
 **/

#ifndef API_API_NEURAY_I_NEURAY_DB_ELEMENT_H
#define API_API_NEURAY_I_NEURAY_DB_ELEMENT_H

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <base/data/db/i_db_tag.h>
#include <base/data/db/i_db_journal_type.h>
#include <base/data/serial/i_serial_classid.h>

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace DB { class Transaction; class Element_base; }

namespace NEURAY {

class Transaction_impl;

/// Possible states for Db_element_impl_base.
///
/// On construction, elements start in state STATE_INVALID. One of the four set_state_*()
/// methods has to be used to initialize the element properly and to switch it to any of the
/// four other states. Elements in state STATE_POINTER change to STATE_INVALID during #store().
enum Db_element_state
{
    STATE_ACCESS,    ///< Element was retrieved from the DB, const access.
    STATE_EDIT,      ///< Element was retrieved from the DB, mutable access.
    STATE_POINTER,   ///< Element is not yet associated with the DB.
    STATE_INVALID    ///< Invalid state, right after construction or after store()
};

/// This interface indicates DB elements and distinguishes them from ordinary classes.
///
/// Used at runtime to find out if an interface can actually be stored in the DB. For example,
/// IOptions can be stored, while INumber cannot be stored.
class IDb_element : public
    mi::base::Interface_declare<0xf50a60fe,0x6ff3,0x4133,0x81,0x4f,0x7c,0xcf,0xaf,0xb8,0xcc,0x51>
{
public:
    /// Connects the API class with an DB element in the DB (const/access).
    ///
    /// The caller is responsible for not to handing out mutable pointers to \c *this.
    ///
    /// \param transaction   The transaction.
    /// \param tag           The tag of the DB element to connect.
    virtual void set_state_access( Transaction_impl* transaction, DB::Tag tag) = 0;

    /// Connects the API class with an DB element in the DB (mutable/edit).
    ///
    /// \param transaction   The transaction.
    /// \param tag           The tag of the DB element to connect.
    virtual void set_state_edit( Transaction_impl* transaction, DB::Tag tag) = 0;

    /// Connects the API class with an DB element in memory.
    ///
    /// The element is not yet associated with the DB in any way (see #store()).
    ///
    /// \param transaction   The transaction.
    /// \param element       The DB element to connect.
    virtual void set_state_pointer( Transaction_impl* transaction, DB::Element_base* element) = 0;

    /// Return the state of this class.
    virtual Db_element_state get_state() const = 0;

    /// Stores the element in the DB.
    ///
    /// May only be called if the element is in state STATE_POINTER.
    ///
    /// \return
    ///           -  0: Success.
    ///           - -6: The element is already in the DB or invalid (state is not STATE_POINTER).
    ///           - -7: The element is to be stored in a different transaction from which it was
    ///                 created
    ///           - -9: There is already an element of name \p name and overwriting elements of that
    ///                 type is not supported. This applies to elements of type
    ///                 #mi::neuraylib::IModule, #mi::neuraylib::IMaterial_definition, and
    ///                 #mi::neuraylib::IFunction_definition.
    virtual mi::Sint32 store(
        Transaction_impl* transaction, const char* name, mi::Uint8 privacy) = 0;

    /// Returns the tag of the element.
    ///
    /// Returns DB::Tag() if the element is not in state STATE_ACCESS or STATE_EDIT.
    virtual DB::Tag get_tag() const = 0;

    /// Returns the corresponding transaction.
    ///
    /// Returns \c NULL if the DB element is in state STATE_INVALID.
    /// \note This method does \em not increase the reference count of the return value.
    virtual Transaction_impl* get_transaction() const = 0;

    /// Returns the corresponding DB transaction.
    ///
    /// Returns \c NULL if the DB element is in state STATE_INVALID.
    /// \note This method does \em not increase the reference count of the return value.
    virtual DB::Transaction* get_db_transaction() const = 0;

    /// Clears the transaction and DB transaction.
    ///
    /// This method can be used as a safety measure to avoid unintended uses of the transaction
    /// pointer. E.g., instances of User_class_api_wrapper in state STATE_ACCESS are shared across
    /// transactions.
    virtual void clear_transaction() = 0;

    /// Adds the given journal flag.
    ///
    /// If the element is in state STATE_EDIT, the given journal flag is added.
    /// If the element is in state STATE_POINTER, the call is ignored (the element is not yet in
    /// the DB and if it is stored it is a new element anyway).
    ///
    /// \pre The element is in state STATE_EDIT or STATE_POINTER.
    virtual void add_journal_flag( DB::Journal_type type) = 0;

    /// Checks whether this DB element can reference the given tag.
    ///
    /// If the element is in state STATE_EDIT, it uses the scope of this version as reference level.
    /// If the element is in state STATE_POINTER, it uses the local scope as reference level.
    ///
    /// \pre The element is in state STATE_EDIT or STATE_POINTER.
    virtual bool can_reference_tag( DB::Tag tag) const = 0;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_I_NEURAY_DB_ELEMENT_H
