/***************************************************************************************************
 * Copyright (c) 2008-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_ACCESS_H
#define BASE_DATA_DB_I_DB_ACCESS_H

#include "i_db_element.h"
#include "i_db_info.h"
#include "i_db_journal_type.h"

#include <base/system/main/i_assert.h>

namespace MI {

namespace SCHED { class Job; }

namespace DB {

class Transaction;

/// Base class for Access and Edit smart pointers.
///
/// The Access and Edit smart pointers simplify accessing and editing database elements and hide the
/// details of DB::Transaction and DB::Info. This base class implements the full Access and Edit
/// logic in a shared and untyped way. It is used by the API. Other users are most likely better off
/// using the derived classes Access<T> and Edit<T>.
///
/// An instance of Access_base represents an Edit whenever m_edit is true. Edits cannot be copied
/// or assigned, they are treated as Access if they appear as source of such an operation (and the
/// target is Access, or Access_base). Accesses can not be assigned to Edits.
///
/// See #mi::neuraylib::ITransaction for semantics of concurrent accesses to the very same database
/// element within one particular transaction.
///
/// Implementation of non-inline methods is in base/data/dblight/dblight_ačcess.cpp.
class Access_base
{
private:
    /// Helper type for the conversion to bool via this intermediate type (such that the bool is
    /// not implicitly converted to another type).
    using unknown_bool_type = bool (Access_base::*)() const;

public:
    /// \name Constructors etc.

    /// Constructor.
    Access_base() = default;

    /// Copy constructor.
    ///
    /// Ignores the \c m_is_edit flag on \p other, i.e., copies an Edit as if it as an Access.
    Access_base( const Access_base& other);

    /// Assignment operator.
    ///
    /// Ignores the \c m_is_edit flag on \p other, i.e., assigns an Edit as if it as an Access.
    /// Asserts if \c m_is_edit on \c *this is set.
    Access_base& operator=( const Access_base& other);

    /// Destructor.
    ///
    /// Finishes edits, see #Transaction::finish_edit().
    ~Access_base();

    //@}
    /// \name Properties
    //@{

    /// Indicates whether the access/edit is valid.
    bool is_valid() const { return m_info != nullptr; }

    /// Conversion to bool, returns \c true for valid accesses/edits.
    operator unknown_bool_type() const { return is_valid() ? &Access_base::is_valid : nullptr; }

    /// Indicates whether this instance represents an edit (or access).
    bool is_edit() const { return m_is_edit; }

    /// Returns the referenced tag.
    Tag get_tag() const { return m_tag; }

    /// Returns the corresponding transaction. Can be \c NULL after construction. RCS:NEU
    Transaction* get_transaction() const { return m_transaction; }

    /// Returns the referenced tag version.
    Tag_version get_tag_version() const;

    /// Returns the referenced DB element (or DB job result). Can be \c NULL. RCS:NEU
    const Element_base* get_base_ptr() const { return m_element; }

    /// Returns the referenced DB element (or DB job result). Can be \c NULL. RCS:NEU
    Element_base* get_base_ptr() { return m_is_edit ? m_element : nullptr; }

    /// Returns the referenced DB job, or \c NULL if this does not reference a DB job. RCS:NEU
    const SCHED::Job* get_job() const;

    /// Returns the journal flags accumulated so far.
    ///
    /// Always JOURNAL_NONE for accesses.
    Journal_type get_journal_flags() const { return m_journal_type; }

    //@}
    /// \name
    //@{

    /// Sets an access to a given tag, possibly within a new transaction.
    ///
    /// Note that this method does not allow to restore the state after construction. A
    /// default-constructed value for \p tag is valid, but a \c NULL value for \p transaction keeps
    /// the current transaction. Use the copy constructor or assignment operator to reset the
    /// state. (Access has a reset() method, but not Edit.)
    ///
    /// \param tag           The new tag.
    /// \param transaction   The new transaction. Pass \c NULL to keep the current transaction.
    ///                      RCS:NEU
    /// \param id            The expected class ID of the new tag. Used for assertions.
    /// \return              The referenced database element. RCS:NEU
    Element_base* set_access(
        Tag tag, Transaction* transaction, SERIAL::Class_id id);

    /// Sets an edit to a given tag, possibly within a new transaction.
    ///
    /// \param tag           The new tag.
    /// \param transaction   The new transaction. Pass \c NULL to keep the current transaction.
    ///                      RCS:NEU
    /// \param id            The expected class ID of the new tag. Used for assertions.
    /// \param journal_type  The initial journal flags of this edit.
    /// \return              The referenced database element. RCS:NEU
    Element_base* set_edit(
        Tag tag, Transaction* transaction, SERIAL::Class_id id, Journal_type journal_type);

    /// Sets the journal flags of an edit.
    ///
    /// Local copy of the journal flags. Transmitted to the database by the destructor.
    void set_journal_flags( Journal_type journal_type)
    {
        MI_ASSERT( m_is_edit);
        m_journal_type = journal_type;
    }

    /// Adds journal flags to the journal flags of an edit.
    ///
    /// Local copy of the journal flags. Transmitted to the database by the destructor.
    void add_journal_flags( Journal_type journal_type)
    {
        MI_ASSERT( m_is_edit);
        m_journal_type.add_journal( journal_type);
    }

    /// Clears the transaction pointer of an access.
    ///
    /// This method is used by the API to avoid unintended use of the transaction pointer. Under
    /// some circumstances, instances of this class are shared across transactions. This method is
    /// a safety measure to avoid unintended use of the transaction pointer in such cases.
    ///
    /// This method must only be called for accesses, not for edits.
    void clear_transaction();

    //@}

protected:
    /// Resets the access or edit to the default-constructed state with exception of the
    /// transaction.
    ///
    /// Finishes edits (and switches back to access).
    void cleanup();

private:
    /// Sets this access to point to the same DB element as another one.
    ///
    /// Used to implement copy constructor and assignment operator.
    ///
    /// \return              The referenced database element. RCS:NEU
    Element_base* set_access( const Access_base& other);

    Element_base* m_element = nullptr;      ///< The referenced DB element.
    Transaction* m_transaction = nullptr;   ///< The corresponding transaction.
    Info* m_info = nullptr;                 ///< The referenced Info.
    Tag m_tag;                              ///< The referenced tag.
    Journal_type m_journal_type;            ///< The journal flags accumulated so far.
    bool m_is_edit = false;                 ///< Edit or access
};

/// Smart pointer for accesses to database elements.
///
/// \note This class only uses assertions to detect mismatches between the template parameters and
///       the type of the database element. Input validation must be done upfront by other means,
///       e.g., DB::Transaction::get_class_id().
///
/// Example:
///
/// \code
/// {
///     Transaction* transaction = ...;
///     // Obtain a (const) smart pointer to a texture identified by the tag 5.
///     Access<Texture> texture( 5, transaction);
///     // Retrieve some property of the texture.
///     int width = texture->get_width();
///     // Leaving the current scope will release the texture held by the smart pointer.
///  }
/// \endcode
template <class T>
class Access : public Access_base
{
public:
    /// Default constructor.
    Access() = default;

    /// Constructs an access to a given tag.
    ///
    /// \param tag           The tag.
    /// \param transaction   The transaction. RCS:NEU
    Access( Tag tag, Transaction* transaction)
    {
        set_access( tag, transaction, T::id);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Copy constructor.
    Access( const Access_base& other)
      : Access_base( other)
    {
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Assignment operator.
    Access<T>& operator=( const Access_base& other)
    {
        Access_base::operator=( other);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
        return *this;
    }

    /// Destructor.
    ~Access() = default;

    /// Sets an access to a given tag, possibly within a new transaction.
    ///
    /// Note that this method does not allow to restore the state after construction. A
    /// default-constructed value for \p tag is valid, but a \c NULL value for \p transaction keeps
    /// the current transaction. Use #reset() for that.
    ///
    /// \param tag           The new tag to access.
    /// \param transaction   The new transaction. Pass \c NULL to keep the current transaction.
    ///                      RCS:NEU
    /// \param id            The expected class ID of the new tag. Used for assertions.
    void set( Tag tag = Tag(), Transaction* transaction = nullptr)
    {
        set_access( tag, transaction, T::id);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Resets the access to the default-constructed state with exception of the transaction.
    void reset() { cleanup(); }

    /// Member access operator (const).
    ///
    /// \return   The referenced DB element. Might not be valid beyond the lifetime of this
    ///           smart pointer. RCS:NEU
    const T* operator->() const
    {
        const T* ptr = static_cast<const T*>( get_base_ptr());
        MI_ASSERT( ptr);
        return ptr;
    }

    /// Returns the referenced DB element (const).
    ///
    /// \return   The referenced DB element. Might not be valid beyond the lifetime of this
    ///           smart pointer. RCS:NEU
    const T* get_ptr() const { return static_cast<const T*>( get_base_ptr()); }
};

/// Smart pointer for edits of database elements.
///
/// \note This class only uses assertions to detect mismatches between the template parameters and
///       the type of the database element. Input validation must be done upfront by other means,
///       e.g., DB::Transaction::get_class_id().
///
/// \note It is not possible to edit database elements representing job results.
///
/// \note The assertions on #is_edit() are supposed to catch misuse due to slicing: Due to the
///       inheritance of Edit from Access it is possible to create an Access<T> reference or an
///       Access_base reference of an instances of Edit. Calling #Access<T>::set() or
///       Access_base::set_access() will turn the Edit into an Access under the hood. This creates
///       problems later e.g. when calling the non-const overload of Edit::get_ptr().
///
/// Example:
///
/// \code
/// {
///     Transaction* transaction = ...;
///     // Obtain a mutable smart pointer to a texture identified by the tag 5.
///     Edit<Texture> texture( 5, transaction);
///     // Set some property of the texture.
///     texture->set_width( 1024);
///     // Leaving the current scope will finish the edit and release the texture held by the smart
///     // pointer.
///  }
/// \endcode
template <class T>
class Edit : public Access<T>
{
public:
    using Access_base::get_base_ptr;

    /// Default constructor.
    Edit()
    {
        this->set_edit( Tag(), nullptr, T::id, JOURNAL_ALL);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Constructs an edit to a given tag.
    ///
    /// \param tag           The new tag.
    /// \param transaction   The new transaction. RCS:NEU
    /// \param journal_type  The initial journal flags of this edit.
    Edit( Tag tag, Transaction* transaction, Journal_type journal_type = JOURNAL_ALL)
    {
        this->set_edit( tag, transaction, T::id, journal_type);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Constructs an edit from an access.
    ///
    /// This can be used to avoid creating an expensive edit upfront: start with an access and
    /// create the edit later only if necessary.
    ///
    /// \param other         The access object.
    /// \param journal_type  The initial journal flags of this edit.
    Edit( const Access<T>& other, Journal_type journal_type = JOURNAL_ALL)
    {
        this->set_edit( other.get_tag(), other.get_transaction(), T::id, journal_type);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Sets an edit to a given tag, possibly within a new transaction.
    ///
    /// Note that this method does not allow to restore the state after construction. A
    /// default-constructed value for \p tag is valid, but a \c NULL value for \p transaction keeps
    /// the current transaction. Use the copy constructor or assignment operator to reset the
    /// state.
    ///
    /// \param tag           The new tag to edit.
    /// \param transaction   The new transaction. Pass \c NULL to keep the current transaction.
    ///                      RCS:NEU
    /// \param journal_type  The initial journal flags of this edit.
    void set(
        Tag tag = DB::Tag(),
        Transaction* transaction = nullptr,
        Journal_type journal_type = JOURNAL_ALL)
    {
        MI_ASSERT( this->is_edit());
        this->set_edit( tag, transaction, T::id, journal_type);
        MI_ASSERT( !T::id || !get_base_ptr() || get_base_ptr()->is_type_of( T::id));
    }

    /// Member access operator (mutable).
    ///
    /// \return   The referenced DB element. Might not be valid beyond the lifetime of this
    ///           smart pointer. RCS:NEU
    T* operator->()
    {
        MI_ASSERT( this->is_edit());
        T* ptr = static_cast<T*>( get_base_ptr());
        MI_ASSERT( ptr);
        return ptr;
    }

    /// Member access operator (const).
    ///
    /// \return   The referenced DB element. Might not be valid beyond the lifetime of this
    ///           smart pointer. RCS:NEU
    const T* operator->() const
    {
        MI_ASSERT( this->is_edit());
        const T* ptr = static_cast<const T*>( get_base_ptr());
        MI_ASSERT( ptr);
        return ptr;
    }

    /// Returns the referenced DB element (mutable).
    ///
    /// \return   The referenced DB element. Might not be valid beyond the lifetime of this
    ///           smart pointer. RCS:NEU
    T* get_ptr()
    {
        MI_ASSERT( this->is_edit());
        return static_cast<T*>( get_base_ptr());
    }

    /// Returns the referenced DB element (const).
    ///
    /// \return   The referenced DB element. Might not be valid beyond the lifetime of this
    ///           smart pointer. RCS:NEU
    const T* get_ptr() const
    {
        MI_ASSERT( this->is_edit());
        return static_cast<const T*>( get_base_ptr());
    }

private:
    /// Delete copy constructor to avoid accidental use.
    Edit( const Edit<T>& other) = delete;

    /// Delete assignment operator to avoid accidental use.
    Edit<T>& operator=( const Access_base& other) = delete;
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_ACCESS_H
