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

/** \file i_db_access.h
 ** \brief This declares the access classes used to access database elements
 **
 ** This file declares the access classes used to access database elements.
 **/

#ifndef BASE_DATA_DB_I_DB_ACCESS_H
#define BASE_DATA_DB_I_DB_ACCESS_H

#include "i_db_element.h"
#include <base/lib/log/i_log_assert.h>

namespace MI
{

namespace SCHED { class Job; }

namespace DB
{

class Transaction;
class Info;
template <class T> class Edit;

/// Base class for Access smart pointers. This is needed to allow
/// obtaining a specialized Access pointer from an Access pointer to
/// Element_base.
///
/// This base class implements the full Access and Edit logic in a
/// shared and untyped way.  Use with great care and prefer the
/// Access<T> and Edit<T> templates below for a type safe way of
/// accessing DB elements whenever possible. This class is useful to
/// efficiently implement things like Access and Edit handling in the
/// neuray C++ API or access caches, which guarantee the type safety
/// themselves.
///
/// An Access_base represents an Edit whenever m_edit is true. Edits
/// cannot be copied or assigned, but they can be copied or assigned
/// to accesses. So, copy and assignment in Access_base have special
/// case handling to keep the m_edit only on the source argument and
/// change the target to be an access.
///
class Access_base
{
  public:
    /// Default constructor
    Access_base();

    /// Copy constructor
    /// Edits become accesses upon copying
    Access_base( const Access_base& access);

    /// Assignment operator
    /// Edits become accesses upon copying
    Access_base& operator=( const Access_base& access);

    /// Destructor
    ~Access_base();

    /// Return the pointer to the job, if this is a job or NULL, otherwise.
    /// \return                         The job
    const SCHED::Job* get_job() const;

    /// Get the transaction this object belongs to
    /// \return                         The transaction
    Transaction* get_transaction() const { return m_transaction; }

    /// Return the tag this object references.
    /// \return                         The tag
    Tag get_tag() const { return m_tag; }

    /// Return the base pointer to the DB element
    const Element_base* get_base_ptr() const { return m_pointer; }

    /// Return the base pointer to the DB element
    Element_base* get_base_ptr() { return m_pointer; }

    /// Check if the pointer is set to a valid tag or to 0.
    /// \return                         True if it is set to a valid tag, 0 otherwise.
    bool is_valid() const { return m_info != nullptr; }

    /// Return whether pointer is set to a valid tag.
    /// \return                         True if it is set to a valid tag, 0 otherwise.
    bool operator!() const { return !this->is_valid(); }

    /// Get the unique id of a certain tag version. The result of a database lookup on a certain
    /// tag depends on the asking transaction and may return different versions for different
    /// transactions. For caching data derived from the tag, such a unique id uniquely
    /// identifies the actual version of the tag obtained from a certain transaction. This
    /// means, that it may be used to identify this version. The database guarantees, that any
    /// legal change to the data (done through an edit) will also change this id.
    /// NOTE: The value obtained is valid host locally, only.
    /// \return                 The tag version
    Tag_version get_tag_version() const;

    /// Set this access to point to a new tag, possibly within a new transaction.
    ///
    /// Will set the class to point to a new element. Note that the execution
    /// context stays the same as before. This will unpin an old element
    /// if necessary.
    /// If the argument is 0, then only the old element will be unpinned (if it
    /// was pointing somewhere) because 0 is no valid tag.
    ///
    /// \param tag                      The new tag
    /// \param transaction              The new transaction
    /// \param id                       The expected class id of the new tag.
    /// \param wait                     Should the access wait if the tag is not yet available
    ///                                 or return immediately?
    Element_base* set_access( Tag tag,
                              Transaction* transaction,
                              SERIAL::Class_id id,
                              bool wait = true);

private:
    /// Set this access to point to the same as source
    ///
    /// Used to implement copy constructor and assignment operator.
    ///
    /// \param source                   The source for copying.
    Element_base* set_access( const Access_base& source);

public:
    /// Set this to edit a new tag, possibly within a new transaction.
    ///
    /// Will set the class to point to a new element. Note that the execution
    /// context stays the same as before. This will unpin an old element
    /// if necessary.
    /// If the argument is 0, then only the old element will be unpinned (if it
    /// was pointing somewhere) because 0 is no valid tag.
    ///
    /// \param tag                      The new tag
    /// \param transaction              The new transaction
    /// \param id                       The expected class id of the new tag.
    /// \param journal type             The initial journal flags of this edit.
    Element_base* set_edit( Tag tag,
                            Transaction* transaction,
                            SERIAL::Class_id id,
                            Journal_type journal_type);

    /// Returns true if this is an edit and false if this is an access.
    bool is_edit() const { return m_is_edit; }

    /// Return the journal flags
    ///
    /// \return                         The journal flags accumulated during this edit.
    Journal_type get_journal_flags() const { return m_journal_type; }

    /// Set the journal flags of the edit. This can be done until the edit goes out of scope or set
    /// is used to let it point to a different tag. This is meant to be used, when journal flags
    /// are not known from the beginning but become clear, later. This is efficient and can be used
    /// many times and always overwrites the old flags. Together with get_journal_flags it can be
    /// used to add the flags successively. Precondition: this has to be an edit.
    ///
    /// \param type                     The new set of journal flags
    void set_journal_flags( Journal_type type) { 
        ASSERT(M_DB, m_is_edit);
        m_journal_type = type;
    }

    /// Add journal flags to the edit. This can be done until the edit goes out of scope or set
    /// is used to let it point to a different tag. This is meant to be used, when journal flags
    /// are not known from the beginning but become clear, later. This is efficient and can be used
    /// many times to aggregate flags. Precondition: this has to be an edit.
    ///
    /// \param type                     The new set of journal flags
    void add_journal_flags( Journal_type type) { 
        ASSERT(M_DB, m_is_edit);
        m_journal_type.add_journal( type);
    }

    /// Clear the transaction pointer.
    ///
    /// This method is used by the API to avoid uninteded use of the transaction pointer.
    /// Under some circumstances, Access_base's are shared across transactions. This method
    /// is a safety measure to avoid unintended use of the transaction pointer in such cases.
    ///
    /// Note that this method must not be called if m_is_edit is true, because cleanup()
    /// will use the transaction pointer in this case.
    void clear_transaction();
    
private:
    /// Unpin old m_info with proper edit cleanup handling before the
    /// access/edit is used for a new tag. No longer an edit afterwards.
    void cleanup();

private:
    /// The pointer to the DB element
    Element_base* m_pointer;

    /// The tag for the access
    Tag           m_tag;

    /// The transaction for the access
    Transaction*  m_transaction;

    /// The info for the accessed version.
    Info*         m_info;

    /// The journal type for changes
    Journal_type  m_journal_type;

    /// The state if this is an edit or an access.
    bool          m_is_edit;
};

/// This is used by jobs and applications to read database elements. It will ensure that the type of
/// the accessed tag and the type of the pointer match. The pointers hides all pinning and unpinning
/// on elements in the cache. It also hides the selection of the correct version of the tag
/// depending on the database context.
/// Example for the usage:
///  {
///      ...
///
///       // Get a read-only pointer to a texture identified by the tag 5. The
///      // database will ensure that tag 5 is of type Texture!
///      Access<Texture> texture(5, transaction);
///
///      // Use the smart pointer in the same way a pointer to a texture would
///      // be used. Note that trying to change the data would result in a
///     // compile time error message!
///      int width = texture->get_width();
///
///      ...
///
///       // Leaving the current scope will unpin the texture
///  }
///
/// The WAIT template parameter decides, if the access will wait for the element to be created
/// or transmitted over the network etc. if it is not locally available
template <class T, bool WAIT = true> class Access : public Access_base
{
  public:
    /// Default constructor.
    Access() {}

    /// Constructor.
    ///
    /// \param tag                      The tag to access
    /// \param transaction              The transaction for the access
    Access( const Typed_tag<T>& tag, Transaction* transaction)
    {
        set_access(tag.get_untyped(), transaction, T::id, WAIT);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Copy constructor.
    ///
    /// \param source                   The source access object.
    Access( const Access_base& source)
        : Access_base( source)
    {
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Copy constructor.
    ///
    /// \param source                   The source access object.
    Access( const Access<T, WAIT>& source)
        : Access_base( source)
    {
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

     /// Destructor
    ~Access() {}

    /// Will set the class to point to a new element. Note that the execution context stays the
    /// same as before. This will unpin an old element if necessary.
    /// If the argument is 0, then only the old element will be unpinned (if it was pointing
    /// somewhere) because 0 is no valid tag.
    ///
    /// \param tag                      The new tag
    /// \param transaction              The new transaction
    void set( Typed_tag<T> const & tag = Typed_tag<T>(),
              Transaction* transaction = 0)
    {
        set_access(tag.get_untyped(), transaction, T::id, WAIT);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Set the access object to the same values as the source
    ///
    /// \param source                   The source access object.
    void operator=( const Access<T,WAIT>& source)
    {
        Access_base::operator=( source);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Set the access object to the same values as the source
    ///
    /// \param source                   The source access object.
    void operator=( const Access_base& source)
    {
        Access_base::operator=( source);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Access operator. This is const because it does not allow jobs and applications to change
    /// database elements.
    ///
    /// \return                         The pointer this points to.
    const T* operator->() const
    {
        ASSERT(M_DB, get_base_ptr()); // catch attempt to dereference an unset Access<T>
        return static_cast<const T*>( get_base_ptr());
    }

    /// Retrieve the internal object pointer. Useful for convenience only. Note that this method
    /// should be used carefully, since it circumvents this smart pointer wrapper - as soon as the
    /// destructor was executed the retrieved pointer is undefined.
    ///
    /// \return                         The pointer this points to.
    const T* get_ptr() const { return static_cast<const T*>( get_base_ptr()); }
};

/// This is used by jobs and applications to edit database elements. It will ensure that the type of
/// the accessed tag and the type of the pointer match. The pointers hides all pinning and unpinning
/// on elements in the cache. It also hides the selection of the correct version of the tag
/// depending on the database context.
///
/// Example for the usage:
///  {
///      ...
///
///      // Get a read-only pointer to a texture identified by the tag 5. The database will ensure
///      // that tag 5 is of type Texture!
///      Edit<Texture> texture(context, 5);
///
///       // Use the smart pointer in the same way a pointer to a texture would be used
///      texture->set_width(100);
///
///      ...
///
///       // Leaving the current scope will unpin the texture
///  }
///
/// Note that it is not possible to edit elements created by jobs.
template <class T> class Edit : public Access<T>
{
  public:
    using Access<T>::get_base_ptr;

    /// Default constructor.
    Edit()
    {
        this->set_edit(Tag(), 0, T::id, JOURNAL_ALL);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Constructor.
    ///
    /// \param tag                      The tag to edit
    /// \param transaction              The transaction for the access
    /// \param journal_type             The type for journal entries
    Edit( const Typed_tag<T>& tag,
          Transaction* transaction,
          Journal_type journal_type = JOURNAL_ALL)
    {
        this->set_edit(tag.get_untyped(), transaction, T::id, journal_type);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }


    /// Constructor. This can be used to get an Edit pointer from an Access pointer, when it gets
    /// clear that the element has to be edited.
    ///
    /// \param source                   The source access object
    /// \param journal_type             The type for journal entries
    Edit( const Access<T>& source,
          Journal_type journal_type = JOURNAL_ALL)
    {
        this->set_edit(source.get_tag(), source.get_transaction(), T::id, journal_type);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }

    /// Destructor. Will unpin an element, if necessary.
    ~Edit()
    {
        // Note: this assert catches mis-uses of Edit<...> and Access<...>
        // where, for example, an Access reference is created for an Edit
        // (possible because of the inheritance of Edit from Access) and
        // on that access the set method is called. This essentially
        // changes the Edit under the hood to an access, which should not
        // happen.
        ASSERT(M_DB, this->is_edit());
    }


    /// Set the class to point to a new element. Note that the execution context stays the same
    /// as before. This will unpin an old element if necessary.
    /// If the argument is 0, then only the old element will be unpinned (if it was pointing
    /// somewhere) because 0 is no valid tag.
    ///
    /// \param tag                      The tag to edit
    /// \param transaction              The transaction for the access
    /// \param journal_type             The type for journal entries
    void set( const Typed_tag<T>& tag = Tag(),
              Transaction* transaction = 0,
              Journal_type journal_type = JOURNAL_ALL)
    {
        this->set_edit(tag.get_untyped(), transaction, T::id, journal_type);
        ASSERT(M_DB, !T::id || !get_base_ptr() || get_base_ptr()->is_type_of(T::id));
    }


    /// Access operator. This is not const because it allows jobs and applications to change
    /// database elements. This means when setting an Edit class element to an element, a new
    /// version of the element will be created.
    ///
    /// \return                         The pointer this points to.
    T* operator->()
    {
        ASSERT(M_DB, get_base_ptr()); // catch attempt to dereference an unset Edit<T>
        return static_cast<T*>(get_base_ptr());
    }

    /// Access operator. This is const and callers are not allowed to change the database
    /// element. Useful if the Edit is a member of a class and you want to call a const method
    /// of T from a const method of that class.
    ///
    /// \return                         The pointer this points to.
    const T* operator->() const
    {
        ASSERT(M_DB, get_base_ptr()); // catch attempt to dereference an unset Edit<T>
        return static_cast<const T*>(get_base_ptr());
    }

    /// Retrieve the internal object pointer. Useful for convenience only.
    /// Note that this method should be used carefully, since it circumvents this smart pointer
    /// wrapper - as soon as the destructor was executed the retrieved pointer is undefined.
    ///
    /// \return                         The pointer this points to.
    T* get_ptr() { return static_cast<T*>(get_base_ptr()); }

    /// Retrieve the internal object pointer. Useful for convenience only.
    /// Note that this method should be used carefully, since it circumvents this smart pointer
    /// wrapper - as soon as the destructor was executed the retrieved pointer is undefined.
    ///
    /// \return                         The pointer this points to.
    const T* get_ptr() const { return static_cast<const T*>(get_base_ptr()); }

private:
    // Copying and assignment is forbidden to avoid the dangerous creation of
    // temporary Edit-pointers. Use 'set' function instead.
    // Copy constructor.
    Edit( const Edit<T>& source);

    // Assignment operators
    void operator=( const Edit<T>& source);

    // Assignment operators
    void operator=( const Access<T>& source);

    // Assignment operators
    Access_base& operator=( const Access_base& source);
};


} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_ACCESS_H
