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

/** \file i_db_element.h
 ** \brief This declares the database element classes.
 **
 ** This file contains the pure virtual base class for any database element as well as templates
 ** which can be used to simplify the implementation of a database element.
 **/

#ifndef BASE_DATA_DB_I_DB_ELEMENT_H
#define BASE_DATA_DB_I_DB_ELEMENT_H

#include "i_db_tag.h"
#include "i_db_journal_type.h"

#include <base/data/serial/i_serial_serializable.h>
#include <string>

namespace MI
{

namespace DB
{

class Transaction;

/// Each database element is derived from the class Element. An element provides a common interface
/// for the access to certain functionality.
/// Attributes may be attached to elements, but the API is not clear, yet.
class Element_base : public SERIAL::Serializable
{
  public:
    /// Destructor.
    virtual ~Element_base() { }

    /// Return the approximate size in bytes of the element including all its substructures. This is
    /// used to make decisions about garbage collection.
    ///
    /// \return                         The size of the element
    virtual size_t get_size() const;

    /// This is used by the database when it needs to create a new version of a tag. It will create
    /// and return a full copy of the element.
    ///
    /// \return                         The new copy of the element
    virtual Element_base* copy() const = 0;

    /// Check, if this object is of the given type. This is true, if either the class id of this
    /// object equals the given class id, or the class is derived from another class which has the
    /// given class id.
    ///
    /// \param id                       The class id to check
    /// \return                         True, if the class is derived from the class with the id
    virtual bool is_type_of(
        SERIAL::Class_id id) const
    {
        return id == 0 ? true : false;
    }

    /// Return which journal flags could be changed in the database element. This is anded later
    /// with the Edits' journal flags to reduce the size of the journals.
    ///
    /// \return                         The journal flags
    virtual Journal_type get_journal_flags() const
    {
        return Journal_type();
    }

    /// The class id for the base class
    static const SERIAL::Class_id id = 0;

    /// The bundle function is used to query which tags will be most probably needed on a host
    /// needing this element. It accepts an array of tags and the size of the array as input. It
    /// should store as many tags as necessary and possible into the array. Note that the tags
    /// should be put into the array sorted by importance.
    /// The return value gives back how many tags have actually been put into the array.
    /// This function may not be helpful for many element classes and thus a default implementation
    /// which does nothing is provided.
    /// Important note: From within the bundle call no accesses to database elements may be done!
    ///
    /// \param results                  A place to store the tags to be bundled
    /// \param size                     The size of the array and as such the limit for the number
    ///                                 of elements which can be bundled
    /// \return                         The actual number of bundled elements.
    virtual Uint bundle(
        Tag* results,
        Uint size) const
    {
        return 0;
    }

    /// Create a list of references this element has stored. The default implementation is supplied
    /// because most elements don't need this.
    ///
    /// \param result                   Store the referenced tags here
    virtual void get_references(
        Tag_set*   result) const
    { }

    /// This function will be called by the database before a new version of a
    /// database element is stored in the database. This function can be used
    /// to check if the members of the database element are consistent and to
    /// fix them if necessary. It is also possible to create new jobs and pass
    /// to them the tag that is used to store this element.
    /// \param trans The current transaction.
    /// \param own_tag The tag that will be used to store this object.
    virtual void prepare_store(
        Transaction* trans,
        Tag own_tag)
    { }

    /// This function specifies whether a data-store element should be distributed
    /// to all nodes via multicasting
    /// \return true if multicast distribution should be enabled.
    virtual bool get_send_to_all_nodes()
    {
        return true;
    }

    /// Indicates how the database element is stored on other nodes in the cluster.
    ///
    /// If disabled, all nodes that receive the database element will keep it in main memory. If
    /// enabled, the owners, i.e., a subset of the nodes that keep a copy of the database element to
    /// fulfil the configured redundancy, will keep a copy of the database element on disk. All 
    /// other nodes will discard the database element. Note that the database element will always
    /// be kept in memory on the host that created or edited it, and on hosts that explicitly
    /// accessed it.
    virtual bool get_offload_to_disk()
    {
        return false;
    }

    /// Return a human readable version of the class id. While the id is enough to identify each
    /// Serializable's type, is it not exactly human readable to read 0x5f43616d for Camera.
    ///
    /// \return                         The name of the class
    virtual std::string get_class_name() const = 0;
};

/// This template defines some functions for derived classes. This makes it easier to define a new
/// Element class:
/// If the class shall directly derived from from Element_base just do it like this:
///     class Test_element : public Element<Test_element, 1000>
///         ...
/// If it is derived from some other class which is derived from Element_base, then do it like this:
///     class Test_element : public Element<Test_element, 1000, Object>
///         ...
/// Note that abstract base classes do not have to be derived from Element but may be derived from
/// Element_base. This is because they will not be instantiated directly.
template <class T, SERIAL::Class_id ID = 0, class P = Element_base>
class Element : public P
{
  public:
    typedef Element<T,ID,P> Element_t;

    /// Class id for this class
    static const SERIAL::Class_id id = ID;

    /// construct an instance of this class
    static SERIAL::Serializable* factory()
    {
        return new T;
    }

    /// default constructor needed because of the copy constructor
    Element();

    /// copy constructor which will call the copy constructor of the base class
    ///
    /// \param source                   The source to copy
    Element(const Element& source);

    /// get the class id of an instance of our class
    ///
    /// \return                         The class id of this class
    SERIAL::Class_id get_class_id() const;

    /// make a copy of this instance
    ///
    /// \return                         The new copy
    Element_base* copy() const;

    /// Return the approximate size in bytes of the element including all its substructures. This is
    /// used to make decisions about garbage collection.
    ///
    /// \return                         The size of the element
    size_t get_size() const
    {
        return sizeof(*this)
            + P::get_size() - sizeof(P);
    }

    /// Check, if this object is of the given type. This is true, if either the class id of this
    /// object equals the given class id, or the class is derived from another class which has the
    /// given class id.
    ///
    /// \param id                       The class id to check
    /// \return                         True, if the class is derived from the class with the id
    bool is_type_of(
        SERIAL::Class_id id) const
    {
        return ID == id ? true : P::is_type_of(id);
    }
};

template <class T, SERIAL::Class_id C, class P>
inline Element<T, C, P>::Element()
{
}

template <class T, SERIAL::Class_id C, class P>
inline Element<T, C, P>::Element(
    const Element& source) :
    P(source)
{
}

template <class T, SERIAL::Class_id C, class P>
inline SERIAL::Class_id Element<T, C, P>::get_class_id() const
{
    return id;
}

template <class T, SERIAL::Class_id C, class P>
inline Element_base* Element<T, C, P>::copy() const
{
    T* element = new T(*(T*)this);
    return element;
}

inline size_t Element_base::get_size() const
{
    return sizeof(*this);
}

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_ELEMENT_H
