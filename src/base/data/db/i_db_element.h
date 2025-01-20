/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_ELEMENT_H
#define BASE_DATA_DB_I_DB_ELEMENT_H

#include <string>

#include "i_db_tag.h"
#include "i_db_journal_type.h"

#include <mi/base/types.h>

#include <base/data/serial/i_serial_serializable.h>

namespace MI {

namespace DB {

class Transaction;

/// Base class for all database elements.
///
/// Database elements are not reference-counted. Storing them in the database passes ownership from
/// the creator to the database.
class Element_base : public SERIAL::Serializable
{
public:
    /// The class ID for this base class.
    static const SERIAL::Class_id id = 0;

    /// Destructor.
    virtual ~Element_base() { }

    /// Indicates whether this object is an instance of the given class.
    ///
    /// \param arg_id   The class given by its class ID.
    /// \return         \c true if this object is directly or indirectly derived from the given
    ///                 class.
    virtual bool is_type_of( SERIAL::Class_id arg_id) const { return arg_id == 0; }

    /// Returns a deep copy of the element.
    ///
    /// This method is invoked when the database needs to create a new version of a tag, e.g., when
    /// an edit operation is started.
    ///
    /// \return   The new copy of the element. RCS:TRO
    virtual Element_base* copy() const = 0;

    /// Returns a human readable version of the class ID.
    virtual std::string get_class_name() const = 0;

    /// Returns all references to other DB elements (or jobs).
    ///
    /// \param result    Adds all references to this tag set. RCS:NEU
    virtual void get_references( Tag_set* result) const { }

    /// Returns a filter for the type of changes to be tracked for elements of this type.
    ///
    /// To avoid unnecessary tracking overhead this filter should contain only those changes you
    /// are actually interested in. The default implementation will cause no changes to be
    //// tracked at all.
    virtual Journal_type get_journal_flags() const { return JOURNAL_NONE; }

    /// Returns the approximate size in bytes of the element including all its substructures.
    ///
    /// Used to make decisions about garbage collection, offloading, etc.
    virtual size_t get_size() const { return sizeof( *this); }

    /// Indicates how the database element should be distributed in the cluster.
    ///
    /// If the method returns \c true the stored or edited database element is distributed to all
    /// nodes in the cluster. If the method returns \c false the database element is only
    /// distributed to the owners, i.e., a subset of the nodes as required to fulfill the
    /// configured redundancy. (Of course, it is later also distributed to those nodes that
    /// explicitly access the database element.)
    virtual bool get_send_to_all_nodes() { return true; }

    /// Indicates how the database element is handled when received via the network without being
    /// explicitly requested.
    ///
    /// If the method returns \c false, a host that receives such a database element will make no
    /// difference between an explicit request or an accidental reception, e.g., because the
    /// element was multicasted. If the method returns \c true, the database will be more
    /// conservative w.r.t. memory usage for elements that it received by accident. The element
    /// will be discarded if the host is not required to keep it for redundancy reasons or if the
    /// element represents the result of a database job. Next, if the disk cache is operational,
    /// the element will be offloaded to disk. Finally, in all other cases, the element will be
    /// kept in memory, as if the element was explicitly requested.
    virtual bool get_offload_to_disk() { return false; }

    /// Indicates which database elements are likely to be requested together with this element.
    ///
    /// This is usually a subset of the elements returned by #get_references(), sorted by
    /// decreasing importance.
    ///
    /// \param results   A place to store the tags to be bundled. RCS:NEU
    /// \param size      The size of \p results.
    /// \return          The number of elements (up to \p size) written to \p results.
    ///
    /// \note No database operations may be performed from within the callback.
    virtual mi::Uint32 bundle( Tag* results, mi::Uint32 size) const { return 0; }

    /// Callback for cleanup before an element is stored or editing finished.
    ///
    /// This callback is invoked by the database from DB::Transaction::store(),
    /// DB::Transaction::store_for_reference_counting(), and DB::Transaction::finish_edit(). It
    /// allows to perform some simple(!) and fast(!) sanity checks and/or pre-computations.
    ///
    /// \param transaction   The corresponding transaction. RCS:NEU
    /// \param own_tag       The tag that will be used to store this object.
    virtual void prepare_store( Transaction* transaction, Tag own_tag) { }
};

/// Helper template that defines some functions for derived classes.
///
/// Examples:
///
/// - If the class shall be directly derived from Element_base:
///     class My_class : public Element<My_class, 1000> { ... }
///
/// - If the class shall be derived from some other class which is derived from Element_base:
///     class My_derived_class : public Element<My_derived_class, 1001, My_class> { ... }
///
template <class T, SERIAL::Class_id ID = 0, class P = Element_base>
class Element : public P
{
public:
    using Element_t = Element<T,ID,P>;

    /// Class ID of this class
    static const SERIAL::Class_id id = ID;

    /// Factory function.
    static SERIAL::Serializable* factory() { return new T; }

    /// Default constructor.
    Element() : P() { }

    /// Copy constructor.
    Element( const Element& other) : P( other) { }

    /// Returns the class ID of this class.
    SERIAL::Class_id get_class_id() const { return id; }

    /// Returns a deep copy of the element.
    ///
    /// This method is invoked when the database needs to create a new version of a tag, e.g., when
    /// an edit operation is started.
    ///
    /// \return   The new copy of the element. RCS:TRO
    Element_base* copy() const { return new T( * reinterpret_cast<const T*>( this)); }

    /// Returns the approximate size in bytes of the element including all its substructures.
    ///
    /// Used to make decisions about garbage collection, offloading, etc.
    size_t get_size() const { return sizeof( *this) + P::get_size() - sizeof( P); }

    /// Indicates whether this object is an instance of the given class.
    ///
    /// \param arg_id   The class given by its class ID.
    /// \return         \c true if this object is directly or indirectly derived from the given
    ///                 class.
    bool is_type_of( SERIAL::Class_id arg_id) const
    {
        return arg_id == ID ? true : P::is_type_of( arg_id);
    }
};

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_ELEMENT_H
