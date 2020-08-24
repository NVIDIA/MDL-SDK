/***************************************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief An unique key for database elements.

#ifndef BASE_DATA_DB_I_DB_TAG_H
#define BASE_DATA_DB_I_DB_TAG_H

#include <base/system/main/types.h>
#include <boost/unordered_map.hpp>
#include <set>
#include "i_db_transaction_id.h"

namespace MI {
namespace DB {

/// A \c Tag is a unique key for database elements. It is used to search the
/// \c Info elements in the database cache.
/// If Tag is derived from Allocatable, PVS suggests to pass Tag as const reference in all
/// signatures, even though it takes only 4 bytes. In neuray, MEM::Allocatable is mapped to operator
/// new and delete anyway, so the easiest way to suppress this warning is to eliminate the base
/// class.
class Tag
{
    /// Pointer to member functions convert to bool but to nothing else.
    typedef bool (Tag::*unknown_bool_type)() const;

public:
    /// Use default constructor to create the invalid tag.
    explicit Tag(Uint32 tag = 0u) : m_tag(tag) { }

    /// Check whether a tag is valid or invalid.
    //@{
    bool is_valid()   const             { return m_tag != 0u; }
    bool is_invalid() const             { return m_tag == 0u; }
    //@}

    /// Allow implicit version to bool. True means "is_valid()" returns true,
    /// false means "is_invalid()" returns true.
    operator unknown_bool_type() const  { return is_valid() ? &Tag::is_valid : 0; }

    /// Implementor's access to the low-level type.
    Uint32 get_uint() const             { return m_tag; }

private:
    Uint32 m_tag;                       ///< the actual unique key of low-level type
};

// A define for printing tags. Use as follows:
// printf("A tag: %10.10" FMT_TAG, value);
#define FMT_TAG "%" FMT_BIT32 "u"

inline bool operator== (Tag const & l, Tag const & r) { return l.get_uint() == r.get_uint(); }
inline bool operator!= (Tag const & l, Tag const & r) { return l.get_uint() != r.get_uint(); }
inline bool operator<  (Tag const & l, Tag const & r) { return l.get_uint() <  r.get_uint(); }
inline bool operator<= (Tag const & l, Tag const & r) { return l.get_uint() <= r.get_uint(); }
inline bool operator>  (Tag const & l, Tag const & r) { return l.get_uint() >  r.get_uint(); }
inline bool operator>= (Tag const & l, Tag const & r) { return l.get_uint() >= r.get_uint(); }

// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption (Tag const &) { return false; }
inline size_t dynamic_memory_consumption (Tag const &) { return 0; }


/// This class can be used to store a tag, when it is known, that only database
/// elements with a given type will be stored below this tag.
template <class T>
class Typed_tag
{
    typedef bool (Typed_tag::*unknown_bool_type)() const;
    Tag m_tag;

public:
    /// This constructor should be explicit. It isn't, because DB::Transaction::store() and similar
    /// methods don't support typed tags very well, so for the sake of convenience we'd like to be
    /// able to construct a typed tag from a tag without further annotation. Implicit down-casting
    /// from a typed tag to a tag, however, is not supported.
    Typed_tag(Tag tag = Tag()) : m_tag(tag) { }

    bool is_valid()   const             { return m_tag.is_valid(); }
    bool is_invalid() const             { return m_tag.is_invalid(); }
    Tag get_untyped() const             { return m_tag; }

    operator unknown_bool_type() const  { return is_valid() ? &Typed_tag::is_valid : 0; }

    /// Implementor's access to the low-level type.
    Uint32 get_uint() const             { return m_tag.get_uint(); }
};

template <class T> inline bool operator== (Typed_tag<T> const & l, Typed_tag<T> const & r)
{
    return l.get_untyped() == r.get_untyped();
}
template <class T> inline bool operator!= (Typed_tag<T> const & l, Typed_tag<T> const & r)
{
    return l.get_untyped() != r.get_untyped();
}
template <class T> inline bool operator<  (Typed_tag<T> const & l, Typed_tag<T> const & r)
{
    return l.get_untyped() <  r.get_untyped();
}
template <class T> inline bool operator<= (Typed_tag<T> const & l, Typed_tag<T> const & r)
{
    return l.get_untyped() <= r.get_untyped();
}
template <class T> inline bool operator>  (Typed_tag<T> const & l, Typed_tag<T> const & r)
{
    return l.get_untyped() >  r.get_untyped();
}
template <class T> inline bool operator>= (Typed_tag<T> const & l, Typed_tag<T> const & r)
{
    return l.get_untyped() >= r.get_untyped();
}

// See base/lib/mem/i_mem_consumption.h
template <class T>
inline bool has_dynamic_memory_consumption (Typed_tag<T> const &) { return false; }
template <class T>
inline size_t dynamic_memory_consumption (Typed_tag<T> const &) { return 0; }

// Abstract type that represents a set of database tags. Required for reference counting.
typedef std::set<Tag> Tag_set;

/// This structure acts as a unique identifier for a certain version of a tag.
/// The database guarantees, that if two database accesses return the same Tag_version, then they
/// have identical contents. Note that this holds only, if only changes are done which are within
/// the limits allowed by the database api: If a database element is changed without using an edit,
/// it will not work.
struct Tag_version
{
    /// default constructor
    Tag_version() : m_tag(Tag()), m_transaction_id(0), m_version(0) {}

    /// the tag
    Tag m_tag;
    /// the transaction creating this version
    Transaction_id m_transaction_id;
    /// the version within the transaction
    Uint32 m_version;
};

/// compare two tag versions
///
/// \param v1                           The first version
/// \param v2                           The second version
inline bool operator==(
    const Tag_version& v1,
    const Tag_version& v2)
{
    if (v1.m_tag != v2.m_tag)  return false;
    if (v1.m_transaction_id != v2.m_transaction_id)  return false;
    if (v1.m_version != v2.m_version)  return false;
    return true;
}

/// compare two tag versions
///
/// \param v1                           The first version
/// \param v2                           The second version
inline bool operator!=(
    const Tag_version& v1,
    const Tag_version& v2)
{
    return !(v1 == v2);
}

/// compare two tag versions
///
/// \param v1                           The first version
/// \param v2                           The second version
inline bool operator<(
    const Tag_version& v1,
    const Tag_version& v2)
{
    if (v1.m_tag < v2.m_tag)  return true;
    if (v1.m_tag > v2.m_tag)  return false;
    if (v1.m_transaction_id < v2.m_transaction_id)  return true;
    if (v1.m_transaction_id > v2.m_transaction_id)  return false;
    return v1.m_version < v2.m_version;
}

/// compare two tag versions
///
/// \param v1                           The first version
/// \param v2                           The second version
inline bool operator>(
    const Tag_version& v1,
    const Tag_version& v2)
{
    return v2 < v1;
}

// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption (Tag_version const &) { return false; }
inline size_t dynamic_memory_consumption (Tag_version const &) { return 0; }

}
}

namespace boost
{

template <>                             // Hash functor for a tag
class hash<MI::DB::Tag>
{
public:
    size_t operator()(const MI::DB::Tag& tag) const
    {
        return tag.get_uint();
    }
};

template <class T>                      // Hash functor for a typed tag
class hash< MI::DB::Typed_tag<T> >
{
  public:
    size_t operator()(const MI::DB::Typed_tag<T>& tag) const
    {
        return tag.get_uint();
    }
};

}

#endif
