/***************************************************************************************************
 * Copyright (c) 2006-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_TAG_H
#define BASE_DATA_DB_I_DB_TAG_H

#include <mi/base/types.h>

#include <functional>
#include <set>

#include "i_db_transaction_id.h"

namespace MI {

namespace DB {

/// Unique key for database elements.
class Tag
{
private:
    /// Helper type for the conversion to bool via this intermediate type (such that the bool is
    /// not implicitly converted to another type).
    using unknown_bool_type = bool (Tag::*)() const;

public:
    /// Default constructor, creates an invalid tag.
    Tag() : m_tag( 0) { }

    /// Constructor.
    explicit Tag( mi::Uint32 tag) : m_tag( tag) { }

    /// Indicates whether the tag is valid.
    bool is_valid() const { return m_tag != 0u; }

    /// Indicates whether the tag is invalid.
    bool is_invalid() const { return m_tag == 0u; }

    /// Conversion to bool, returns \c true for valid tags.
    operator unknown_bool_type() const { return is_valid() ? &Tag::is_valid : nullptr; }

    /// Returns the tag value.
    mi::Uint32 operator()() const { return m_tag; }

    /// Returns the tag value.
    mi::Uint32 get_uint() const { return m_tag; }

private:
    mi::Uint32 m_tag;
};

/// Comparison operators for Tag
//@{

inline bool operator==( const Tag& lhs, const Tag& rhs) { return lhs.get_uint() == rhs.get_uint(); }
inline bool operator!=( const Tag& lhs, const Tag& rhs) { return lhs.get_uint() != rhs.get_uint(); }
inline bool operator< ( const Tag& lhs, const Tag& rhs) { return lhs.get_uint() <  rhs.get_uint(); }
inline bool operator<=( const Tag& lhs, const Tag& rhs) { return lhs.get_uint() <= rhs.get_uint(); }
inline bool operator> ( const Tag& lhs, const Tag& rhs) { return lhs.get_uint() >  rhs.get_uint(); }
inline bool operator>=( const Tag& lhs, const Tag& rhs) { return lhs.get_uint() >= rhs.get_uint(); }

//@}

} }

/// Hash functor for Tag
template<>
struct std::hash<class MI::DB::Tag>
{
    size_t operator()( const MI::DB::Tag& tag) const noexcept
    { return std::hash<mi::Uint32>()( tag.get_uint()); }
};

namespace MI { namespace DB {

/// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption( const Tag&) { return false; }
inline size_t dynamic_memory_consumption( const Tag&) { return 0; }

/// The printf-format string suitable for tags.
///
/// For example:
///   printf( "A tag: " FMT_TAG, tag.get_uint());
#define FMT_TAG "%" FMT_BIT32 "u"


/// Unique typed key for database elements.
///
/// \note The template type is only a \em hint for the type of the corresponding database element,
///       but a typed tag does not guarantee the correct type in any way. Input validation must be
///       done by other means, e.g., DB::Transaction::get_class_id().
template <class T>
class Typed_tag
{
private:
    /// Helper type for the conversion to bool via this intermediate type (such that the bool is
    /// not implicitly converted to another type).
    using unknown_bool_type = bool (Typed_tag::*)() const;

public:
    /// Default constructor, creates an invalid tag.
    Typed_tag() = default;

    /// Constructor.
    explicit Typed_tag( Tag tag) : m_tag( tag) { }

    /// Indicates whether the tag is valid.
    bool is_valid() const { return m_tag.is_valid(); }

    /// Indicates whether the tag is invalid.
    bool is_invalid() const { return m_tag.is_invalid(); }

    /// Conversion to bool.
    operator unknown_bool_type() const { return is_valid() ? &Typed_tag::is_valid : nullptr; }

    /// Conversion to Tag.
    operator Tag() const { return m_tag; }

    /// Returns the tag value.
    mi::Uint32 get_uint() const { return m_tag.get_uint(); }

private:
    Tag m_tag;
};

/// Comparison operators for Typed_tag
//@{

template <class T> inline bool operator==( const Typed_tag<T>& lhs, const Typed_tag<T>& rhs)
{
    return static_cast<Tag>( lhs) == static_cast<Tag>( rhs);
}
template <class T> inline bool operator!=( const Typed_tag<T>& lhs, const Typed_tag<T>& rhs)
{
    return static_cast<Tag>( lhs) != static_cast<Tag>( rhs);
}
template <class T> inline bool operator< ( const Typed_tag<T>& lhs, const Typed_tag<T>& rhs)
{
    return static_cast<Tag>( lhs) <  static_cast<Tag>( rhs);
}
template <class T> inline bool operator<=( const Typed_tag<T>& lhs, const Typed_tag<T>& rhs)
{
    return static_cast<Tag>( lhs) <= static_cast<Tag>( rhs);
}
template <class T> inline bool operator> ( const Typed_tag<T>& lhs, const Typed_tag<T>& rhs)
{
    return static_cast<Tag>( lhs) >  static_cast<Tag>( rhs);
}
template <class T> inline bool operator>=( const Typed_tag<T>& lhs, const Typed_tag<T>& rhs)
{
    return static_cast<Tag>( lhs) >= static_cast<Tag>( rhs);
}

//@}

} }

/// Hash functor for Typed_tag
template <class T>
struct std::hash<class MI::DB::Typed_tag<T>>
{
    size_t operator()( const MI::DB::Typed_tag<T>& tag) const noexcept
    { return std::hash<mi::Uint32>()( tag.get_uint()); }
};

namespace MI { namespace DB {

/// See base/lib/mem/i_mem_consumption.h
template <class T>
inline bool has_dynamic_memory_consumption( const Typed_tag<T>&) { return false; }
template <class T>
inline size_t dynamic_memory_consumption( const Typed_tag<T>&) { return 0; }


/// Set of tags. Used for reference counting.
using Tag_set = std::set<Tag>;


/// Unique identifier for a certain version of a database element or job.
///
/// The database guarantees that if two database accesses return the same tag version, then they
/// have identical contents.
///
/// Note that this guarantee holds only if all changes are done within the limits allowed by the
/// database API, e.g., changing database elements only via DB::Edit.
struct Tag_version
{
    /// Default constructor.
    Tag_version() : m_tag( Tag()), m_transaction_id( 0), m_version( 0) { }
    /// Constructor.
    Tag_version( Tag tag, Transaction_id transaction_id, mi::Uint32 version)
      : m_tag( tag), m_transaction_id( transaction_id), m_version( version) { }

    /// The tag.
    Tag m_tag;
    /// The transaction creating this tag version.
    Transaction_id m_transaction_id;
    /// The version within the transaction.
    mi::Uint32 m_version;
};

/// Comparison operators
///
/// Based on lexicographic comparison by tag, transaction ID, and version.
//@{

inline bool operator==( const Tag_version& lhs, const Tag_version& rhs)
{
    if( lhs.m_tag != rhs.m_tag) return false;
    if( lhs.m_transaction_id != rhs.m_transaction_id) return false;
    if( lhs.m_version != rhs.m_version) return false;
    return true;
}

inline bool operator!=( const Tag_version& lhs, const Tag_version& rhs)
{
    return ! (lhs == rhs);
}

inline bool operator<( const Tag_version& lhs, const Tag_version& rhs)
{
    if( lhs.m_tag < rhs.m_tag) return true;
    if( lhs.m_tag > rhs.m_tag) return false;
    if( lhs.m_transaction_id < rhs.m_transaction_id) return true;
    if( lhs.m_transaction_id > rhs.m_transaction_id) return false;
    return lhs.m_version < rhs.m_version;
}

inline bool operator>( const Tag_version& lhs, const Tag_version& rhs)
{
    return rhs < lhs;
}

//@}

/// See base/lib/mem/i_mem_consumption.h
inline bool has_dynamic_memory_consumption( Tag_version const &) { return false; }
inline size_t dynamic_memory_consumption( Tag_version const &) { return 0; }

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_TAG_H
