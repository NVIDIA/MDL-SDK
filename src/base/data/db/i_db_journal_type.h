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

/// \file i_db_journal_type.h
/// \brief The definition of the Journal_type

#ifndef BASE_DATA_DB_JOURNAL_TYPE_H
#define BASE_DATA_DB_JOURNAL_TYPE_H

#include <base/system/main/types.h>

namespace MI {
namespace DB {

/// This Journal_type is used for internal book-keeping functionality.
///
/// Structure to hold a journal type entry. This is in principal a Uint32. It has been wrapped into
/// a structure to avoid confused parameter orders to various functions
class Journal_type
{
  public:
    /// Constructor.
    /// \param type bitmask for journal entry
    inline explicit Journal_type(
        Uint32 type=0);

    /// Copy constructor.
    /// \param j the other journal
    inline Journal_type(
        const Journal_type& j);

    /// Assignment operator.
    /// \param j the other journal
    inline Journal_type& operator=(
        const Journal_type& j);

    /// Add type.
    /// \param type new journal entry type
    inline void add_journal(
        Uint32 type);

    /// Add type.
    /// \param j new journal entry
    inline void add_journal(
        Journal_type j);

    /// Restrict to given types.
    /// \param mask restrict to this mask
    inline void restrict_journal(
        Uint32 mask);

    /// Restrict to given types.
    /// \param mask restrict to this journal
    inline void restrict_journal(
        Journal_type mask);

    /// Retrieve type.
    /// \return the stored journal type entry
    inline Uint32 get_type() const;

    /// Retrieve whether the given mask is set.
    /// \param mask the flag we are interested in
    /// \return whether the given mask is set
    inline bool is_set(
        Journal_type mask) const;

  private:
    Uint32 m_type;                              ///< bitmask for journal entry
};

/// Equality operator for Journal_types.
/// \param one the one
/// \param other the other
/// \return true when equal, false else
inline bool operator==(
    const Journal_type& one,
    const Journal_type& other);

/// Unequality operator for Journal_types.
/// \param one the one
/// \param other the other
/// \return true when not equal, false else
inline bool operator!=(
    const Journal_type& one,
    const Journal_type& other);


/// Constant for journal types
static const Journal_type JOURNAL_NONE(0);
/// Constant for journal types
static const Journal_type JOURNAL_ALL(0xffffffff);

// Constructor.
inline Journal_type::Journal_type(
    Uint32 type)                        // bitmask for journal entry
  : m_type(type)
{}


// Copy constructor.
inline Journal_type::Journal_type(
    const Journal_type& j)              // the other journal
  : m_type(j.get_type())
{}


// Assignment operator.
inline Journal_type& Journal_type::operator=(
    const Journal_type& j)              // the other journal
{
    if (this == &j)
        return *this;
    m_type = j.get_type();
    return *this;
}


// Add type.
inline void Journal_type::add_journal(
    Uint32 type)                        // new journal entry type
{
    m_type |= type;
}


// Add type.
inline void Journal_type::add_journal(
    Journal_type j)                     // new journal entry
{
    m_type |= j.get_type();
}


// Restrict to given types.
inline void Journal_type::restrict_journal(
    Uint32 mask)                        // restrict to this mask
{
    m_type &= mask;
}


// Restrict to given types.
inline void Journal_type::restrict_journal(
    Journal_type mask)                  // restrict to this journal
{
    m_type &= mask.get_type();
}


// Retrieve type.
inline Uint32 Journal_type::get_type() const
{
    return m_type;
}


// Retrieve whether the given mask is set.
inline bool Journal_type::is_set(
    Journal_type mask) const
{
    return (m_type & mask.get_type()) == mask.get_type();
}


// Equality operator for Journal_types.
inline bool operator==(
    const Journal_type& one,            // the one
    const Journal_type& other)          // the other
{
    return one.get_type() == other.get_type();
}


// Unequality operator for Journal_types.
inline bool operator!=(
    const Journal_type& one,            // the one
    const Journal_type& other)          // the other
{
    return !(one == other);
}

}
}

#endif
