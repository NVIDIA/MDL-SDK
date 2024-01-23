/***************************************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_JOURNAL_TYPE_H
#define BASE_DATA_DB_I_DB_JOURNAL_TYPE_H

#include <mi/base/types.h>

namespace MI {

namespace DB {

/// The journal type is a bit field that is used to track types of changes per DB element.
class Journal_type
{
public:
    /// Constructor.
    ///
    /// \param type   Initial value.
    explicit Journal_type( mi::Uint32 type = 0) : m_type( type) { }

    /// Default copy constructor.
    Journal_type( const Journal_type&) = default;

    /// Default assignment operator.
    Journal_type& operator=( const Journal_type&) = default;

    /// Sets the bits set in \p other.
    void add_journal( mi::Uint32 other) { m_type |= other; }

    /// Sets the bits set in \p other.
    void add_journal( Journal_type other) {  m_type |= other.get_type(); }

    /// Clears the bits not set in \p other.
    void restrict_journal( mi::Uint32 other) { m_type &= other; }

    /// Clears the bits not set in \p other.
    void restrict_journal( Journal_type other) { m_type &= other.get_type(); }

    /// Returns the bit field.
    mi::Uint32 get_type() const { return m_type; }

    /// Indicates whether all the bits set in \p other are set in this journal type.
    bool is_set( Journal_type other) const
    { return (m_type & other.get_type()) == other.get_type(); }

private:
    /// Representation of the bit field.
    mi::Uint32 m_type;
};

/// Equality operator for journal_types.
inline bool operator==( const Journal_type& lhs, const Journal_type& rhs)
{
    return lhs.get_type() == rhs.get_type();
}

/// Inequality operator for journal_types.
inline bool operator!=( const Journal_type& lhs, const Journal_type& rhs)
{
    return !(lhs == rhs);
}

/// Constant for a journal type with no bits set.
static const Journal_type JOURNAL_NONE( 0u);

/// Constant for a journal type with all bits set.
static const Journal_type JOURNAL_ALL( ~0u);

} // namespace DB

} // namespace MI

#endif // BASE_DATA_DB_I_DB_JOURNAL_TYPE_H
