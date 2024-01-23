/***************************************************************************************************
 * Copyright (c) 2003-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_DATA_DB_I_DB_TRANSACTION_ID_H
#define BASE_DATA_DB_I_DB_TRANSACTION_ID_H

#include <mi/base/types.h>

#include <climits>

namespace MI {

namespace DB {

/// A wrap-around counter based on mi::Uint32.
//
/// This implements a wrap-around counter that allows the counters to overflow while letting
/// comparisons still return meaningful result. This is based on the assumption that the compared
/// IDs are never expected to be more than UINT_MAX/2 apart from each other. If they are, than an
/// overflow is assumed and compare operators behave accordingly.
//
/// \note These comparison operators do \em not implement a strict total ordering. Standard library
///       containers for Transaction_id require an explicit (different) key comparison function.
///
/// TODO: It would be better to skip ID 0 on initialization and wrap-around since some parts of the
/// code base use ID 0 for an invalid/unavailable transaction ID.
class Transaction_id
{
public:
    /// Default constructor.
    Transaction_id() = default;

    /// Constructor from mi::Uint32.
    explicit Transaction_id( mi::Uint32 id) : m_id( id) { }

    /// Copy constructor.
    Transaction_id( const Transaction_id&) = default;

    /// Assignment operator.
    Transaction_id& operator=( const Transaction_id& other) = default;

    /// Assignment from mi::Uint32.
    Transaction_id& operator=( mi::Uint32 other)
    {
        m_id = other;
        return *this;
    }

    /// Returns the id value.
    mi::Uint32 operator()() const { return m_id; }

    /// Returns the id value.
    mi::Uint32 get_uint() const { return m_id; }

    /// Postfix increment operator.
    const Transaction_id operator++( int)
    {
        Transaction_id id( m_id);
        m_id++;
        return id;
    }

    /// Prefix increment operator.
    Transaction_id& operator++() { ++m_id; return *this; }

    /// Postfix decrement operator.
    const Transaction_id operator--( int)
    {
        Transaction_id id( m_id);
        m_id--;
        return id;
    }

    /// Prefix decrement operator.
    Transaction_id& operator--() { --m_id; return *this; }

private:
    mi::Uint32 m_id = 0;
};

inline Transaction_id operator+( const Transaction_id& lhs, const Transaction_id& rhs)
{
    return Transaction_id( lhs() + rhs());
}

inline Transaction_id operator-( const Transaction_id& lhs, const Transaction_id& rhs)
{
    return Transaction_id( lhs() - rhs());
}

inline Transaction_id operator+( const Transaction_id& lhs, const mi::Uint32 rhs)
{
    return Transaction_id( lhs() + rhs);
}

inline Transaction_id operator-( const Transaction_id& lhs, const mi::Uint32 rhs)
{
    return Transaction_id( lhs() - rhs);
}

inline Transaction_id operator+( const mi::Uint32 lhs, const Transaction_id& rhs)
{
    return Transaction_id( lhs + rhs());
}

inline Transaction_id operator-( const mi::Uint32 lhs, const Transaction_id& rhs)
{
    return Transaction_id( lhs - rhs());
}

/// \name Comparison operators
///
/// If the difference between the two operands is greater than half of what the counter can hold, an
/// overflow is assumed.
///
/// \note These comparison operators do \em not implement a strict total ordering. Standard library
///       containers for Transaction_id require an explicit (different) key comparison function.
//@{

inline bool operator==( const Transaction_id& lhs, const Transaction_id& rhs)
{
    return lhs() == rhs();
}

inline bool operator!=( const Transaction_id& lhs, const Transaction_id& rhs)
{
    return lhs() != rhs();
}

inline bool operator<( const Transaction_id& lhs, const Transaction_id& rhs)
{
    if( lhs() == rhs())
        return false;
    if( lhs() > rhs())
        return lhs() - rhs() > UINT_MAX / 2;
    return rhs() - lhs() <= UINT_MAX / 2;
}

inline bool operator<=( const Transaction_id& lhs, const Transaction_id& rhs)
{
    if( lhs() == rhs())
        return true;
    if( lhs() > rhs())
        return lhs() - rhs() > UINT_MAX / 2;
    return rhs() - lhs() <= UINT_MAX / 2;
}

inline bool operator>( const Transaction_id& lhs, const Transaction_id& rhs)
{
    if( lhs() == rhs())
        return false;
    if( lhs() > rhs())
        return lhs() - rhs() <= UINT_MAX / 2;
    return rhs() - lhs() > UINT_MAX / 2;
}

inline bool operator>=( const Transaction_id& lhs, const Transaction_id& rhs)
{
    if( lhs() == rhs())
        return true;
    if( lhs() > rhs())
        return lhs() - rhs() <= UINT_MAX / 2;
    return rhs() - lhs() > UINT_MAX / 2;
}

//@}

} /// namespace DB

} /// namespace MI

#endif /// BASE_DATA_DB_I_DB_TRANSACTION_ID_H
