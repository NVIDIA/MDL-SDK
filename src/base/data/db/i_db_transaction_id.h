/***************************************************************************************************
 * Copyright (c) 2003-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Transaction ID class, a Uint32 wrap around counter

#ifndef BASE_DATA_DB_I_DB_TRANSACTION_ID_H
#define BASE_DATA_DB_I_DB_TRANSACTION_ID_H

#include <climits>
#include <base/system/main/types.h> // Uint32

namespace MI {
namespace DB {

// Transaction ID class, a Uint32 wrap around counter
//
// This implements a wrap around counter that allows the counters to
// overflow while letting comparisons still return valid result. This is
// based on the assumption that the compared ids are never expected to
// be more than UINT32_MAX/2 apart from each other. If they are, than an
// overflow is assumed and compare operators behave accordingly.
//
// Warning: the comparison operators do _not_ implement an order 
// relationship. For large distributions of IDs, things like
// sorted STL containers or sort algorithms will fall apart.
//
class Transaction_id
{
  private:
    Uint32 m_id;                           // the actual counter

  public:
    // constructor
    inline explicit Transaction_id( Uint32 id) : m_id( id) {}

    // default constructor
    inline Transaction_id() : m_id(0) {}

    // return id value
    inline Uint32 operator()() const { return m_id; }

    // return id value
    Uint32 get_uint() const { return m_id; }

    // assignment operator
    inline Transaction_id &operator=( Uint32 value)
        {
            m_id = value;
            return *this;
        }

    // copy assignment operator
    inline Transaction_id &operator=( const Transaction_id& source)
        {
            m_id = source.m_id;
            return *this;
        }

    // postfix increment operator
    inline const Transaction_id operator++(int)
        {
            Transaction_id id(m_id);
            m_id++;
            return id;
        }

    // prefix increment operator
    inline Transaction_id &operator++() { ++m_id; return *this; }

    // postfix decrement operator
    inline const Transaction_id operator--(int)
        {
            Transaction_id id(m_id);
            m_id--;
            return id;
        }

    // prefix decrement operator
    inline Transaction_id &operator--() { --m_id; return *this; }

};

inline Transaction_id operator+( const Transaction_id &id1, const Transaction_id &id2)
{
    return Transaction_id(id1() + id2());
}

inline Transaction_id operator-( const Transaction_id &id1, const Transaction_id &id2)
{
    return Transaction_id(id1() - id2());
}

inline Transaction_id operator+( const Transaction_id &id1, const Uint32 id2)
{
    return Transaction_id(id1() + id2);
}

inline Transaction_id operator-( const Transaction_id &id1, const Uint32 id2)
{
    return Transaction_id(id1() - id2);
}

inline Transaction_id operator+( const Uint32 id1, const Transaction_id &id2)
{
    return Transaction_id(id1 + id2());
}

inline Transaction_id operator-( const Uint32 id1, const Transaction_id &id2)
{
    return Transaction_id(id1 - id2());
}


// comparison operator for wrap ids
inline bool operator==( const Transaction_id &id1, const Transaction_id &id2)
{
    return id1() == id2();
}

// comparison operator for wrap ids
inline bool operator!=( const Transaction_id &id1, const Transaction_id &id2)
{
    return id1() != id2();
}


// comparison operator for IDs. If the difference between the
// two operands is greater than half of what the counter can hold, an
// overflow is assumed.
// Warning: this is _not_ implementing an order relationship
inline bool operator<( const Transaction_id &id1, const Transaction_id &id2)
{
    if (id1() == id2())
        return false;
    if (id1() > id2())
        return id1() - id2() > UINT_MAX / 2;
    return id2() - id1() <= UINT_MAX / 2;
}


// comparison operator for IDs. If the difference between the
// two operands is greater than half of what the counter can hold, an
// overflow is assumed.
// Warning: this is _not_ implementing an order relationship
inline bool operator<=( const Transaction_id &id1, const Transaction_id &id2)
{
    if (id1() == id2())
        return true;
    if (id1() > id2())
        return id1() - id2() > UINT_MAX / 2;
    return id2() - id1() <= UINT_MAX / 2;
}


// comparison operator for IDs. If the difference between the
// two operands is greater than half of what the counter can hold, an
// overflow is assumed.
// Warning: this is _not_ implementing an order relationship
inline bool operator>( const Transaction_id &id1, const Transaction_id &id2)
{
    if (id1() == id2())
        return false;
    if (id1() > id2())
        return id1() - id2() <= UINT_MAX / 2;
    return id2() - id1() > UINT_MAX / 2;
}


// comparison operator for IDs. If the difference between the
// two operands is greater than half of what the counter can hold, an
// overflow is assumed.
// Warning: this is _not_ implementing an order relationship
inline bool operator>=( const Transaction_id &id1, const Transaction_id &id2)
{
    if (id1() == id2())
        return true;
    if (id1() > id2())
        return id1() - id2() <= UINT_MAX / 2;
    return id2() - id1() > UINT_MAX / 2;
}

} // namespace DB
} // namespace MI

#endif // BASE_DATA_DB_I_DB_TRANSACTION_ID_H
