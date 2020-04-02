/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

#ifndef MDL_COMPILERCORE_BITSET_H
#define MDL_COMPILERCORE_BITSET_H 1

#include <cstring>
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"

namespace mi {
namespace mdl {

///
/// A simple memory efficient bitset of a flexible size.
/// \tparam V  a VLA type
/// \tparam A  an allocator type for the VLA
///
///
/// \note VC9 miscompiles this in x64 release mode if the unsigned cast
///       in "unsigned(bit & 7)" is removed :-(   
template<typename V, typename A>
class Bitset_base {
    typedef Bitset_base<V, A> _Myt;

protected:
    V m_bits;
    size_t const N;

public:
    /// Default constructor: creates an empty bitset.
    ///
    /// \param alloc  the allocator
    /// \param N      the size of the bitset
    Bitset_base(A alloc, size_t N)
    : m_bits(alloc, (N + 7) / 8)
    , N(N)
    {
        memset(m_bits.data(), 0, (N + 7) / 8);
    }

    /// Constructor with initializer.
    template<typename OV, typename OA>
    Bitset_base(A alloc, Bitset_base<OV, OA> const &o)
    : m_bits(alloc, (o.get_size() + 7) / 8)
    , N(o.get_size())
    {
        copy_data(o);
    }

    /// Copy data from another bitset of the same size.
    template<typename OV, typename OA>
    void copy_data(Bitset_base<OV, OA> const &o)
    {
        MDL_ASSERT(get_size() == o.get_size());
        memcpy(m_bits.data(), o.get_data(), (N + 7) / 8);
    }

    /// Tests if the given bit is set.
    bool test_bit(size_t bit) const {
        return (m_bits[bit >> 3] & (1U << unsigned(bit & 7))) != 0;
    }

    bool is_any_set() const {
        for (size_t i = 0, n_bytes = N / 8; i < n_bytes; ++i)
            if (m_bits[i] != 0) return true;
        return false;
    }

    /// Return the full byte containing the information of the given bit.
    /// Useful for hashing.
    unsigned get_raw_byte(size_t bit) const {
        return m_bits[bit >> 3];
    }

    /// Sets the given bit.
    void set_bit(size_t bit) {
        m_bits[bit >> 3] |= 1U << unsigned(bit & 7);
    }

    /// Clears the given bit.
    void clear_bit(size_t bit) {
        m_bits[bit >> 3] &= ~(1U << unsigned(bit & 7));
    }

    /// Sets all bits
    void set_bits() {
        memset(m_bits.data(), 255, (N + 7) / 8);
    }

    /// Clears all bits.
    void clear_bits() {
        memset(m_bits.data(), 0, (N + 7) / 8);
    }

    /// Returns the immutable size of the bitset.
    size_t get_size() const {
        return N;
    }

    void const *get_data() const {
        return m_bits.data();
    }

    /// Returns true, iff the content the given bitset is identical to the content of this bitset.
    bool operator==(_Myt const &o) const {
        if (get_size() != o.get_size()) return false;
        return memcmp(m_bits.data(), o.m_bits.data(), get_size()) == 0;
    }

    /// Returns true, iff the content the given bitset is not identical to the content of this
    /// bitset.
    bool operator!=(_Myt const &o) const {
        return !operator==(o);
    }

    /// Returns true, iff this bitset is a subset of the given bitset.
    bool operator<=(_Myt const &o) const {
        if (get_size() != o.get_size()) return false;
        for (size_t i = 0, n_bytes = (N + 7) / 8; i < n_bytes; ++i) {
            if ((m_bits[i] & o.m_bits[i]) != m_bits[i])
                return false;
        }
        return true;
    }


    // non copyable without allocator
    Bitset_base(Bitset_base const &bitset) MDL_DELETED_FUNCTION;
    Bitset_base &operator=(Bitset_base const &) MDL_DELETED_FUNCTION;
};

///
/// A simple memory efficient bitset of a flexible size, allocated on an allocator.
///
class Bitset : public Bitset_base<VLA<unsigned char>, IAllocator *>
{
    typedef Bitset_base<VLA<unsigned char>, IAllocator *> Base;
public:
    /// Default constructor: creates an empty bitset.
    ///
    /// \param alloc  the allocator
    /// \param N      the size of the bitset
    Bitset(IAllocator *alloc, size_t N)
    : Base(alloc, N)
    {
    }

    /// Copy constructor.
    Bitset(Bitset const &o)
    : Base(o.m_bits.get_allocator(), o.get_size())
    {
        memcpy(m_bits.data(), o.m_bits.data(), (N + 7) / 8);
    }

    /// Copy constructor.
    template<typename V, typename A>
    Bitset(IAllocator *alloc, Bitset_base<V, A> const &o)
    : Base(alloc, o)
    {
    }

    /// Copy data from another bitset of the same size.
    template<typename V, typename A>
    void copy_data(Bitset_base<V, A> const &o)
    {
        Base::copy_data(o);
    }
};

///
/// A simple memory efficient bitset of a flexible size, allocated on a memory arena.
///
class Arena_Bitset : public Bitset_base<Arena_VLA<unsigned char>, Memory_arena &>
{
    typedef Bitset_base<Arena_VLA<unsigned char>, Memory_arena &> Base;
public:
    /// Default constructor: creates an empty bitset.
    ///
    /// \param arena  the memory arena
    /// \param N      the size of the bitset
    Arena_Bitset(Memory_arena &arena, size_t N)
    : Base(arena, N)
    {
    }
};

///
/// A static sized bitset
///
template<size_t N>
class Static_bitset
{
    unsigned char m_bits[(N + 7) / 8];

public:
    /// Default constructor: creates an empty bitset.
    Static_bitset()
    {
        memset(m_bits, 0, sizeof(m_bits));
    }

    /// Copy constructor.
    Static_bitset(Static_bitset const &other)
    {
        memcpy(m_bits, other.m_bits, sizeof(m_bits));
    }

    /// Tests if the given bit is set.
    bool test_bit(size_t bit) const {
        return (m_bits[bit >> 3] & (1U << unsigned(bit & 7))) != 0;
    }

    /// Sets the given bit.
    void set_bit(size_t bit) {
        m_bits[bit >> 3] |= 1U << unsigned(bit & 7);
    }

    /// Clears the given bit.
    void clear_bit(size_t bit) {
        m_bits[bit >> 3] &= ~(1U << unsigned(bit & 7));
    }

    /// Clears all bits.
    void clear_bits() {
        memset(m_bits, 0, sizeof(m_bits));
    }

    /// Returns the immutable size of the bitset.
    size_t get_size() const {
        return N;
    }

    /// == for static bitsets.
    bool operator==(Static_bitset const &rhs) const {
        return memcmp(m_bits, rhs.m_bits, sizeof(m_bits)) == 0;
    }

    /// != for static bitsets.
    bool operator!=(Static_bitset const &rhs) const {
        return !operator==(rhs);
    }

    /// Assignment operator.
    Static_bitset &operator=(Static_bitset const &other) {
        memcpy(m_bits, other.m_bits, sizeof(m_bits));
        return *this;
    }
};

///
/// A variable size bitset.
///
class Dynamic_bitset
{
    IAllocator    *m_alloc;
    unsigned char *m_bits;
    size_t        m_size;

public:
    /// Constructor: creates an empty bitset.
    Dynamic_bitset(IAllocator *alloc)
    : m_alloc(alloc)
    , m_bits(NULL)
    , m_size(0)
    {
    }

    /// Destructor.
    ~Dynamic_bitset() {
        if (m_bits != NULL) {
            m_alloc->free(m_bits);
            m_bits = 0;
        }
    }

    /// Tests if the given bit is set.
    bool test_bit(size_t bit) const {
        if (bit >= m_size)
            return 0;

        return (m_bits[bit >> 3] & (1U << unsigned(bit & 7))) != 0;
    }

    /// Sets the given bit.
    void set_bit(size_t bit) {
        if (bit >= m_size)
            grow(bit);
        m_bits[bit >> 3] |= 1U << unsigned(bit & 7);
    }

    /// Clears the given bit.
    void clear_bit(size_t bit) {
        if (bit >= m_size)
            return;
        m_bits[bit >> 3] &= ~(1U << unsigned(bit & 7));
    }

    /// Clears all bits.
    void clear_bits() {
        if (m_bits != NULL) {
            memset(m_bits, 0, (m_size + 7) >> 3);
        }
    }

    /// Returns the immutable size of the bitset.
    size_t get_size() const {
        return m_size;
    }

private:
    /// Grow the set to at least bit bits
    void grow(size_t bit) {
        if (bit < m_size)
            return;

        size_t n_size = (bit >> 8) + 8;

        unsigned char *data = static_cast<unsigned char *>(m_alloc->malloc(n_size));

        if (m_size > 0) {
            memcpy(data, m_bits, m_size >> 3);
        }
        memset(data + (m_size >> 3), 0, (n_size - m_size) >> 3);

        m_alloc->free(m_bits);

        m_bits = data;
        m_size = n_size;
    }

    // non copyable
    Dynamic_bitset(Dynamic_bitset const &bitset) MDL_DELETED_FUNCTION;
    Dynamic_bitset &operator=(Dynamic_bitset const &) MDL_DELETED_FUNCTION;
};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_BITSET_H
