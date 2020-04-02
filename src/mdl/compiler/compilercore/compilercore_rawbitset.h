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

#ifndef MDL_COMPILERCORE_RAWBITSET_H
#define MDL_COMPILERCORE_RAWBITSET_H 1

#include <cstring>

namespace mi {
namespace mdl {

/// A simple memory efficient raw bitset of a fixed size.
///
/// \tparam N  number of bits to allocate
///
/// \note VC9 miscompiles this in x64 release mode if the unsigned cast
///       in "unsigned(bit & 7)" is removed :-(   
template<size_t N>
class Raw_bitset {
    unsigned char m_bits[(N + 7) / 8];

public:
    /// Default constructor: creates an empty bitset.
    Raw_bitset() {
        memset(m_bits, 0, sizeof(m_bits));
    }

    /// Copy constructor.
    Raw_bitset(Raw_bitset<N> const &other) {
        memcpy(m_bits, other.m_bits, sizeof(m_bits));
    }

    /// Assignment operator.
    Raw_bitset<N> &operator=(Raw_bitset<N> const &other) {
        memcpy(m_bits, other.m_bits, sizeof(m_bits));
        return *this;
    }

    /// Tests if the given bit is set.
    bool test_bit(size_t bit) const {
        return (m_bits[bit >> 3] & (1U << unsigned(bit & 7))) != 0;
    }

    /// Sets if the given bit is set.
    void set_bit(size_t bit) {
        m_bits[bit >> 3] |= 1U << unsigned(bit & 7);
    }

    /// Clears if the given bit is set.
    void clear_bit(size_t bit) {
        m_bits[bit >> 3] &= ~(1U << unsigned(bit & 7));
    }

    /// Returns the immutable size of the bitset.
    size_t get_size() const {
        return N;
    }

    /// Get the raw data.
    unsigned char const *raw_data() const {
        return m_bits;
    }

    /// Get the raw data.
    unsigned char *raw_data() {
        return m_bits;
    }
};

}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_RAWBITSET_H
