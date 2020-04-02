/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_HASH_H
#define MDL_COMPILERCORE_HASH_H

#include <cstring>
#include <mi/base/types.h>

namespace mi {
namespace mdl {

/// Simple implementation of RFC 1321, also known as MD5 Message-Digest Algorithm
class MD5_hasher {
public:
    MD5_hasher()
    : m_a(0x67452301)
    , m_b(0xefcdab89)
    , m_c(0x98badcfe)
    , m_d(0x10325476)
    , m_count(0)
    {
    }

    /// Update the MD5 sum by a data block.
    ///
    /// \param data  points to a data block
    /// \param size  the size of the block
    void update(unsigned char const *data, size_t size);

    /// Update the MD5 sum by a character.
    void update(char c) { update((unsigned char const *)&c, 1); }

    /// Update the MD5 sum by a string.
    void update(char const *s) {
        if (s == NULL)
            update(char(0));
        else
            update((unsigned char const *)s, strlen(s));
    }

    /// Update the MD5 sum by an unsigned 32bit.
    void update(mi::Uint32 v) {
        unsigned char buf[4] = {
                static_cast<unsigned char>(v),
                static_cast<unsigned char>(v >> 8),
                static_cast<unsigned char>(v >> 16),
                static_cast<unsigned char>(v >> 24) };
        update(buf, 4);
    }

    /// Update the MD5 sum by an unsigned 64bit.
    void update(mi::Uint64 v) {
        unsigned char buf[8] = {
            static_cast<unsigned char>(v),
            static_cast<unsigned char>(v >> 8),
            static_cast<unsigned char>(v >> 16),
            static_cast<unsigned char>(v >> 24),
            static_cast<unsigned char>(v >> 32),
            static_cast<unsigned char>(v >> 40),
            static_cast<unsigned char>(v >> 48),
            static_cast<unsigned char>(v >> 56) };
        update(buf, 8);
    }

    /// Update the MD5 sum by a signed 32bit.
    void update(mi::Sint32 v) {
        update(mi::Uint32(v));
    }

    /// Update the MD5 sum by an 32bit float.
    void update(mi::Float32 f) {
        // FIXME: handle LE/BE
        union { mi::Float32 f; unsigned char buf[4]; } u;
        u.f = f;
        update(u.buf, sizeof(u.buf));
    }

    /// Update the MD5 sum by an 64bit float.
    void update(mi::Float64 f) {
        // FIXME: handle LE/BE
        union { mi::Float64 f; unsigned char buf[8]; } u;
        u.f = f;
        update(u.buf, sizeof(u.buf));
    }

    /// Finishes the calculation and returns the MD5 hash
    void final(unsigned char result[16]);

    /// Restart the hasher.
    void restart() {
        m_a = 0x67452301;
        m_b = 0xefcdab89;
        m_c = 0x98badcfe;
        m_d = 0x10325476;
        m_count = 0;
    }

private:
    /// The F function.
    mi::Uint32 F(mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) {
        // use xor instead of andnot
        return z ^ (x & (y ^ z));
    }

    /// The G function.
    mi::Uint32 G(mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) {
        // use xor instead of andnot
        return y ^ (z & (x ^ y));
    }

    /// The H function.
    mi::Uint32 H(mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) {
        return x ^ y ^ z;
    }

    /// The I function.
    mi::Uint32 I(mi::Uint32 x, mi::Uint32 y, mi::Uint32 z) {
        return y ^ (x | ~z);
    }

    /// MD5 basic transformation.
    ///
    /// \param data  the data block
    /// \param size  the size of the data block
    ///
    /// \return pointer to first non-processed byte
    unsigned char const *transform(unsigned char const *data, size_t size);

private:
    mi::Uint32    m_a, m_b, m_c, m_d;
    mi::Uint64    m_count;
    unsigned char m_buffer[64]; // PVS: -V730_NOINIT
};

} // mdl
} // mi

#endif
