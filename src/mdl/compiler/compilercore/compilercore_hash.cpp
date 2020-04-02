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

#include "pch.h"

#include "compilercore_cc_conf.h"

#include "compilercore_hash.h"

namespace mi {
namespace mdl {

unsigned char const *MD5_hasher::transform(unsigned char const *data, size_t size)
{
#define SET(n) \
    (block[n] =  (mi::Uint32)data[(n) * 4 + 0]        | ((mi::Uint32)data[(n) * 4 + 1] << 8) | \
    ((mi::Uint32)data[(n) * 4 + 2] << 16) | ((mi::Uint32)data[(n) * 4 + 3] << 24))
#define GET(n) block[n]

    // ROTATE_LEFT rotates x left n bits.
#define ROTATE_LEFT(x, n) ((x) << (n) | (((x) & 0xffffffff) >> (32 - (n))))

    // The MD5 transformation for all four rounds.
#define STEP(func, a, b, c, d, x, s, t) \
    (a) += func((b), (c), (d)) + (x) + (t); \
    (a) = ROTATE_LEFT((a), (s)); \
    (a) += (b);

    mi::Uint32 block[16];

    mi::Uint32 a = m_a;
    mi::Uint32 b = m_b;
    mi::Uint32 c = m_c;
    mi::Uint32 d = m_d;

    do {
        mi::Uint32 saved_a = a;
        mi::Uint32 saved_b = b;
        mi::Uint32 saved_c = c;
        mi::Uint32 saved_d = d;

        // Round 1
        STEP(F, a, b, c, d, SET( 0),  7, 0xd76aa478)
            STEP(F, d, a, b, c, SET( 1), 12, 0xe8c7b756)
            STEP(F, c, d, a, b, SET( 2), 17, 0x242070db)
            STEP(F, b, c, d, a, SET( 3), 22, 0xc1bdceee)
            STEP(F, a, b, c, d, SET( 4),  7, 0xf57c0faf)
            STEP(F, d, a, b, c, SET( 5), 12, 0x4787c62a)
            STEP(F, c, d, a, b, SET( 6), 17, 0xa8304613)
            STEP(F, b, c, d, a, SET( 7), 22, 0xfd469501)
            STEP(F, a, b, c, d, SET( 8),  7, 0x698098d8)
            STEP(F, d, a, b, c, SET( 9), 12, 0x8b44f7af)
            STEP(F, c, d, a, b, SET(10), 17, 0xffff5bb1)
            STEP(F, b, c, d, a, SET(11), 22, 0x895cd7be)
            STEP(F, a, b, c, d, SET(12),  7, 0x6b901122)
            STEP(F, d, a, b, c, SET(13), 12, 0xfd987193)
            STEP(F, c, d, a, b, SET(14), 17, 0xa679438e)
            STEP(F, b, c, d, a, SET(15), 22, 0x49b40821)

            // Round 2
            STEP(G, a, b, c, d, GET( 1),  5, 0xf61e2562)
            STEP(G, d, a, b, c, GET( 6),  9, 0xc040b340)
            STEP(G, c, d, a, b, GET(11), 14, 0x265e5a51)
            STEP(G, b, c, d, a, GET( 0), 20, 0xe9b6c7aa)
            STEP(G, a, b, c, d, GET( 5),  5, 0xd62f105d)
            STEP(G, d, a, b, c, GET(10),  9, 0x02441453)
            STEP(G, c, d, a, b, GET(15), 14, 0xd8a1e681)
            STEP(G, b, c, d, a, GET( 4), 20, 0xe7d3fbc8)
            STEP(G, a, b, c, d, GET( 9),  5, 0x21e1cde6)
            STEP(G, d, a, b, c, GET(14),  9, 0xc33707d6)
            STEP(G, c, d, a, b, GET( 3), 14, 0xf4d50d87)
            STEP(G, b, c, d, a, GET( 8), 20, 0x455a14ed)
            STEP(G, a, b, c, d, GET(13),  5, 0xa9e3e905)
            STEP(G, d, a, b, c, GET( 2),  9, 0xfcefa3f8)
            STEP(G, c, d, a, b, GET( 7), 14, 0x676f02d9)
            STEP(G, b, c, d, a, GET(12), 20, 0x8d2a4c8a)

            // Round 3
            STEP(H, a, b, c, d, GET( 5),  4, 0xfffa3942)
            STEP(H, d, a, b, c, GET( 8), 11, 0x8771f681)
            STEP(H, c, d, a, b, GET(11), 16, 0x6d9d6122)
            STEP(H, b, c, d, a, GET(14), 23, 0xfde5380c)
            STEP(H, a, b, c, d, GET( 1),  4, 0xa4beea44)
            STEP(H, d, a, b, c, GET( 4), 11, 0x4bdecfa9)
            STEP(H, c, d, a, b, GET( 7), 16, 0xf6bb4b60)
            STEP(H, b, c, d, a, GET(10), 23, 0xbebfbc70)
            STEP(H, a, b, c, d, GET(13),  4, 0x289b7ec6)
            STEP(H, d, a, b, c, GET( 0), 11, 0xeaa127fa)
            STEP(H, c, d, a, b, GET( 3), 16, 0xd4ef3085)
            STEP(H, b, c, d, a, GET( 6), 23, 0x04881d05)
            STEP(H, a, b, c, d, GET( 9),  4, 0xd9d4d039)
            STEP(H, d, a, b, c, GET(12), 11, 0xe6db99e5)
            STEP(H, c, d, a, b, GET(15), 16, 0x1fa27cf8)
            STEP(H, b, c, d, a, GET( 2), 23, 0xc4ac5665)

            // Round 4
            STEP(I, a, b, c, d, GET( 0),  6, 0xf4292244)
            STEP(I, d, a, b, c, GET( 7), 10, 0x432aff97)
            STEP(I, c, d, a, b, GET(14), 15, 0xab9423a7)
            STEP(I, b, c, d, a, GET( 5), 21, 0xfc93a039)
            STEP(I, a, b, c, d, GET(12),  6, 0x655b59c3)
            STEP(I, d, a, b, c, GET( 3), 10, 0x8f0ccc92)
            STEP(I, c, d, a, b, GET(10), 15, 0xffeff47d)
            STEP(I, b, c, d, a, GET( 1), 21, 0x85845dd1)
            STEP(I, a, b, c, d, GET( 8),  6, 0x6fa87e4f)
            STEP(I, d, a, b, c, GET(15), 10, 0xfe2ce6e0)
            STEP(I, c, d, a, b, GET( 6), 15, 0xa3014314)
            STEP(I, b, c, d, a, GET(13), 21, 0x4e0811a1)
            STEP(I, a, b, c, d, GET( 4),  6, 0xf7537e82)
            STEP(I, d, a, b, c, GET(11), 10, 0xbd3af235)
            STEP(I, c, d, a, b, GET( 2), 15, 0x2ad7d2bb)
            STEP(I, b, c, d, a, GET( 9), 21, 0xeb86d391)

            a += saved_a;
        b += saved_b;
        c += saved_c;
        d += saved_d;

        data += 64;
        size -= 64;
    } while (size > 0);

    m_a = a;
    m_b = b;
    m_c = c;
    m_d = d;

    return data;

#undef STEP
#undef ROTATE_LEFT
#undef GET
#undef SET
}

// Update the MD5 sum by a data block.
void MD5_hasher::update(unsigned char const *data, size_t size)
{
    size_t old = size_t(m_count);

    m_count += size;

    size_t used = old & 0x3f;

    if (used > 0) {
        size_t free = 64 - used;

        if (size < free) {
            // just copy
            memcpy(&m_buffer[used], data, size);
            return;
        }

        // copy the first part
        memcpy(&m_buffer[used], data, free);
        data += free;
        size -= free;

        transform(m_buffer, 64);
    }

    if (size >= 64) {
        data = transform(data, size & ~size_t(0x3f));
        size &= size_t(0x3f);
    }
    memcpy(m_buffer, data, size);
}

// Finishes the calculation and returns the MD5 hash
void MD5_hasher::final(unsigned char result[16])
{
    size_t used = size_t(m_count) & 0x3f;

    // fill up
    m_buffer[used++] = 0x80;

    size_t free = 64 - used;

    if (free < 8) {
        // no space for length
        memset(&m_buffer[used], 0, free);
        transform(m_buffer, 64);
        used = 0;
        free = 64;
    }
    memset(&m_buffer[used], 0, free - 8);

    mi::Uint64 bits = m_count << 3u;
    mi::Uint32 lo   = mi::Uint32(bits);
    mi::Uint32 hi   = mi::Uint32(bits >> 32);

    m_buffer[56] = lo;
    m_buffer[57] = lo >> 8;
    m_buffer[58] = lo >> 16;
    m_buffer[59] = lo >> 24;
    m_buffer[60] = hi;
    m_buffer[61] = hi >> 8;
    m_buffer[62] = hi >> 16;
    m_buffer[63] = hi >> 24;

    transform(m_buffer, 64);

    result[ 0] = m_a;
    result[ 1] = m_a >> 8;
    result[ 2] = m_a >> 16;
    result[ 3] = m_a >> 24;
    result[ 4] = m_b;
    result[ 5] = m_b >> 8;
    result[ 6] = m_b >> 16;
    result[ 7] = m_b >> 24;
    result[ 8] = m_c;
    result[ 9] = m_c >> 8;
    result[10] = m_c >> 16;
    result[11] = m_c >> 24;
    result[12] = m_d;
    result[13] = m_d >> 8;
    result[14] = m_d >> 16;
    result[15] = m_d >> 24;

    restart();
}

}  // mdl
}  // mi
