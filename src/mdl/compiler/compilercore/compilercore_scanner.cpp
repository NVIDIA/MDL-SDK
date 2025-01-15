/******************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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

// include the generated scanner from compilercore_parser.atg
#include "Scanner.cpp"

namespace mi {
namespace mdl {

void Scanner::ReportBufferDecoderError()
{
    errors->Error(line, col + 1, UTF8_DECODER_ERROR, Error_params(arena.get_allocator()));
}

int UTF8Buffer::Read() {
    int ch;
    int pos    = 0;
    bool error = false;

    do {
        ch = Buffer::Read();
        // until we find a utf8 start (0xxxxxxx or 11xxxxxx)
    } while ((ch >= 0x80) && ((ch & 0xC0) != 0xC0) && (ch != EoF));

    if (ch <= 0x7F || ch == EoF) {
        // nothing to do, first 127 chars are the same in ascii and utf8
        // 0xxxxxxx or end of file character
    } else if ((ch & 0xF8) == 0xF0) {
        // 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        int c1 = ch & 0x07;

        pos = Buffer::GetPos();
        ch  = Buffer::Read();
        if ((ch & 0xC0) != 0x80) {
            error = true;
        } else {
            int c2 = ch & 0x3F;

            pos = Buffer::GetPos();
            ch = Buffer::Read();
            if ((ch & 0xC0) != 0x80) {
                error = true;
            } else {
                int c3 = ch & 0x3F;

                pos = Buffer::GetPos();
                ch = Buffer::Read();
                error |= (ch & 0xC0) != 0x80;

                int c4 = ch & 0x3F;
                ch = (c1 << 18) | (c2 << 12) | (c3 << 6) | c4;

                // must be U+10000 .. U+10FFFF
                error |= (ch < 0x1000) || (ch > 0x10FFFF);

                // Because surrogate code points are not Unicode scalar values, any UTF-8 byte
                // sequence that would otherwise map to code points U+D800..U+DFFF is illformed
                error |= (0xD800 <= ch) && (ch <= 0xDFFF);
            }
        }
    } else if ((ch & 0xF0) == 0xE0) {
        // 1110xxxx 10xxxxxx 10xxxxxx
        int c1 = ch & 0x0F;

        pos = Buffer::GetPos();
        ch  = Buffer::Read();
        if ((ch & 0xC0) != 0x80) {
            error = true;
        } else {
            int c2 = ch & 0x3F;

            pos = Buffer::GetPos();
            ch  = Buffer::Read();

            error |= (ch & 0xC0) != 0x80;

            int c3 = ch & 0x3F;
            ch = (c1 << 12) | (c2 << 6) | c3;

            // must be U+0800 .. U+FFFF
            error |= ch < 0x0800;

            // Because surrogate code points are not Unicode scalar values, any UTF-8 byte
            // sequence that would otherwise map to code points U+D800..U+DFFF is illformed
            error |= (0xD800 <= ch) && (ch <= 0xDFFF);
        }
    } else if ((ch & 0xE0) == 0xC0) {
        // 110xxxxx 10xxxxxx
        int c1 = ch & 0x1F;

        pos = Buffer::GetPos();
        ch  = Buffer::Read();

        error |= (ch & 0xC0) != 0x80;

        int c2 = ch & 0x3F;
        ch = (c1 << 6) | c2;

        // must be U+0080 .. U+07FF
        error |= ch < 0x80;
    } else {
        // error
        pos = Buffer::GetPos();
        error = true;
    }
    if (error) {
        ch = 0xFFFD;  // replacement character
        Buffer::SetPos(pos);

        owner->ReportBufferDecoderError();
    }
    return ch;
}

} // mdl
} // mi
