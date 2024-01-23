/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_WCHAR_SUPPORT_H
#define MDL_COMPILERCORE_WCHAR_SUPPORT_H 1

#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// Get the length of an utf16 encoded string in code points, not in words.
size_t utf16_len(wchar_t const *p);

/// Get the length of an utf8 encoded string in code points, not in bytes.
size_t utf8_len(char const *p);

/// Converts an utf16 string into an utf8 encoded string.
///
/// \param res   the result string
/// \param src   the utf816 string to convert
///
/// \return the result utf8 string
char const *utf16_to_utf8(string &res, wchar_t const *src);

/// Converts one utf8 character to a utf32 encoded unicode character.
///
/// \param[in]  up   pointer to the utf8 character
/// \param[out] res  unicode result in utf32 encoding
///
/// \return pointer to the next utf8 character
char const *utf8_to_unicode_char(char const *up, unsigned &res);

/// Converts a utf8 encoded string into a utf16 encoded string.
///
/// \param[inout] res  the result string
/// \param[in]    src  the utf8 string to convert
///
/// \return the result utf16 string
wchar_t const *utf8_to_utf16(wstring &res, char const *src);

/// Converts a utf8 encoded string into a utf32 encoded string.
///
/// \param[inout] res  the result string
/// \param[in]    src  the utf8 string to convert
///
/// \return the result utf32 string
unsigned const *utf8_to_utf32(u32string &res, char const *src);

/// Converts a u32 string into an utf8 encoded string.
///
/// \param[inout] res   the result string
/// \param[in]    src   the u32 string to convert
///
/// \return the result utf8 string
char const *utf32_to_utf8(string &res, unsigned const *src);

/// Add an unicode utf32 character to an utf16 string.
///
/// \param[inout] s  the utf16 string to be modified
/// \param[in]    c  the utf32 unicode character
///
/// \note Adds one or two wchars to s.
void utf16_append(wstring &s, unsigned c);

/// Get the next unicode character from an utf16 string.
///
/// \param[inout] p  pointer to an utf16 string
unsigned utf16_next(wchar_t const *&p);

/// Add an unicode utf32 character to an utf8 string.
///
/// \param[inout] s  the utf16 string to be modified
/// \param[in]    c  the utf32 unicode character
///
/// \note Adds one, two, three, or four chars to s.
void utf8_append(string &s, unsigned c);

/// Get the next unicode character from an utf8 string.
///
/// \param[inout] p  pointer to an utf8 string
unsigned utf8_next(char const *&p);

}  // mdl
}  // mi

#endif
