/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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

/**
 * Converts a wchar_t * string into an utf8 encoded string.
 *
 * @param res   the result string
 * @param src   the wchar_t string to convert
 *
 * @return the result utf8 string
 *
 * @note handles utf16
 */
char const *wchar_to_utf8(string &res, wchar_t const *src);

/**
 * Converts one utf8 character to a utf32 encoded unicode character.
 *
 * \param[in]  up   pointer to the utf8 character
 * \param[out] res  unicode result in utf32 encoding
 *
 * \return pointer to the next utf8 character
 */
char const *utf8_to_unicode_char(char const *up, unsigned &res);

/**
 * Converts a utf8 encoded string into a utf16 encoded string.
 *
 * \param res  the result string
 * \param src  the utf8 string to convert
 *
 * @return the result utf16 string
 */
wchar_t const *utf8_to_utf16(wstring &res, char const *src);

/**
 * Converts a utf8 encoded string into a utf32 encoded string.
 *
 * \param res  the result string
 * \param src  the utf8 string to convert
 *
 * @return the result utf32 string
 */
unsigned const *utf8_to_utf32(u32string &res, char const *src);

/**
 * Converts a u32 string into an utf8 encoded string.
 *
 * @param res   the result string
 * @param src   the u32 string to convert
 *
 * @return the result utf8 string
 *
 * @note handles utf16
 */
char const *utf32_to_utf8(string &res, unsigned const *src);

}  // mdl
}  // mi

#endif
