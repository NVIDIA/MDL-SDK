/******************************************************************************
 * Copyright (c) 2006-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file
/// \brief basic types definitions

#ifndef BASE_SYSTEM_MAIN_TYPES_H
#define BASE_SYSTEM_MAIN_TYPES_H

// Get MI_ARCH_ definitions (e.g. MI_ARCH_64BIT)
#include <mi/base/config.h>

#include <mi/base/types.h>

#ifdef __cplusplus
namespace MI {
#endif

using mi::Sint8;                         //< 8-bit signed integer
using mi::Uint8;                         //< 8-bit unsigned integer
using mi::Sint16;                        //< 16-bit signed integer
using mi::Uint16;                        //< 16-bit unsigned integer
using mi::Sint32;                        //< 32-bit signed integer
using mi::Uint32;                        //< 32-bit unsigned integer
using mi::Sint64;                        //< 64-bit signed integer
using mi::Uint64;                        //< 64-bit unsigned integer
using mi::Float32;                       //< 32-bit IEEE-754 single precision float
using mi::Float64;                       //< 64-bit IEEE-754 double precision float
using mi::Size;                          //< architecture-dependent unsigned word size
using mi::Difference;                    //< architecture-dependent signed word size

typedef Float32			Scalar;  //< 32-bit IEEE-754 single precision float
typedef Float64			Dscalar; //< 64-bit IEEE-754 double precision float
typedef mi::Uint32		Uint;    //< unsigned integer
typedef unsigned char		Uchar;	 //< unsigned character
typedef unsigned short		Ushort;	 //< unsigned short
typedef signed int		Sint;	 //< signed integer

#ifdef MI_PLATFORM_WINDOWS
#ifdef MI_ARCH_64BIT
typedef __int64			ssize_t;
#else
typedef long			ssize_t;
#endif
#endif

static const Uint null_index = (Uint)~0;  //< default index-1
static const Uint max_index  = (Uint)~1;  //< maximum index

// =============================================================================
//
// Integer types with the same sizeof() as pointers.
#ifdef __cplusplus
template <unsigned int Bytesize> struct Pointer_as_uint_;
template <> struct Pointer_as_uint_<4u> { typedef Uint32 Type; };
template <> struct Pointer_as_uint_<8u> { typedef Uint64 Type; };
typedef Pointer_as_uint_<(unsigned int)sizeof(void*)>::Type Pointer_as_uint;

template <unsigned int Bytesize> struct Pointer_as_sint_;
template <> struct Pointer_as_sint_<4u> { typedef Sint32 Type; };
template <> struct Pointer_as_sint_<8u> { typedef Sint64 Type; };
typedef Pointer_as_sint_<(unsigned int)sizeof(void*)>::Type Pointer_as_sint;
#endif

// =============================================================================
//
// Define macros for printf format strings which differ on 32-bit and 64-bit
// platforms. Use these defines as follows:
//
// printf("A 64 bit unsigned integer: %10.10" FMT_BIT64 "u", value);
#ifdef MI_PLATFORM_WINDOWS
#  define FMT_BIT64 "I64"
#  define FMT_BIT32 ""
#else
#  define FMT_BIT64 "ll"
#  define FMT_BIT32 ""
#endif

// For size_t, ssize_t, mi::Size, and mi::Difference. Use like below:
// size_t         value = ...; printf("Memory size is %" FMT_SIZE_T        " bytes", value);
// ssize_t        value = ...; printf("Memory size is %" FMT_SSIZE_T       " bytes", value);
// mi::Size       value = ...; printf("Memory size is %" MI_BASE_FMT_MI_SIZE       " bytes", value);
// mi::Difference value = ...; printf("Memory size is %" MI_BASE_FMT_MI_DIFFERENCE " bytes", value);
#if defined(MI_ARCH_64BIT) || defined(MI_PLATFORM_MACOSX)
#ifdef MI_PLATFORM_WINDOWS
#  define FMT_SIZE_T        "llu"
#  define FMT_SSIZE_T       "lld"
#else
#  define FMT_SIZE_T        "zu"
#  define FMT_SSIZE_T       "zd"
#endif
#else
#  define FMT_SIZE_T        FMT_BIT32 "u"
#  define FMT_SSIZE_T       FMT_BIT32 "d"
#endif

// For backward compatibility
#define FMT_MI_SIZE         MI_BASE_FMT_MI_SIZE
#define FMT_MI_DIFFERENCE   MI_BASE_FMT_MI_DIFFERENCE

// gcc allows to match the parameters of a printf like function against the
// format strings. The following defines can be used for this. They have to
// be put between a function prototype and the closing semicolon.
// The number behind the PRINTFLIKE specifies the position of the format string
// and assumes, that the ... is the next argument. Note that for class member
// you have to include the - invisible - "this" argument. So in that case if the
// format string is the first argument, use PRINTFLIKE2.
#ifdef __GNUC__
#define PRINTFLIKE1 __attribute__((format(printf, 1, 2)))
#define PRINTFLIKE2 __attribute__((format(printf, 2, 3)))
#define PRINTFLIKE3 __attribute__((format(printf, 3, 4)))
#define PRINTFLIKE4 __attribute__((format(printf, 4, 5)))
#define PRINTFLIKE5 __attribute__((format(printf, 5, 6)))
#else
#define PRINTFLIKE1
#define PRINTFLIKE2
#define PRINTFLIKE3
#define PRINTFLIKE4
#define PRINTFLIKE5
#endif

#ifdef __cplusplus
} // namespace MI
#endif

#endif
