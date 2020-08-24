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
/// \brief define platform-specific macros
///
/// - #DEBUG, #NDEBUG          signify building of a debug or release variant
/// - #MI_RESTRICTED_PTR       declare a restricted pointer in C++
/// - #MI_RESTRICTED_THIS_PTR  declare a restricted 'this' pointer
///
/// Guess the current OS and compiler, and define various preprocessor
/// symbols accordingly to abstract access to platform specific features.

#ifndef BASE_SYSTEM_MAIN_PLATFORM_H
#define BASE_SYSTEM_MAIN_PLATFORM_H

#include <mi/base/config.h>

// The ANSI standard X3.159-1989 ("ANSI C") mandates use of the preprocessor
// symbol "NDEBUG" to signify that a non-debugging version (release version) of
// the software is being built. Defining NDEBUG will typically disable calls to
// assert() and other consistency checks. 3rd-party software
// relies on this behavior. Proprietary Mental Images code,
// however, uses the preprocessor define "DEBUG" with a reversed meaning: a
// non-debugging build is signified by the absence of that symbol.
//
// In order to guarantee that all software is built consistently, the following
// preprocessor code enforced that exactly _one_ of the symbols NDEBUG or DEBUG
// is defined; the assumption defined(NDEBUG) <==> !defined(DEBUG) holds.

#ifdef DEBUG
#  ifdef NDEBUG
#    error "preprocessor defines DEBUG and NDEBUG are mutually exclusive"
#  endif
#else
#  ifndef NDEBUG
#    error "one of DEBUG and NDEBUG must be defined"
#  endif
#endif

// The C99 standard added the "restrict" keyword to specify that a given
// pointer does not alias to anything else in the current context, which is an
// information quite useful to the optimizer. ISO C++ does not offer that
// keyword, but many compilers support it nonetheless through alternative
// spellings. The define MI_RESTRICTED_PTR provides compiler-independent access
// to this extension. Use it as follows:
//
//   void f(char * MI_RESTRICTED_PTR cptr, int * MI_RESTRICTED_PTR iptr)
//   {
//      /* ... */
//   }
//
// In addition, some compilers allow to specify that the 'this' pointer in a
// member function call is restricted. The define MI_RESTRICTED_THIS_PTR
// captures the feature:
//
//   void Class::method() MI_RESTRICTED_THIS_PTR
//   {
//      /* ... */
//   }
//
// When 'this' is used inside of Class::method(), the pointer is effectively
// interpreted as having the type "Class * MI_RESTRICTED_PTR const". Note that
// the const qualifier applies to the _pointer_, not to the object.

#if defined(__GNUC__)			// GNU C/C++ Compiler
#  define MI_RESTRICTED_PTR      __restrict__
#  define MI_RESTRICTED_THIS_PTR __restrict__
#else
#  define MI_RESTRICTED_PTR
#  define MI_RESTRICTED_THIS_PTR
#endif

#endif // BASE_SYSTEM_MAIN_PLATFORM_H
