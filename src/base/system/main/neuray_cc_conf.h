/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef BASE_SYSTEM_MAIN_NEURAY_CC_CONF_H
#define BASE_SYSTEM_MAIN_NEURAY_CC_CONF_H 1

/// Compiler specific stuff, mostly C++1X support, inspired by LLVM.

#ifndef __has_feature
# define __has_feature(x) 0
#endif

#ifndef __has_attribute
# define __has_attribute(x) 0
#endif

#ifndef __has_builtin
# define __has_builtin(x) 0
#endif

/// \macro __GNUC_PREREQ
/// Defines __GNUC_PREREQ if glibc's features.h isn't available.
#ifndef __GNUC_PREREQ
# if defined(__GNUC__) && defined(__GNUC_MINOR__)
#  define __GNUC_PREREQ(maj, min) ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
# else
#  define __GNUC_PREREQ(maj, min) 0
# endif
#endif

/// NEURAY_DELETED_FUNCTION - Expands to = delete if the compiler supports it.
/// Use to mark functions as uncallable.
#if (__has_feature(cxx_deleted_functions) || defined(__GXX_EXPERIMENTAL_CXX0X__))
// No version of MSVC currently supports this.
#define NEURAY_DELETED_FUNCTION = delete
#else
#define NEURAY_DELETED_FUNCTION
#endif

/// NEURAY_FINAL - Expands to 'final' if the compiler supports it.
/// Use to mark classes or virtual methods as final.
#if __has_feature(cxx_override_control) || (defined(_MSC_VER) && _MSC_VER >= 1700)
#define NEURAY_FINAL final
#else
#define NEURAY_FINAL
#endif

/// NEURAY_OVERRIDE - Expands to 'override' if the compiler supports it.
/// Use to mark virtual methods as overriding a base class method.
#if __has_feature(cxx_override_control) || (defined(_MSC_VER) && _MSC_VER >= 1700)
#define NEURAY_OVERRIDE override
#else
#define NEURAY_OVERRIDE
#endif

/// NEURAY_CONSTEXPR - Expands to 'constexpr' if the compiler supports it.
/// Use to mark functions computing constant expressions.
#if __has_feature(cxx_constexpr) || (defined(_MSC_VER) && (_MSC_VER >= 1900))
# define NEURAY_CONSTEXPR constexpr
#else
# define NEURAY_CONSTEXPR
#endif

/// NEURAY_WARN_UNUSED_RESULT - Expands to '__attribute__((__warn_unused_result__))'
/// if the compiler supports it.
/// Use to mark functions whose result should not be ignored.
#if __has_attribute(warn_unused_result) || __GNUC_PREREQ(3, 4)
#define NEURAY_WARN_UNUSED_RESULT __attribute__((__warn_unused_result__))
#else
#define NEURAY_WARN_UNUSED_RESULT
#endif

/// NEURAY_NOINLINE - On compilers where we have a directive to do so,
/// mark a method "not for inlining".
#if __has_attribute(noinline) || __GNUC_PREREQ(3, 4)
#define NEURAY_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define NEURAY_NOINLINE __declspec(noinline)
#else
#define NEURAY_NOINLINE
#endif

/// NEURAY_ALWAYS_INLINE - On compilers where we have a directive to do
/// so, mark a method "always inline" because it is performance sensitive. GCC
/// 3.4 supported this but is buggy in various cases and produces unimplemented
/// errors, just use it in GCC 4.0 and later.
#if __has_attribute(always_inline) || __GNUC_PREREQ(4, 0)
#define NEURAY_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define NEURAY_ALWAYS_INLINE __forceinline
#else
#define NEURAY_ALWAYS_INLINE
#endif

#endif  // BASE_SYSTEM_MAIN_NEURAY_CC_CONF_H
