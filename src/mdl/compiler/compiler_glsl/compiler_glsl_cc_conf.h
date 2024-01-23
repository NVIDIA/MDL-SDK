/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_CC_CONF_H_H
#define MDL_COMPILER_GLSL_CC_CONF_H_H 1

// compiler specific stuff

#ifndef __has_feature
# define __has_feature(x) 0
#endif

/// GLSL_DELETED_FUNCTION - Expands to = delete if the compiler supports it.
/// Use to mark functions as uncallable.
#if (__has_feature(cxx_deleted_functions) || defined(__GXX_EXPERIMENTAL_CXX0X__))
// No version of MSVC currently supports this.
#define GLSL_DELETED_FUNCTION = delete
#else
#define GLSL_DELETED_FUNCTION
#endif

/// GLSL_FINAL - Expands to 'final' if the compiler supports it.
/// Use to mark classes or virtual methods as final.
#if __has_feature(cxx_override_control) || (defined(_MSC_VER) && _MSC_VER >= 1700)
#define GLSL_FINAL final
#else
#define GLSL_FINAL
#endif

/// GLSL_OVERRIDE - Expands to 'override' if the compiler supports it.
/// Use to mark virtual methods as overriding a base class method.
#if __has_feature(cxx_override_control) || (defined(_MSC_VER) && _MSC_VER >= 1700)
#define GLSL_OVERRIDE override
#else
#define GLSL_OVERRIDE
#endif

/// GLSL_CONSTEXPR - Expands to 'constexpr' if the compiler supports it.
/// Use to mark functions computing constant expressions.
#if __has_feature(cxx_constexpr) || defined(__GXX_EXPERIMENTAL_CXX0X__)
# define GLSL_CONSTEXPR constexpr
#else
# define GLSL_CONSTEXPR
#endif

#endif // MDL_COMPILER_GLSL_CC_CONF_H_H
