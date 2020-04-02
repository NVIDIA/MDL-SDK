/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief Some basic definitions and macros required for the type traits.

#ifndef BASE_SYSTEM_STLEXT_TYPE_TRAITS_DEFS_H
#define BASE_SYSTEM_STLEXT_TYPE_TRAITS_DEFS_H

/// The MI_BOOST_STATIC_CONSTANT workaround.
/// On compilers which don't allow in-class initialization of static integral constant members,
/// we must use enums as a workaround if we want the constants to be available at compile-time.
/// This macro gives us a convenient way to declare such constants.
#ifdef MI_BOOST_NO_INCLASS_MEMBER_INITIALIZATION
#  define MI_BOOST_STATIC_CONSTANT(type, assignment) enum { assignment }
#else
#  define MI_BOOST_STATIC_CONSTANT(type, assignment) static const type assignment
#endif

/// The MI_IS_POD macro which checks whether a given type is a POD or not.
/// This works reliable only on compilers with built-in intrinsics, hence the default is \c false.
#undef MI_IS_POD
#if defined(_MSC_VER) && (_MSC_VER >=1400)                  // since VC++ 8
#   define MI_IS_POD(T) (!__is_class(T) || (__is_pod(T) && __has_trivial_constructor(T)) )
#endif
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 2)   // since g++ 4.3
#   define MI_IS_POD(T) __is_pod(T)
#endif
#ifndef MI_IS_POD
#   define MI_IS_POD(T) false
#endif


#endif
