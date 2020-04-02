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
/// \file mi/base/version.h
/// \brief Major and minor version number and an optional qualifier.
/// 
/// See \ref mi_base_version.

#ifndef MI_BASE_VERSION_H
#define MI_BASE_VERSION_H

#include <mi/base/config.h>

/** \defgroup mi_base_version Versioning of the Base API
    \ingroup mi_base
    \brief The Base API has a major and minor version number and an optional qualifier.

    Version numbers and what they tell about compatibility across versions 
    is explained in \ref mi_base_intro_versioning.

    \par Include File:
    <tt> \#include <mi/base/version.h></tt>

    @{
 */

// No DLL API yet, make this API version 0 and do not document it in the manual.
//
// API version number.
//
// A change in this version number indicates that the binary compatibility
// of the interfaces offered through the shared library have changed. 
// Older versions can then explicitly asked for.
#define MI_BASE_API_VERSION  0


// The following three to four macros define the API version.
// The macros thereafter are defined in terms of the first four.

/// Base API major version number
///
/// \see \ref mi_base_intro_versioning
#define MI_BASE_VERSION_MAJOR  1

/// Base API minor version number
///
/// \see \ref mi_base_intro_versioning
#define MI_BASE_VERSION_MINOR  0

/// Base API version qualifier
///
/// The version qualifier is a string such as \c "alpha",
/// \c "beta", or \c "beta2", or the empty string \c "" if this is a final 
/// release, in which case the macro \c MI_BASE_VERSION_QUALIFIER_EMPTY
/// is defined as well.
///
/// \see  \ref mi_base_intro_versioning
#define MI_BASE_VERSION_QUALIFIER  ""

// This macro is defined if #MI_BASE_VERSION_QUALIFIER is the empty string \c "".
#define MI_BASE_VERSION_QUALIFIER_EMPTY

/// Base API major and minor version number without qualifier in a
/// string representation, such as \c "1.1".
#define MI_BASE_VERSION_STRING  MI_BASE_STRINGIZE(MI_BASE_VERSION_MAJOR) "." \
                                MI_BASE_STRINGIZE(MI_BASE_VERSION_MINOR)

/// \def MI_BASE_VERSION_QUALIFIED_STRING
/// Base API major and minor version number and qualifier in a
/// string representation, such as \c "1.1" or \c "1.2-beta2".
#ifdef MI_BASE_VERSION_QUALIFIER_EMPTY
#define MI_BASE_VERSION_QUALIFIED_STRING  MI_BASE_VERSION_STRING
#else
#define MI_BASE_VERSION_QUALIFIED_STRING  MI_BASE_VERSION_STRING "-" MI_BASE_VERSION_QUALIFIER
#endif // MI_BASE_VERSION_QUALIFIER_EMPTY


/*@}*/ // end group mi_base_version

#endif // MI_BASE_VERSION_H
