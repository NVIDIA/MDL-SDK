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
/// \brief Major and minor version number and an optional qualifier.
///
/// See \ref mi_neuray_version

#ifndef MI_NEURAYLIB_VERSION_H
#define MI_NEURAYLIB_VERSION_H

#include <mi/base/config.h>

/** \defgroup mi_neuray_version Versioning of the \neurayApiName
    \ingroup mi_neuray
    \brief The \neurayApiName has a major and minor version number and an optional qualifier.

    Version numbers and what they tell about compatibility across versions
    is explained in \ref mi_base_intro_versioning.

    \par Include File:

    @{
 */

/// API version number.
///
/// A change in this version number indicates that the binary compatibility
/// of the interfaces offered through the shared library have changed.
#define MI_NEURAYLIB_API_VERSION  43

// The following three to four macros define the API version.
// The macros thereafter are defined in terms of the first four.

/// \NeurayApiName major version number
///
/// \see \ref mi_base_intro_versioning
#define MI_NEURAYLIB_VERSION_MAJOR  4

/// \NeurayApiName minor version number
///
/// \see \ref mi_base_intro_versioning
#define MI_NEURAYLIB_VERSION_MINOR  1

/// \NeurayApiName version qualifier
///
/// The version qualifier is a string such as \c "alpha",
/// \c "beta", or \c "beta2", or the empty string \c "" if this is a final
/// release, in which case the macro \c MI_NEURAYLIB_VERSION_QUALIFIER_EMPTY
/// is defined as well.
///
/// \see \ref mi_base_intro_versioning
#define MI_NEURAYLIB_VERSION_QUALIFIER  ""

// This macro is defined if #MI_NEURAYLIB_VERSION_QUALIFIER is the empty string \c "".
#define MI_NEURAYLIB_VERSION_QUALIFIER_EMPTY

/// \NeurayApiName major and minor version number without qualifier in a
/// string representation, such as \c "2.0".
#define MI_NEURAYLIB_VERSION_STRING  MI_BASE_STRINGIZE(MI_NEURAYLIB_VERSION_MAJOR) "." \
                                     MI_BASE_STRINGIZE(MI_NEURAYLIB_VERSION_MINOR)

/// \def MI_NEURAYLIB_VERSION_QUALIFIED_STRING
/// \NeurayApiName major and minor version number and qualifier in a
/// string representation, such as \c "2.0" or \c "2.0-beta2".
#ifdef MI_NEURAYLIB_VERSION_QUALIFIER_EMPTY
#define MI_NEURAYLIB_VERSION_QUALIFIED_STRING  MI_NEURAYLIB_VERSION_STRING
#else
#define MI_NEURAYLIB_VERSION_QUALIFIED_STRING  MI_NEURAYLIB_VERSION_STRING "-" \
                                               MI_NEURAYLIB_VERSION_QUALIFIER
#endif // MI_NEURAYLIB_VERSION_QUALIFIER_EMPTY

/// \NeurayProductName product version number in a string representation, such as \c "2.0".
#define MI_NEURAYLIB_PRODUCT_VERSION_STRING  "2020.1.2"

/// Type of plugins for the \NeurayApiName.
/// \see #mi::base::Plugin::get_type().
#define MI_NEURAYLIB_PLUGIN_TYPE "neuray API v33"

// Enables features that were deprecated with version 11.1.
// #define MI_NEURAYLIB_DEPRECATED_11_1

// Prefixes names of deprecated methods unless deprecated features are enabled.
#ifdef MI_NEURAYLIB_DEPRECATED_11_1
#define MI_NEURAYLIB_DEPRECATED_METHOD_11_1(X) X
#else
#define MI_NEURAYLIB_DEPRECATED_METHOD_11_1(X) deprecated_##X
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_LEGACY_MDL_API
#warning Support for macro \
    MI_NEURAYLIB_DEPRECATED_LEGACY_MDL_API \
    has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_ITRANSACTION_STORE_DEFAULT_PRIVACY_LEVEL_ZERO
#warning Support for macro \
    MI_NEURAYLIB_DEPRECATED_ITRANSACTION_STORE_DEFAULT_PRIVACY_LEVEL_ZERO \
    has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_IDICE_TRANSACTION_STORE_DEFAULT_PRIVACY_LEVEL_ZERO
#warning Support for macro \
    MI_NEURAYLIB_DEPRECATED_IDICE_TRANSACTION_STORE_DEFAULT_PRIVACY_LEVEL_ZERO \
    has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_NAMESPACE_MI_TRANSITION
#warning Support for macro MI_NEURAYLIB_DEPRECATED_NAMESPACE_MI_TRANSITION has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_NO_EXPLICIT_TRANSACTION
#warning Support for macro MI_NEURAYLIB_DEPRECATED_NO_EXPLICIT_TRANSACTION has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_7_1
#warning Support for macro MI_NEURAYLIB_DEPRECATED_7_1 has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_7_2
#warning Support for macro MI_NEURAYLIB_DEPRECATED_7_2 has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_7_3
#warning Support for macro MI_NEURAYLIB_DEPRECATED_7_3 has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_8_0
#warning Support for macro MI_NEURAYLIB_DEPRECATED_8_0 has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_8_1
#warning Support for macro MI_NEURAYLIB_DEPRECATED_8_1 has been removed
#endif

#ifdef MI_NEURAYLIB_DEPRECATED_9_1
#warning Support for macro MI_NEURAYLIB_DEPRECATED_9_1 has been removed
#endif

/*@}*/ // end group mi_neuray_version

#endif // MI_NEURAYLIB_VERSION_H
