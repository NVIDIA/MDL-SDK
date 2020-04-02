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
/// \file mi/math/assert.h
/// \brief Assertions and compile-time assertions.
///
/// See \ref mi_math_assert.

#ifndef MI_MATH_ASSERT_H
#define MI_MATH_ASSERT_H

#include <mi/base/config.h>
#include <mi/base/assert.h>

namespace mi {

namespace math {

/** \defgroup mi_math_assert \MathApiName Assertions
    \ingroup mi_math

    Assertions.

    \par Include File:
    <tt> \#include <mi/math/assert.h></tt>

    \MathApiName supports quality software development with assertions. They are contained in
    various places in the %math API include files.

    These tests are mapped to corresponding %base API assertions by default, which in turn are
    switched off by default to have the performance of a release build. To activate the tests, you
    need to define the two macros #mi_math_assert and #mi_math_assert_msg, or correspondingly the
    #mi_base_assert and #mi_base_assert_msg macros, before including the relevant include files.
    Defining only one of the two macros is considered an error.

    See also \ref mi_math_intro_config and \ref mi_base_intro_config in
    %general.

    @{
*/

#if       defined( mi_math_assert) && ! defined( mi_math_assert_msg) \
     || ! defined( mi_math_assert) &&   defined( mi_math_assert_msg)
error "Only one of mi_math_assert and mi_math_assert_msg has been defined. Please define both."
#else
#ifndef mi_math_assert

/// Math API assertion macro (without message).
///
/// If \c expr evaluates to \c true this macro shall have no effect. If \c expr evaluates to \c
/// false this macro may print a diagnostic message and change the control flow of the program, such
/// as aborting the program or throwing an exception. But it may also have no effect at all, for
/// example if assertions are configured to be disabled.
///
/// By default, this macro maps to #mi_base_assert, which in turn does nothing by default. You can
/// (re-)define this macro to perform possible checks and diagnostics within the specification given
/// in the previous paragraph.
///
/// \see \ref mi_math_intro_assert and
///      \ref mi_base_intro_assert
#define mi_math_assert(expr) mi_base_assert(expr)

/// Math API assertion macro (with message).
///
/// If \c expr evaluates to \c true this macro shall have no effect. If \c expr evaluates to \c
/// false this macro may print a diagnostic message and change the control flow of the program, such
/// as aborting the program or throwing an exception. But it may also have no effect at all, for
/// example if assertions are configured to be disabled.
///
/// The \c msg text string contains additional diagnostic information that may be shown with a
/// diagnostic message. Typical usages would contain \c "precondition" or \c "postcondition" as
/// clarifying context information in the \c msg parameter.
///
/// By default, this macro maps to #mi_base_assert_msg, which in turn does nothing by default. You
/// can (re-)define this macro to perform possible checks and diagnostics within the specification
/// given in the previous paragraph.
///
/// \see \ref mi_math_intro_assert and
///      \ref mi_base_intro_assert
#define mi_math_assert_msg(expr, msg) mi_base_assert_msg(expr, msg)

// Just for doxygen, begin

/// Indicates whether assertions are actually enabled. This symbol gets defined if and only if you
/// (re-)defined #mi_math_assert and #mi_math_assert_msg (or #mi_base_assert and
/// #mi_base_assert_msg). Note that you can not simply test for #mi_math_assert or
/// #mi_math_assert_msg, since these macros get defined in any case (if you do not (re-)define them,
/// they evaluate to a dummy statement that has no effect).
#define mi_math_assert_enabled
#undef mi_math_assert_enabled

// Just for doxygen, end

#ifdef mi_base_assert_enabled
#define mi_math_assert_enabled
#endif // mi_math_assert_enabled

#else // mi_math_assert

#define mi_math_assert_enabled

#endif // mi_math_assert
#endif // mi_math_assert xor mi_math_assert_msg


/*@}*/ // end group mi_math_assert

} // namespace math

} // namespace mi

#endif // MI_MATH_ASSERT_H
