/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The definition of some type traits.
///
/// Type traits allow developers to determine various characteristics about types at compile time.

#ifndef BASE_SYSTEM_STLEXT_TYPE_TRAITS_H
#define BASE_SYSTEM_STLEXT_TYPE_TRAITS_H

#include "i_stlext_cv_traits.h"
#include "i_stlext_type_traits_base_types.h"
#include "stlext_type_traits_helper.h"
#include <cstddef>

namespace MI
{
namespace STLEXT
{

//--------------------------------------------------------------------------------------------------

/// Is a given type \p T volatile?
/// Usage
/// \code
///  MI_CHECK_EQUAL(is_volatile<int>::value, false);
///  MI_CHECK_EQUAL(is_volatile<volatile int>::value, true);
/// \endcode
template <typename T>
struct is_volatile : Integral_constant<bool, ::MI::STLEXT::detail::cv_traits_imp<T*>::is_volatile>
{};
/// Is a given type \p T volatile? This is the partial specialization for non-const references.
template <typename T>
struct is_volatile<T&> : False_type     // was originally integral_constant<bool, false>
{};


//--------------------------------------------------------------------------------------------------

/// Is a given type \p T const?
/// Usage
/// \code
///  MI_CHECK_EQUAL(is_const<int const*>::value, false);
///  MI_CHECK_EQUAL(is_const<int const* const>::value, true);
/// \endcode
template <typename T>
struct is_const : Integral_constant<bool, ::MI::STLEXT::detail::cv_traits_imp<T*>::is_const>
{};
/// Is a given type \p T const? Partial specialization for non-const references.
template <typename T>
struct is_const<T&> : False_type        // was originally integral_constant<bool, false>
{};


//--------------------------------------------------------------------------------------------------

namespace detail {

/// Little utility using \c is_volatile and hence in this file at this place.
template <typename T>
struct remove_const_impl
{
    typedef typename remove_const_helper<
        typename cv_traits_imp<T*>::unqualified_type,
        ::MI::STLEXT::is_volatile<T>::value
    >::type type;
};

} // namespace detail


/// Remove \c const from the given type \p T.
template <typename T>
struct remove_const
{
    typedef typename detail::remove_const_impl<T>::type type;
};
/// Remove \c const from the given type \p T - partial specialization for non-const references.
template <typename T>
struct remove_const<T&>
{
    typedef T& result;
};

/// Remove \c const from the given type \p T - partial specialization for arrays.
template <typename T, size_t N>
struct remove_const<T const[N]>
{
    typedef T type[N];
};

/// Remove \c const from the given type \p T - partial specialization for volatile arrays.
template <typename T, size_t N>
struct remove_const<T const volatile[N]>
{
    typedef T volatile type[N];
};


//--------------------------------------------------------------------------------------------------

/// Retrieve whether the given type \p T is a POD or not. Since some compilers start providing
/// intrinsics for it this type trait provides a wrapper around the different (sometimes non-)
/// implementations.
///
/// Usage
/// \code
///  MI_CHECK_EQUAL(is_pod<char*>::value, true);
///  MI_CHECK_EQUAL(is_pod<std::string>::value, false);
/// \endcode
template <typename T>
struct is_pod : Integral_constant<bool, ::MI::STLEXT::detail::is_pod_impl<T>::value>
{};


//--------------------------------------------------------------------------------------------------


namespace detail {

template <typename From, typename To>
class is_convertible_impl
{
    struct Two_chars {
        char dummy[2];
    };
    static char       test(To);
    static Two_chars  test(...);
    static From m_from;
public:
    enum {
        // If 'From' is convertible to 'To', then the first overload of the 'test' function
        // will be used, which returns char, thus comparing its size to sizeof(char) will
        // evaluate to true. In any other case the second function is used, which returns
        // Two_chars, thus comparing its size to sizeof(char) evaluates to false.
        // See Alexandrescu's article "Generic<Programming>: Mappings between Types and Values"
        // (Dr. Dobb's 1/10/2000) for a detailed description.
        value = sizeof(test(m_from)) == sizeof(char)
    };
};

} // namespace detail

/// Retrieve whether the given type \p From is convertible to the given type \p To
template <typename From, typename To>
class is_convertible: public Integral_constant<bool, detail::is_convertible_impl<From, To>::value>
{
    enum {
        value = detail::is_convertible_impl<From, To>::value
    };
};


}
}

#endif
