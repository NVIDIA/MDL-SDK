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
/// \brief The definition of some implementation helpers.

#ifndef BASE_SYSTEM_STLEXT_TYPE_TRAITS_HELPER_H
#define BASE_SYSTEM_STLEXT_TYPE_TRAITS_HELPER_H

#include "stlext_type_traits_defs.h"
#include "stlext_mpl.h"

#include <cstddef>

namespace MI
{
namespace STLEXT
{

namespace detail
{

//==================================================================================================

/// Little helper to remove the const from the given type T.
template <typename T, bool is_vol>
struct remove_const_helper
{
    typedef T type;
};

/// Specialization of the above helper for volatile types T.
template <typename T>
struct remove_const_helper<T, true>
{
    typedef T volatile type;
};


//==================================================================================================

//--------------------------------------------------------------------------------------------------

/// Is type T simple? Default is no.
template <typename T>
struct is_simple : False_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<bool> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<char> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<signed char> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<unsigned char> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<short> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<unsigned short> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<int> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<unsigned int> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<long> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<unsigned long> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<long long> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<unsigned long long> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<float> : True_type
{};
/// Is type T simple? Specializations.
template <>
struct is_simple<double> : True_type
{};
/// Is type T simple? Partial specializations.
template <typename T>
struct is_simple<T*> : True_type
{};


//--------------------------------------------------------------------------------------------------

/// Little helper for implementing \c is_pod<>.
template <typename T>
struct is_pod_impl
{
    // OR-concatenating both MI_IS_POD() and is_simple<>
    MI_BOOST_STATIC_CONSTANT(bool, value = (
        ::MI::STLEXT::mpl::bool_func_or<
            MI_IS_POD(T),
            is_simple<T>::value
         >::value
        )
    );
};

/// Specialization for array types. It is based simply on its single-type equivalent.
template <typename T, std::size_t sz>
struct is_pod_impl<T[sz]> : is_pod_impl<T>
{};

/// Specialization for void type.
template <>
struct is_pod_impl<void>
{
    MI_BOOST_STATIC_CONSTANT(bool, value = true);
};

/// Specialization for void const type.
template <>
struct is_pod_impl<void const>
{
    MI_BOOST_STATIC_CONSTANT(bool, value = true);
};

/// Specialization for void volatile type.
template <>
struct is_pod_impl<void volatile>
{
    MI_BOOST_STATIC_CONSTANT(bool, value = true);
};

/// Specialization for void const volatile type.
template <>
struct is_pod_impl<void const volatile>
{
    MI_BOOST_STATIC_CONSTANT(bool, value = true);
};


// Since we haven't defined this yet it should always be initialized here.
#if !defined(MI_BOOST_NO_INCLASS_MEMBER_INITIALIZATION)
template <typename T> bool const is_pod_impl<T>::value;
//template <> bool const is_pod_impl<void>::value;
//template <> bool const is_pod_impl<void const>::value;
//template <> bool const is_pod_impl<void volatile>::value;
//template <> bool const is_pod_impl<void const volatile>::value;
#endif

}

}
}

#endif
