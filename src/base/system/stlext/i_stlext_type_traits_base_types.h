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
/// \brief The definition of some basic types, eg \c True_type and \c False_type.

#include "stlext_type_traits_defs.h"

#ifndef BASE_SYSTEM_STLEXT_TYPE_TRAITS_BASE_TYPES_H
#define BASE_SYSTEM_STLEXT_TYPE_TRAITS_BASE_TYPES_H

namespace MI {
namespace STLEXT {

/// The base class for value-based type traits.
template <typename T, T val>
struct Integral_constant
{
    typedef Integral_constant<T, val>  type;
    typedef T                          value_type;
    MI_BOOST_STATIC_CONSTANT(T, value = val);
};

// Since we haven't defined this yet it should always be initialized here.
#if !defined(MI_BOOST_NO_INCLASS_MEMBER_INITIALIZATION)
template <typename T, T val>
T const Integral_constant<T, val>::value;
#endif

/// The convenient typedef for the \c true case.
typedef Integral_constant<bool, true>  True_type;
/// The convenient typedef for the \c false case.
typedef Integral_constant<bool, false> False_type;

}
}

#endif
