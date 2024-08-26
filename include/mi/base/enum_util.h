/***************************************************************************************************
 * Copyright (c) 2009-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MI_BASE_ENUM_UTIL_H
#define MI_BASE_ENUM_UTIL_H

#if (__cplusplus >= 201402L)
#include "config.h"

#include <type_traits>

// internal utility for MI_MAKE_ENUM_BITOPS
#define MI_MAKE_ENUM_BITOPS_PAIR(Enum,OP) \
MI_HOST_DEVICE_INLINE constexpr Enum operator OP(const Enum l, const Enum r) { \
    using Basic = std::underlying_type_t<Enum>; \
    return static_cast<Enum>(static_cast<Basic>(l) OP static_cast<Basic>(r)); } \
MI_HOST_DEVICE_INLINE Enum& operator OP##=(Enum& l, const Enum r) { \
    using Basic = std::underlying_type_t<Enum>; \
    return reinterpret_cast<Enum&>(reinterpret_cast<Basic&>(l) OP##= static_cast<Basic>(r)); }

// internal utility for MI_MAKE_ENUM_BITOPS
#define MI_MAKE_ENUM_SHIFTOPS_PAIR(Enum,OP) \
template <typename T, std::enable_if_t<std::is_integral<T>::value,bool> = true> \
MI_HOST_DEVICE_INLINE constexpr Enum operator OP(const Enum e, const T s) { \
    return static_cast<Enum>(static_cast<std::underlying_type_t<Enum>>(e) OP s); } \
template <typename T, std::enable_if_t<std::is_integral<T>::value,bool> = true> \
MI_HOST_DEVICE_INLINE constexpr Enum& operator OP##=(Enum& e, const T s) { \
    return reinterpret_cast<Enum&>(reinterpret_cast<std::underlying_type_t<Enum>&>(e) OP##= s); }


/// Utility to define binary operations on enum types.
///
/// Note that the resulting values may not have names in the given enum type.
#define MI_MAKE_ENUM_BITOPS(Enum) \
MI_MAKE_ENUM_BITOPS_PAIR(Enum,|) \
MI_MAKE_ENUM_BITOPS_PAIR(Enum,&) \
MI_MAKE_ENUM_BITOPS_PAIR(Enum,^) \
MI_MAKE_ENUM_BITOPS_PAIR(Enum,+) \
MI_MAKE_ENUM_BITOPS_PAIR(Enum,-) \
MI_MAKE_ENUM_SHIFTOPS_PAIR(Enum,<<) \
MI_MAKE_ENUM_SHIFTOPS_PAIR(Enum,>>) \
MI_HOST_DEVICE_INLINE constexpr Enum operator ~(const Enum e) { \
    return static_cast<Enum>(~static_cast<std::underlying_type_t<Enum>>(e)); }

namespace mi {

/** \brief Converts an enumerator to its underlying integral value. */
template <typename T, std::enable_if_t<std::is_enum<T>::value,bool> = true>
constexpr auto to_underlying(const T val) { return std::underlying_type_t<T>(val); }

template <typename T, std::enable_if_t<std::is_integral<T>::value,bool> = true>
constexpr auto to_underlying(const T val) { return val; }

}

#else
#define MI_MAKE_ENUM_BITOPS(Enum)
#endif

#endif //MI_BASE_ENUM_UTIL_H
