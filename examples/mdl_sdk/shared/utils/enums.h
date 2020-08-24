/******************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

 // examples/mdl_sdk/shared/utils/enum_flags.h
 //
 // Code shared by all examples

#ifndef EXAMPLE_SHARED_UTILS_ENUMS_H
#define EXAMPLE_SHARED_UTILS_ENUMS_H

#include <type_traits>
#include <stdint.h>

namespace mi { namespace examples { namespace enums
{
    template<typename TEnum>
    using is_scoped_enum =
        std::integral_constant<bool, std::is_enum<TEnum>::value &&
        !std::is_convertible<TEnum, size_t>::value>;

    // Cast an enum (mask) to the underlying integer type
    template<class TEnum, typename = typename std::enable_if<is_scoped_enum<TEnum>::value>::type>
    constexpr typename std::underlying_type<TEnum>::type to_integer(TEnum e) noexcept {
        return static_cast<typename std::underlying_type<TEnum>::type>(e);
    }

    // Cast from the underlying integer type to an enum (mask).
    template<class TEnum, typename = typename std::enable_if<is_scoped_enum<TEnum>::value>::type>
    constexpr TEnum from_integer(typename std::underlying_type<TEnum>::type v) noexcept {
        return static_cast<TEnum>(v);
    }

    // Check if an enum bit-mask has a certain bit set.
    template<class TEnum, typename = typename std::enable_if<is_scoped_enum<TEnum>::value>::type>
    inline bool has_flag(const TEnum mask, const TEnum flag_to_check) {
        return (to_integer(mask) & to_integer(flag_to_check)) > 0;
    }

    // Set a certain bit in an enum bit-mask.
    template<class TEnum, typename = typename std::enable_if<is_scoped_enum<TEnum>::value>::type>
    inline TEnum set_flag(const TEnum& mask, const TEnum flag_to_add) {
        return static_cast<TEnum>(to_integer(mask) | to_integer(flag_to_add));
    }

    // Remove a certain bit from an enum bit-mask.
    template<class TEnum, typename = typename std::enable_if<is_scoped_enum<TEnum>::value>::type>
    inline TEnum remove_flag(const TEnum& mask, const TEnum flag_to_add) {
        return static_cast<TEnum>(to_integer(mask) & ~to_integer(flag_to_add));
    }

    // Toggle a certain bit from an enum bit-mask.
    template<class TEnum, typename = typename std::enable_if<is_scoped_enum<TEnum>::value>::type>
    inline TEnum toggle_flag(const TEnum& mask, const TEnum flag_to_add) {
        return static_cast<TEnum>(to_integer(mask) ^ to_integer(flag_to_add));
    }

}}}
#endif
