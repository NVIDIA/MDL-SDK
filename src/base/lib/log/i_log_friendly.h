/******************************************************************************
 * Copyright (c) 2010-2023, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief

#ifndef BASE_LIB_LOG_I_LOG_FRIENDLY_H
#define BASE_LIB_LOG_I_LOG_FRIENDLY_H

#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>

#include <base/lib/mem/i_mem_size.h>


namespace MI {
namespace LOG {

/** iostream helper to print large numbers with a prefix.

 Numbers of this type will be printed with the appropriate SI prefix.

 \sa #MI::LOG::Prefixed
 \sa #MI::LOG::round
 \sa #MI::LOG::prefix_threshold
 */
template <typename T>
struct Prefixed
{
    using base_type = T;

    base_type value;

    explicit Prefixed(const T val={})
    : value(val) {}
};

template <typename T>
inline bool operator==(const Prefixed<T>& l, const Prefixed<T>& r)
{ return l.value == r.value; }


/** iostream helper to print a number with a unit.

 If the underlying type is \c Prefixed<T>, the unit will receive SI unit
 prefixes as appropriate.
 */
template <typename Tag, typename T=Prefixed<size_t>>
struct Quantity
{
    using base_type = T;
    base_type value;

    Quantity()
    : value{} {}

    explicit Quantity(const T val)
    : value{val} {}
};
template <typename Tag, typename T>
struct Quantity<Tag,Prefixed<T>>
{
    Prefixed<T> value;

    Quantity()
    : value{} {}

    explicit Quantity(const T val)
    : value{Prefixed<T>{val}} {}

    explicit Quantity(const Prefixed<T> val)
    : value{val} {}
};


// implementation details
namespace DETAIL {


inline long& get_prefix_threshold(std::ostream& str)
{
    static const int idx = std::ios_base::xalloc();
    return str.iword(idx);
}


inline long& get_rounding(std::ostream& str)
{
    static const int idx = std::ios_base::xalloc();
    return str.iword(idx);
}


struct Threshold
{
    size_t value = 0;
};


inline std::ostream& operator<<(std::ostream& str, const Threshold t)
{
    get_prefix_threshold(str) = t.value;
    return str;
}


struct Config
{
    long& threshold;
    long& round;

    Config(std::ostream& str)
    : threshold{get_prefix_threshold(str)}
    , round{get_rounding(str)}
    {}

    ~Config()
    {
        threshold = 0;
        round = false;
    }

    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
};


template <typename T>
inline std::ostream& print(std::ostream& str, const Prefixed<T> num)
{
    const Config cfg{str};
    if (num.value < size_t(cfg.threshold)) {
        return str << num.value;
    }

    unsigned i = 0;
    const char* prefix[] = {" ", " k", " M", " G", " T", " P", " E"};
    double val = (double)num.value;
    for (; val >= 1000.; ++i) {
        val /= 1000.;
    }

    str << std::fixed << std::setprecision(3);
    return (cfg.round ? (str << (T)std::round(val)) : (str << val)) << prefix[i];
}

}


/** Utility macro to generate a new unit type.

 This macro generates a unit type called \p N of underlying type \p T.
 Printing instances of \p N will yield the value and the unit symbol \p S.

 \see #MAKE_PREFIXED_UNIT
 */
#define MAKE_UNIT(N,S,T) \
using N = MI::LOG::Quantity<struct N##_tag,T>; \
inline const char* get_unit_symbol(const N&) { return S; }


/** Utility macro to generate a new unit type.

 This macro generates a new unit type like #MAKE_UNIT does. In contrast
 to that macro, however, this macro generates a unit that is printed with
 SI prefixes.
 */
#define MAKE_PREFIXED_UNIT(N,S,T) MAKE_UNIT(N,S,MI::LOG::Prefixed<T>)


/** iostream helper to print human-readable byte counts. */
using MEM::Bytes;


/** iostream helper to print human-readable large numbers.

 Large numbers greater than 100000 will receive an SI prefix and will be rounded.
 */
struct Large_number : public Prefixed<size_t>
{
    using Prefixed<size_t>::Prefixed;
};


/** iostream helper to print item counts.

 Pushing instances of this class to an ostream will print the count
 followed by the correct singular or plural form. */
template <typename T=size_t>
struct Item_count
{
    T count;
    const char* const name;
    const char* const plural;

    Item_count(T c, const char* const n, const char* const p=nullptr)
    : count(c), name(n), plural(p) {}

    template <typename T_=T, typename U=typename T_::base_type>
    Item_count(U c, const char* const n, const char* const p=nullptr)
    : Item_count{T{c},n,p} {}
};


/** iostream helper to print large item counts.

 Like \c Item_count, large item counts will be printed with the appropriate singular
 or plural form. However, the number will be printed with the appropriate SI prefix and
 rounded, e.g. `Large_item_count{123456,"foo"}` will print as "123 k foos".
 */
struct Large_item_count : public Item_count<Large_number>
{
    Large_item_count(size_t c, const char* const n, const char* const p=nullptr)
    : Item_count<Large_number>(Large_number{c},n,p) {}
};


/** iostream helper to print numbers of seconds. */
MAKE_UNIT(Seconds,"s",double)

/** iostream helper to print prefixed Hertz (e.g. kHz) */
MAKE_PREFIXED_UNIT(Hertz,"Hz",size_t)


/** iostream manipulator that controls rounding of prefixed numbers.

 Note that this manipulator controls the output of the next \c Prefixed number
 and is then reset.

 \sa #MI::LOG::Prefixed
 */
inline std::ostream& round(std::ostream& str)
{
    DETAIL::get_rounding(str) = true;
    return str;
}


/** iostream manipulator that controls when numbers are prefixed.

 This manipulator sets a threshold for number prefixes. Values lower than the
 provided threshold will not receive a prefix.

 Note that this manipulator controls the output of the next \c Prefixed number
 and is then reset.

 \sa #MI::LOG::Prefixed
 */
inline DETAIL::Threshold prefix_threshold(const size_t t)
{
    return DETAIL::Threshold{t};
}

}
namespace MEM {

/** \brief Prints a human-readable presentation of the provided number of bytes to the given stream.

 This function converts \p bytes to kibi-, mebi-, ..., exbibytes as appropriate and prints the
 result.
 */
inline std::ostream& operator<<(std::ostream& str, const Bytes bytes)
{
    if (bytes.is_unknown()) {
        return str << "unknown";
    }

    int i = 0;
    const char* units[] = {" B", " KiB", " MiB", " GiB", " TiB", " PiB", " EiB"};
    double size = (double)bytes.get_count();
    while (size > 1024.) {
        size /= 1024.;
        ++i;
    }
    return str << std::fixed << std::setprecision(i==0?0:3) << size << units[i];
}

}
namespace LOG {
using MEM::operator<<;


template <typename T>
inline std::ostream& operator<<(std::ostream& str, const Prefixed<T> num)
{
    return DETAIL::print(str,num);
}


inline std::ostream& operator<<(std::ostream& str, const Large_number num)
{
    // Note: large numbers are automatically thresholded and rounded. Could make this configurable.
    str << round << prefix_threshold(100000);
    return DETAIL::print(str,num);
}


template <typename Tag, typename T>
inline std::ostream& operator<<(std::ostream& str, const Quantity<Tag,T>& value)
{
    return str << std::fixed << std::setprecision(3)
               << value.value << ' ' << get_unit_symbol(value);
}


template <typename Tag, typename T>
inline std::ostream& operator<<(std::ostream& str, const Quantity<Tag,Prefixed<T>>& value)
{
    return str << value.value << get_unit_symbol(value);
}


template <typename T>
inline std::ostream& operator<<(std::ostream& str, const Item_count<T>& value)
{
    str << value.count << ' ';
    if (value.count == T(1))
        return (str << value.name);
    else if (value.plural)
        return (str << value.plural);
    return (str << value.name << 's');
}


/// Converts an integer memory size into a human-readable form, e.g., 16777216 -> "16 MiB".
inline std::string get_readable_memory_size(const size_t mem_size)
{
    std::ostringstream str;
    str << Bytes(mem_size);
    return str.str();
}


/// Converts number into a human-readable form, e.g., 16777216 -> "16 M".
inline std::string get_readable_amount(const size_t num)
{
    std::ostringstream str;
    str << Large_number(num);
    return str.str();
}


inline std::ostream& operator<<(std::ostream& str, const Seconds& value)
{
    if (value.value >= 60) {
        std::chrono::duration<Seconds::base_type> rest{value.value};
        const auto h = std::chrono::duration_cast<std::chrono::hours>(rest);
        const auto min = std::chrono::duration_cast<std::chrono::minutes>(rest -= h);
        const auto s = std::chrono::duration_cast<std::chrono::seconds>(rest -= min);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(rest -= s);

        if (h.count()) str << h.count() << ':';
        if (h.count() || min.count()) str << std::setw(2) << std::setfill('0') << min.count() << ':';
        str << std::setw(2) << std::setfill('0') << s.count();
        if (ms.count()) str << '.' << std::setw(3) << std::setfill('0') << ms.count();
        return str;
    }
    else {
        unsigned i = 0;
        const char* units[] = {" s", " ms", " us", " ns"};
        double scaled = value.value;
        for (; i+1 < sizeof(units)/sizeof(units[0]) && scaled < 1.; ++i) {
            scaled *= 1000.;
        }
        return str << std::fixed << std::setprecision(3) << scaled << units[i];
    }
}



} // namespace LOG
}  // namespace MI

#endif // BASE_LIB_LOG_I_LOG_FRIENDLY_H
