/******************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

namespace MI {
namespace LOG {

/** iostream helper to print human readable byte counts. */
struct Bytes
{
    size_t bytes;

    explicit Bytes(size_t bytes=~size_t(0))
    : bytes(bytes) {}
};


/** iostream helper to print human readable numbers. */
struct Large_number
{
    size_t count;

    explicit Large_number(size_t val=0)
    : count(val) {}
};

inline bool operator==(const Large_number& l, const Large_number& r)
{ return l.count == r.count; }


/** iostream helper to print item counts.

 Pushing instances of this class to an ostream will print the count
 followed by the correct singular or plural form. */
template <typename T=size_t>
struct Item_count
{
    T count;
    const char* const name;
    const char* const plural;

    Item_count(size_t c, const char* const n, const char* const p=nullptr)
    : count(c), name(n), plural(p) {}
};


/** iostream helper to print large item counts. */
struct Large_item_count : public Item_count<Large_number>
{
    Large_item_count(size_t c, const char* const n, const char* const p=nullptr)
    : Item_count<Large_number>(c,n,p) {}
};


/** iostream helper to print numbers of seconds.

 This is mainly used to make printing of timings consistent and may be replaced
 by std::chrono.
 */
struct Seconds
{
    double seconds;

    Seconds(double s)
    : seconds(s) {}
};



/** \brief Prints a 'readable' presentation of the provided number of bytes to the given stream.

 This function converts \p bytes to kibi-, mebi-, ..., exbibytes as appropriate and prints the
 result.
 */
inline std::ostream& operator<<(std::ostream& str, const Bytes bytes)
{
    if (bytes.bytes == ~size_t(0)) {
        return str << "unknown";
    }

    int i = 0;
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    double size = (double)bytes.bytes;
    while (size > 1024.) {
        size /= 1024.;
        ++i;
    }
    return str << std::fixed << std::setprecision(3) << size << ' ' << units[i];
}


/// Converts an integer memory size into a human-readable form, e.g., 16777216 -> "16 MiB".
inline std::string get_readable_memory_size(size_t mem_size)
{
    std::ostringstream str;
    str << Bytes(mem_size);
    return str.str();
}


inline std::ostream& operator<<(std::ostream& str, const Large_number num)
{
    if (num.count < 100000)
        str << num.count;
    else if (num.count < 100000000)
        str << (num.count + 999) / 1000 << "k";
    else
        str << (num.count + 999999) / 1000000 << "M";

    return str;
}


/// Converts number into a human-readable form, e.g., 16777216 -> "16 M".
inline std::string get_readable_amount(const size_t num)
{
    std::ostringstream str;
    str << Large_number(num);
    return str.str();
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


inline std::ostream& operator<<(std::ostream& str, const Seconds value)
{
    return str << std::fixed << std::setprecision(3) << value.seconds << 's';
}


} // namespace LOG
}  // namespace MI

#endif // BASE_LIB_LOG_I_LOG_FRIENDLY_H
