/******************************************************************************
 * Copyright (c) 2003-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Time interface
///
/// Provide abstract time class with an interface which is not system dependent.
/// The time can be converted from doubles. The class defines operators
/// which may be used for arithmetic operations and comparisons on times.
/// Static functions may be used to query the current time. The current time is
/// cached internally to avoid costly system calls each time the current time is
/// needed. To actually update the time, the update_cache_system_time() function
/// can be used or directly use the additionally thread safe, but uncached get_time() instead.
///

#ifndef BASE_HAL_TIME_TIME_H
#define BASE_HAL_TIME_TIME_H

#include <string>
#include <base/system/main/types.h>

namespace MI
{
namespace TIME
{

// System independent time class
class Time
{
public:
    // Constructor for a time class. The time given is in seconds.
    Time(double seconds = 0);

    // Get the seconds of the class. This may only be used within the hal layer
    // itself, because
    double get_seconds() const;

    // Compare a time to another time
    bool operator== (const Time & other) const;
    bool operator!= (const Time & other) const;
    bool operator<  (const Time & other) const;
    bool operator<= (const Time & other) const;
    bool operator>  (const Time & other) const;
    bool operator>= (const Time & other) const;

    // Arithmetic for times
    Time & operator+= (const Time & other);
    Time & operator-= (const Time & other);
    Time   operator+  (const Time & other) const;
    Time   operator-  (const Time & other) const;

    Time & operator*= (const double & scalar);
    Time & operator/= (const double & scalar);
    Time   operator*  (const double & scalar) const;
    Time   operator/  (const double & scalar) const;

    // Convert a time value to a human readable string.
    std::string to_string() const { return to_string_rfc_2822(); }

    // Convert a time value to a human readable string as specified in RFC 2822.
    //
    // Uses timezone settings. Example: Wed, 30 Nov 2016 11:19:37 +0100
#ifndef WIN_NT
    std::string to_string_rfc_2822() const { return to_string("%a, %d %b %Y %H:%M:%S %z",false); }
#else
    std::string to_string_rfc_2822() const;
#endif

    // Convert a time value to a human readable string as specified in RFC 2616.
    //
    // Always uses GMT, independent of timezone settings. Example: Wed, 30 Nov 2016 10:19:37 GMT
    std::string to_string_rfc_2616() const { return to_string("%a, %d %b %Y %H:%M:%S GMT",true); }

    // Convert a time value to a human readable string.
    //
    // \param format   Format string passed to strftime(). The internal buffer is limited to
    //                 1024 characters.
    // \param gmt      Use GMT or timezone settings.
    //
    // Note that strftime() on Windows handles %z/%Z differently from what's mandated by the
    // C++11/C99 standard. We could implement our own handling of %z/%Z on Windows (that's what
    // e.g. Python does), or use Boost instead.
    std::string to_string(const char* format, bool gmt) const;

    // Convert a time value to a readable string. The time value is considered
    // being an interval and not an absolute moment in time.
    std::string interval_to_string() const;

    // Create time.
    static Time mktime(
	struct tm   *tm);   // the time struct

    // Report local time.
    static int localtime(
	struct tm   *now);  // the local time

private:
    // The seconds of the timer class. This might differ beyond different
    // operating systems to adapt to the native implementation of time.
    double m_seconds;
};

/// The Time utilities, use this instead of gettimeofday() and similiar on other
/// OS because this one is much cheaper if used correctly (and OS independent).

/// Update the time to the current seconds provided by the systems. This
/// should only be called when necessary, because it may have a severe
/// performance impact.
/// update_cached_system_time() will be called automatically by the Selector module
/// before and after it waits for events. Thus for most applications it should
/// never be necesssary to call this.
void update_cached_system_time();

/// Get the current seconds of the time as it was cached last time update_cached_system_time()
/// was called (unless the parameter is true). Not thread safe.
/// \param call_update_first set to true if the cached value is not good enough
/// \return Time
Time get_cached_system_time(const bool call_update_first = false);

/// Get the current seconds of the time. Thread safe.
/// \return Time
Time get_time();

/// Get the current seconds since 1970.
/// \return Time since 1970 in seconds
Time get_wallclock_time();

/// Suspend current thread for the given time interval
/// Uses high resolution sleep functions (Unix: ::usleep(), Windows: ::Sleep())
void sleep (
     const Time interval);		// sleep time in seconds

}} // namespace MI::TIME

#include "time_inline.h"
#include "time_stopwatch.h"

#endif // BASE_HAL_TIME_TIME_H
