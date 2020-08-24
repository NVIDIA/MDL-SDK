/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief Abstract time class and the implementation of the time functions
///
/// Provide abstract time class with an interface which is not system dependent. The time can be
/// converted from doubles. The class defines operators which may be used for arithmetic operations
/// and comparisons on times. Static functions may be used to query the current time. The current
/// time is cached internally to avoid costly system calls each time the current time is needed. To
/// actually update the time, the update_cache_system_time() function can be used or directly
/// use the additionally thread safe, but uncached get_time() instead.

#include "pch.h"

#include <stdio.h>
#include <base/hal/time/i_time.h>

#ifndef WIN_NT
#include <sys/time.h>
#include <time.h>
#include <unistd.h>		// for usleep()
#include <cerrno> // EINVAL
#else
#include <mi/base/miwindows.h>
#endif
#include <ctime>
#include <cstring>
#include <atomic>


namespace MI
{
namespace TIME
{

// Create time.
Time Time::mktime(
    struct tm *tm)	// the time struct
{
    time_t t = ::mktime(tm);
    return Time(static_cast<double>(t));
}

// Report local time.
int Time::localtime(
    struct tm *now)	// time descriptor
{
    Time sys_time = get_wallclock_time();
    time_t sys_time_sec = static_cast<time_t>(sys_time.get_seconds());
#ifdef WIN_NT
    return ::localtime_s(now, &sys_time_sec);
#else
    return ::localtime_r(&sys_time_sec, now) ? 0 : EINVAL;
#endif
}

#ifdef WIN_NT
std::string Time::to_string_rfc_2822() const
{
    // strftime() on Windows generates "W. Europe Standard Time" instead of "+0100" for %z.
    // Therefore we omit %z at the end here and add the corresponding time zone offset ourselves.
    std::string result = to_string("%a, %d %b %Y %H:%M:%S ", false);

    int minutes = - static_cast<int>(_timezone / 60);
    int hours   = minutes / 60;
    minutes     -= hours * 60;
    bool minus  = false;

    if (hours < 0 || minutes < 0) {
        hours   = -hours;
        minutes = -minutes;
        minus   = true;
    }

    char buffer[16];
    snprintf(&buffer[0], sizeof(buffer), "%c%02d:%02d", minus?'-':'+', hours, minutes);
    result += buffer;

    return result;
}
#endif

std::string Time::to_string(const char* format, bool gmt) const
{
    time_t time_stamp_time_t = static_cast<time_t>(m_seconds);

    tm time_stamp_tm;
#ifdef WIN_NT
    (gmt ? ::gmtime_s : ::localtime_s)(&time_stamp_tm, &time_stamp_time_t);
#else
    (gmt ? ::gmtime_r : ::localtime_r)(&time_stamp_time_t, &time_stamp_tm);
#endif

    char buffer[1024];
    size_t result = ::strftime(&buffer[0], sizeof(buffer), format, &time_stamp_tm);
    return result > 0 ? std::string(buffer) : std::string("");
}

// Convert a time value to a readable string. The time value is considered being an interval and
// not an absolute moment in time.
std::string Time::interval_to_string() const
{
    Uint seconds = static_cast<Uint>(m_seconds);
    Uint const days = seconds / 86400u;
    Uint const hours = (seconds % 86400u) / 3600u;
    Uint const minutes= (seconds % 3600u) / 60u;
    Uint const fraction	= static_cast<Uint>(m_seconds * 100u) % 100u;
    seconds %= 60u;

    char buffer[256];
    int	 len = 0;
    if (days > 0) {
	len = snprintf(buffer, sizeof(buffer), "%u days, ", days);
    }
    snprintf( buffer + len, sizeof(buffer) - len, "%02u:%02u:%02u.%02u", hours, minutes, seconds, fraction);
    return buffer;
}

//
// Get the cached time or, if the boolean flag is set, update it first.
Time get_cached_system_time(const bool call_update_first)
{
    static std::atomic<double> cached_time;

    if (call_update_first)
    {
#ifndef WIN_NT
	struct timeval tv;
	gettimeofday(&tv, NULL);
	cached_time = (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
#else
	static bool init = false;
	static double inv_frequency;
	if (!init)
	{
	    _tzset();
            static LARGE_INTEGER freq={0};
            if (freq.QuadPart == 0)
                QueryPerformanceFrequency(&freq);
	    inv_frequency = 1.0/(double)freq.QuadPart;
	    init = true;
	}
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	cached_time = (double)counter.QuadPart * inv_frequency;
#endif
    }

    return Time(cached_time);
}

Time get_time()
{
#ifndef WIN_NT
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return Time((double)tv.tv_sec + (double)tv.tv_usec*1.0e-6);
#else
    static LARGE_INTEGER freq={0};
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return Time((double)counter.QuadPart / (double)freq.QuadPart);
#endif
}

// Backward compatibility
void update_cached_system_time()
{
    (void)get_cached_system_time(true);
}

// Suspend current thread for the given time interval
// Uses high resolution sleep functions (Unix: ::usleep(), Windows: ::Sleep())
void sleep(
    const Time time)
{
#ifdef WIN_NT
    // The windows version expects milliseconds here.
    Uint millis = (Uint)(time.get_seconds() * 1000);
    if (millis == 0)
	millis = 1;
    ::Sleep(millis);
#else
    useconds_t micros = (useconds_t)(time.get_seconds() * 1000000);
    ::usleep((useconds_t) micros);
#endif
}

// Get the number of seconds since 1970.
Time get_wallclock_time()
{
    time_t current_time;
    ::time(&current_time);
    return Time(double(current_time));
}

}} // namespace MI::TIME
