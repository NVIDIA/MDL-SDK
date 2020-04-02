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
/// \brief Provide the inline functions for the time module. Inline functions for time module.

#ifndef BASE_HAL_TIME_INLINE_H
#define BASE_HAL_TIME_INLINE_H

namespace MI { namespace TIME {

// Constructor
inline Time::Time(double seconds) : m_seconds(seconds)
{
}

// Get the seconds of the class.
inline double Time::get_seconds() const
{
    return m_seconds;
}

// Compare a time to another time
inline bool Time::operator== (const Time & other) const
{
    return m_seconds == other.m_seconds;
}
inline bool Time::operator!= (const Time & other) const
{
    return m_seconds != other.m_seconds;
}
inline bool Time::operator< (const Time & other) const
{
    return m_seconds < other.m_seconds;
}
inline bool Time::operator<= (const Time & other) const
{
    return m_seconds <= other.m_seconds;
}
inline bool Time::operator> (const Time & other) const
{
    return m_seconds > other.m_seconds;
}
inline bool Time::operator>= (const Time & other) const
{
    return m_seconds >= other.m_seconds;
}

// Arithmetic for times
inline Time & Time::operator+= (const Time & other)
{
    m_seconds += other.m_seconds;
    return *this;
}
inline Time & Time::operator-= (const Time & other)
{
    m_seconds -= other.m_seconds;
    return *this;
}
inline Time Time::operator+ (const Time & other) const
{
    return m_seconds + other.m_seconds;
}
inline Time Time::operator- (const Time & other) const
{
    return m_seconds - other.m_seconds;
}
inline Time & Time::operator*= (const double & scalar)
{
    m_seconds *= scalar;
    return *this;
}
inline Time & Time::operator/= (const double & scalar)
{
    m_seconds /= scalar;
    return *this;
}
inline Time Time::operator* (const double & scalar) const
{
    return m_seconds * scalar;
}
inline Time Time::operator/ (const double & scalar) const
{
    return m_seconds / scalar;
}

}} // namespace MI::TIME

#endif // BASE_HAL_TIME_INLINE_H
