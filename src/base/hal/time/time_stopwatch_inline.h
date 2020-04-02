/******************************************************************************
 * Copyright (c) 2004-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Stop Watch for statistics

#ifndef BASE_HAL_TIME_STOPWATCH_INLINE_H
#define BASE_HAL_TIME_STOPWATCH_INLINE_H

namespace MI {
namespace TIME {

//////////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////////
inline Stopwatch::Stopwatch()
    : m_is_running(false)
    , m_ctime(0)
{}

//////////////////////////////////////////////////////////////////////////
// Start the counter
//////////////////////////////////////////////////////////////////////////
inline void Stopwatch::restart()
{
    m_is_running = true;
    m_ctime = 0;
    m_time = get_time();
}

//////////////////////////////////////////////////////////////////////////
// Resume the counter
//////////////////////////////////////////////////////////////////////////
inline void Stopwatch::resume()
{
    if(!m_is_running)
    {
        m_is_running = true;
        m_time = get_time();
    }
}

//////////////////////////////////////////////////////////////////////////
// Resume the counter
//////////////////////////////////////////////////////////////////////////
inline void Stopwatch::start()
{
    return resume();
}

//////////////////////////////////////////////////////////////////////////
// Stop the counter
//////////////////////////////////////////////////////////////////////////
inline void Stopwatch::stop()
{
    if(m_is_running)
    {
        m_is_running = false;
        m_ctime += get_time() - m_time;
    }
}

//////////////////////////////////////////////////////////////////////////
// Reset the counter
//////////////////////////////////////////////////////////////////////////
inline void Stopwatch::reset(Time time)
{
    m_ctime = time;
    if(m_is_running)
    {
        m_time = get_time();
    }
}

//////////////////////////////////////////////////////////////////////////
// Get elapse time
//////////////////////////////////////////////////////////////////////////
inline double Stopwatch::elapsed() const
{
    double time = m_ctime.get_seconds();

    if( m_is_running )
    {
        Time passed_time = (get_time() - m_time);
        time += passed_time.get_seconds();
    }
    
    // can never fully trust timers
    // just make sure that elapsed times are at least never negative
    return time > 0 ? time : 0;
}

//////////////////////////////////////////////////////////////////////////
// Get state
//////////////////////////////////////////////////////////////////////////
inline bool Stopwatch::is_running() const
{
    return m_is_running;
}

//////////////////////////////////////////////////////////////////////////
// Run the stopwatch for a limited scope
//////////////////////////////////////////////////////////////////////////
inline Stopwatch::Scoped_run::Scoped_run(
    Stopwatch & watch)                  // Stopwatch to run in current scope
    : m_watch(watch)
{
    m_was_running = m_watch.is_running();
    m_watch.resume();
}

inline Stopwatch::Scoped_run::~Scoped_run()
{
    if(!m_was_running)
        m_watch.stop();
}

//////////////////////////////////////////////////////////////////////////
// Pause the stopwatch for a limited scope
//////////////////////////////////////////////////////////////////////////
inline Stopwatch::Scoped_stop::Scoped_stop(
    Stopwatch & watch)                  // Stopwatch to pause in current scope
    : m_watch(watch)
{
    m_was_running = m_watch.is_running();
    m_watch.stop();
}

inline Stopwatch::Scoped_stop::~Scoped_stop()
{
    if(m_was_running)
        m_watch.resume();
}

}} // MI::TIME

#endif   // BASE_HAL_TIME_STOP_WATCH_INLINE_H
