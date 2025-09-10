/******************************************************************************
 * Copyright (c) 2004-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Stop watch for statistics.

#ifndef BASE_HAL_TIME_STOPWATCH_H
#define BASE_HAL_TIME_STOPWATCH_H

#include <base/hal/time/i_time.h>

namespace MI {
namespace TIME {

class Stopwatch
{
public:
    /// Starts the stopwatch. No effect if already running.
    inline void start();

    /// Stops the stopwatch. No effect if already stopped.
    inline void stop();

    /// Identical to #start().
    inline void resume() { return start(); }

    /// Sets the cumulative time to \p time. Does not affect the running status.
    inline void reset( double time = 0);

    /// Stops the stopwatch, set the cumulative time to 0, and starts it again.
    inline void restart();

    /// Returns the total elapsed time so far.
    inline double elapsed() const;

    /// Indicates whether the stopwatch is currently running.
    inline bool is_running() const { return m_is_running; }

    /// Scoped operations on a stopwatch.
    class Scoped_run;
    class Scoped_stop;

private:
    /// Returns the current time used by the stopwatch (in seconds).
    inline static double get_current_time();

    bool m_is_running = false;     ///< Indicates whether the stopwatch is running.
    double m_accumulated_time = 0; ///< Time accumulated up to the last stop in seconds.
    double m_start_time = 0;       ///< Time of the last start/resume/reset/restart call in seconds.
};

/// Run a stopwatch for the lifetime of this object.
///
/// Run scopes can be nested. If the stopwatch was already running, this object will have no effect.
///
/// Example:
/// \code
///   Stopwatch watch;
///   {
///       Stopwatch::Scoped_run timer(watch);
///       run_some_elaborate_function();
///   }
///   std::cout << watch.elapsed() << std::endl;
/// \endcode
class Stopwatch::Scoped_run
{
public:
    /// \param watch stopwatch to run in current scope
    Scoped_run( Stopwatch &watch);
    ~Scoped_run();

    Scoped_run( const Scoped_run&) = delete;
    Scoped_run& operator=( const Scoped_run&) = delete;

private:
    Stopwatch& m_watch;
    bool m_was_running;
};

/// Stop a stopwatch for the lifetime of this object.
///
/// Stop scopes can be nested. If the stopwatch was not running, this object will have no effect.
///
/// Example:
/// \code
///   Stopwatch watch;
///   {
///       Stopwatch::Scoped_run timer1(watch);
///       {
///           run_some_elaborate_function();
///           {
///               Stopwatch::Scoped_stop timer1(watch);
///               run_an_elaborate_function_but_dont_time_it();
///           }
///           run_another_elaborate_function();
///       }
///   }
///   std::cout << watch.elapsed() << std::endl;
/// \endcode

class Stopwatch::Scoped_stop
{
public:
    /// \param watch stopwatch to pause in current scope
    Scoped_stop( Stopwatch& watch);
    ~Scoped_stop();

    Scoped_stop( const Scoped_stop&) = delete;
    Scoped_stop& operator=( const Scoped_stop&) = delete;

private:
    Stopwatch& m_watch;
    bool m_was_running;
};

inline void Stopwatch::start()
{
    if( m_is_running)
        return;

    m_is_running = true;
    m_start_time = get_current_time();
}

inline void Stopwatch::stop()
{
    if( !m_is_running)
        return;

    m_is_running = false;
    m_accumulated_time += get_current_time() - m_start_time;
}

inline void Stopwatch::reset( double time)
{
    m_accumulated_time = time;
    if( m_is_running)
        m_start_time = get_current_time();
}

inline void Stopwatch::restart()
{
    m_is_running = true;
    m_accumulated_time = 0;
    m_start_time = get_current_time();
}

inline double Stopwatch::elapsed() const
{
    double time = m_accumulated_time;
    if( m_is_running )
        time += get_current_time() - m_start_time;

    // Avoid returning negative values in case the clock is not monotonic.
    return time > 0 ? time : 0;
}

inline double Stopwatch::get_current_time()
{
    return TIME::get_time().get_seconds();
}

inline Stopwatch::Scoped_run::Scoped_run( Stopwatch& watch)
  : m_watch( watch)
{
    m_was_running = m_watch.is_running();
    m_watch.resume();
}

inline Stopwatch::Scoped_run::~Scoped_run()
{
    if( !m_was_running)
        m_watch.stop();
}

inline Stopwatch::Scoped_stop::Scoped_stop( Stopwatch& watch)
  : m_watch( watch)
{
    m_was_running = m_watch.is_running();
    m_watch.stop();
}

inline Stopwatch::Scoped_stop::~Scoped_stop()
{
    if( m_was_running)
        m_watch.resume();
}

}} // MI::TIME

#endif // BASE_HAL_TIME_STOPWATCH_H
