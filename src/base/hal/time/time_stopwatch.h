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
/// \brief Stop Watch for statistics.
///
/// A stop watch is a system timer that can be started, resumed, stopped, reset,
/// and queried for the time period the instance was running.

#ifndef BASE_HAL_TIME_STOPWATCH_H
#define BASE_HAL_TIME_STOPWATCH_H

#include <base/hal/time/i_time.h>

namespace MI {
namespace TIME {

//////////////////////////////////////////////////////////////////////////
/// Class for statistic, simulate a StopWatch
//////////////////////////////////////////////////////////////////////////
class Stopwatch
{
public:
    /// Constructor, initializes the time module member
    Stopwatch();
    
    /// Reset the stopwatch and start the counter.
    inline void         restart();

    /// Resume the stopwatch and start the counter.
    inline void         resume();
    /// Same as resume. Here for backwards compatibility.
    inline void         start();

    /// Stop the counter. Do nothing if already stopped.
    inline void         stop();

    /// Reset the counter.
    inline void         reset(Time time = 0);

    /// Get elapsed time.
    /// \return elapsed time
    inline double       elapsed() const;

    /// return true iff counting
    inline bool         is_running() const;

    /// Scoped operations on Stopwatch.
    class Scoped_run;
    class Scoped_stop;

private:
    bool                m_is_running;   ///< true if running
    Time                m_ctime;        ///< cumulative time
    Time                m_time;         ///< start time
};

/// Run a stopwatch for the lifetime of this object. Use the class as follows:
///
/// \code
///
///   Stopwatch watch;
///   {
///       Stopwatch::Scoped_run timer(watch);
///       run_some_elaborate_function();
///   }
///   cout << watch.elapsed() << endl;
///
/// \endcode
///
/// Run scopes can be nested. If the stopwatch was already running, 
/// this objects will have no effect.

class Stopwatch::Scoped_run
{
public:
    /// \param watch stopwatch to run in current scope
    Scoped_run(
        Stopwatch &     watch);
    ~Scoped_run();

private:
    Stopwatch & m_watch;
    bool m_was_running;
};

/// This class implements a timer scope similar to Scoped_run, but instead it
/// will stop a running watch for the lifetime of the class. Here is a usage
/// example:
///
/// \code
///
///   Stopwatch watch;
///   {
///       Stopwatch::Scoped_run timer(watch);
///       {
///           run_some_elaborate_function();
///           {
///               Stopwatch::Scoped_stop timer(watch);
///               run_an_elaborate_function_but_dont_time_it();
///           }
///           run_another_elaborate_function();
///       }
///   }
///   cout << watch.elapsed() << endl;
///
/// \endcode
///
/// Stop scopes can be nested. If the stopwatch was not running, 
/// this objects will have no effect.

class Stopwatch::Scoped_stop
{
public:
    /// \param watch stopwatch to pause in current scope
    Scoped_stop(
        Stopwatch &     watch);
    ~Scoped_stop();

private:
    Stopwatch & m_watch;
    bool m_was_running;
};

}} // MI::TIME

#include "time_stopwatch_inline.h"

#endif // BASE_HAL_TIME_STOPWATCH_H
