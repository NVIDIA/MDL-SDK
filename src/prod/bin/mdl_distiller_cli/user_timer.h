/******************************************************************************
 * Copyright (c) 2017-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \file user_timer.h
/// \brief User process timer class.

#pragma once

/// User process timer.
///
class User_timer {
private:
    double      m_elapsed;  // in seconds
    double      m_started;  // in seconds
    int         m_count;    // start-stop cycles
    bool        m_running;

    static bool m_failed;

    /// Returns the user process time in seconds. In case of a failure
    /// of the underlying OS call, it returns 0 and sets m_failed to true.
    double user_process_time() const;

    /// Returns the precision of the timer as the smallest time in seconds
    /// that the timer can measure by dynamically detecting it through polling.
    double compute_precision() const;

public:
    /// Default timer in stopped state.
    User_timer() :
        m_elapsed(0.0), m_started(0.0), m_count(0), m_running(false) {}

    /// Start the timer.
    void   start() {
        m_started = user_process_time();
        m_running = true;
        ++ m_count;
    }

    /// Stop the timer.
    void   stop () {
        double t = user_process_time();
        m_elapsed += (t - m_started);
        m_started = 0.0;
        m_running = false;
    }

    /// Reset the timer to zero time.
    void   reset() {
        m_count  = 0;
        m_elapsed = 0.0;
        if (m_running) {
            m_started = user_process_time();
            ++ m_count;
        } else {
            m_started = 0.0;
        }
    }

    /// Returns if the timer is currently running.
    bool   is_running() const { return m_running; }

    /// Returns time in seconds.
    double time() const {
        if (m_running) {
            double t = user_process_time();
            return m_elapsed + (t - m_started);
        }
        return m_elapsed;
    }

    /// Returns the number of iterations measured, e.g., start-stop cycles.
    int    count() const { return m_count; }

    /// Returns the precision of the timer as the smallest time in seconds
    /// that the timer can measure. Returns -1 is the underlying OS timer fails.
    double precision()  const;

    /// Returns maximum time measurable in seconds.
    double max_time() const;
};

