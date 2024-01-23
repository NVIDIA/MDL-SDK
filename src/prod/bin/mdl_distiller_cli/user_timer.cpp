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

#include "user_timer.h"

// Use getrusage if available and std::clock() otherwise.
#include <climits>
#ifdef _POSIX_ARG_MAX // defined for POSIX
#include <unistd.h>
#ifdef _POSIX_VERSION
#ifdef _XOPEN_UNIX // XSI: X/Open System Interfaces Extension
#define GETRUSAGE 1
#endif
#endif
#endif

#ifdef GETRUSAGE
// types, function prototype and constants for the POSIX function
// int getrusage (int who, struct rusage *usage);
#include <sys/resource.h>
#else //  GETRUSAGE //
// used for clock()
#include <ctime>
#endif //  GETRUSAGE //

// For the numerical limits
#include <float.h>


bool User_timer::m_failed = false;

double User_timer::user_process_time() const {
#ifdef GETRUSAGE
    struct rusage usage;
    int ret = getrusage( RUSAGE_SELF, &usage);
    if ( ret == 0) {
        return double( usage.ru_utime.tv_sec)               // seconds
             + double( usage.ru_utime.tv_usec) / 1000000.0; // microseconds
    }
#else // __GETRUSAGE //
    std::clock_t clk = std::clock();
    if ( clk != (std::clock_t)-1) {
        return double(clk) / CLOCKS_PER_SEC;
    }
#endif // __GETRUSAGE //
    m_failed = true;
    return 0.0;
}

double User_timer::compute_precision() const {
    double min_res = DBL_MAX;
    for ( int i = 0; i < 5; ++i) {
        double current = user_process_time();
        if ( m_failed)
            return -1.0;
        double next    = user_process_time();
        while ( current >= next) { // poll timer until it increases
            next = user_process_time();
            if ( m_failed)
                return -1.0;
        }
        // minimum timing of all runs.
        if ( min_res > next - current)
            min_res = next - current;
    }
    return min_res;
}

double User_timer::precision() const {
    static double prec = compute_precision();
    return prec;
}

double User_timer::max_time() const {
#ifdef __GETRUSAGE
    return DBL_MAX;
#else // __GETRUSAGE //
    return 2146.0;
#endif // __GETRUSAGE //
}
