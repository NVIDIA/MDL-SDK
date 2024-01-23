/******************************************************************************
 * Copyright (c) 2012-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Exposes MI::TEST::install_sigrtmin_handler() which (on Linux) installs a signal handler
///        that ignores RTMIN signals.
   
#ifndef BASE_SYSTEM_TEST_SIGRTMIN_HANDLER_H
#define BASE_SYSTEM_TEST_SIGRTMIN_HANDLER_H

#include <mi/base/config.h>
#ifdef MI_PLATFORM_LINUX
#define MI_INSTALL_RTMIN_HANDLER
#endif

#ifdef MI_INSTALL_RTMIN_HANDLER
#include <stdio.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif // MI_INSTALL_RTMIN_HANDLER

namespace MI {

namespace TEST {

#ifdef MI_INSTALL_RTMIN_HANDLER
void sigrtmin_handler (int id, siginfo_t* info, void* context)
{
    fprintf (stderr, "Received signal %d from process %d (own PID is %d).\n",
        info->si_signo, info->si_pid, getpid());
}
#endif // MI_INSTALL_RTMIN_HANDLER

void install_sigrtmin_handler()
{
#ifdef MI_INSTALL_RTMIN_HANDLER
    struct sigaction sa;  
    sa.sa_sigaction = sigrtmin_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset (&sa.sa_mask);
    sa.sa_restorer = 0;
    sigaction (SIGRTMIN, &sa, 0);
#endif // MI_INSTALL_RTMIN_HANDLER
}

} // namespace TEST

} // namespace MI

#endif // BASE_SYSTEM_TEST_SIGRTMIN_HANDLER_H
