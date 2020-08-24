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
 /// \brief ANSI color management. Useful for error messages and the like.
 ///
 /// See #set_console_color to change the text color of the console

#include "pch.h"
#include "hal.h"
#include <base/lib/log/i_log_assert.h>
#include <base/system/main/i_module_id.h>
#include <cstdio>


#ifdef WIN_NT
#include <mi/base/miwindows.h>
#endif // WIN_NT

namespace MI
{
namespace HAL
{

//
// colorization: ANSI color switching strings for printing to a terminal.
// This is used for error messages, for example (see base/data/log).
//

static const char * const colorstrings[] = {
    "\033[0m",          // default
    "\033[30m",         // black
    "\033[31m",         // red
    "\033[32m",         // green
    "\033[33m",         // yellow
    "\033[34m",         // blue
    "\033[35m",         // magenta
    "\033[36m",         // cyan
    "\033[37m"          // white
};

#ifdef WIN_NT
static WORD colorcodes[] = {
    // Default
    FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE,
    // Black
    FOREGROUND_INTENSITY | 0,
    // Red
    FOREGROUND_INTENSITY | FOREGROUND_RED,
    // Green
    FOREGROUND_INTENSITY | FOREGROUND_GREEN,
    // Yellow
    FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN,
    // Blue
    FOREGROUND_INTENSITY | FOREGROUND_BLUE,
    // Magenta
    FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_BLUE,
    // Cyan
    FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE,
    // White
    FOREGROUND_INTENSITY | FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE
};
#endif // WIN_NT

// colorization: ANSI color switching strings for printing to a terminal
// helper method.
const char *get_ansi_color(
    Color               c)              // one of C_*
{
    ASSERT(M_HAL, sizeof(colorstrings) == C_NUM * sizeof(char *));
    ASSERT(M_HAL, c >= 0 && c < C_NUM);
    return colorstrings[c];
}

//
// change the text color of the console
//

void set_console_color(
    Color               c)              // one of C_*
{
#ifdef WIN_NT
    HANDLE std_out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (std_out == INVALID_HANDLE_VALUE)
    {
        ASSERT(M_HAL, !"Can't obtain std out handle");
        return;
    }
    SetConsoleTextAttribute(std_out, colorcodes[c]);
#else
    fprintf(stderr, "%s", get_ansi_color(c));
#endif // WIN_NT
}

}} // end of namespaces
