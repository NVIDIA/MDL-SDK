/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief Several system-specific utilites incl the path-related Ospath

#ifndef BASE_HAL_HAL_HAL_H
#define BASE_HAL_HAL_HAL_H

#include "i_hal_ospath.h"

#include <string>

namespace MI {
namespace HAL{

//------------------------------------ system errors ----------------------
/// return system error code, like standard thread-local errno variable but
/// this works on weird operating systems too.
int get_errno();

/// return an error string describing a system error code, like get_errno().
/// The Disk module has error() functions that return system error codes.
/// \param err the system error code
/// \return error string describing the given system error code
std::string strerror(
    int         err);

/// Indicate whether we have somewhere to write stderr. Windows apps don't
/// have a console and can not easily write msgs to stderr. No stderr-having
/// apps may decide they want to do something other than quietly doing
/// nothing for critical messages. Note that this function will return true
/// if stderr is open but has been redirected (including to /dev/null)
bool has_stderr();

/// Perform a fatal exit -- this should only be called to terminate the app
/// based on really fatal errors
/// \param exit_code exit code to return
void fatal_exit(
    int exit_code);


//------------------------------------ ui ---------------------------------
/// Pop a message box with the given caption and message.
/// Applications should be extremely judicious about popping message boxes;
/// for server applications this should never happen and applications should
/// not pop an endless stream of these (or even more than one)
void message_box(
    const char *caption,                // the caption for the box
    const char *msg);           // the message to display

enum Color {
    C_DEFAULT, C_BLACK, C_RED, C_GREEN, C_YELLOW,
    C_BLUE, C_MAGENTA, C_CYAN, C_WHITE, C_NUM };

/// change the text color of the console
/// \param one of C_*
void set_console_color(
    Color c);

/// Emits am UTF8-encoded message to stderr.
///
/// On Windows, if stderr is connected to the console, WriteConsoleW is used instead of fprintf() to
/// ensure correct output of non-ASCII characters.
void fprintf_stderr_utf8( const char* message);

//------------------------------------ misc -------------------------------
/// return the name of a writable temp directory (/tmp on Unix/Linux)
std::string get_tmpdir();

/// return the name of a suitable userdata directory
std::string get_userdata_dir();

/// return the value of an environment variable or empty string else
/// \param name name of env variable
std::string get_env(
    const std::string& name);


} // namespace HAL

} // namespace MI

#endif // BASE_HAL_HAL_HAL_H

