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
/// \brief Several system-specific utilites

#include "pch.h"
#include "hal.h"
#include <cerrno>               // for errno
#include <cstdio>               // for fflush/stderr
#include <cstring>              // for strerror.h
#include <cstdlib>

#ifndef WIN_NT
#include <unistd.h>
#else
#include <mi/base/miwindows.h>
#include <shlobj.h>
#include <io.h>
#include <wincon.h>
// define non existing errno code (must be mapped to what is being set in msg_socket.cpp)
#ifndef EADDRINUSE
#define EADDRINUSE WSAEADDRINUSE
#endif
#ifndef ECONNREFUSED
#define ECONNREFUSED WSAENOTCONN
#endif
#endif

namespace MI
{

namespace HAL
{

//
// return system error code, like standard thread-local errno variable but
// this works on weird operating systems too.
//

int get_errno()
{
    return errno;
}


//
// return an error hal describing a system error code, like get_errno().
// The Disk module has error() functions that return system error codes.
//

std::string strerror(
    int         err)                    // system error code
{
    char buffer[1024];
#ifdef WIN_NT
    // There is no EADDRINUSE in windows errno.h
    if (err == EADDRINUSE)
        ::sprintf_s(buffer, sizeof(buffer), "Address already in use");
    else if (err == ECONNREFUSED)
        ::sprintf_s(buffer, sizeof(buffer), "Connection refused");
    else
        ::strerror_s(buffer, sizeof(buffer), err);
#else
#ifdef _GNU_SOURCE
    // Use GLIBC unstandard strerror_r which is defined if
    // _GNU_SOURCE is set. If will not use the supplied buffer if
    // the supplied err value is within a well know range. Instead it
    // will return a pointer to an internal static buffer.
    char *p = ::strerror_r(err, buffer, sizeof(buffer));
    if(p)
        return p;
#else
    ::strerror_r(err, buffer, sizeof(buffer));
#endif
#endif
    return buffer;
}

//
// Indicate whether we have somewhere to write stderr. Windows apps don't
// have a console and can not easily write msgs to stderr. No stderr-having
// apps may decide they want to do something other than quietly doing
// nothing for critical messages. Note that this function will return true
// if stderr is open but has been redirected (including to /dev/null)
//

bool has_stderr()
{
#ifdef WIN_NT
    // on windows, we might be a WINDOWS (ie non-console) app, which means
    // that there is no stdout or stderr. We can't check this at precompile
    // time, so we check at runtime -- if the file descriptor for stderr is
    // -2 (this is the case for a windows app compiled under VC2005) or -1
    // (other compilers) then we want to show a message box -- for a console
    // or server application then this file descriptor should be valid, and
    // the caller may want to display a message box for fatal errors or
    // similar
    int fd = _fileno(stderr);
    if (fd == -1 || fd == -2)
        return false;
    else
        return true;
#else
    // It is possible that a non-windows client would have closed stderr, but
    // for now we're not going to worry about this case
    return true;
#endif
}


//
// Perform a fatal exit -- this should only be called to terminate the app
// based on really nasty fatal errors which the app can not possibly
// recover from
//

void fatal_exit(
    int exit_code)                      // exit code to return
{
    // There might be important information in stderr that we want to see
    // and since stderr is line buffered on Windows we want to flush it.
    // POSIX only says that stderr is not _fully_ buffered so we flush
    // on all platforms.
    if (has_stderr())
        fflush(stderr);

    ::_exit(exit_code);
}


//
// Pop a message box with the given caption and message
// This method is only implemented on windows right now
//

void message_box(
    const char *caption,                // the caption for the box
    const char *msg)                    // the message to display
{
#ifdef WIN_NT
    ::MessageBox(0, msg, caption, MB_OK);
#else
    // For now we do nothing on linux, implement this with XmMessageBox for
    // motif-enabled platforms. Applications should be very judicious about
    // popping message boxes; especially in the case that there may be more
    // than one
#endif
}


// return the name of a temp directory (/tmp on Unix/Linux).
// existence and write permissions are not checked.
std::string get_tmpdir()
{
#ifndef WIN_NT
    std::string env = get_env("TMPDIR");
    if (!env.empty())
        return env;
    env = get_env("TMP");
    if (!env.empty())
        return env;
    env = get_env("TEMP");
    if (!env.empty())
        return env;
    return std::string("/tmp");
#else
    const DWORD bufsize = 4096;
    char buf[bufsize];
    DWORD len = ::GetTempPath(bufsize, buf);

    if (len <= 0 || len > bufsize)
        return std::string(".");

    // remove trailing '\' since _stat() gets confused
    std::string dir(buf, (buf[len-1] == '\\' || buf[len-1] == '/')? len-1 : len);

    return Ospath::convert_to_forward_slashes(dir);
#endif
}


//
// return the name of a suitable directory for storing user data
//

std::string get_userdata_dir()
{
#ifndef WIN_NT
    std::string env = get_env("HOME");
    return !env.empty() ? env : std::string(".");
#else
    char sz_path[MAX_PATH];
    HRESULT hresult = ::SHGetFolderPath(NULL,
        CSIDL_APPDATA,
        NULL,
        SHGFP_TYPE_CURRENT,
        sz_path);

    if (hresult == S_OK) {
        // convert to forward slashes
        for (char *p=sz_path; *p; p++)
            if (*p == '\\')
                *p = '/';
        return std::string(sz_path);
    }
    else
        return std::string(".");
#endif
}


//
// return the value of an environment variable or empty string else
//

std::string get_env(
    const std::string& name)            // name of variable
{
    char* value = getenv(name.c_str());
    if (!value)
        return std::string();
    std::string result = value;

    // if the getenv'ed value is wrapped with a leading and a trailing "
    // then remove those two - watched this so far only on windows on
    // paths with spaces inbetween
    char first = *result.begin();
    char last = *result.rbegin();
    if (first == '\"' && last == '\"')
        result = result.substr(1, result.size()-2);

    return result;
}

void fprintf_stderr_utf8( const char* message)
{
#ifndef WIN_NT
    fprintf( stderr, "%s", message);
#else
    DWORD dummy;
    HANDLE h = (HANDLE) _get_osfhandle( fileno( stderr));
    if( GetConsoleMode( h, &dummy) == 0) {
        fprintf( stderr, "%s", message);
        return;
    }

    // If stderr is connected to the console, convert UTF8-encoded message to wide chars and use
    // WriteConsoleW instead of fprintf().
    int length = MultiByteToWideChar( CP_UTF8, MB_ERR_INVALID_CHARS, message, -1, NULL, 0);
    if( length > 0) {
        wchar_t* wbuf = new wchar_t[length];
        MultiByteToWideChar( CP_UTF8, MB_ERR_INVALID_CHARS, message, -1, wbuf, length);
        BOOL success = WriteConsoleW( h, wbuf, length, &dummy, NULL);
        if( success == 0)
            fprintf( stderr, "%s", message); // fallback for failing WriteConsoleW()
        delete[] wbuf;
    } else
        fprintf( stderr, "%s", message); // fallback for failing MultiByteToWideChar()
#endif
}

} // namespace HAL

} // namespace MI

