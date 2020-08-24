/***************************************************************************************************
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
 **************************************************************************************************/

/// \file
/// \brief operation on open files
///
/// Represents a single file. All the usual file operations are here. There is one instance per
/// open file. The open method is not part of the constructor because we need to return a failure
/// code.

#include "pch.h"
#include "disk.h"

#include <base/hal/hal/hal.h>
#include <base/lib/log/log.h>
#include <base/lib/log/i_log_macros.h>
#include <base/util/string_utils/i_string_utils.h>

#include <cerrno>
#include <cstdio>
#include <cstdarg>
#include <limits>

namespace MI {
namespace DISK {

namespace {
/// A hard-coded mapping of the public \c File::Mode to the clib's string.
const char * const fopen_mode[] = {"rb", "wb", "w+b", "ab", "r+b"};
}

const char* IFile::get_mode_string(
    Mode mode)
{
    ASSERT(M_DISK, M_READ <= mode);
    ASSERT(M_DISK, mode <= M_NONE);
    return fopen_mode[mode];
}


//
// constructor and destructor for File.
//

File::File()
{
    m_fp    = 0;
    m_error = 0;
    m_mode = M_NONE;
}


File::~File()
{
    close();
}


// open a file by path. If the file was already open, close it first.
bool File::open(
    const std::string& path,
    Mode mode)
{
    const char* filepath = 0;
    if (!path.empty())
        filepath = path.c_str();
    return open(filepath, mode);
}


bool File::open(
    const char          *path,          // path to open
    Mode                mode)           // read, write, modify, append?
{
    ASSERT(M_DISK, mode != M_NONE);
    if (m_fp && !close())
        return false;

    m_error = 0;
    if (!path) {
        // either in XOR out
        ASSERT(M_DISK, (mode==M_READ)||(mode==M_WRITE)||(mode==M_APPEND));
        m_path.clear();
        m_fp = mode == M_READ ? stdin : stdout;
    }
    else {
        m_path = path;
        if (mode != M_READ) {
           std::string dir = HAL::Ospath::dirname(path);
           if (!dir.empty() && !is_directory(dir.c_str()))
               mkdir(dir.c_str());
        }
        if (!(m_fp = DISK::fopen(m_path.c_str(), get_mode_string(mode)))) {
            m_error = HAL::get_errno();
            if (LOG::mod_log)
                LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                                   "open file \"%s\", mode %s: %s",
                                   path, get_mode_string(mode),
                                   HAL::strerror(m_error).c_str());
            return false;
        }
        // reject directories, pipes, etc.
        if (!DISK::is_file(path)) // NOT: File::is_file(path)
            return false;
    }
    if (LOG::mod_log)
        LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                           "open file \"%s\", mode %s, ok",
                           path, get_mode_string(mode));
    m_mode = mode;
    return true;
}


//
// close the file. It is safe to close a file that is not open.
//

bool File::close()
{
    // from the man page about the return value of flcose()
    // Upon successful completion 0 is returned.  Otherwise, EOF is returned
    // and errno is set to indicate the error.  In either case any further
    // access (including another call to fclose()) to the stream results in
    // undefined behavior.
    if (!m_fp || !fclose(m_fp)) {
        m_error = 0;
        m_mode  = M_NONE;
    }
    else {
        m_error = HAL::get_errno();
    }
    // see the above comment - any further access to the stream results in UB
    m_fp = 0;
    return !m_error;
}


//
// read data from the file into a buffer. Reading 0 bytes is fine; this
// can happen at EOF, but getting -1 back indicates an error.
//

Sint64 File::read(
    char                *buf,           // copy data here
    Uint64               num)           // copy this many bytes
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK,
        m_mode != M_WRITE && m_mode != M_APPEND && m_mode != M_NONE);

    // check file stream for non-zero, otherwise ferror() will invoke
    // invalid parameter handler
    if (m_fp == 0 || buf == 0) {
        m_error = EINVAL;
        return -1;
    }

#ifdef BIT64
    // num has only 63 significant bits
    ASSERT(M_DISK, (num >> 63) == 0); // only 63 significant bits
    if (num >> 63) {
        MI_STREAM_DEBUG(M_DISK, LOG::Mod_log::C_DISKTRACE)
            << "File::read() error: Only 63 significant bits supported"
            " for filesize [file: " << m_path << "]";
        m_error = EINVAL;
        return -1;
    }
#else
    // 32 bit platform
    if (num >= Uint64(std::numeric_limits<size_t>::max())) {
        MI_STREAM_DEBUG(M_DISK, LOG::Mod_log::C_DISKTRACE)
            << "File::read() error: Filesize too large (" << num << ")"
            " [file: " << m_path << "]";
        m_error = EINVAL;
        return -1;
    }
#endif

    size_t nread = fread(buf, size_t(1), size_t(num), m_fp);
    // check error with ferror(), not by inspecting return of fread()
    if (ferror(m_fp) == 0) {
        m_error = 0;
        return Sint64(nread);  // safe to cast
    }
    else {
        m_error = HAL::get_errno();
        return -1;
    }
}


//
// read a line from the file. Lines are terminated by \r and/or \n or EOF.
// The newline characters are not removed here. The returned string is always
// 0-terminated. This cannot fail; at EOF zero-length strings are returned.
//

bool File::readline(
    char                *line,          // copy data here
    int                 num)            // copy at most this many bytes
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK,
        m_mode != M_WRITE && m_mode != M_APPEND && m_mode != M_NONE);

    m_error = 0;
    if (!fgets(line, num, m_fp))
        line[0] = 0; // eof
    return true;
}


bool File::read_line(
    std::string& line,
    bool strip)
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK,
        m_mode != M_WRITE && m_mode != M_APPEND && m_mode != M_NONE);
    m_error = 0;
    char l[4096];
    readline(l, sizeof(l));
    if (*l) {
        if (strip)
            line = STRING::strip(l);
        else
            line = l;
    }
    else
        line.clear();
    return true;
}

//
// write raw data to a file.
//

Sint64 File::write(
    const char          *buf,           // write this buffer
    Uint64               num)           // write this many bytes
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK, m_mode != M_READ);
    ASSERT(M_DISK, (num >> 63) == 0);

    // check file stream for non-zero, otherwise ferror() will invoke
    // invalid parameter handler:
    if (m_fp == 0 || buf == 0) {
        m_error = EINVAL;
        return -1;
    }

#ifdef BIT64
    // num has only 63 significant bits
    ASSERT(M_DISK, (num >> 63) == 0); // only 63 significant bits
    if (num >> 63) {
        MI_STREAM_DEBUG(M_DISK, LOG::Mod_log::C_DISKTRACE)
            << "File::write() error: Only 63 significant bits supported"
            " for filesize [file: " << m_path << "]";
        m_error = EINVAL;
        return -1;
    }
#else
    // 32 bit platform
    if (num >= Uint64(std::numeric_limits<size_t>::max())) {
        MI_STREAM_DEBUG(M_DISK, LOG::Mod_log::C_DISKTRACE)
            << "File::write() error: Filesize too large (" << num << ")"
            " [file: " << m_path << "]";
        m_error = EINVAL;
        return -1;
    }
#endif


    size_t nwritten = fwrite(buf, size_t(1), size_t(num), m_fp);
    if (ferror(m_fp) == 0) {
        m_error = 0;
        return Sint64(nwritten);  // safe to cast
    }
    else {
        m_error = HAL::get_errno();
        return -1;
    }
}


//
// write a line to open file
//

bool File::writeline(
    const char          *line)          // write this string
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK, m_mode != M_READ);

    if (fputs(line, m_fp) != EOF) {
        m_error = 0;
        return true;
    } else {
        m_error = HAL::get_errno();
        return false;
    }
}


//
// write a formatted string to open file
//

int File::printf(
    const char          *fmt,           // format string with %X
    ...)                                // arguments for %X
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK, m_mode != M_READ);

    int count;
    va_list args;
    va_start(args, fmt);
    count = vfprintf(m_fp, fmt, args);
    va_end(args);
    if (count < 0)
        m_error = HAL::get_errno();
    return count;
}


//
// write a formatted string to open file
//

int File::vprintf(
    const char          *fmt,           // format string with %X
    va_list              args)          // arguments for %X
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK, m_mode != M_READ);

    int count = vfprintf(m_fp, fmt, args);
    if (count < 0)
        m_error = HAL::get_errno();
    return count;
}


// flush buffered output data to the file

bool File::flush()
{
    ASSERT(M_DISK, m_fp != 0);
    ASSERT(M_DISK, m_mode != M_READ);

    return fflush(m_fp) ? false : true;
}


//
// return true if we are at the end of the file. Unknown before reading or
// writing for the first time.
//

bool File::eof() const
{
    ASSERT(M_DISK, m_fp != 0);
    //ASSERT(M_DISK, !m_path.empty() || m_mode == M_READ);//no eof() for stdout

    return feof(m_fp) ? true : false;
}


//
// seek the file to a byte offset. If whence is 0, the offset is counted from
// the beginning of the file; if whence is 1, it is counted from the current
// position in the file; and if whence is 2, it is counted from EOF. Using a
// 64-bit int is rather optimistic here since not all systems support that.
//

bool File::seek(
    Sint64              offset,         // seek to this byte offset
    int                 whence)         // 0=absolute, 1=relative, 2=rel to eof
{
    ASSERT(M_DISK, m_fp != 0);
    //ASSERT(M_DISK, !m_path.empty());  // no seek() for stdin/stdout
#if defined(LINUX) || defined(MACOSX)
    mi_static_assert(sizeof(Sint64) == sizeof(off_t));
    if (!fseeko(m_fp, off_t(offset), whence)) {
#elif defined(WIN_NT)
    if (!_fseeki64(m_fp, offset, whence)) {
#else
    if (!fseek(m_fp, offset, whence)) {
#endif
        m_error = 0;
        return true;
    } else {
        m_error = HAL::get_errno();
        return false;
    }
}


//
// return the current absolute position in the file where the next byte would
// be read or written. This is 0 after opening except in append mode.
//

Sint64 File::tell()
{
    ASSERT(M_DISK, m_fp != 0);
    //ASSERT(M_DISK, !m_path.empty());  // no tell() for stdin/stdout

    m_error = 0;
#if defined(LINUX) || defined(MACOSX)
    mi_static_assert(sizeof(Sint64) == sizeof(off_t));
    return ftello(m_fp);
#elif defined(WIN_NT)
    return _ftelli64(m_fp);
#else
    return ftell(m_fp);
#endif
}


//
// return the size of the file. This is implemented by seeking to the end,
// retrieving the position there, and seeking back. It's fast.
//

Sint64 File::filesize()
{
    ASSERT(M_DISK, m_fp != 0);
    //ASSERT(M_DISK, !m_path.empty());  // no filesize() for stdin/stdout

    Sint64 curr = tell();
    seek(0, 2);
    Sint64 size = tell();
    seek(curr);
    return size;
}


//
// return the system's file descriptor. Don't use this except when third
// party code (e.g. tifflib) forces it. File provides all functions to
// access files, which should be used in all other cases.
//

int File::get_file_descriptor() const
{
    return fileno(get_file_pointer());
}


//
// return whether this File represents a file or stdinput/stdoutput.
//

bool File::is_file() const
{
    return (m_fp != stdin) && (m_fp != stdout) && m_fp;
}


}
}
