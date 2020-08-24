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
/// \brief operation on open directories
///
/// Represents a single directory. All the usual directory operations are here. There is one
/// instance per open directory. The open method is not part of the constructor because we
/// need to return a failure code.

#include "pch.h"

#include "disk.h"

#ifdef LINUX
#include <dirent.h>
#else
#ifdef WIN_NT
#include <mi/base/miwindows.h>
#else
#include <sys/dir.h>            // for DIR (from opendir)
#endif
#endif

#include <cstdio>
#include <base/hal/hal/hal.h>
#include <base/lib/log/log.h>
#include <base/lib/mem/i_mem_allocatable.h>
#include <base/util/string_utils/i_string_utils.h>

namespace MI {
namespace DISK {

using namespace HAL;

#ifndef WIN_NT

// Wrapper for the Unix DIR structure.
struct Hal_dir : public MEM::Allocatable

{
    DIR *m_dp;                  // open directory, 0 if not open
};


//
// constructor and destructor for Directory.
//

Directory::Directory()
{
    m_dp_wrapper = new Hal_dir;
    m_dp_wrapper->m_dp = 0;
    m_error = 0;
    m_eof = true;
}


Directory::~Directory()
{
    close();
    delete m_dp_wrapper;
}


//
// open a directory by path. If the directory was already open, close it first.
// Open directories are mainly useful for iterating over the files or
// directories in them.
//

bool Directory::open(
    const char          *path)          // path to open
{
    m_eof = false;
    if (m_dp_wrapper->m_dp && !close())
        return false;

    m_path = path? path : "";

    if ((m_dp_wrapper->m_dp = opendir(m_path.c_str())) != 0) {
        m_error = 0;
        if (LOG::mod_log)
            LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                               "open directory \"%s\", ok", path);
        return true;
    } else {
        m_error = HAL::get_errno();
        if (LOG::mod_log)
            LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                               "open directory \"%s\": %s",
                               path, HAL::strerror(m_error).c_str());
        return false;
    }
}


//
// close the directory. It is safe to close a directory that is not open.
//

bool Directory::close()
{
    if (m_dp_wrapper->m_dp) {
        closedir(m_dp_wrapper->m_dp);
        m_dp_wrapper->m_dp = 0;
    }
    return true;
}


//
// read the name of the next file or directory in the currently opened
// directory into a buffer. (this Directory must be opened before calling
// this method). File or directory names beginning with a dot are skipped
// if requested. When there are no more names to read, return a null pointer.
//
// Note that if a file is successfully read, the const char * returned is still
// owned by *this*, so caller should take care if something is done with *this*
// (including deleting but also closing the dir may affect the returned char *)
//


std::string Directory::read(
    bool                nodot)          // ignore files beginning with '.'
{
    ASSERT(M_DISK, m_dp_wrapper->m_dp != 0);
    m_error = 0;
    for (;;) {
        struct dirent *entry = readdir(m_dp_wrapper->m_dp);
        if ((m_eof = (0==entry)))                       // eof: return false
            return std::string();
        if (!(nodot && entry->d_name[0] == '.'))        // not a rejected dot
            return entry->d_name;
    }
}


//
// rewind the directory to the beginning.
//

bool Directory::rewind()
{
    ASSERT(M_DISK, m_dp_wrapper->m_dp != 0);
    rewinddir(m_dp_wrapper->m_dp);
    m_eof = false;
    return true;
}


#else // WIN_NT

struct Hal_dir
{
    WIN32_FIND_DATAW m_find_data;
    HANDLE m_first_handle;
    bool m_opened;
    bool m_first_file;
};

//
// constructor and destructor for Directory. They keep the directory list in
// Mod_disk up to date.
//

Directory::Directory()
  :   m_error(0)
{
    m_eof = true;
    m_dir = new Hal_dir;
    memset(&(m_dir->m_find_data), 0, sizeof(m_dir->m_find_data));
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;
    m_dir->m_opened = false;
    m_dir->m_first_file = true;
}


Directory::~Directory()
{
    close();
    delete m_dir;
}


//
// open a directory by path. If the directory was already open, close it first
// Open directories are mainly useful for iterating over the files or
// directories in them.
// The main thing we do here on the windows side of things is to make
// sure that the path is formatted properly. One windows, the file searching
// expects a wildcard, so the path should look something like "c:\users\*".
// But on the other side of things, UNIX users are not typically going to give
// the input as this type of path, so we add the * on to the end (and another
// dir separator if necessary). We do allow users to use the wildcard notation
// -- if we find a '*' in the path we assume it's ok and leave it be.
//
// returns true on success, and false otherwise, which will typically only
// happen if this directory doesn't exist
//

bool Directory::open(
    const char          *path)          // path to open
{
    m_eof = false;
    if ( m_dir->m_opened && !close() )
        return false;

    std::string new_path(path? path : "");

    // if we find a '*', just leave things as they are
    // note that this will likely not work for a 'c:/users/*/log' call
    if (strchr(new_path.c_str(), '*') == NULL) {
        size_t len = path ? strlen(path) : 0;

        // need this as m_path is const char *
        char *temp_path;

        if (len == 0) { // empty string -- assume they just want the curr dir
            temp_path = new char[len+2];
            strcpy(temp_path, "*");
        }
        // otherwise check if there is a trailing delimiter -- if not
        // add one (and also the '*')
        else if (new_path[len-1] != '/' && new_path[len-1] != '\\') {
            temp_path = new char[len+3];
            strcpy(temp_path, new_path.c_str());
            strcat(temp_path, "/*");
        }
        // there is a trailing delimiter, so we just add the wildcard
        else {
            temp_path = new char[len+2];
            strcpy(temp_path, new_path.c_str());
            strcat(temp_path, "*");
        }

        m_path = temp_path;
        delete[] temp_path;
    } else
        m_path = new_path;

    // check for existence -- user is not going to be able to find anything
    // in a directory that isn't there
    if (!Directory::exists(new_path.c_str())) {
        if (LOG::mod_log)
            LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                               "open directory \"%s\", not found",
                               new_path.c_str());
        return false;
    }

    // This flag tells the readdir method whether it should invoke
    // FindFirstFile or FindNextFile
    m_dir->m_first_file = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;

    // and now we indicate we've been opened -- we don't really
    // do much with this open call, it's the first search that matters
    m_dir->m_opened = true;
    if (LOG::mod_log)
        LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                           "open directory \"%s\", ok", m_path.c_str());
    return true;
}


//
// close the directory. It is safe to close a directory that is not open.
//

bool Directory::close()
{
    bool ret_val = true;

    if (m_dir->m_opened && m_dir->m_first_handle != INVALID_HANDLE_VALUE) {
        // FindClose returns BOOL not bool, so we check this way
        ret_val = (::FindClose(m_dir->m_first_handle) != 0);
    }

    m_dir->m_opened = false;
    m_dir->m_first_file = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;
    if (ret_val == false &&  LOG::mod_log)
        LOG::mod_log->debug(M_DISK, LOG::Mod_log::C_DISKTRACE,
                           "close directory \"%s\", failed", m_path.c_str());
    return ret_val;
}


//
// Internal method that encapsulates windows file/dir reading, and
// allows the read() method to easily call us. Note that this method
// returns true on successful Find{First|Next}File read, which includes
// subdirectories also. We leave it up to the caller to determine what
// they are looking for.
// return path is stored in the m_dir's m_find_data structure
// returns true on success, false otherwise
//

bool Directory::read_next_file()
{
    ASSERT(M_DISK, m_dir->m_opened);

    // return if we haven't been opened already
    if (!m_dir->m_opened)
        return false;

    bool success = false;
    if (m_dir->m_first_file) {
        ASSERT(M_DISK, m_path.c_str());
        // to be on the save side we interpret each path as UTF-8 and use the Unicode functionality
        std::wstring path = STRING::utf8_to_wchar(m_path.c_str());
        m_dir->m_first_handle = ::FindFirstFileW(path.c_str(), &m_dir->m_find_data);

        if (m_dir->m_first_handle != INVALID_HANDLE_VALUE) {
            success = true;
            // so we don't call this block again
            m_dir->m_first_file = false;
        }
        else {
            m_error = GetLastError();
            if (m_error == ERROR_NO_MORE_FILES) { // not really an error
                m_error = 0;
                m_eof = true;
            }
        }
    }
    else {
        // FindNextFile returns BOOL not bool, so we check this way
        if (::FindNextFileW(m_dir->m_first_handle, &m_dir->m_find_data) != 0 )
            success = true;
        else {
            m_error = GetLastError();
            if (m_error == ERROR_NO_MORE_FILES) { // not really an error
                m_error = 0;
                m_eof = true;
            }
        }
    }
    return success;
}


// read the name of the next file in the directory into a buffer. File names
// beginning with a dot are skipped if requested. When there are no more names
// to read, return a null pointer.
//
// Note that if a file is successfully read, the const char * returned is still
// owned by *this*, so caller should take care if something is done with *this*
// (including deleting but also closing the dir may affect the returned char *)
std::string Directory::read(
    bool nodot)                                 // ignore files beginning with '.'
{
    ASSERT(M_DISK, m_dir->m_opened);

    m_error = 0;

    if (m_dir->m_opened) {
        while (read_next_file()) {
            if ( nodot ) {
                if (m_dir->m_find_data.cFileName[0] == '.')
                    continue;
            }
            return STRING::wchar_to_utf8(m_dir->m_find_data.cFileName);
        }
    }
    return std::string();
}


//
// rewind the directory to the beginning.
//

bool Directory::rewind()
{
    // Could (and perhaps should) do some assertions here that the dir is
    // already opened, but in this case it doesn't in fact matter, and
    // subsequent read calls will return the first one found
    m_dir->m_first_file = true;
    m_eof = false;
    return true;
}


//
// static method to see if Directory exists on Windows
//

bool Directory::exists(
    const char *path)               // the path to check
{
    if (!path || *path == '\0')  return false;

    std::string newpath(path);
    char *new_path = &newpath[0];

    // let's strip off any trailing *'s, forward- or back-slashes
    size_t len = strlen(new_path);
    while (len>0) {
        if (new_path[len-1] == '*' ||
            new_path[len-1] == '/' ||
            new_path[len-1] == '\\')
                new_path[len-1] = '\0'; // effectively shortens the string
        else
            break;
        len = strlen(new_path);
    }

    if (is_directory(new_path))
        return true;
    else
        return false;
}

#endif


//
// return true if the dir has been successfully opened. This is useful for
// debugging code that loops over all open dirs, and prints the open ones.
//

bool Directory::is_open() const
{
#ifndef WIN_NT
    return m_dp_wrapper->m_dp;
#else
    return m_dir->m_opened;
#endif
}


}
}
