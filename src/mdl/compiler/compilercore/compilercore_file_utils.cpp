/******************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "compilercore_allocator.h"
#include "compilercore_wchar_support.h"
#include "compilercore_file_utils.h"

#include <mi/base/miwindows.h>

#include <sys/types.h>
#include <sys/stat.h>

#ifndef MI_PLATFORM_WINDOWS
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#endif

namespace mi {
namespace mdl {

// Join two paths using the OS specific separator.
string join_path(
    string const &path1,
    string const &path2)
{
    if (path1.empty())
        return path2;
    if (path2.empty())
        return path1;
    size_t l = path1.length() - 1;
    if (path1[l] == os_separator())
        return path1 + path2;
    return path1 + os_separator() + path2;
}

// Check if a file name matches a file mask in UTF-8 encoding.
bool utf8_match(
    char const *file_mask,
    char const *file_name)
{
    unsigned mask_c, name_c;

    file_mask = utf8_to_unicode_char(file_mask, mask_c);
    file_name = utf8_to_unicode_char(file_name, name_c);

    for (;;) {
        if (name_c == '\0')
            return mask_c == '\0';
        if (mask_c == '\0')
            return name_c == '\0';

        // we do NOT support the whole regular expression set here, just the cases we need
        // for UDIM support:
        // -?      - one or zero '-'
        // [0-9]   - one digit
        // [0-9]+  - any number of digits

        if (mask_c == '[') {
            char const *p = file_mask;

            if (strncmp(p, "0-9]", 4) == 0) {
                if (p[4] == '+') {
                    // [0-9]+: match any number of digits
                    file_mask += 5;

                    if (!isdigit(name_c))
                        return false;
                    do {
                        file_name = utf8_to_unicode_char(file_name, name_c);
                    } while (isdigit(name_c));

                    file_mask = utf8_to_unicode_char(file_mask, mask_c);
                    continue;
                } else {
                    // [0-9]: match ONE digit
                    file_mask += 4;

                    if (!isdigit(name_c))
                        return false;

                    file_mask = utf8_to_unicode_char(file_mask, mask_c);
                    file_name = utf8_to_unicode_char(file_name, name_c);
                    continue;
                }
            }
        } else if (mask_c == '-') {
            unsigned n_mask_c;
            char const *n_file_mask = utf8_to_unicode_char(file_mask, n_mask_c);

            if (n_mask_c == '?') {
                // optional '-'
                file_mask = utf8_to_unicode_char(n_file_mask, mask_c);

                if (name_c == '-') {
                    file_name = utf8_to_unicode_char(file_name, name_c);
                }
                continue;
            }
        }

        // normal match
        if (mask_c != name_c)
            return false;

        file_mask = utf8_to_unicode_char(file_mask, mask_c);
        file_name = utf8_to_unicode_char(file_name, name_c);
    }
}

/// Opens a file and returns a pointer to the opened file.
///
/// \param alloc  an allocator
/// \param path   an UTF8 encoded path
/// \param mode   open mode
FILE *fopen_utf8(
    IAllocator *alloc,
    char const *path,
    char const *mode)
{
#ifdef MI_PLATFORM_WINDOWS
    wstring p(alloc);
    utf8_to_utf16(p, path);

    wstring m(alloc);
    utf8_to_utf16(m, mode);

    return _wfopen(p.c_str(), m.c_str());
#else
    return ::fopen(path, mode);
#endif
}

// Check if the given file name (UTF8 encoded) names a file on the file system.
bool is_file_utf8(
    IAllocator *alloc,
    char const *fname)
{
#ifdef MI_PLATFORM_WINDOWS
    struct _stat st;

    wstring path(alloc);
    utf8_to_utf16(path, fname);

    if (!::_wstat(path.c_str(), &st)) {
        return (st.st_mode & S_IFREG) != 0;
    }
#else
    struct stat st;

    // assume native UTF8-support
    if (!::stat(fname, &st)) {
        return (st.st_mode & (S_IFREG | S_IFLNK)) != 0;
    }
#endif
    return false;
}

// Check if in the given directory a file matching the given mask exists.
bool has_file_utf8(
    IAllocator *alloc,
    char const *directory,
    char const *mask)
{
    Directory dir(alloc);

    string dname(directory, alloc);

    char const *p = strrchr(mask, os_separator());
    if (p != NULL) {
        dname += os_separator();
        dname += string(mask, p - mask, alloc);
        mask = p + 1;
    }

    if (!dir.open(dname.c_str()))
        return false;

    for (;;) {
        char const *name = dir.read();

        if (dir.eof() || name == NULL)
            break;

        if (utf8_match(mask, name)) {
            string fname = join_path(dname, string(name, alloc));
            if (is_file_utf8(alloc, fname.c_str()))
                return true;
        }
    }
    return false;
}

// Check if the given name (UTF8 encoded) names a directory on the file system.
bool is_directory_utf8(
    IAllocator *alloc,
    char const *dname)
{
#ifdef MI_PLATFORM_WINDOWS
    struct _stat st;

    wstring path(alloc);
    utf8_to_utf16(path, dname);

    if (!::_wstat(path.c_str(), &st)) {
        return (st.st_mode & S_IFDIR) != 0;
    }
#else
    struct stat st;

    // assume native UTF8-support
    if (!::stat(dname, &st)) {
        return (st.st_mode & S_IFDIR) != 0;
    }
#endif
    return false;
}

// Creates a directory on the file system.
bool mkdir_utf8(
    IAllocator *alloc,
    char const *dname)
{
#ifdef MI_PLATFORM_WINDOWS
    wstring path(alloc);
    utf8_to_utf16(path, dname);

    if (::_wmkdir(path.c_str()) != 0) {
        return false;
    }
#else
    // assume native UTF8-support
    if (mkdir(dname, 0755) != 0) {
        return false;
    }
#endif
    return true;
}

// Get the current working directory
string get_cwd(IAllocator *alloc)
{
    string result(alloc);
#ifdef MI_PLATFORM_WINDOWS
    // retrieve required buffer size first
    DWORD size = GetCurrentDirectoryW(0, NULL); // return value contains terminating null char
    wstring buf(alloc);
    buf.resize(size);

    if (!GetCurrentDirectoryW(size, buf.data())) {
        return string(alloc);
    }
    wchar_to_utf8(result, buf.data());
#else
    char buf[PATH_MAX];
    if (getcwd(buf, PATH_MAX)) {
        buf[sizeof(buf) -1] = '\0';
        result = buf;
    }
#endif
    return result;
}

// test if character is file path separator
static inline bool is_path_separator(
    char  c )
{
    return c == '/' || c == '\\';
}

// return true if a path is absolute, ie. begins with / or X:/
bool is_path_absolute(
    char const *path)
{
    if (path == NULL || path[0] == 0)
        return false;

    if (is_path_separator(path[0]))
        return true;

    if (path[1] == ':' && is_path_separator(path[2]) &&
        ((path[0] >= 'a' && path[0] <= 'z') || (path[0] >= 'A' && path[0] <= 'Z')))
        return true;

    return false;
}

// Simplifies a file path.
string simplify_path(
    IAllocator   *alloc,
    string const &file_path,
    string const &sep)
{
    MDL_ASSERT(!file_path.empty());

    vector<string>::Type directory_names(alloc);

    size_t start = 0;
    size_t length = file_path.size();
    size_t sep_length = sep.size();

    size_t slash;
    do {
        slash = file_path.find(sep, start);
        if (slash == string::npos)
            slash = length;
        string directory_name = file_path.substr(start, slash - start);
        if (directory_name == ".") {
            // ignore
        } else if (directory_name == "..") {
            // drop one
            if (directory_names.empty()) {
                // trying to go above root. Linux AND windows allow this.
                // In Unix, '..' is always a link, and in the case of root a link to itself
                // Windows handles it the same way, so allow it
            } else {
                directory_names.pop_back();
            }
        }
        else if (!directory_name.empty()) {
            directory_names.push_back(directory_name);
        }
        start = slash + sep_length;
    } while (slash != length);

    string result(alloc);
    if (file_path.find(sep) == 0) {
        result += sep;
        if (file_path.find(sep, sep_length) == sep_length) {
            // handle '//' at start
            result += sep;
        }
    }
    if (!directory_names.empty())
        result += directory_names[0];
    for (size_t i = 1, n = directory_names.size(); i < n; ++i) {
        result += sep + directory_names[i];
    }
    if (file_path.find(sep, length - sep_length) == (length - sep_length) && 
        (result.length() != sep_length || result.find(sep, 0) != 0))
            result += sep;

    return result;
}

// Simplifies a file path.
string simplify_path(
    IAllocator   *alloc,
    string const &file_path,
    char         sep)
{
    string sep_string(alloc);
    sep_string.append(sep);
    return simplify_path(alloc, file_path, sep_string);
}

// Converts OS-specific directory separators into slashes.
string convert_os_separators_to_slashes(string const &s)
{
    char sep = os_separator();
    if (sep == '/')
        return s;

    string r(s);

    for (size_t i = 0, n = r.length(); i < n; ++i) {
        if (r[i] == sep)
            r[i] = '/';
    }
    return r;
}

// Converts slashes into OS-specific directory separators.
string convert_slashes_to_os_separators(string const &s)
{
    char sep = os_separator();
    if (sep == '/')
        return s;

    string r(s);

    for (size_t i = 0, n = r.length(); i < n; ++i) {
        if (r[i] == '/')
            r[i] = sep;
    }
    return r;
}

#ifdef MI_PLATFORM_WINDOWS

struct Directory::Hal_dir
{
    WIN32_FIND_DATAW m_find_data;
    HANDLE           m_first_handle;
    bool             m_opened;
    bool             m_first_file;
};

/// Check if the given directory path exists.
///
/// \param alloc      the allocator
/// \param utf8_path  an UTF8 encoded directory path
static bool directory_exists(
    IAllocator *alloc,
    char const *utf8_path)
{
    wstring newpath(alloc);
    utf8_to_utf16(newpath, utf8_path);

    // let's strip off any trailing *'s, forward- or back-slashes
    size_t len = newpath.size();
    while (len > 0) {
        wchar_t c = newpath[len-1];
        if (c != L'*' && c != L'/' && c != L'\\')
            break;
        --len;
    }
    newpath = newpath.substr(0, len);
    DWORD res = ::GetFileAttributesW(newpath.c_str());
    if (res == INVALID_FILE_ATTRIBUTES)
        return false;
    return (res & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

// Constructor.
Directory::Directory(IAllocator *alloc)
: m_alloc(alloc)
, m_dir(NULL)
, m_path(alloc)
, m_opath(alloc)
, m_tmp(alloc)
, m_error(0)
, m_eof(true)
{
    Allocator_builder builder(alloc);

    m_dir = builder.create<Hal_dir>();
    memset(&(m_dir->m_find_data), 0, sizeof(m_dir->m_find_data));
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;
    m_dir->m_opened       = false;
    m_dir->m_first_file   = true;
}

// Destructor.
Directory::~Directory()
{
    Allocator_builder builder(m_alloc);

    close();
    builder.destroy(m_dir);
}

// Open a directory for reading names in it.
bool Directory::open(
    char const *utf8_path,
    char const *utf8_filter)
{
    m_eof = false;
    if (m_dir->m_opened && !close())
        return false;

    string new_path(utf8_path != NULL ? utf8_path : "", m_alloc);
    m_opath = new_path;

    // if we find a '*', just leave things as they are
    // note that this will likely not work for a 'c:/users/*/log' call
    if (new_path.find('*') == string::npos) {
        size_t len = new_path.length();

        // need this as m_path is const char *
        string temp_path(new_path);

        if (len == 0) { // empty string -- assume they just want the curr dir
            temp_path = utf8_filter == NULL ? "*" : utf8_filter;
        } else if (new_path[len - 1] == '/' || new_path[len - 1] == '\\') {
            // there is a trailing delimiter, so we just add the wildcard
            temp_path += utf8_filter == NULL ? "*" : utf8_filter;
        } else {
            // no trailing delimiter -- add one (and also the '*')
            temp_path += '/';
            temp_path += utf8_filter == NULL ? "*" : utf8_filter;
        }

        m_path = temp_path;
    } else
        m_path = new_path;

    // check for existence -- user is not going to be able to find anything
    // in a directory that isn't there
    if (!directory_exists(m_alloc, new_path.c_str())) {
        return false;
    }

    // This flag tells the readdir method whether it should invoke
    // FindFirstFile or FindNextFile
    m_dir->m_first_file   = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;

    // and now we indicate we've been opened -- we don't really
    // do much with this open call, it's the first search that matters
    m_dir->m_opened = true;
    return true;
}

// Close directory.
bool Directory::close()
{
    bool ret_val = true;

    if (m_dir->m_opened && m_dir->m_first_handle != INVALID_HANDLE_VALUE) {
        // FindClose returns BOOL not bool, so we check this way
        ret_val = (::FindClose(m_dir->m_first_handle) != 0);
    }

    m_dir->m_opened       = false;
    m_dir->m_first_file   = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;
    return ret_val;
}

// Windows-specific helper method to encapsulate the directory reading code.
bool Directory::read_next_file()
{
    // return if we haven't been opened already
    if (!m_dir->m_opened)
        return false;

    bool success = false;
    if (m_dir->m_first_file) {
        wstring w_path(m_alloc);
        utf8_to_utf16(w_path, m_path.c_str());

        m_dir->m_first_handle = ::FindFirstFileW(
            w_path.c_str(),             // our path
            &m_dir->m_find_data);       // where windows puts the results

        if (m_dir->m_first_handle != INVALID_HANDLE_VALUE) {
            success = true;
            // so we don't call this block again
            m_dir->m_first_file = false;
        } else {
            m_error = GetLastError();
            if (m_error == ERROR_NO_MORE_FILES) { // not really an error
                m_error = 0;
                m_eof = true;
            }
        }
    } else {
        // FindNextFileW returns BOOL not bool, so we check this way
        if (::FindNextFileW(m_dir->m_first_handle, &m_dir->m_find_data) != 0) {
            success = true;
        } else {
            m_error = GetLastError();
            if (m_error == ERROR_NO_MORE_FILES) { // not really an error
                m_error = 0;
                m_eof = true;
            }
        }
    }
    return success;
}

// Read the next filename from the directory. Names are unsorted.
char const *Directory::read()
{
    m_error = 0;
    if (m_dir->m_opened) {
        if (read_next_file()) {
            // We don't dup the returned data
            return wchar_to_utf8(m_tmp, m_dir->m_find_data.cFileName);
        }
    }
    return NULL;
}

bool Directory::rewind()
{
    // Could (and perhaps should) do some assertions here that the dir is
    // already opened, but in this case it doesn't in fact matter, and
    // subsequent read calls will return the first one found
    m_dir->m_first_file = true;
    m_eof               = false;
    return true;
}

#else // !MI_PLATFORM_WINDOWS

// Wrapper for the Unix DIR structure.
struct Directory::Hal_dir
{
    // open directory, NULL if not open
    DIR *m_dp;
};

// Constructor.
Directory::Directory(IAllocator *alloc)
: m_alloc(alloc)
, m_dir(NULL)
, m_path(alloc)
, m_opath(alloc)
, m_tmp(alloc)
, m_error(0)
, m_eof(true)
{
    Allocator_builder builder(alloc);
    m_dir = builder.create<Hal_dir>();
    m_dir->m_dp = NULL;
}

// Destructor.
Directory::~Directory()
{
    Allocator_builder builder(m_alloc);
    close();
    builder.destroy(m_dir);
}

// Open a directory for reading names in it.
bool Directory::open(
    char const *utf8_path,
    char const *utf8_filter)
{
    m_eof = false;
    if (m_dir->m_dp && !close())
        return false;

    m_path  = utf8_path != NULL ? utf8_path : "";
    m_opath = m_path;

    if ((m_dir->m_dp = opendir(m_path.c_str())) != NULL) {
        m_error = 0;
        return true;
    } else {
        m_error = errno;
        return false;
    }
}

// Close directory.
bool Directory::close()
{
    if (m_dir->m_dp != NULL) {
        closedir(m_dir->m_dp);
        m_dir->m_dp = NULL;
    }
    return true;
}

char const *Directory::read()
{
    m_error = 0;
    for (;;) {
        struct dirent *entry = readdir(m_dir->m_dp);
        if (entry == NULL) {
            m_eof = true;
            return NULL;
        }
        return entry->d_name;
    }
}

bool Directory::rewind()
{
    rewinddir(m_dir->m_dp);
    m_eof = false;
    return true;
}

#endif // MI_PLATFORM_WINDOWS

}  // mdl
}  // mi
