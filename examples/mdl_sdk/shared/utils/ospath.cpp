/***************************************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The implementation of useful path' utilities.

#include "ospath.h"

#include "sys/types.h"
#include "sys/stat.h"
#include <sstream>
#include <cassert>

#include "strings.h"
using namespace mi::examples::strings;

#ifndef WIN_NT
#include <unistd.h>
#include <dirent.h>
#else
#include <mi/base/miwindows.h>
#endif
#include <algorithm>
#include <vector>

namespace mi { namespace examples { namespace ospath {

namespace {

// Replace all occurrence of a string with another one
std::string replace_all(
    const std::string& str,             // input string
    const std::string& fstr,            // find string
    const std::string& rstr)            // replace string
{
    if (str.empty())
        return std::string("");

    std::string fs(str);
    size_t pos = 0;
    while(pos != std::string::npos) {
        pos = fs.find(fstr, pos);
        if (pos != std::string::npos) {
            fs.replace(pos, fstr.size(), rstr);
        }
    }

    return fs;
}

void splitpath(std::string path, std::vector<std::string>& components)
{
    components.clear();

    std::string npath = Ospath::convert_to_forward_slashes(path);
    npath = replace_all(npath, std::string("//"), std::string("/"));

    size_t p0 = 0, p1;
    while ((p1 = npath.find('/', p0)) != std::string::npos) {
        if (p1 - p0 > 0)
            components.push_back(npath.substr(p0, p1 - p0));
        p0 = p1 + 1;
    }
    if (p0 < npath.size())
        components.push_back(npath.substr(p0));
}

}

// Return the OS independent separator. On Windows the separator is '\',
// while on the other operating system is is '/'.
std::string Ospath::sep()
{
#ifdef WIN_NT
        return std::string("\\");
#else
        return std::string("/");
#endif
}

// Return the OS independent path set separator.
// on Linux and MacOSX, path set is separated by ":", e.g.,
// "/home/user0:/home/user0", on Windows,
// "C:\home\user0;C:\home\user0"
std::string Ospath::get_path_set_separator()
{
#ifdef _WIN32
    return ";";
#else
    return ":";
#endif
}

// Return the base name of pathname path. This is the second half of the
// pair returned by split(path). Note that the result of this function is
// different from the Unix basename program; where basename for '/foo/bar/'
// returns 'bar', the basename() function returns an empty string ('').
std::string Ospath::basename(
    const std::string& path)                    // Incoming path
{
    std::string head, tail;
    split(path, head, tail);
    return tail;
}

// Return the directory name of pathname path. This is the first half of
// the pair returned by split(path).
std::string Ospath::dirname(
    const std::string& path)                    // Incoming path
{
    std::string head, tail;
    split(path, head, tail);
    return head;
}

// Joins one or more path components intelligently. If any component is an
// absolute path, all previous components are thrown away, and joining
// continues. The return value is the concatenation of path1, and
// optionally path2, etc., with exactly one directory separator (os.sep)
// inserted between components, unless path2 is empty.
std::string Ospath::join(
    const std::string& path,            // Incoming path1
    const std::string& path2)           // Incoming path2
{
    if (path.empty())
        return path2;
    else if (path2.empty())
        return path;
    else
        return path + sep() + path2;
}


#if 0
// Normalize a pathname. This collapses redundant separators and up-level
// references, e.g. A//B, A/./B and A/foo/../B all become A/B. It does not
// normalize the case. On Windows, it converts forward slashes to backward slashes.
std::string Ospath::normpath(
    const std::string& orig_in_path)            // Incoming path
{
    std::vector<std::string> token_list;
    const std::string separator = HAL::Ospath::sep();
    const std::string in_path = Ospath::normpath_only(orig_in_path);
    std::string out_path;

    UTIL::Tokenizer::parse(in_path, separator, token_list);

    std::vector<std::string> last_dir;
    std::vector<std::string>::const_iterator it = token_list.begin();

    // iterate over all elements, skipping . and resolving ..
    while(it != token_list.end()) {
        if (it->empty()) { // We do not care about empty tokens (i.e. //)
            ++it;
            continue;
        }
        // valid dir names are appended and saved on a stack (last_dir)
        if (*it != ".." && *it != ".") {
            last_dir.push_back(*it);
            if (!out_path.empty() && out_path[out_path.length() - 1] != separator[0])
                out_path.append(separator);
            out_path.append(*it);
        }
        // ".." means we have to pop the last valid dir name from the stack if we can
        else if (*it == "..") {
            if (last_dir.empty())
                return orig_in_path;
            size_t ind = out_path.rfind(last_dir.back());
            if (ind == std::string::npos)
                return orig_in_path;
            out_path = out_path.substr(0, ind);
            last_dir.pop_back();
        }
        ++it;
    }

    // special case: path ended with /
    if (!out_path.empty() && !in_path.empty() && in_path[in_path.length() - 1] == separator[0])
        out_path.append(separator);
    // special case: path was absolute, i.e. started with /
    if (!in_path.empty() && in_path[0] == separator[0])
      out_path.insert(0, separator);

#if WIN_NT
    // Check for UNC and put back the '\\' which have been removed.
    // First convert back to backslashes since siteconfig converts
    // paths to forward slashes which means the test against sep()
    // which returns backslashes on windows would not work.
    std::string win_path = convert_to_backward_slashes(orig_in_path);
    if (win_path.length() >= 2 &&
        (win_path[0] == sep()[0]) && (win_path[1] == sep()[0]))
        out_path.insert(0, sep());
    return convert_to_backward_slashes(out_path);
#else
    return out_path;
#endif
}
#endif


// Normalize a pathname. On Windows, it converts forward slashes to
// backward slashes.
std::string Ospath::normpath_only(
    const std::string& path)            // Incoming path
{
    std::string npath = convert_to_forward_slashes(path);
    npath = replace_all(npath, std::string("//"), std::string("/"));

#if WIN_NT
    // Check for UNC and put back the '\\' which have been removed.
    // First convert back to backslashes since siteconfig converts
    // paths to forward slashes which means the test against sep()
    // which returns backslashes on windows would not work.
    std::string win_path = convert_to_backward_slashes(path);
    if (win_path.length() >= 2 &&
        (win_path[0] == sep()[0]) && (win_path[1] == sep()[0]))
        npath.insert(0, sep());
    return convert_to_backward_slashes(npath);
#else
    return npath;
#endif
}

std::string Ospath::normpath_v2(const std::string& path)
{
    std::string p = Ospath::convert_to_forward_slashes(path);
    std::string npath = replace_all(p, std::string("//"), std::string("/"));

    std::vector<std::string> path_components;
    size_t p0 = 0, p1;
    while ((p1 = npath.find('/', p0)) != std::string::npos) {
        if (p1 - p0 > 0)
            path_components.push_back(npath.substr(p0, p1 - p0));
        p0 = p1 + 1;
    }
    if (p0 < npath.size())
        path_components.push_back(npath.substr(p0));

    // resolved current and parent directory references
    std::vector<std::string> result_components;
    for (size_t i = 0; i < path_components.size(); ++i) {
        const std::string& component = path_components[i];

        // skip empty path components or current directory references
        if (component.empty() || component == ".")
            continue;

        // handle parent directory references
        if (component == "..") {
            if (result_components.empty() || result_components.back() == "..")
                result_components.push_back("..");
            else
                result_components.pop_back();
            continue;
        }

        // handle regular path components
        result_components.push_back(component);
    }

    // convert result_components into a string
    std::string result = result_components.empty() ? "" : result_components[0];
    for (size_t i = 1; i < result_components.size(); ++i)
        result += "/" + result_components[i];

    // re-add leading separator for absolute paths
    if (!p.empty() && p[0] == '/')
        result.insert(0, "/");

#if WIN_NT
    // re-add another leading separator for UNC paths on Windows
    if (p.length() >= 2 && p[0] == '/' && p[1] == '/')
        result.insert(0, "/");
#endif

    return result.empty() ? "." : result;
}


// Split the pathname path into a pair, (head, tail) where tail is the last
// pathname component and head is everything leading up to that. The tail
// part will never contain a slash; if path ends in a slash, tail will be
// empty. If there is no slash in path, head will be empty. If path is
// empty, both head and tail are empty. Trailing slashes are stripped from
// head unless it is the root (one or more slashes only). In nearly all
// cases, join(head, tail) equals path (the only exception being when there
// were multiple slashes separating head from tail).
void Ospath::split(
    const std::string& path,                    // Incoming path
    std::string& head_ref,                      // Out head part
    std::string& tail_ref)                      // Out tail part
{
    std::string head, tail;

    if (path.empty()) {
        head_ref = head;
        tail_ref = tail;
        return;
    }

    std::string filepath;
    filepath = convert_to_forward_slashes(path);

    std::string::size_type i = filepath.find_last_of('/');
    if (i == std::string::npos) {
        head_ref = std::string("");
        tail_ref = path;
        return;
    }

    // Remove from one behind '/' to the end
    head = path;
    head.erase(i+1, std::string::npos);
    // remove trailing '/' iff size()>1 and if this is not a Windows drive, eg C:/
    if (head.size() > 1
#ifdef WIN_NT
        // watch out for a drive
        && !(head.size() > 2 && head[head.size()-2] == ':')
#endif
        )
        head.erase(i);

    // Remove from the beginning of the string to the '/' included
    tail = path;
    tail.erase(0, i+1);

    head_ref = head;
    tail_ref = tail;
}

// Split the pathname path into a pair, (head, tail) where tail is the
// last pathname component and head is everything leading up to that.
// This version leaves both parts untouched and does only the splitting.
void Ospath::split_only(
    const std::string& path,            // Incoming path
    std::string& head,                  // Out head part
    std::string& tail)                  // Out tail part
{
    // set pos to the 1st delimiter
    std::string::size_type pos_win = path.rfind('\\');
    std::string::size_type pos_lin = path.rfind('/');
    std::string::size_type pos = std::string::npos;
    if (pos_win == std::string::npos && pos_lin != std::string::npos)
        pos = pos_lin;
    else if (pos_win != std::string::npos && pos_lin == std::string::npos)
        pos = pos_win;
    else
        pos = std::max(pos_win, pos_lin);

    head = path.substr(0, pos);                         // this spares the trailing delimiter
    tail = path.substr(pos == std::string::npos? 0 : pos+1);
}

// Split the pathname path into a pair (drive, tail) where drive is either
// a drive specification or the empty string. On systems which do not use
// drive specifications, drive will always be the empty string. In all
// cases, drive + tail will be the same as path.
void Ospath::splitdrive(
    const std::string& path,            // Incoming path
    std::string& drive,                 // Out drive part
    std::string& tail)                  // Out tail part
{
    drive = std::string("");
    tail = std::string("");
    if (path.empty())
        return;

    std::string::size_type i = path.find_last_of(':');
    if (i == std::string::npos)
    {
        tail = path;
        return;
    }

    drive = path.substr(0,i+1);
    tail = path.substr(i+1);
}

// Split the pathname path into a pair (root, ext) such that
// root + ext == path, and ext is empty or begins with a period and
// contains at most one period.
void Ospath::splitext(
    const std::string& path,            // Incoming path
    std::string& root,                  // Out root part
    std::string& ext)                   // Out extension part
{
    root = std::string("");
    ext = std::string("");
    if (path.empty())
        return;

    std::string::size_type i = path.find_last_of('.');
    if (i == std::string::npos)
        return;

    root = path;
    root.erase(i, std::string::npos);
    ext = path;
    ext.erase(0, i);
}


std::string Ospath::get_ext(
    const std::string& path)
{
    std::string::size_type i = path.find_last_of('.');
    if (i == std::string::npos)
        return std::string();
    return path.substr(i);
}


// Convert the given path to use forward slashes
std::string Ospath::convert_to_forward_slashes(
    const std::string& path)            // Convert this path
{
    return replace_all(path, std::string("\\"), std::string("/"));
}


// Convert the given path to use forward slashes
std::string Ospath::convert_to_backward_slashes(
    const std::string& path)            // Convert this path
{
    return replace_all(path, std::string("/"), std::string("\\"));
}


namespace {

// Test if character \p c is file path separator.
// \param c character to test
inline bool is_path_separator(
    char  c)
{
    return c == '/' || c == '\\';
}

}

// return true if a path is absolute, ie. begins with / or X:/
bool is_path_absolute(
    const std::string& path)
{
    if (path.empty())
        return false;

    if (is_path_separator(path[0]))
        return true;

    if (((path[0] >= 'a' && path[0] <= 'z') ||
        (path[0] >= 'A' && path[0] <= 'Z'))
        && path[1] == ':'
        && is_path_separator(path[2]))
        return true;

    // check for uri's - not exactly a bullet-proof test
    if (path.size() > 5) {
        if (path.find("file:") == 0 || path.find("FILE:") == 0)
            return true;
    }

    return false;
}


bool find_path(
    const std::string& file)
{
    struct stat buf;
    return ::stat(file.c_str(), &buf) == 0;
}


bool is_directory(
    const std::string& path)
{
    struct stat buf;
    if (::stat(path.c_str(), &buf) == 0)
        return (buf.st_mode & S_IFDIR) != 0;
    return false;
}


bool is_file(
    const std::string& path)
{
    struct stat buf;
    if (::stat(path.c_str(), &buf) == 0)
        return (buf.st_mode & S_IFREG) != 0;
    return false;
}

std::string get_cwd()
{
    std::string result;
#ifdef WIN_NT
    // retrieve required buffer size first
    const DWORD size = GetCurrentDirectoryW(0, nullptr); // return value contains terminating null char
    std::vector<wchar_t> buf(size);
    if (!GetCurrentDirectoryW(size, buf.data()))
        return std::string();

    result = wchar_to_utf8(buf.data());
#else
    char buf[PATH_MAX];
    if (getcwd(buf, PATH_MAX)) {
        buf[sizeof(buf) - 1] = '\0';
        result = buf;
    }
#endif
    return result;
}


bool make_path_relative(
    const std::string& abs_base_dir,
    const std::string& abs_path,
    std::string& rel_path)
{
    if (!is_path_absolute(abs_base_dir) || !(is_path_absolute(abs_path))) {
        rel_path = abs_path;
        return false;
    }
    std::string abs_base_dir_norm = Ospath::normpath_v2(abs_base_dir);
    std::string abs_path_norm = Ospath::normpath_v2(abs_path);

    std::string abs_dir_norm, filename;
    Ospath::split(abs_path_norm, abs_dir_norm, filename);

    if (abs_base_dir_norm == abs_dir_norm) {
        rel_path = filename;
        return true;
    }
    std::vector<std::string> abs_base_dir_comp, abs_dir_comp;

    splitpath(abs_base_dir_norm, abs_base_dir_comp);
    splitpath(abs_dir_norm, abs_dir_comp);
    int n = std::min(abs_base_dir_comp.size(), abs_dir_comp.size());
    int c = 0;
    for (int i = 0; i < n; ++i) {
        if (abs_base_dir_comp[i] == abs_dir_comp[i])
            c++;
        else
            break;
    }
#if WIN_NT
    if (c == 0) { // nothing in common
        rel_path = abs_path;
        return false;
    }
#endif
    std::string up;
    for (int i = c; i < abs_base_dir_comp.size(); ++i) {
        up += "../";
    }
    rel_path = up;
    for (int i = c; i < abs_dir_comp.size(); ++i) {
        rel_path += abs_dir_comp[i] + "/";
    }
    rel_path += filename;
    return true;
}


#ifndef WIN_NT

// Wrapper for the Unix DIR structure.
struct Hal_dir

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

    m_path = path ? path : "";

    if ((m_dp_wrapper->m_dp = opendir(m_path.c_str())) != 0) {
        m_error = 0;
        return true;
    }
    else {
        m_error = -1;
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
    m_error = 0;
    for (;;) {
        struct dirent *entry = readdir(m_dp_wrapper->m_dp);
        if ((m_eof = (0 == entry)))                     // eof: return false
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
    : m_error(0)
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
    if (m_dir->m_opened && !close())
        return false;

    std::string new_path(path ? path : "");

    // if we find a '*', just leave things as they are
    // note that this will likely not work for a 'c:/users/*/log' call
    if (strchr(new_path.c_str(), '*') == NULL) {
        size_t len = path ? strlen(path) : 0;

        // need this as m_path is const char *
        char *temp_path;

        if (len == 0) { // empty string -- assume they just want the curr dir
            temp_path = new char[len + 2];
            strcpy(temp_path, "*");
        }
        // otherwise check if there is a trailing delimiter -- if not
        // add one (and also the '*')
        else if (new_path[len - 1] != '/' && new_path[len - 1] != '\\') {
            temp_path = new char[len + 3];
            strcpy(temp_path, new_path.c_str());
            strcat(temp_path, "/*");
        }
        // there is a trailing delimiter, so we just add the wildcard
        else {
            temp_path = new char[len + 2];
            strcpy(temp_path, new_path.c_str());
            strcat(temp_path, "*");
        }

        m_path = temp_path;
        delete[] temp_path;
    }
    else
        m_path = new_path;

    // check for existence -- user is not going to be able to find anything
    // in a directory that isn't there
    if (!Directory::exists(new_path.c_str())) {
        return false;
    }

    // This flag tells the readdir method whether it should invoke
    // FindFirstFile or FindNextFile
    m_dir->m_first_file = true;
    m_dir->m_first_handle = INVALID_HANDLE_VALUE;

    // and now we indicate we've been opened -- we don't really
    // do much with this open call, it's the first search that matters
    m_dir->m_opened = true;

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
    assert(m_dir->m_opened);

    // return if we haven't been opened already
    if (!m_dir->m_opened)
        return false;

    bool success = false;
    if (m_dir->m_first_file) {
        // to be on the save side we interpret each path as UTF-8 and use the Unicode functionality
        const std::wstring path = utf8_to_wchar(m_path.c_str());

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
        if (::FindNextFileW(m_dir->m_first_handle, &m_dir->m_find_data) != 0)
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
    assert(m_dir->m_opened);

    m_error = 0;

    if (m_dir->m_opened) {
        while (read_next_file()) {
            if (nodot) {
                if (m_dir->m_find_data.cFileName[0] == '.')
                    continue;
            }
            return wchar_to_utf8(m_dir->m_find_data.cFileName);
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
    const char *path)     // the path to check
{
    if (!path || *path == '\0')  return false;

    std::string newpath(path);
    char *new_path = &newpath[0];

    // let's strip off any trailing *'s, forward- or back-slashes
    size_t len = strlen(new_path);
    while (len>0) {
        if (new_path[len - 1] == '*' ||
            new_path[len - 1] == '/' ||
            new_path[len - 1] == '\\')
            new_path[len - 1] = '\0'; // effectively shortens the string
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

}}}
