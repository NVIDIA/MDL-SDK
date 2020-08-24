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
/// \brief implementation of some disk functionality

#include "pch.h"
#include "disk.h"

#include <base/lib/log/log.h>
#include <base/hal/hal/hal.h>
#include <base/system/stlext/i_stlext_no_unused_variable_warning.h>
#include <base/util/string_utils/i_string_utils.h>

#include <cerrno>
#include <cstring>

#ifndef WIN_NT

#ifdef MACOSX
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <climits>
#include <sys/vfs.h>            // for statfs (Linux)
#include <sys/statfs.h>         // for statfs (Irix)
#endif
#include <sys/stat.h>           // for mkdir, stat
#include <unistd.h>             // for access, link, unlink, rmdir

#else
#include <crtdbg.h>             // For _CrtSetReportMode
#include <mi/base/miwindows.h>
#include <shlobj.h>
#include <io.h>
#endif

#include <sys/types.h>          // for statfs (Irix), mkdir
#include <sys/stat.h>           // for _stat

namespace MI {
namespace DISK {

using namespace LOG;

//-----------------------------------------------------------------------------

// Set error to given value.
void set_error(
    int value)                                          // the value
{
    // just to avoid any compiler warnings - this whole function should be removed.
    STLEXT::no_unused_variable_warning_please(value);
}

//-----------------------------------------------------------------------------

// test if character is file path separator
inline bool is_path_separator(
    char  c )                                           // character to test
{
    return c == '/' || c == '\\';
}


//-----------------------------------------------------------------------------

// return true if a path is absolute, ie. begins with / or X:/
bool is_path_absolute(
    const char* path)                                   // check this path
{
    if (!path)
        return false;

    if (is_path_separator(path[0]))
        return true;

    if (((path[0] >= 'a' && path[0] <= 'z') ||
         (path[0] >= 'A' && path[0] <= 'Z'))
        && path[1] == ':'
        && is_path_separator(path[2]))
        return true;

    return false;
}

bool is_path_absolute(
    const std::string& path)
{
    return is_path_absolute(path.c_str());
}


//-----------------------------------------------------------------------------

// given a path, return true if that path is a directory, false otherwise
bool is_directory(
    const char* path)                                   // path to check
{
    Stat file_stat;
    if (stat(path, &file_stat) && file_stat.m_is_dir)
        return true;
    else
        return false;
}

//-----------------------------------------------------------------------------

// given a path, return true if that path is a regular file or a symlink, false otherwise
bool is_file(
    const char* path)                                   // path to check
{
    Stat file_stat;
    if (stat(path, &file_stat) && file_stat.m_is_file)
        return true;
    else
        return false;
}


#ifndef WIN_NT

//-----------------------------------------------------------------------------

// delete a file
bool file_remove(
    const char* path)                   // path of file to remove
{
    std::string npath(path? path : "");

    if (!unlink(npath.c_str())) {
        set_error(0);
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}

//-----------------------------------------------------------------------------

// rename a file or directory
bool rename(
    const char          *opath,         // path of file to rename
    const char          *npath)         // new path; must be on the same disk
{
    std::string nopath(opath? opath : "");
    std::string nnpath(npath? npath : "");

    if (!::rename(nopath.c_str(), nnpath.c_str())) {
        set_error(0);
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}


//-------------------------------------------------------------------------------------------------

/// Create a directory. Since the POSIX standard C library call mkdir() doesn't support the
/// creation of intermediate directories as required - ie, something the utility 'mkdir -p'
/// offers - we use the utility's OpenBSD implementation.
/// \p path path of directory to create
/// \p mode file mode of terminal directory
/// \p dir_mode file mode of intermediate directories
/// \return 0 on success, or -1 if an error occurred
int mkpath(char *path, mode_t mode, mode_t dir_mode)
{
     struct stat sb;
     char *slash;
     int done, exists;

     slash = path;

     for (;;) {
         slash += strspn(slash, "/");
         slash += strcspn(slash, "/");

         done = (*slash == '\0');
         *slash = '\0';

         /* skip existing path components */
         exists = !stat(path, &sb);
         if (!done && exists && S_ISDIR(sb.st_mode)) {
             *slash = '/';
             continue;
         }

         if (::mkdir(path, done ? mode : dir_mode) < 0) {
             if (!exists) {
                 /* Not there */
//               warn("%s", path);
                 return (-1);
             }
             if (!S_ISDIR(sb.st_mode)) {
                 /* Is there, but isn't a directory */
                 errno = ENOTDIR;
//               warn("%s", path);
                 return (-1);
             }
         }

         if (done)
             break;

         *slash = '/';
     }

     return (0);
}


//-----------------------------------------------------------------------------

// create a directory with a given set of permissios, default 0755
bool mkdir(
    const char          *path,          // path of directory to create
    int                 mode)           // rwxrwxrwx permissions
{
    std::string npath(path? path : "");

    //if (!::mkdir(npath.c_str(), mode)) {
    if (!mkpath(const_cast<char*>(npath.c_str()), mode, mode)) {
        set_error(0);
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}


//-----------------------------------------------------------------------------

// delete a directory. The directory must be empty.
bool rmdir(
    const char          *path)          // path of directory to delete
{
    std::string npath(path? path : "");

    if (!::rmdir(npath.c_str())) {
        set_error(0);
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}


//-----------------------------------------------------------------------------

// set the current working directory
bool chdir(
    const char  *path)          // path of directory to make current
{
    if (path == NULL)
        return false;

    std::string npath(path);

    if (!::chdir(npath.c_str())) {
        set_error(0);
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}

//-----------------------------------------------------------------------------

// get the current working directory
static const char *getcurdir()
{
    static char buf[PATH_MAX];

    if (!getcwd(buf, PATH_MAX))
        return 0;

    return buf;
}


//--------------------------------------------------------------------------------------------------

// Get the current working directory
std::string get_cwd()
{
    std::string result;
    if (const char* cwd = getcurdir()) {
        result = cwd;
        set_error(0);
    }
    else
        set_error(HAL::get_errno());
    return result;
}


//-----------------------------------------------------------------------------

// given a path, return information about the file. Return true, if the
// operation succeeded or false, if not. In the latter case the file_stat
// structure is left unchanged.
bool stat(
    const char          *path,          // path to get information for
    Stat                *file_stat)     // store the results here
{
    struct stat         st;             // results of inode check

    // Be defensive and check for NULL since std::string will
    // ASSERT if handed a NULL string. Set m_error to ENOENT since
    // that is what stat would have returned.
    if (!path) {
        set_error(ENOENT);
        return false;
    }

    std::string npath(path);

    if (!::stat(npath.c_str(), &st)) {
        set_error(0);
        file_stat->m_size = st.st_size;
        file_stat->m_modification_time = st.st_mtime;
        file_stat->m_access_time = st.st_atime;
        file_stat->m_change_time = st.st_ctime;
        file_stat->m_is_dir = !!(st.st_mode & S_IFDIR);
        file_stat->m_is_file = !!(st.st_mode & (S_IFREG | S_IFLNK));
        file_stat->m_is_readable = st.st_mode & S_IRUSR;
        file_stat->m_is_writable = st.st_mode & S_IWUSR;
        file_stat->m_is_executable = st.st_mode & S_IXUSR;
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}


//-----------------------------------------------------------------------------

// return free space in bytes on the file system that <path> resides on. This
// can be any file or directory in the filesystem, including its mountpoint.
Sint64 freespace(
    const char          *path)          // path of a file/dir on disk to test
{
    struct statfs       fs;             // collect file system stats here

    std::string npath(path? path : "");

    if (!statfs(npath.c_str(), &fs)) {
        set_error(0);
        return (Sint64)fs.f_bfree * (Sint64)fs.f_bsize;
    } else {
        set_error(HAL::get_errno());
        return -1;
    }
}


//-----------------------------------------------------------------------------

// copy a file. Return false but do not attempt to delete the target file if
// something goes wrong, because it's possible that we can't write to someone
// else's file but we can delete it. Mustn't lose data in such a case. This
// means that we may leave partial files behind, that's up to the caller.
bool file_copy(
    const char          *opath,         // path of file to copy
    const char          *npath)         // path of target copy
{
    char                buf[4096];      // copy buffer
    File                src, tar;       // source and target file
    int                 n;              // number of bytes read/written

    std::string nopath(opath? opath : "");

    if (!src.open(nopath.c_str(), File::M_READ)) {      // open source file
        set_error(src.error());
        return false;
    }

    std::string nnpath(npath? npath : "");
    if (!tar.open(nnpath.c_str(), File::M_WRITE)) {     // open target file
        set_error(tar.error());
        src.close();
        return false;
    }

    do {                                                // copy loop:
        if ((n = src.read(buf, sizeof(buf))) < 0) {     // read source data
            set_error(src.error());
            src.close();
            tar.close();
            return false;
        }
        if (n != tar.write(buf, n)) {                   // write target data
            set_error(tar.error());
            src.close();
            tar.close();
            return false;
        }
    } while (n && !src.eof());

    if (!src.close()) {                                 // close source file
        set_error(src.error());
        tar.close();
        return false;
    }
    if (!tar.close()) {                                 // close target file
        set_error(tar.error());
        return false;
    }
    return true;
}

#else

//-----------------------------------------------------------------------------

// delete a file
bool file_remove(
    const char          *path)          // path of file to remove
{
    std::string npath(path? path : "");

    bool success = DeleteFile(npath.c_str()) != FALSE;
    if (!success)
        set_error(HAL::get_errno());
    return success;
}

//-----------------------------------------------------------------------------

// rename a file or directory
bool rename(
    const char          *opath,         // path of file to rename
    const char          *npath)         // new path; must be on the same disk
{
    std::string nopath(opath? opath : "");
    std::string nnpath(npath? npath : "");

    return MoveFile(nopath.c_str(), nnpath.c_str()) != FALSE;
}

//-----------------------------------------------------------------------------

// create a directory with a given set of permissions, default 0755
bool mkdir(
    const char          *path,          // path of directory to create
    int                 mode)           // rwxrwxrwx permissions
{
    std::string npath(path? path : "");
    // path has to be absolute!
    if (!is_path_absolute(npath))
        npath = HAL::Ospath::join(get_cwd(), npath);

    // normalize the path, SHCreateDirectoryEx() does not support slashes
    npath = HAL::Ospath::normpath(npath);

    // TODO: Translate mode flags to Win32 security descriptor
    HWND window_handle = 0;                             // I think we get along without one ;-)
    SECURITY_ATTRIBUTES* sec = 0;
    int result = SHCreateDirectoryEx(window_handle, npath.c_str(), sec);

#ifdef DEBUG
    char buf[2048];
    if (result != ERROR_SUCCESS) {
        // to get at least some kinda understanding what might had gone wrong
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, 0, result, 0, buf, 2048, 0);
    }
#endif
    return result == ERROR_SUCCESS;
}


//-----------------------------------------------------------------------------

// delete a directory. The directory must be empty.
bool rmdir(
    const char          *path)          // path of directory to delete
{
    std::string npath(path? path : "");

    return RemoveDirectory(npath.c_str()) != FALSE;
}


//-----------------------------------------------------------------------------

// set the current working directory
bool chdir(
    const char  *path)          // path of directory to make current
{
    if (path == NULL)
        return false;

    std::string npath(path);

    if (SetCurrentDirectory(npath.c_str())) {
        set_error(0);
        return true;
    } else {
        set_error(HAL::get_errno());
        return false;
    }
}


//-------------------------------------------------------------------------------------------------

// Get the current working directory
std::string get_cwd()
{
    // retrieve required buffer size first
    DWORD size = GetCurrentDirectory(0, 0); // return value contains terminating null char
    std::vector<char> buf(static_cast<size_t>(size), '\0');

    if (!GetCurrentDirectory(size, &buf[0])) {
        set_error(HAL::get_errno());
        return std::string();
    }
    set_error(0);
    return std::string(&buf[0]);
}


//-----------------------------------------------------------------------------

// given a path, return information about the file. Return true, if the
// operation succeeded or false, if not. In the latter case the file_stat
// structure is left unchanged.
bool stat(
    const char          *path,          // path to get information for
    Stat                *file_stat)     // store the results here
{
    struct _stat64      st;             // results of inode check

    // Be defensive and check for NULL since strlen will segfault
    // if handed a NULL string. Set m_error to ENOENT since that
    // is what stat would have returned.
    if (!path) {
        set_error(ENOENT);
        return false;
    }

    // stat() does not work for (real) directories with a trailing backslash
    std::string new_path = STRING::rstrip(path, "*/\\");
    // stat() does not work for drives without a trailing backslash, eg C:. Hence adding it again.
    // But only, and hence the first comparison, if something was stripped indeed.
    if (path != new_path && !new_path.empty() && new_path[new_path.size() - 1] == ':')
        new_path += '/';

    // Note: according to MSDN, _wstat fails on symlinks on Windows 7, while _stat works.
    std::wstring p(STRING::utf8_to_wchar(new_path.c_str()));
    if (!::_wstat64(p.c_str(), &st)) {
        set_error(0);
        file_stat->m_size               = st.st_size;
        file_stat->m_modification_time  = double(st.st_mtime);
        file_stat->m_access_time        = double(st.st_atime);
        file_stat->m_change_time        = double(st.st_ctime);
        file_stat->m_is_dir             = !!(st.st_mode & S_IFDIR);
        file_stat->m_is_file            = !!(st.st_mode & S_IFREG);
        file_stat->m_is_readable        = !!(st.st_mode & S_IREAD);
        file_stat->m_is_writable        = !!(st.st_mode & S_IWRITE);
        file_stat->m_is_executable      = !!(st.st_mode & S_IEXEC);
        return true;
    }
    else {
        set_error(HAL::get_errno());
        return false;
    }
    return false;
}

//-----------------------------------------------------------------------------

// return free space in bytes on the file system that <path> resides on. This
// can be any file or directory in the filesystem, including its mountpoint.
Sint64 freespace(
    const char          *path)          // path of a file/dir on disk to test
{
    Uint64 free_bytes_available_to_caller;
    Uint64 total_number_of_bytes;
    Uint64 total_number_of_free_bytes;

    if (!GetDiskFreeSpaceEx(path,
        (PULARGE_INTEGER) &free_bytes_available_to_caller,
        (PULARGE_INTEGER) &total_number_of_bytes,
        (PULARGE_INTEGER) &total_number_of_free_bytes))
    {
        ASSERT(M_DISK, !"DISK::freespace: GetDiskFreeSpaceEx() failed");
        return -1;
    }

    return Sint64(free_bytes_available_to_caller);
}


//-----------------------------------------------------------------------------

// copy a file. Return false but do not attempt to delete the target file if
// something goes wrong, because it's possible that we can't write to someone
// else's file but we can delete it. Mustn't lose data in such a case. This
// means that we may leave partial files behind, that's up to the caller.
bool file_copy(
    const char          *opath,         // path of file to copy
    const char          *npath)         // path of target copy
{
    return CopyFile(opath, npath, FALSE) != FALSE;
}


#endif

//-----------------------------------------------------------------------------

// check whether a named file is readable or writable
bool access(
    const char          *path,          // path of a file/dir
    bool                write)          // have permission to read or write?
{
    std::string npath(path? path : "");

#ifdef WIN_NT
    std::wstring p(STRING::utf8_to_wchar(npath.c_str()));
    if (!::_waccess(p.c_str(), write ? 0x02 : 0x00)) {
#else
    if (!::access(npath.c_str(), write ? W_OK : R_OK)) {
#endif
        set_error(0);
        if (LOG::mod_log)
            LOG::mod_log->debug(M_DISK, Mod_log::C_DISKTRACE,
                               "access \"%s\", mode %c, ok",
                               path, write ? 'w' : 'r');
        return true;
    }
    else {
        int error = 0;
        set_error(HAL::get_errno());
        if (LOG::mod_log)
            LOG::mod_log->debug(M_DISK, Mod_log::C_DISKTRACE,
                "access \"%s\", mode %c: %s",
                path, write ? 'w' : 'r',
                HAL::strerror(error).c_str());
        return false;
    }

    return true;
}


#ifdef WIN_NT
void invalid_parameter_handler(
    const wchar_t* expression,
    const wchar_t* function,
    const wchar_t* file,
    unsigned int line,
    uintptr_t pReserved)
{
    if (!LOG::mod_log)
        return;
    LOG::mod_log->error(M_DISK, Mod_log::C_DISKTRACE, 777,
        "Invalid parameter detected in function %s. File: %s Line: %d\n",
        STRING::wchar_to_utf8(function).c_str(), STRING::wchar_to_utf8(file).c_str(), line);
    LOG::mod_log->error(M_DISK, Mod_log::C_DISKTRACE, 777,
        "Expression: %s", STRING::wchar_to_utf8(expression).c_str());
}

struct Reset_guard
{
    Reset_guard(
        int old_report_mode,
        _invalid_parameter_handler old_handler)
      : m_orig_report_mode(old_report_mode), m_orig_handler(old_handler)
    {}
    ~Reset_guard() {
        _set_invalid_parameter_handler(m_orig_handler);
        _CrtSetReportMode(_CRT_ASSERT, m_orig_report_mode);
    }
private:
    int m_orig_report_mode;
    _invalid_parameter_handler m_orig_handler;
};

#endif

// Opens a file and returns a pointer to the opened file.
FILE* fopen(
    const char* path,
    const char* mode)
{
#ifdef WIN_NT
    std::wstring p(STRING::utf8_to_wchar(path));
    std::wstring m(STRING::utf8_to_wchar(mode));
    // set own error reporting handler and disable the message box for assertions.
    Reset_guard guard(
        _CrtSetReportMode(_CRT_ASSERT, 0),
        _set_invalid_parameter_handler(invalid_parameter_handler));
    return _wfopen(p.c_str(), m.c_str());
#else
    return ::fopen(path, mode);
#endif
}


std::string find_file_on_paths(
    const char* file_name,
    int paths_count,
    const char* const* search_paths)
{
    if (!file_name || file_name[0] == '\0')
        return std::string();

    // transform search_paths into string list such that we can use the already existing
    // find_file_on_paths() impl
    std::vector<std::string> dirs;
    for (int i=0; i<paths_count; ++i) {
        if (search_paths[i])
            dirs.push_back(search_paths[i]);
    }

    return find_file_on_paths(file_name, dirs);
}


std::string find_file_on_path(
    const char* file_name,
    const std::string& dir)
{
    if (!file_name || file_name[0] == '\0')
        return std::string();

    std::string fullpath;
    if (access(file_name))
        fullpath = file_name;
    if (fullpath.empty()) {
        // concatenate the current dir with file_name
        std::string file(HAL::Ospath::join(dir, file_name));
        if (access(file.c_str()))
            fullpath = file;
    }
    return fullpath;
}


std::string find_file_on_paths(
    const char* file_name,
    const std::vector<std::string>& dirs)
{
    return find_file_on_paths(file_name, dirs.begin(), dirs.end());
}

std::string convert_to_forward_slashes(
    const std::string &in)      // input
{
    std::string fs(in);
    size_t pos = 0;
    while(pos != std::string::npos) {
        pos = fs.find('\\', pos);
        if (pos != std::string::npos) {
            fs.replace(pos, 1, "/");
        }
    }
    return fs;
}

bool rmdir_r(const char* path)
{
    if (!is_directory(path))
        return false;

    Directory dir;
    if (!dir.open(path))
        return false;

    std::string next = dir.read();
    bool success = true;
    while (!next.empty())
    {
        if (is_directory(next.c_str()))
            success &= rmdir_r(next.c_str());
        else
            success &= file_remove(next.c_str());

        next = dir.read();
    }
    return success && rmdir(path);
}

}
}
