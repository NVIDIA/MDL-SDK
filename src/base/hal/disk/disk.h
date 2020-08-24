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
/// \brief The definition of disk standalone helpers.

#ifndef BASE_HAL_DISK_H
#define BASE_HAL_DISK_H

#include "i_disk_ifile.h"
#include "i_disk_file.h"

#include <sys/types.h>

#include <base/system/main/types.h>
#include <base/hal/time/i_time.h>

#include <cstdarg>
#include <string>
#include <vector>

namespace MI {
namespace DISK {

// Wrapper for the Unix DIR - at least on SGI the sys/dir.h headers
// contains dangerous macros requiring a forward declaration, for example
//   #define seekdir BSDseekdir
// We'll also use the same-named structure on Windows, tho its contents
// are quite a bit different
struct Hal_dir;


/// Represents a single directory. It's like File, except that the only useful
/// operation is looping over the file names in the directory.
class Directory
{
  public:
    /// Constructor.
    Directory();

    /// Destructor.
    ~Directory();

    /// Open a directory for reading names in it
    /// \param path to open
    /// \return success
    bool open(
        const char* path);

    /// Close directory
    /// \return success
    bool close();

    /// Restart reading at the first file
    /// \return success
    bool rewind();                      // true=ok, false=fail (see error())

    /// Read the next filename from the directory. Names are unsorted.
    /// \param nodot ignore files beginning with '.'
    /// \return the next filename, or empty string if at eof
    std::string read(
        bool nodot=true);

    /// Retrieve whether reading has hit the end of the directory.
    /// \return true if reading has hit the end of the directory
    bool eof() const;

    /// Retrieve true if the dir has been successfully opened. This is useful for
    /// debugging code that loops over all open dirs, and prints the open ones.
    /// \return true if the dir is open
    bool is_open() const;

    /// Retrieve the last path given to this class using open(), even if the open
    /// failed or the directory was closed in the meantime. This is the final
    /// path from substitution that was used to access the file system. This is
    /// useful for error messages.
    /// \return the last path
    const char *path() const;

    /// Retrieve last system error code.
    /// \return last system error code
    int error() const;

  private:
    std::string m_path;                 ///< last path passed to open()
    int                 m_error;                ///< last error, 0 if none
    bool                m_eof;                  ///< hit EOF while reading?
#ifndef WIN_NT
    Hal_dir*            m_dp_wrapper;           ///< open directory, 0 if not open
#else
    Hal_dir*            m_dir;                  ///< information for windows-based dir searching

    /// An internal, windows-specific helper method to encapsulate
    /// the directory reading code.
    /// \return success
    bool read_next_file();

    /// Retrieve whether a given path exists.
    /// For now this is simply a (static) helper method, but might be a useful
    /// method to expose publicly and for all platforms
    /// \param path path in question
    /// \return true, if path exists
    static bool exists(
        const char* path);

#endif
};


/// this is used to hold information gathered by the stat method
struct Stat
{
    Sint64      m_size;                 ///< size in bytes of the file
    TIME::Time  m_modification_time;    ///< time of last data modification
    TIME::Time  m_access_time;          ///< time of last access
    TIME::Time  m_change_time;          ///< time of last file status change
    bool        m_is_dir;               ///< is the file a directory?
    bool        m_is_file;              ///< is the file a regular file or a symlink?
    bool        m_is_writable;          ///< is the file writable?
    bool        m_is_readable;          ///< is the file readable?
    bool        m_is_executable;        ///< is the file executable?
};


/// Retrieve whether a path is absolute, ie. begins with / or X:/.
/// \param path the path
/// \return true if a path is absolute
bool is_path_absolute(
    const char* path);
/// Retrieve whether a path is absolute, ie. begins with / or X:/.
/// \param path the path
/// \return true if a path is absolute
bool is_path_absolute(
    const std::string& path);

/// Retrieve whether path is a directory.
/// \param path the path
/// \return true if path is a directory
bool is_directory(
    const char* path);

/// Retrieve whether path is a regular file or a symlink.
/// \param path the path
/// \return true if path is a regular file or a symlink
bool is_file(
    const char* path);

/// check whether a named file is readable or writable
/// \param path the path
/// \return success
bool access(
    const char* path,
    bool        write = false);         // have permission to read or write?

/// Opens a file and returns a pointer to the opened file. This is a wrapper around the C lib
/// function. Under Windows, it handles UTF-8 strings and validates its parameters. In addition
/// to that it disables the message box for assertions.
/// While we should strive for using the DISK::File class our code makes use of fopen frequently.
FILE* fopen(
    const char* path,
    const char* mode);

/// Delete a file.
/// \param path the path
/// \return success
bool file_remove(
    const char* path);

/// Rename a file or directory.
/// \param opath path of file to rename
/// \param npath new path; must be on the same disk
/// \return success
bool rename(
    const char* opath,
    const char* npath);

/// create a directory
/// \param path the path
/// \param mode rwxrwxrwx permissions
/// \return success
bool mkdir(
    const char* path,
    int mode=0755);

/// delete a directory. The directory must be empty.
/// \param path the path
/// \return success
bool rmdir(
    const char* path);

bool rmdir_r(
    const char* path);

/// Set the current working directory.
/// \param path the path
/// \return success
bool chdir(
    const char* path);

/// Retrieve the current working directory.
/// \return the current working directory or the empty string else
std::string get_cwd();

/// Given a path, return information about the file. Return true, if the
/// operation succeeded or false, if not. In the latter case the file_stat
/// structure is left unchanged.
/// \param[out] file_stat store the results here
/// \return success
bool stat(
    const char* path,
    Stat* file_stat);

/// Retrieve free space in bytes on the file system that <path> resides on.
/// \param path the path
/// \return free space in bytes, -1 if error
Sint64 freespace(
    const char* path);

/// Copy a file.
/// \param opath path of file to copy
/// \param npath path of target copy
/// \return success
bool file_copy(
    const char* opath,
    const char* npath);

/// Convert given path into a path using only forward slashes.
/// \param path incoming path
/// \return converted path
std::string convert_to_forward_slashes(
    const std::string& path);

/// Enable disktrace verbosity.
/// \note This implementation is based on a (global) static variable!
void set_disktrace_enabled();

/// Find a file on a given set of search paths.
//@{
/// Find a file either by its own name only or on the given directory \p dir.
/// \param filename name of the file
/// \param dir the given directory
/// \return absolute path to the file when found, empty string else
std::string find_file_on_path(
    const char* file_name,
    const std::string& dir=std::string());

/// Find a file by name and given set of search paths.
/// \param filename name of the file
/// \param search_paths the list of search paths
/// \return absolute path to the file when found, empty string else
std::string find_file_on_paths(
    const char* file_name,
    const std::vector<std::string>& search_paths);

/// Find a file by name and given set of search paths.
/// \param filename name of the file
/// \param begin begin of the list of search paths
/// \param end end of the list of search paths
/// \return absolute path to the file when found, empty string else
template <typename InputIterator>
std::string find_file_on_paths(
    const char* file_name,
    InputIterator begin,
    InputIterator end);

/// Find a file by name and given set of search paths. This function is for the sake of
/// convenience only when using some of the interface functions of the path module or
/// the IMill, eg \c const char* const* IMill::get_shader_path(int*).
/// \param filename name of the file
/// \param paths_count number of passed-in paths
/// \param search_paths the list of search paths
/// \return absolute path to the file when found, empty string else
std::string find_file_on_paths(
    const char* filename,
    int paths_count,
    const char* const* search_paths);
//@}

}
}

#include "disk_inline.h"                // the really trivial bits

#endif
