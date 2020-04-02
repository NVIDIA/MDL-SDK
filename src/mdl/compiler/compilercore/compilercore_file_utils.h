/******************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_FILE_UTILS_H
#define MDL_COMPILERCORE_FILE_UTILS_H 1

#include <cstdio>

#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// Return the OS-dependent path separator. This is the character separating the single
/// directories from each other. E.g.  "\\" or "/".
inline char os_separator()
{
#ifdef MI_PLATFORM_WINDOWS
    return '\\';
#else
    return '/';
#endif
}

/// Join two paths using the OS specific separator.
///
/// \param path1  incoming path
/// \param path2  incoming path
///
/// \return the resulting string; if path1 or path2 is empty returns the other component, else
///         both component connected by OS-specific path separator
string join_path(
    string const &path1,
    string const &path2);

/// Check if a file name matches a file mask in UTF-8 encoding.
///
/// \param file_mask  the file mask
/// \param file_name  the file name
///
/// \note supports only [0-9], [0-9]+, and -? regex so far
bool utf8_match(
    char const *file_mask,
    char const *file_name);

/// Opens a file and returns a pointer to the opened file.
///
/// \param alloc  an allocator
/// \param path   an UTF8 encoded path
/// \param mode   open mode
FILE *fopen_utf8(
    IAllocator *alloc,
    char const *path,
    char const *mode);

/// Check if the given file name (UTF8 encoded) names a file on the file system.
///
/// \param alloc  an allocator
/// \param fname  an UTF8 encoded file name
bool is_file_utf8(
    IAllocator *alloc,
    char const *fname);

/// Check if in the given directory a file matching the given mask exists.
///
/// \param alloc      an allocator
/// \param directory  an UTF8 encoded directory name
/// \param mask       an UTF8 encoded file mask
bool has_file_utf8(
    IAllocator *alloc,
    char const *directory,
    char const *mask);

/// Check if the given name (UTF8 encoded) names a directory on the file system.
///
/// \param alloc  an allocator
/// \param path   an UTF8 encoded file path
bool is_directory_utf8(
    IAllocator *alloc,
    char const *path);

/// Creates a directory on the file system.
///
/// \param alloc  an allocator
/// \param path   an UTF8 encoded file path
bool mkdir_utf8(
    IAllocator *alloc,
    char const *path);

/// Retrieve the current working directory.
///
/// \param alloc  an allocator
///
/// \return the current working directory or the empty string else
string get_cwd(IAllocator *alloc);

/// Return true if a path is absolute, i.e. begins with / or X:/
///
/// \param path   check this path
bool is_path_absolute(
    char const *path);

/// Simplifies a file path by removing directory names "." and pairs of directory names like
/// ("foo", ".."). Slashes are used as separators. Leading and trailing slashes in the input are
/// preserved.
///
/// \param file_path  the file path to simplify
/// \param sep        the separator (single character)
/// The input must be valid w.r.t. to the number of directory names "..".
string simplify_path(
    IAllocator   *alloc,
    string const &file_path,
    char         sep);

/// Simplifies a file path by removing directory names "." and pairs of directory names like
/// ("foo", ".."). Slashes are used as separators. Leading and trailing slashes in the input are
/// preserved.
///
/// \param file_path  the file path to simplify
/// \param sep        the separator (as string, possibly multiple characters like '::')
/// The input must be valid w.r.t. to the number of directory names "..".
string simplify_path(
    IAllocator   *alloc,
    string const &file_path,
    string const &sep);

/// Converts OS-specific directory separators into slashes.
///
/// \param s  the string to convert
string convert_os_separators_to_slashes(string const &s);

/// Converts slashes into OS-specific directory separators.
///
/// \param s  the string to convert
string convert_slashes_to_os_separators(string const &s);

/// Represents a single directory in a OS independent way.
class Directory
{
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Directory(IAllocator *alloc);

    /// Destructor.
    ~Directory();

    /// Open a directory for reading names in it.
    ///
    /// \param utf8_path    UFT8 encoded path to open
    /// \param utf8_filter  an optional filter for the entries to be read, will be ignored
    ///                    if not supported
    ///
    /// \return true on success
    bool open(
        char const *utf8_path,
        char const *utf8_filter = NULL);

    /// Close directory.
    /// \return success
    bool close();

    /// Restart reading at the first file
    /// \return success
    bool rewind();      // true=ok, false=fail (see error())

    /// Read the next filename from the directory. Names are unsorted.
    /// \return the UTF8-encoded next filename, or NULL if at eof
    char const *read();

    /// Retrieve whether reading has hit the end of the directory.
    /// \return true if reading has hit the end of the directory
    bool eof() const { return m_eof; }

    /// Retrieve last system error code.
    /// \return last system error code
    int error() const { return m_error; }

    /// Get the current directory.
    string const &get_curr_dir() const { return m_opath; }

private:
#ifdef MI_PLATFORM_WINDOWS
    /// Windows-specific helper method to encapsulate the directory reading code.
    /// \return success
    bool read_next_file();
#endif

private:
    struct Hal_dir;

    IAllocator *m_alloc;       ///< The allocator.
    Hal_dir    *m_dir;         ///< information for OS-specific dir searching
    string     m_path;         ///< path used to OS dependend opendir() call
    string     m_opath;        ///< last path passed to open()
    string     m_tmp;          ///< temporary result buffer
    int        m_error;        ///< last error, 0 if none
    bool       m_eof;          ///< hit EOF while reading?
};


}  // mdl
}  // mi

#endif
