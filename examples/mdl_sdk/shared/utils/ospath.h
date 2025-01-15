/***************************************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The definition of useful path' utilities.


#ifndef EXAMPLE_SHARED_UTILS_OSPATH_H
#define EXAMPLE_SHARED_UTILS_OSPATH_H

#include <string>

namespace mi { namespace examples { namespace ospath {

/// This class implements some useful functions on pathnames.
/// Many of those are close copies of their Python equivalents.
class Ospath
{
  public:
    /// Return the base name of pathname path. This is the second half of the
    /// pair returned by split(path). Note that the result of this function is
    /// different from the Unix basename program; where basename for '/foo/bar/'
    /// returns 'bar', the basename() function returns an empty string ('').
    /// \param path incoming path
    /// \return the resulting string
    static std::string basename(
        const std::string& path);

    /// Return the directory name of pathname path. This is the first half of
    /// the pair returned by split(path).
    /// \param path incoming path
    /// \return the resulting string
    static std::string dirname(
        const std::string& path);

    /// Joins one or more path components intelligently. If any component is an
    /// absolute path, all previous components are thrown away, and joining
    /// continues. The return value is the concatenation of path1, and
    /// optionally path2, etc., with exactly one directory separator (os.sep)
    /// inserted between components, unless path2 is empty.
    /// \param path1 incoming path
    /// \param path2 incoming path
    /// \return the resulting string
    static std::string join(
        const std::string& path1,
        const std::string& path2);
#if 0
    /// Normalize a pathname. This collapses redundant separators and up-level
    /// references, e.g. A//B, A/./B and A/foo/../B all become A/B. It does not
    /// normalize the case. On Windows, it converts forward slashes to backward slashes.
    /// Note that paths like . and A/.. are normalized to the empty string.
    /// \param path incoming path
    /// \return the resulting string
    static std::string normpath(
        const std::string& path);
#endif
    /// Normalize a pathname. It does not normalize the case. On Windows, it converts forward
    /// slashes to backward slashes.
    /// Note that paths like . and A/.. are normalized to the empty string.
    /// \param path incoming path
    /// \return the resulting string
    static std::string normpath_only(
        const std::string& path);

    /// Normalizes a path name.
    ///
    /// The function collapses redundant directory separators and folds current and parent directory
    /// references as possible. For example, "a//b", "a/./b", and "a/c/../b" are converted into
    /// "a/b". Paths like "a/.." are converted into ".".
    ///
    /// Differences to #normpath():
    /// - Conversion of "\" to "/" on Windows.
    /// - Different representation of current directory ("." vs "").
    /// - Different handling of unresolved parent directory references ("a/../.." => ".." vs
    ///   "a/../.."). Questionable for absolute paths but quite useful for relative paths.
    /// - Path separators at the end are not retained.
    static std::string normpath_v2(const std::string& path);

    /// Split the pathname path into a pair, (head, tail) where tail is the
    /// last pathname component and head is everything leading up to that. The
    /// tail part will never contain a slash; if path ends in a slash, tail
    /// will be empty. If there is no slash in path, head will be empty. If
    /// path is empty, both head and tail are empty. Trailing slashes are
    /// stripped from head unless it is the root (one or more slashes only).
    /// In nearly all cases, join(head, tail) equals path (the only exception
    /// being when there were multiple slashes separating head from tail).
    /// \param path incoming path
    /// \param[out] head the head part
    /// \param[out] tail the rest
    static void split(
        const std::string& path,
        std::string& head,
        std::string& tail);
    /// Split the pathname path into a pair, (head, tail) where tail is the
    /// last pathname component and head is everything leading up to that.
    /// This version leaves both parts untouched and does only the splitting.
    /// \param path incoming path
    /// \param[out] head the head part
    /// \param[out] tail the rest
    static void split_only(
        const std::string& path,
        std::string& head,
        std::string& tail);

    /// Split the pathname path into a pair (drive, tail) where drive is either
    /// a drive specification or the empty string. On systems which do not use
    /// drive specifications, drive will always be the empty string. In all
    /// cases, drive + tail will be the same as path.
    /// \param path incoming path
    /// \param[out] drive the drive part
    /// \param[out] tail the rest
    static void splitdrive(
        const std::string& path,
        std::string& drive,
        std::string& tail);

    /// Split the pathname path into a pair (root, ext) such that root + ext == path, and \p ext
    /// is empty or begins with a period and contains at most one period.
    /// \param path incoming path
    /// \param[out] root the root part
    /// \param[out] ext the extension
    static void splitext(
        const std::string& path,
        std::string& root,
        std::string& ext);

    /// Retrieve the extension from a given \p path such that the returned result is empty or begins
    /// with a period.
    /// \param path incoming path
    /// \return the extension string, or the empty string else
    static std::string get_ext(
        const std::string& path);

    /// Return the OS-dependent path separator. This is the character separating the single
    /// directories from each other. E.g.  "\\" or "/".
    static std::string sep();

    /// Return the OS independent path set separators. This is a string containing all characters
    /// which can be used to separate single paths from each other and might contain more than
    /// one character.On Linux and MacOSX, path set is separated by ":" and/or ";", e.g.,
    /// "/home/user0:/home/user0", on Windows "C:\home\user0;C:\home\user0"
    static std::string get_path_set_separator();

    /// Convert all separators to forward slashes.
    /// \param path incoming path
    /// \return the resulting string
    static std::string convert_to_forward_slashes(
        const std::string& path);
    /// Convert all separators to backward slashes.
    /// \param path incoming path
    /// \return the resulting string
    static std::string convert_to_backward_slashes(
        const std::string& path);
};

/// Find out whether \p path is absolute or not
/// \return true if \p path is absolute
bool is_path_absolute(
    const std::string& path);
/// Check whether the given path \p path exists or not.
bool find_path(
    const std::string& path);
/// Find out whether \p path is a directory or not
/// \return true if \p path is a directory
bool is_directory(
    const std::string& path);
/// Find out whether \p path is a file or not
/// \return true if \p path is a file
bool is_file(
    const std::string& path);

/// Returns the current working directory.
std::string get_cwd();


/// Makes the given \p abs_path relative to \p abs_base_dir. Returns true
/// if \p abs_path can be expressed relative to \p abs_base_dir, false
/// otherwise. In the later case \p rel_path is set to \p abs_path.
bool make_path_relative(
    const std::string& abs_base_dir,
    const std::string& abs_path,
    std::string& rel_path);

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

}}}

#endif
