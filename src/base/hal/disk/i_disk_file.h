/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief The implementation for a plain file.

#ifndef BASE_HAL_DISK_I_DISK_FILE_H
#define BASE_HAL_DISK_I_DISK_FILE_H

#include "i_disk_ifile.h"

namespace MI {
namespace DISK {

/// Represents a single file. All the usual file operations are here. There is
/// one instance per open file. The open method is not part of the constructor
/// because we need to return a failure code. Since files and directories are
/// used a whole lot more than Mod_disk, the extra complication of going through
/// a Mod_disk factory and a subclass was avoided.
class File : public IFile
{
  public:
    /// Constructor.
    File();
    /// Destructor.
    virtual ~File();

    /// Open a file by path. To open stdin/stdout, set path to 0 and pass the
    /// corresponding mode, ie M_READ for stdin and M_WRITE for stdout.
    /// \param path (already substituted) path to file
    /// \param mode opening mode
    /// \return true=ok, false=fail (see error())
    virtual bool open(
        const char* path,
        IFile::Mode mode = M_READ);
    /// Open a file by path. To open stdin/stdout, set path to empty string and pass the
    /// corresponding mode, ie M_READ for stdin and M_WRITE for stdout.
    /// \param path (already substituted) path to file
    /// \param mode opening mode
    /// \return true=ok, false=fail (see error())
    virtual bool open(
        const std::string& path,
        IFile::Mode mode = M_READ);

    /// Close the file.
    /// \return success, ie false if internal close failed.
    virtual bool close();

    /// Read raw bytes from an open file.
    /// 64 bit platform:
    ///    'num' has 63 significant bits, it's an error to use the MSB
    /// 32 bit platform:
    ///    'num' has 32 significant bits, it's an error to use more
    /// (we use Uint64 as convenience, it's always compatible to size_t)
    /// \param buf copy data here
    /// \param num copy this many bytes (63 valid bits)
    /// \return # chars read, -1 = error
    virtual Sint64 read(
        char* buf,
        Uint64 num);

    /// Read a line ending with a newline or EOF from the open file. Leave
    /// newlines in the returned buffer.
    /// \param line copy data here
    /// \param num copy this many bytes or until \n\0
    /// \return true=ok, false=fail (see error())
    virtual bool readline(
        char* line,
        int num);
    /// Read a line ending with an EOL or EOF from the open file. Note that the line will be
    /// stripped of whitespaces, if not told otherwise.
    /// \param line copy data here
    /// \param strip strip whitespaces?
    /// \return true=ok, false=fail (see error())
    virtual bool read_line(
        std::string& line,
        bool strip=true);

    /// Write raw data to a file
    /// 64 bit platform:
    ///    'num' has 63 significant bits, it's an error to use the MSB
    /// 32 bit platform:
    ///    'num' has 32 significant bits, it's an error to use more
    /// (we use Uint64 as convenience, it's always compatible to size_t)
    /// \param buf write this buffer
    /// \param num write this many bytes(63 valid bits)
    /// \return # chars written, -1 = error
    virtual Sint64 write(
        const char* buf,
        Uint64 num);

    /// Write a line to open file. Any newlines must be in <line>, writeline
    /// doesn't add one.
    /// \param line write this string
    /// \return true=ok, false=fail (see error())
    virtual bool writeline(
        const char* line);

    /// Write a formatted string to an open file.
    /// \param fmt format string with %X
    virtual int printf(
        const char* fmt,
        ...) PRINTFLIKE2;

    /// Write a formatted string to an open file.
    /// \param fmt format string with %X
    /// \param args arguments for %X
    virtual int vprintf(
        const char* fmt,
        va_list args);

    /// Flush buffered output data to the file.
    /// \return success
    virtual bool flush();

    /// Retrieve whether we are at the end of the file. Unknown before reading or
    /// writing for the first time.
    /// \return true if we are at the end of the file
    virtual bool eof() const;

    /// Seek the file to a byte offset. If whence is 0, the offset is counted
    /// from the beginning of the file; if whence is 1, it is counted from the
    /// current position in the file; and if whence is 2, it is counted from
    /// EOF. Not all systems support files > 2 GB.
    /// \param offset seek to this byte offset
    /// \param whence 0=absolute, 1=relative, 2=rel to eof
    /// \return success
    virtual bool seek(
        Sint64 offset,
        int whence=0);

    /// Retrieve the current absolute position in the file where the next byte
    /// would be read or written. This is 0 after opening except in append mode.
    /// \return the current absolute position
    virtual Sint64 tell();

    /// Retrieve the size of the file. This is implemented by seeking to the end,
    /// retrieving the position there, and seeking back. It's fast.
    /// \return the size of the file
    virtual Sint64 filesize();

    /// Retrieve true if the file has been successfully opened. This is useful for
    /// debugging code that loops over all open files, and prints the open ones.
    /// \return true, if successfully open
    virtual bool is_open() const;

    /// Retrieve whether this File represents a file or stdinput/stdoutput.
    /// \return whether it is a file
    virtual bool is_file() const;

    /// Retrieve the last path given to this class using open(), even if the open
    /// failed or the file was closed in the meantime. This is the final path
    /// from substitution that was used to access the file system. This is
    /// useful for error messages
    /// \return the last path
    virtual const char* path() const;

    /// Retrieve the last system error code.
    /// \return last system error code
    virtual int error() const;

    /// Retrieve the system's file pointer. Don't use this except when third party
    /// code (e.g. jpeg) forces it. File provides all functions to access files,
    /// which should be used in all other cases.
    /// \return the system's file pointer
    virtual FILE* get_file_pointer() const;

    /// Retrieve the system's file desciptor. Don't use this except when third
    /// party code (e.g. tifflib) forces it. File provides all functions to
    /// access files, which should be used in all other cases.
    /// \return the system's file desciptor
    virtual int get_file_descriptor() const;

  private:
    std::string m_path;                 ///< last path passed to open()
    int                 m_error;                ///< last error, 0 if none
    FILE*               m_fp;                   ///< open file, 0 if not open
    IFile::Mode         m_mode;                 ///< current opend mode

    File(const File&);
    File& operator=(const File&);
};

}
}

#endif
