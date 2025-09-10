/***************************************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Header for implementations of mi::neuraylib::IReader and mi::neuraylib::IWriter
///        backed by an instance of FILE*.

#ifndef BASE_HAL_DISKDISK_FILE_READER_WRITER_BASE_IMPL_H
#define BASE_HAL_DISKDISK_FILE_READER_WRITER_BASE_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iwriter.h>

#include <boost/core/noncopyable.hpp>

#include <cstdio>
#include <string>

namespace mi { namespace neuraylib { class IReader; class IStream_position; class IWriter; } }

namespace MI {

namespace DISK {

/// Creates a random-access reader for the given file path.
///
/// \param path   The path to be handled by the reader.
/// \return       A reader for \p path, or \c nullptr in case of errors, e.g., the file path is
///               invalid, or the file path is valid, but the file denoted by the file path could
///               could not be opened for reading
mi::neuraylib::IReader* create_reader( const char* path);

/// Creates random-access writer for the given file path.
///
/// \param path    The path to be handled by the writer.
/// \return        A writer for \p path, or \c nullptr in case of errors, e.g., the file path is
///                invalid, or the file path is valid, but the file denoted by the file path could
///                not be opened for writing
mi::neuraylib::IWriter* create_writer( const char* path);

/// Base class of File_reader_impl and File_writer_impl.
template <typename T>
class File_reader_writer_base_impl : public T, public boost::noncopyable
{
public:

    /// Destructor
    ///
    /// Closes the file if it is still open.
    ~File_reader_writer_base_impl();

    // public API methods

    mi::Sint32 get_error_number() const;

    const char* get_error_message() const;

    bool eof() const;

    mi::Sint32 get_file_descriptor() const;

    /// Returns \c true in this implementation.
    bool supports_recorded_access() const;

    const mi::neuraylib::IStream_position* tell_position() const;

    bool seek_position( const mi::neuraylib::IStream_position* stream_position);

    bool rewind();

    /// Returns \c true in this implementation.
    bool supports_absolute_access() const;

    mi::Sint64 tell_absolute() const;

    bool seek_absolute( mi::Sint64 pos);

    mi::Sint64 get_file_size() const;

    bool seek_end();

    // internal methods

    /// Opens the file.
    ///
    /// \param path          The file to open.
    /// \param for_reading   \c true for readers, \c false for writers
    /// \return              \c true in case of success, \c false otherwise.
    bool open( const char* path, bool for_reading);

    /// Returns the path of the file (or empty string if not available).
    const char* get_path();

    // Closes the file.
    /// \return      \c true on success, \c false on failure
    bool close();

    /// Repositions the file position indicator.
    ///
    /// \param pos      The new position.
    /// \param whence   Supports the same values as fseek():
    ///                 -  0: \p pos is relative to the start of the file.
    ///                 -  1: \p pos is relative to the current position.
    ///                 -  2: \p pos is relative to the end of the file.
    /// \return         \c true in case of success, \c false otherwise.
    bool seek_absolute_internal( mi::Sint64 pos, int whence);

protected:

    /// Last error code (or 0 if none).
    mutable int m_error = 0;
    /// File pointer (or nullptr if not open).
    FILE* m_fp = nullptr;

private:

    /// File path (or empty string if not open).
    std::string m_path;
    /// Caches the error message.
    mutable std::string m_error_message;
};

/// This implementation of mi::neuraylib::IReader wraps FILE*.
class File_reader_impl
  : public File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IReader> >
{
public:

    using Base
        = File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IReader> >;

    // public API methods

    mi::Sint64 read( char* buffer, mi::Sint64 size);

    bool readline( char* buffer, mi::Sint32 size);

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    bool supports_lookahead() const;

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    mi::Sint64 lookahead( mi::Sint64 size, const char** buffer) const;

    // internal methods

    /// Opens the file.
    ///
    /// \param path  The file to open.
    /// \return      \c true on success, \c false on failure
    bool open( const char* path);
};

/// This implementation of mi::neuraylib::IWriter wraps FILE*.
class File_writer_impl
  : public File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IWriter> >
{
public:

    using Base
        = File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IWriter> >;

    // public API methods

    mi::Sint64 write( const char* buffer, mi::Sint64 size);

    bool writeline( const char* str);

    bool flush();

    // internal methods

    /// Opens the file.
    ///
    /// \param path  The file to open.
    /// \return      \c true on success, \c false on failure
    bool open( const char* path);
};

} // namespace DISK

} // namespace MI

#endif // BASE_HAL_DISKDISK_FILE_READER_WRITER_BASE_IMPL_H
