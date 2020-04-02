/***************************************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Base interface common for readers and writers.

#ifndef MI_NEURAYLIB_IREADER_WRITER_BASE_H
#define MI_NEURAYLIB_IREADER_WRITER_BASE_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

class IStream_position;

/** \if MDL_SDK_API \addtogroup mi_neuray_mdl_sdk_misc
    \else \addtogroup mi_neuray_impexp
    \endif
@{
*/

/// Base interface for readers and writers.
///
/// Readers and writers are an abstraction of plain byte oriented data access to files or network
/// streamed data. They separate the data access from the data parsing and formatting.
///
/// Readers and writers can have certain capabilities. The capabilities are hierarchical, meaning
/// that if a capability is supported, all capabilities above it are supported as well. Possible
/// capabilities are:
///
/// - Streaming: The minimal capability is sequential unbuffered reading or writing of data,
///   suitable for streaming or stdin/stdout usage.
///
/// - Lookahead: Applicable to readers only, lookahead gives access to a flexible, but maybe limited
///   amount of data in the input stream that has not been read yet. It is usually a byproduct of
///   buffering or random access, where after reading of the lookahead the read position is reset to
///   the position before the lookahead. It can be used for examining a file for a magic header that
///   identifies the file format.
///
/// - Random recorded access: The reader or writer is capable of jumping to a position that has
///   previously been recorded. The recorded position uses an opaque representation of a position
///   in a stream.
///
/// - Random absolute access: The reader or writer is capable of jumping to an absolute position
///   indexed from the beginning of the stream.
///
/// The different capabilities are not expressed in a class hierarchy but in query functions and
/// well-defined behavior of all API functions for all capabilities.
///
/// Readers and writers are exposed as three interfaces to the importers and exporters:
/// #mi::neuraylib::IReader_writer_base (this interface), #mi::neuraylib::IReader and
/// #mi::neuraylib::IWriter, where the latter two derive from the first.
///
/// Readers and writers operate in binary mode only. They do not perform any automatic conversions,
/// for example, such as newline to CR/LF for text mode files on Windows. If you require such
/// conversions for a specific file format, you need to add the necessary control sequences yourself
/// and have your parser accept them correctly.
///
/// Read and write sizes, as well as seek and tell positions, are consistently #mi::Sint64 types. It
/// is a signed integer since some functions will use -1 as error indicator. Note: On 32 bit
/// machines, all #mi::Sint64 sizes are limited to 32 bits and thus file sizes to 4 GB.
///
/// The #mi::neuraylib::IReader_writer_base class deals with the common part between reader and
/// writer classes. This includes random access functions, access to an optionally available file
/// descriptor, and handling of error and end-of-file conditions. The file descriptor is provided to
/// support 3rd party libraries that require a file descriptor. It may not be available for all
/// readers/writers.
class IReader_writer_base :
    public base::Interface_declare<0x919370c2,0x2bb4,0x40db,0x81,0xff,0xd3,0x1c,0x52,0x10,0x54,0x64>
{
public:
    /// Returns the error number of the last error that happened in this reader or writer,
    /// or 0 if no error occurred.
    virtual Sint32 get_error_number() const = 0;

    /// Returns the error message of the last error that happened in this reader or writer.
    /// Returns \c NULL if #get_error_number() returns 0.
    virtual const char* get_error_message() const = 0;

    /// Returns \c true if the end of the file has been reached.
    /// The result is undefined before reading or writing for the first time.
    virtual bool eof() const = 0;

    /// Returns the file descriptor of the stream, or -1 if it is not available.
    virtual Sint32 get_file_descriptor() const = 0;

    /// \name Random recorded access
    //@{

    /// Returns \c true if random recorded access is supported, and \c false otherwise.
    virtual bool supports_recorded_access() const = 0;

    /// Returns the current position in this stream.
    virtual const IStream_position* tell_position() const = 0;

    /// Repositions the stream to the position \p stream_position.
    /// \return \c true in case of success, or \c false in case of errors, e.g., if
    ///         \p stream_position is not valid or recorded access is not supported and the state of
    ///         the stream remains unchanged.
    virtual bool seek_position( const IStream_position* stream_position) = 0;

    /// Resets the stream position to the beginning.
    /// \return \c true in case of success (and clears the error condition), and \c false in case
    /// of errors.
    virtual bool rewind() = 0;

    //@}
    /// \name Random absolute access
    //@{

    /// Returns \c true if random absolute access is supported, and \c false otherwise.
    virtual bool supports_absolute_access() const = 0;

    /// Returns the absolute position in bytes from the beginning of the stream beginning, or -1
    /// if absolute access is not supported.
    virtual Sint64 tell_absolute() const = 0;

    /// Repositions the stream to the absolute position \p pos.
    /// \return \c true in case of success, or \c false in case of errors, e.g., if \p pos is not
    ///         valid or absolute access is not supported and the state of the stream remains
    ///         unchanged.
    virtual bool seek_absolute(Sint64 pos) = 0;

    /// Returns the size in bytes of the data in the stream.
    /// Based on random access, this is a fast operation.
    virtual Sint64 get_file_size() const = 0;

    /// Sets the stream position to the end of the file.
    /// \return \c true in case of success, or \c false in case of errors.
    virtual bool seek_end() = 0;

    //@}
};

/*@}*/ // end group mi_neuray_impexp

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IREADER_WRITER_BASE_H
