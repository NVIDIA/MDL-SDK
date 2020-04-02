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
/// \brief Header for implementations of mi::neuraylib::IReader and mi::neuraylib::IWriter
///        backed by an instance of mi::neuraylib::IBuffer.

#ifndef BASE_HAL_DISK_DISK_FILE_READER_WRITER_BASE_IMPL_H
#define BASE_HAL_DISK_DISK_FILE_READER_WRITER_BASE_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ireader.h>
#include <mi/neuraylib/iwriter.h>

#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class IBuffer; } }

namespace MI {

namespace DISK {

class Buffer_impl;

/// Base class of Memory_reader_impl and Memory_writer_impl.
template <typename T, typename B>
class Memory_reader_writer_base_impl : public T, public boost::noncopyable
{
public:

    /// Destructor
    ~Memory_reader_writer_base_impl();

    // public API methods of IReader_writer_base

    /// Always returns 0 in this implementation.
    mi::Sint32 get_error_number() const;

    /// Always returns \c NULL in this implementation.
    const char* get_error_message() const;

    bool eof() const;

    /// Always returns -1 in this implementation.
    mi::Sint32 get_file_descriptor() const;

    /// Returns \c true in this implementation.
    bool supports_recorded_access() const;

    const mi::neuraylib::IStream_position* tell_position() const;

    bool seek_position( const mi::neuraylib::IStream_position* stream_position);

    /// Always succeeds in this implementation.
    bool rewind();

    /// Returns \c true in this implementation.
    bool supports_absolute_access() const;

    mi::Sint64 tell_absolute() const;

    bool seek_absolute( mi::Sint64 pos);

    mi::Sint64 get_file_size() const;

    /// Always succeeds in this implementation.
    bool seek_end();

protected:

    /// The internally used buffer.
    mi::base::Handle<B> m_buffer;

    /// Current position of the file pointer.
    mi::Size m_position;
};

/// This implementation of mi::neuraylib::IReader wraps mi::neuraylib::IBuffer.
class Memory_reader_impl
  : public Memory_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IReader>,
                                          const mi::neuraylib::IBuffer>
{
public:

    /// Constructor
    Memory_reader_impl( const mi::neuraylib::IBuffer* buffer);

    // public API methods

    mi::Sint64 read( char* buffer, mi::Sint64 size);

    bool readline( char* buffer, mi::Sint32 size);

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    bool supports_lookahead() const;

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    mi::Sint64 lookahead( mi::Sint64 size, const char** buffer) const;
};

/// This implementation of mi::neuraylib::IWriter wraps mi::neuraylib::IBuffer.
class Memory_writer_impl
  : public Memory_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IWriter>,
                                          Buffer_impl>
{
public:

    /// Constructor
    Memory_writer_impl();

    // public API methods

    mi::Sint64 write( const char* buffer, mi::Sint64 size);

    /// Always succeeds in this implementation.
    bool writeline( const char* str);

    /// Always succeeds in this implementation.
    bool flush();

    // internal methods

    /// Returns the internally used buffer.
    mi::neuraylib::IBuffer* get_buffer() const;
};

} // namespace DISK

} // namespace MI

#endif // BASE_HAL_DISK_DISK_FILE_READER_WRITER_BASE_IMPL_H
