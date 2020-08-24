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
/// \brief Source for implementations of mi::neuraylib::IReader and mi::neuraylib::IWriter
///        backed by an instance of mi::neuraylib::IBuffer.

#include "pch.h"

#include "disk_memory_reader_writer_impl.h"
#include "disk_stream_position_impl.h"

#include <cstring>
#include <vector>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/istream_position.h>
#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace DISK {

/// An implementation of mi::neuraylib::IBuffer as used by Memory_writer_impl.
class Buffer_impl : public mi::base::Interface_implement<mi::neuraylib::IBuffer>
{
public:
    const mi::Uint8* get_data() const { return m_buffer.empty() ? 0 : &m_buffer[0]; }

    mi::Size get_data_size() const { return m_buffer.size(); }

    std::vector<mi::Uint8> m_buffer;
};

template <typename T, typename B>
Memory_reader_writer_base_impl<T,B>::~Memory_reader_writer_base_impl()
{
    // nothing to do
}

template <typename T, typename B>
mi::Sint32 Memory_reader_writer_base_impl<T,B>::get_error_number() const
{
    return 0;
}

template <typename T, typename B>
const char* Memory_reader_writer_base_impl<T,B>::get_error_message() const
{
    return 0;
}

template <typename T, typename B>
bool Memory_reader_writer_base_impl<T,B>::eof() const
{
    return m_position == m_buffer->get_data_size();
}

template <typename T, typename B>
mi::Sint32 Memory_reader_writer_base_impl<T,B>::get_file_descriptor() const
{
    return -1;
}

template <typename T, typename B>
bool Memory_reader_writer_base_impl<T,B>::supports_recorded_access() const
{
    return true;
}

template <typename T, typename B>
const mi::neuraylib::IStream_position* Memory_reader_writer_base_impl<T,B>::tell_position() const
{
    return new Stream_position_impl( m_position, true);
}

template <typename T, typename B>
bool Memory_reader_writer_base_impl<T,B>::seek_position(
    const mi::neuraylib::IStream_position* stream_position)
{
    if( !stream_position)
        return false;
    if( !stream_position->is_valid())
        return false;

    const Stream_position_impl* stream_position_impl
        = static_cast<const Stream_position_impl*>( stream_position);
    m_position = static_cast<mi::Size>( stream_position_impl->get_stream_position());
    return true;
}

template <typename T, typename B>
bool Memory_reader_writer_base_impl<T,B>::rewind()
{
    m_position = 0;
    return true;
}

template <typename T, typename B>
bool Memory_reader_writer_base_impl<T,B>::supports_absolute_access() const
{
    return true;
}

template <typename T, typename B>
mi::Sint64 Memory_reader_writer_base_impl<T,B>::tell_absolute() const
{
    return m_position;
}

template <>
bool Memory_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IReader>,
                                    const mi::neuraylib::IBuffer>::seek_absolute( mi::Sint64 pos)
{
    if( pos < 0 || static_cast<mi::Size>( pos) > m_buffer->get_data_size())
        return false;

    m_position = static_cast<mi::Size>( pos);
    return true;
}

template <>
bool Memory_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IWriter>,
                                    Buffer_impl>::seek_absolute( mi::Sint64 pos)
{
    if( pos < 0)
        return false;

    if( static_cast<mi::Size>( pos) > m_buffer->get_data_size())
        m_buffer->m_buffer.resize( static_cast<mi::Size>( pos), 0);
    m_position = static_cast<mi::Size>( pos);
    return true;
}

template <typename T, typename B>
mi::Sint64 Memory_reader_writer_base_impl<T,B>::get_file_size() const
{
    return m_buffer->get_data_size();
}

template <typename T, typename B>
bool Memory_reader_writer_base_impl<T,B>::seek_end()
{
    m_position = m_buffer->get_data_size();
    return true;
}

// explicit template instantiation for Memory_reader_writer_base_impl<T>
template class Memory_reader_writer_base_impl<
    mi::base::Interface_implement<mi::neuraylib::IReader>, const mi::neuraylib::IBuffer>;
template class Memory_reader_writer_base_impl<
    mi::base::Interface_implement<mi::neuraylib::IWriter>, Buffer_impl>;

Memory_reader_impl::Memory_reader_impl( const mi::neuraylib::IBuffer* buffer)
{
    ASSERT( M_DISK, buffer);

    m_buffer = mi::base::make_handle_dup( buffer);
    m_position = 0;
}

mi::Sint64 Memory_reader_impl::read( char* buffer, mi::Sint64 size)
{
    size = std::min( size, static_cast<mi::Sint64>( m_buffer->get_data_size() - m_position));
    if( size == 0)
        return 0;

    const mi::Uint8* data = m_buffer->get_data();
    memcpy( buffer, data + m_position, static_cast<mi::Size>( size));
    m_position += static_cast<mi::Size>( size);
    return size;
}

bool Memory_reader_impl::readline( char* buffer, mi::Sint32 size)
{
    if( size == 0)
        return false;

    const mi::Uint8* data = m_buffer->get_data();
    mi::Size data_size = m_buffer->get_data_size();
    mi::Size end = m_position;
    bool more_input       = end < data_size;
    bool more_output      = end-m_position < static_cast<mi::Size>( size-1);
    bool newline_not_seen = true; // not data[end] == '\n' !
    while( more_input && more_output && newline_not_seen) {
        newline_not_seen  = data[end] != '\n';
        ++end;
        more_input        = end < data_size;
        more_output       = end-m_position < static_cast<mi::Size>( size-1);
    }
    mi::Sint64 result = read( buffer, end-m_position);
    buffer[result] = '\0';
    return true;
}

bool Memory_reader_impl::supports_lookahead() const
{
    return false;
}

mi::Sint64 Memory_reader_impl::lookahead( mi::Sint64 size, const char** buffer) const
{
    return 0;
}

Memory_writer_impl::Memory_writer_impl()
{
    m_buffer = new Buffer_impl();
    m_position = 0;
}

mi::Sint64 Memory_writer_impl::write( const char* buffer, mi::Sint64 size)
{
    if( size == 0)
        return 0;
    if( m_position + size > m_buffer->get_data_size())
        m_buffer->m_buffer.resize( m_position + static_cast<mi::Size>( size));
    memcpy( &m_buffer->m_buffer[m_position], buffer, static_cast<mi::Size>( size));
    m_position += static_cast<mi::Size>( size);
    return size;
}

bool Memory_writer_impl::writeline( const char* str)
{
    write( str, strlen( str));
    return true;
}

bool Memory_writer_impl::flush()
{
    // nothing to do
    return true;
}

mi::neuraylib::IBuffer* Memory_writer_impl::get_buffer() const
{
    m_buffer->retain();
    return m_buffer.get();
}

} // namespace DISK

} // namespace MI
