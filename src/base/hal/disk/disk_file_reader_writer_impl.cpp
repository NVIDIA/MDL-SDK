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
///        backed by DISK::File.

#include "pch.h"

#include "disk_file_reader_writer_impl.h"
#include "disk_stream_position_impl.h"

#include <cstring>
#include <base/hal/hal/hal.h>

namespace MI {

namespace DISK {

template <typename T>
File_reader_writer_base_impl<T>::~File_reader_writer_base_impl()
{
    if( m_file.is_open())
        m_file.close();
}

template <typename T>
mi::Sint32 File_reader_writer_base_impl<T>::get_error_number() const
{
    return m_file.error();
}

template <typename T>
const char* File_reader_writer_base_impl<T>::get_error_message() const
{
    m_error_message = HAL::strerror( m_file.error());
    return m_error_message.c_str();
}

template <typename T>
bool File_reader_writer_base_impl<T>::eof() const
{
    return m_file.eof();
}

template <typename T>
mi::Sint32 File_reader_writer_base_impl<T>::get_file_descriptor() const
{
    return m_file.get_file_descriptor();
}

template <typename T>
bool File_reader_writer_base_impl<T>::supports_recorded_access() const
{
    return true;
}

template <typename T>
const mi::neuraylib::IStream_position* File_reader_writer_base_impl<T>::tell_position() const
{
    return new Stream_position_impl( m_file.tell(), m_file.tell() >= 0);
}

template <typename T>
bool File_reader_writer_base_impl<T>::seek_position(
    const mi::neuraylib::IStream_position* stream_position)
{
    if( !stream_position)
        return false;
    if( !stream_position->is_valid())
        return false;

    const Stream_position_impl* stream_position_impl
        = static_cast<const Stream_position_impl*>( stream_position);
    return m_file.seek( stream_position_impl->get_stream_position());
}

template <typename T>
bool File_reader_writer_base_impl<T>::rewind()
{
    return m_file.seek( 0);
}

template <typename T>
bool File_reader_writer_base_impl<T>::supports_absolute_access() const
{
    return true;
}

template <typename T>
mi::Sint64 File_reader_writer_base_impl<T>::tell_absolute() const
{
    return m_file.tell();
}

template <typename T>
bool File_reader_writer_base_impl<T>::seek_absolute( mi::Sint64 pos)
{
    return m_file.seek( pos);
}

template <typename T>
mi::Sint64 File_reader_writer_base_impl<T>::get_file_size() const
{
    return m_file.filesize();
}

template <typename T>
bool File_reader_writer_base_impl<T>::seek_end()
{
   return m_file.seek( 0, 2);
}

// explicit template instantiation for File_reader_writer_base_impl<T>
template class File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IReader> >;
template class File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IWriter> >;

mi::Sint64 File_reader_impl::read( char* buffer, mi::Sint64 size)
{
    return m_file.read( buffer, size);
}

bool File_reader_impl::readline( char* buffer, mi::Sint32 size)
{
    return m_file.readline( buffer, size);
}

bool File_reader_impl::supports_lookahead() const
{
    return false;
}

mi::Sint64 File_reader_impl::lookahead( mi::Sint64 size, const char** buffer) const
{
    return 0;
}

bool File_reader_impl::open( const char* path)
{
    if( !path)
        return false;

    return DISK::is_file( path) && m_file.open( path, DISK::IFile::M_READ);
}

const char* File_reader_impl::get_path()
{
    return m_file.path();
}

bool File_reader_impl::close()
{
    return m_file.close();
}

mi::Sint64 File_writer_impl::write( const char* buffer, mi::Sint64 size)
{
    return m_file.write( buffer, size);
}

bool File_writer_impl::writeline( const char* str)
{
    return m_file.writeline( str);
}

bool File_writer_impl::flush()
{
    return m_file.flush();
}

bool File_writer_impl::open( const char* path)
{
    if( !path)
        return false;

    return m_file.open( path, DISK::IFile::M_WRITE);
}

const char* File_writer_impl::get_path()
{
    return m_file.path();
}

bool File_writer_impl::close()
{
    return m_file.close();
}

} // namespace DISK

} // namespace MI
