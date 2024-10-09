/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "disk_file_reader_writer_impl.h"
#include "disk_stream_position_impl.h"

#include <cstring>
#include <filesystem>

#include <base/system/main/i_assert.h>
#include <base/hal/hal/hal.h>

namespace fs = std::filesystem;

namespace MI {

namespace DISK {

template <typename T>
File_reader_writer_base_impl<T>::~File_reader_writer_base_impl()
{
    close();
}

template <typename T>
mi::Sint32 File_reader_writer_base_impl<T>::get_error_number() const
{
    return m_error;
}

template <typename T>
const char* File_reader_writer_base_impl<T>::get_error_message() const
{
    m_error_message = HAL::strerror( m_error);
    return m_error_message.c_str();
}

template <typename T>
bool File_reader_writer_base_impl<T>::eof() const
{
    MI_ASSERT( m_fp);
    return feof( m_fp) ? true : false;
}

template <typename T>
mi::Sint32 File_reader_writer_base_impl<T>::get_file_descriptor() const
{
    MI_ASSERT( m_fp);
    return fileno( m_fp);
}

template <typename T>
bool File_reader_writer_base_impl<T>::supports_recorded_access() const
{
    return true;
}

template <typename T>
const mi::neuraylib::IStream_position* File_reader_writer_base_impl<T>::tell_position() const
{
    mi::Sint64 pos = tell_absolute();
    return new Stream_position_impl( pos, pos >= 0);
}

template <typename T>
bool File_reader_writer_base_impl<T>::seek_position(
    const mi::neuraylib::IStream_position* stream_position)
{
    MI_ASSERT( m_fp);

    if( !stream_position)
        return false;
    if( !stream_position->is_valid())
        return false;

    const auto* stream_position_impl = static_cast<const Stream_position_impl*>( stream_position);
    mi::Sint64 pos = stream_position_impl->get_stream_position();
    return seek_absolute( pos);
}

template <typename T>
bool File_reader_writer_base_impl<T>::rewind()
{
    return seek_absolute( 0);
}

template <typename T>
bool File_reader_writer_base_impl<T>::supports_absolute_access() const
{
    return true;
}

template <typename T>
mi::Sint64 File_reader_writer_base_impl<T>::tell_absolute() const
{
    MI_ASSERT( m_fp);

    m_error = 0;
#if defined( MI_PLATFORM_LINUX) || defined( MI_PLATFORM_MACOSX)
    mi_static_assert( sizeof( mi::Sint64) == sizeof( off_t));
    mi::Sint64 pos = ftello( m_fp);
#elif defined( MI_PLATFORM_WINDOWS)
    mi::Sint64 pos = _ftelli64( m_fp);
#else
    mi::Sint64 pos = ftell( m_fp);
#endif

    return pos;
}

template <typename T>
bool File_reader_writer_base_impl<T>::seek_absolute( mi::Sint64 pos)
{
    return seek_absolute_internal( pos, 0);
}

template <typename T>
mi::Sint64 File_reader_writer_base_impl<T>::get_file_size() const
{
    auto* self = const_cast<File_reader_writer_base_impl<T>*>( this);
    mi::Sint64 current = tell_absolute();
    self->seek_absolute_internal( 0, 2);
    mi::Sint64 size = tell_absolute();
    self->seek_absolute_internal( current, 0);
    return size;
}

template <typename T>
bool File_reader_writer_base_impl<T>::seek_end()
{
   return seek_absolute_internal( 0, 2);
}


template <typename T>
bool File_reader_writer_base_impl<T>::seek_absolute_internal( mi::Sint64 pos, int whence)
{
    MI_ASSERT( m_fp);

#if defined( MI_PLATFORM_LINUX) || defined( MI_PLATFORM_MACOSX)
    mi_static_assert( sizeof( mi::Sint64) == sizeof( off_t));
    bool success = fseeko( m_fp, static_cast<off_t>( pos), whence) == 0;
#elif defined( MI_PLATFORM_WINDOWS)
    bool success = _fseeki64( m_fp, pos, whence) == 0;
#else
    bool success = fseek( m_fp, pos, whence) == 0;
#endif

    m_error = success ? 0 : HAL::get_errno();
    return success;
}

template <typename T>
bool File_reader_writer_base_impl<T>::open( const char* path, bool for_reading)
{
    if( !path)
        return false;

    if( m_fp && !close())
        return false;

    // For writing, create corresponding directory if necessary.
    fs::path fs_path( fs::u8path( path));
    if( !for_reading) {
        std::error_code ec;
        fs::path directory = fs_path.parent_path();
        if( !directory.empty() && !fs::is_directory( directory, ec))
            fs::create_directories( directory, ec);
    }

#ifdef MI_PLATFORM_WINDOWS
    const std::wstring& wpath = fs_path.wstring();
    m_fp = _wfopen( wpath.c_str(), for_reading ? L"rb" : L"wb");
#else
    m_fp = ::fopen( path, for_reading ? "rb" : "wb");
#endif

    if( !m_fp) {
        m_error = HAL::get_errno();
        return false;
    }

    std::error_code ec;
    if( !fs::is_regular_file( fs_path, ec)) {
        m_error = EINVAL;
        return false;
    }

    m_error = 0;
    m_path = path;
    return true;
}

template <typename T>
const char* File_reader_writer_base_impl<T>::get_path()
{
    return m_path.c_str();
}

template <typename T>
bool File_reader_writer_base_impl<T>::close()
{
    if( !m_fp)
        return true;

    m_error = fclose( m_fp) == 0 ? 0 : HAL::get_errno();
    m_fp = nullptr;
    m_path.clear();
    return m_error == 0;
}


// explicit template instantiation for File_reader_writer_base_impl
template class File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IReader> >;
template class File_reader_writer_base_impl<mi::base::Interface_implement<mi::neuraylib::IWriter> >;

mi::Sint64 File_reader_impl::read( char* buffer, mi::Sint64 size)
{
    MI_ASSERT( m_fp);

    // Reject invalid m_fp due to ferror() below.
    if( !m_fp || !buffer || size < 0) {
        m_error = EINVAL;
        return -1;
    }

    size_t result = fread( buffer, 1, static_cast<size_t>( size), m_fp);
    if( ferror( m_fp) == 0) {
        m_error = 0;
        return result;
    } else {
        m_error = HAL::get_errno();
        return -1;
    }
}

bool File_reader_impl::readline( char* buffer, mi::Sint32 size)
{
    MI_ASSERT( m_fp);

    if( fgets( buffer, size, m_fp) == nullptr)
        buffer[0] = '\0';
    m_error = 0;
    return true;
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
    return Base::open( path, /*for_reading*/ true);
}

mi::Sint64 File_writer_impl::write( const char* buffer, mi::Sint64 size)
{
    MI_ASSERT( m_fp);

    // Reject invalid m_fp due to ferror() below.
    if( !m_fp || !buffer || size < 0) {
        m_error = EINVAL;
        return -1;
    }

    size_t result = fwrite( buffer, 1, static_cast<size_t>( size), m_fp);
    if( ferror( m_fp) == 0) {
        m_error = 0;
        return mi::Sint64( result);
    } else {
        m_error = HAL::get_errno();
        return -1;
    }
}

bool File_writer_impl::writeline( const char* str)
{
    MI_ASSERT( m_fp);

    if( fputs( str, m_fp) != EOF) {
        m_error = 0;
        return true;
    } else {
        m_error = HAL::get_errno();
        return false;
    }
}

bool File_writer_impl::flush()
{
    MI_ASSERT( m_fp);

    return fflush( m_fp) == 0;
}

bool File_writer_impl::open( const char* path)
{
    return Base::open( path, /*for_reading*/ false);
}

} // namespace DISK

} // namespace MI
