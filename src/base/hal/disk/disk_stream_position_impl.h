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
/// \brief Header for an implementations of mi::neuraylib::IStream_position.

#ifndef BASE_HAL_DISKDISK_STREAM_POSITION_IMPL_H
#define BASE_HAL_DISKDISK_STREAM_POSITION_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/istream_position.h>

#include <boost/core/noncopyable.hpp>

namespace MI {

namespace DISK {

/// This implementation of mi::neuraylib::IStream_position is used by File_reader_writer_base_impl
/// and Memory_reader_writer_base_impl.
class Stream_position_impl
  : public mi::base::Interface_implement<mi::neuraylib::IStream_position>,
    public boost::noncopyable
{
public:

    /// Constructor
    Stream_position_impl( mi::Sint64 position, bool valid)
      : m_position( position), m_valid( valid) { }

    // public API methods

    bool is_valid() const { return m_valid; }

    // internal methods

    /// Returns the stream position.
    mi::Sint64 get_stream_position() const { return m_position; }

private:

    /// The stream position
    mi::Sint64 m_position;

    /// Indicates whether the stream position is valid
    bool m_valid;
};

} // namespace DISK

} // namespace MI

#endif // BASE_HAL_DISKDISK_STREAM_POSITION_IMPL_H
