/******************************************************************************
 * Copyright (c) 2009-2020, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************/
/// \file
/// \brief Functions for (de)serializing and (de)compressing a vector using zlib.

#ifndef BASE_DATA_SERIAL_I_COMPRESSED_SERIALIZATION_H
#define BASE_DATA_SERIAL_I_COMPRESSED_SERIALIZATION_H

#include "serial.h"
#include <vector>
#include <base/lib/zlib/zlib.h>

namespace MI {
namespace SERIAL {

// Compress and serialize a vector, overwriting (and possibly resizing)
// the given temporary buffer. Returns false if the compression
// failed (in which case nothing is serialized) and true otherwise.
// Note that the \p data vector is treated as a chunk of memory,
// i.e. no conversion is taking place.
// See the zlib header regarding the compression level settings,
// e.g. use Z_BEST_SPEED or Z_BEST_COMPRESSION, etc.
template <typename T>
bool compress_and_serialize(
    SERIAL::Serializer* const serial,
    const std::vector<T>& data,
    std::vector<unsigned char>& temp_buffer,
    const int compression_level = Z_BEST_SPEED)
{
    // make sure the result buffer is large enough
    const size_t data_size = data.size();
    const uLong data_bytes = static_cast<uLong>(data_size * sizeof(T));
    const uLong compress_bound = compressBound(data_bytes);
    temp_buffer.resize(compress_bound);

    // compress the data
    uLong compressed_size = compress_bound;
    if (compress2(reinterpret_cast<Byte*>(&temp_buffer[0]),
                  &compressed_size, 
                  reinterpret_cast<const Byte*>(&data[0]),
                  data_bytes,
                  compression_level) != Z_OK)
    {
        // compression failed
        return false;
    }

    // serialize uncompressed data size
    serial->write_size_t(data_size);

    // serialize compressed data
    temp_buffer.resize(static_cast<size_t>(compressed_size));
    SERIAL::write(serial, temp_buffer);

    return true;
}

// Deserialize and decompress a vector, overwriting (and possibly resizing)
// the given temporary buffer. Returns false if the decompression
// failed (in which case the result vector contents are undefined)
// and true otherwise.
// Note that the \p data vector is treated as a chunk of memory,
// i.e. no conversion is taking place.
template <typename T>
bool deserialize_and_decompress(
    SERIAL::Deserializer* const deser,
    std::vector<T>& data,
    std::vector<unsigned char>& temp_buffer)
{
    // deserialize uncompressed data size
    size_t data_size = 0;
    deser->read_size_t(&data_size);
    data.resize(data_size);

    // deserialize compressed data
    SERIAL::read(deser, &temp_buffer);

    // decompress the data
    uLong dest_len(data_size * sizeof(T));
    return uncompress(reinterpret_cast<Byte*>(&data[0]),
                      &dest_len,
                      reinterpret_cast<const Byte*>(&temp_buffer[0]),
                      static_cast<uLong>(temp_buffer.size())) == Z_OK;
}

} // namespace SERIAL
} // namespace MI

#endif // BASE_DATA_SERIAL_I_COMPRESSED_SERIALIZATION_H

