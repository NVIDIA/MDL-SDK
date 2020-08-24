/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief 

#include "pch.h"

#include "serial_marker_helpers.h"
#include "i_serial_serializable.h"
#include "serial.h"

#include <base/lib/zlib/i_zlib.h>

namespace MI
{
namespace SERIAL
{

void Serializer_checksummer::start()
{
    m_crc_stack.push(0);
    m_crc_len = 0;
}

void Serializer_checksummer::end()
{
    ASSERT(M_SERIAL, !m_crc_stack.empty());
    mi::Uint32 nested_crc = m_crc_stack.top();
    m_crc_stack.pop();

    if (m_crc_stack.empty())
        return;

    m_crc_stack.top() = ZLIB::crc32_combine(m_crc_stack.top(), nested_crc, m_crc_len);
    m_crc_len = 0;
}

mi::Uint32 Serializer_checksummer::get()
{
    if (m_crc_stack.empty())
        return 0;

    return m_crc_stack.top();
}

void Serializer_checksummer::update(const char* buffer, size_t size)
{
    if (m_crc_stack.empty())
        return;

    m_crc_stack.top() = ZLIB::crc32(m_crc_stack.top(), buffer, size);
    m_crc_len += size;
}

void Serializer_marker_helper::update_checksum(const char* buffer, size_t size)
{
    m_checksum.update(buffer, size);
}

void Serializer_marker_helper::set_extension_marker(Serializer* serializer)
{
    Uint32 marker = EXTENSION_MARKER;
    serializer->write(marker);
}

void Serializer_marker_helper::serialize_with_end_marker(Serializer* serializer,
                                                         const Serializable* serializable)
{
    m_checksum.start();

    serializable->serialize(serializer);

    Uint32 marker = END_MARKER;
    serializer->write(marker);

    mi::Uint32 crc = m_checksum.get();
    serializer->write(crc);
    m_checksum.end();
}

void Serializer_marker_helper::serialize_with_end_marker(mi::neuraylib::ISerializer* serializer,
                                                 const mi::neuraylib::ISerializable* serializable)
{
    m_checksum.start();

    serializable->serialize(serializer);

    Uint32 marker = END_MARKER;
    serializer->write(&marker);

    mi::Uint32 crc = m_checksum.get();
    serializer->write(&crc);
    m_checksum.end();
}

/// Internal wrapper for treating IDeserializer and Deserializer objetcs the same.
class Deserializer_wrapper
{
public:
    Deserializer_wrapper(Deserializer* deserializer)
        : m_int_deserializer(deserializer),
          m_ext_deserializer(NULL),
          m_valid(true)
    {}

    Deserializer_wrapper(mi::neuraylib::IDeserializer* deserializer)
    : m_int_deserializer(NULL),
      m_ext_deserializer(deserializer),
      m_valid(true)
    {}

    bool is_valid() const
    {
        return m_int_deserializer ? m_int_deserializer->is_valid() : m_valid;
    }

    template <typename T>
    void read(T* byte)
    {
        if (m_int_deserializer)
            m_int_deserializer->read(byte);
        else
            m_valid = m_ext_deserializer->read(byte);

    }

private:
    Deserializer* m_int_deserializer;
    mi::neuraylib::IDeserializer* m_ext_deserializer;
    bool m_valid;
};

void Deserializer_marker_helper::update_checksum(const char* buffer, size_t size)
{
    m_checksum.update(buffer, size);
}

Marker_status Deserializer_marker_helper::read_extension_marker(Deserializer* deserializer)
{
    // Avoid going past the end marker.
    if (m_last_marker == END_MARKER)
        return MARKER_NOT_FOUND;

    deserializer->read(reinterpret_cast<Uint32*>(&m_last_marker));
    return m_last_marker == EXTENSION_MARKER ? MARKER_FOUND : MARKER_NOT_FOUND;
}


Marker_status Deserializer_marker_helper::skip_to_marker(Deserializer_wrapper* deserializer,
                                                         Markers marker)
{
    const Uint8 marker_byte = reinterpret_cast<Uint8*>(&marker)[0];
    Uint8 found = 0;

    // Do it byte by byte to make sure we don't go past the marker if there's a
    // serialization error which messes up the alignment.
    while (deserializer->is_valid() && found < sizeof(marker))
    {
        Uint8 byte = 0;
        deserializer->read(&byte);
        if (byte == marker_byte)
            ++found;
        else
            found = 0;
    }

    return found == sizeof(marker) ? MARKER_FOUND : MARKER_NOT_FOUND;
}

Marker_status Deserializer_marker_helper::read_end_marker_once(Deserializer_wrapper* deserializer)
{
    // Read end marker only if we didn't get there already.
    if (m_last_marker != END_MARKER)
    {
        Uint32 marker = 0;

        deserializer->read(&marker);
        if (marker == EXTENSION_MARKER)
        {
            Marker_status status = skip_to_marker(deserializer, END_MARKER);
            if (status != MARKER_FOUND)
                return status;
        }
        else if (marker != END_MARKER)
            // Didn't find an end or extension marker next, so this is an error.
            return MARKER_NOT_FOUND;
    }

    // Don't include the checksum's cheksum in the comparison, use previous value.
    Uint32 buffer_crc = m_checksum.get();
    Uint32 read_crc = 0;
    deserializer->read(&read_crc);

    return (buffer_crc == read_crc) ? MARKER_FOUND : MARKER_BAD_CHECKSUM;
}

Marker_status Deserializer_marker_helper::read_end_marker_internal(
        Deserializer_wrapper* deserializer)
{
    Marker_status status = MARKER_NOT_FOUND;
    Marker_status prev_status;

    // Read end marker until we find one with a good checksum.
    do
    {
        prev_status = status;
        status = read_end_marker_once(deserializer);
    }
    while (deserializer->is_valid() && status == MARKER_BAD_CHECKSUM);

    if (status == MARKER_NOT_FOUND && prev_status == MARKER_BAD_CHECKSUM)
        // When an end marker with a bad checksum is encountered, the above loop
        // will keep on searching until it finds a valid end marker.
        // However, if the valid end marker is not found, the status will be "not
        // found" when it should actually be "bad checksum".
        status = MARKER_BAD_CHECKSUM;

    return status;
}

Marker_status Deserializer_marker_helper::deserialize_with_end_marker(Deserializer* deserializer,
                                                                      Serializable* serializable)
{
    Deserializer_wrapper wrapper(deserializer);

    m_checksum.start();
    serializable->deserialize(deserializer);
    Marker_status status = read_end_marker_internal(&wrapper);
    m_checksum.end();

    return status;
}

Marker_status Deserializer_marker_helper::deserialize_with_end_marker(
        mi::neuraylib::IDeserializer* deserializer, mi::neuraylib::ISerializable* serializable)
{
    Deserializer_wrapper wrapper(deserializer);

    m_checksum.start();
    serializable->deserialize(deserializer);
    Marker_status status = read_end_marker_internal(&wrapper);
    m_checksum.end();

    return status;
}


} // namespace SERIAL
} // namespace MI
