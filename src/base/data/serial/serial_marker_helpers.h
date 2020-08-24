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
/// \brief Helpers for setting/reading serialization markers.

#ifndef SERIAL_MARKER_HELPERS_H
#define SERIAL_MARKER_HELPERS_H

#include <cstddef>

#include "i_serializer.h"

#include <base/system/main/types.h>
#include <mi/neuraylib/iserializer.h>
#include <mi/neuraylib/ideserializer.h>

#include <stack>

namespace MI
{
namespace SERIAL
{

class Serializer;
class Deserializer;
class Serializable;

class Deserializer_wrapper;

/// Helper class for computing checksums from a stream of bytes with scoping
/// via start()/end().
class Serializer_checksummer
{
public:

    /// Start a scope.
    void start();

    /// End a scope.
    void end();

    /// Get crc for current scope.
    mi::Uint32 get();

    /// Update crc with given buffer of given size.
    void update(const char* buffer, size_t size);

private:

    /// Stack of crcs.
    std::stack<mi::Uint32> m_crc_stack;

    /// Number of checksummed bytes in this scope.
    size_t m_crc_len;
};

enum Markers
{
    NO_MARKER = 0x0,
    END_MARKER = 0xABABABAB,
    EXTENSION_MARKER = 0xB5B5B5B5
};

/// Helper for writing markers on the serialization side.
class Serializer_marker_helper
{
public:

    /// Update crc for given partial buffer.
    void update_checksum(const char* buffer, size_t size);

    /// Set an extension marker on the given serializer.
    void set_extension_marker(Serializer* serializer);

    /// Serialize given object with the given serializer and set the end marker
    /// and checksum.
    ///
    /// Note: The id's of the serializable object are not checksummed. They are
    /// being written from the outside. But if it's a nested scope, the outer scope
    /// will checksum the inner's class ids.
    ///
    /// \param serializer		Serializer to be used.
    /// \param serializable	Object to be serialized.
    void serialize_with_end_marker(Serializer* serializer, const Serializable* serializable);

    /// Serialize given object with the given serializer and set the end marker
    /// and checksum.
    ///
    /// Note: The id's of the serializable object are not checksummed. They are
    /// being written from the outside. But if it's a nested scope, the outer scope
    /// will checksum the inner's class ids.
    ///
    /// \param serializer		Serializer to be used.
    /// \param serializable	Object to be serialized.
    void serialize_with_end_marker(mi::neuraylib::ISerializer*,
                                   const mi::neuraylib::ISerializable* serializable);

private:
    Serializer_checksummer m_checksum;
};

/// Helper for reading markers on the deserialization side.
class Deserializer_marker_helper
{
public:
    Deserializer_marker_helper()
        : m_checksum()
        , m_last_marker(NO_MARKER)
    {}

    /// Update crc for given partial buffer.
    void update_checksum(const char* buffer, size_t size);

    /// Read extension marker from given deserializer.
    ///
    /// \return     			-  0 On success.
    ///             			- -1 Marker not found.
    ///             			- -2 Checksum failed.
    Marker_status read_extension_marker(Deserializer* deserializer);

    /// Deserialize given object with the given deserializer and looks for the
    /// correct end marker.
    ///
    /// Stops reading after finding the end marker.
    ///
    /// \param deserializer   Deserializer to be used.
    /// \param serializable   Object to be deserialized.
    /// \return     			-  0 On success.
    ///             			- -1 Marker not found.
    ///             			- -2 Checksum failed.
    Marker_status deserialize_with_end_marker(Deserializer* deserializer,
                                              Serializable* serializable);

    /// Deserialize given object with the given deserializer and looks for the
    /// correct end marker.
    ///
    /// Stops reading after finding the end marker.
    ///
    /// \param deserializer   Deserializer to be used.
    /// \param serializable   Object to be deserialized.
    /// \return     			-  0 On success.
    ///             			- -1 Marker not found.
    ///             			- -2 Checksum failed.
    Marker_status deserialize_with_end_marker(mi::neuraylib::IDeserializer* deserializer,
                                              mi::neuraylib::ISerializable* serializable);

private:
    /// Implementation used by IDeserializer and Deserializer.
    Marker_status read_end_marker_internal(Deserializer_wrapper* deserializer);

    Marker_status read_end_marker_once(Deserializer_wrapper* deserializer);

    /// Fast forward to given marker, skipping all extension markers.
    Marker_status skip_to_marker(Deserializer_wrapper* deserializer, Markers marker);

    Serializer_checksummer m_checksum;

    /// Store last read marker. Used to avoid going past the end marker.
    Markers m_last_marker;
};

} // namespace SERIAL
} // namespace MI

#endif // SERIAL_MARKER_HELPERS_H
