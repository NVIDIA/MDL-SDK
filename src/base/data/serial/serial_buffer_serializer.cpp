/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Provide serializer/deserializer pair writing to / reading from memory
///
/// Provide a serializer/deserializer pair which serializes to and deserializes from memory.
/// This can be used for different purposes including implementing tests for correct serialization.
/// The following example explains how to use it:
/// 	Some_serializable serializable(...);
/// 	Buffer_serializer serializer;
/// 	serializable.serialize(serializer);
/// 	Buffer_deserializer deserializer;
/// 	Some_serializable deserialized;
/// 	deserializer.deserialize(&deserialized, serializer->get_buffer(),
///	    serializer->get_buffer_size());

#include "pch.h"

#include <base/lib/mem/i_mem_allocatable.h>
#include <base/lib/log/log.h>

#include "i_serial_buffer_serializer.h"


namespace MI
{

namespace SERIAL
{

// Constructor
Buffer_serializer::Buffer_serializer()
{
    m_buffer = NULL;
    reset();
}

// Destructor
Buffer_serializer::~Buffer_serializer()
{
    MEM::delete_array<Uint8>(m_buffer);
}

// Reset it so that it can be reused
void Buffer_serializer::reset()
{
    MEM::delete_array<Uint8>(m_buffer);
    m_buffer_size = 1024;
    m_buffer = MEM::new_array<Uint8>(m_buffer_size);
    m_written_size = 0;
    clear_shared_objects();
}

// Get the buffer holding the serialized data
Uint8* Buffer_serializer::get_buffer()
{
    return m_buffer;
}

// Get the buffer, detaching it from the serializer
Uint8* Buffer_serializer::takeover_buffer()
{
    Uint8* buffer = m_buffer;
    m_buffer = NULL;
    return buffer;
}

// Get the size of the buffer holding the serialized data
size_t Buffer_serializer::get_buffer_size()
{
    return m_written_size;
}

// ensure that the buffer has the needed number of bytes free
void Buffer_serializer::ensure_size(
    size_t needed_size)					// the needed size
{
    if (m_buffer_size - m_written_size < needed_size)
    {
    while (m_buffer_size - m_written_size < needed_size)
        m_buffer_size *= 2;
    Uint8* new_buffer = MEM::new_array<Uint8>(m_buffer_size);
    memcpy(new_buffer, m_buffer, m_written_size);
    MEM::delete_array<Uint8>(m_buffer);
    m_buffer = new_buffer;
    }
}

// Write out various value types
void Buffer_serializer::write_impl(
    const char* buffer,					// read data from here
    size_t size)					// write this amount of data
{
    ensure_size(size);
    memcpy(m_buffer + m_written_size, buffer, size);
    m_written_size += size;
}

void Buffer_serializer::reserve(
    size_t needed_size)
{
    if (m_buffer_size - m_written_size < needed_size)
    {
    m_buffer_size = m_written_size + needed_size;
    Uint8* new_buffer = MEM::new_array<Uint8>(m_buffer_size);
    memcpy(new_buffer, m_buffer, m_written_size);
    MEM::delete_array<Uint8>(m_buffer);
    m_buffer = new_buffer;
    }
}

// This deserializer's read() and write() functions work without having a Deserialization_manager,
// but the deserialize() method does not because it needs to look up the class's constructor
// function.
Buffer_deserializer::Buffer_deserializer(
    Deserialization_manager*	manager) 		// the deserialization manager
    : Deserializer_impl(manager),
      m_buffer(NULL),
      m_read_pointer(NULL),
      m_buffer_size(0),
      m_valid(true)
{
}

// Destructor
Buffer_deserializer::~Buffer_deserializer()
{
}

// Set the deserializer to use the given buffer for input
void Buffer_deserializer::reset(
    const Uint8* buffer,                // the buffer
    size_t buffer_size)                 // the size of the buffer
{
    m_buffer = buffer;
    m_buffer_size = buffer_size;
    m_read_pointer = buffer;
    m_valid = true;
    clear_shared_objects();
}

// ensure that the buffer has the needed number of bytes free
bool Buffer_deserializer::ensure_size(
    size_t needed_size)					// the needed size
{
    return needed_size <= (m_buffer_size - (m_read_pointer - m_buffer));
}

// Deserialize from a buffer of known size
Serializable* Buffer_deserializer::deserialize(
    const Uint8* buffer,				// buffer storing the serialized data
    size_t buffer_size)					// size of the given buffer
{
    m_buffer = buffer;
    m_buffer_size = buffer_size;
    m_read_pointer = buffer;

    Serializable* serializable = Deserializer_impl::deserialize();

    if (m_read_pointer == m_buffer + buffer_size)
    {
    clear_shared_objects();
    return serializable;
    }

    //fprintf(stderr, "Buffer underflow while deserializing object %u %u\n",
    //buffer_size, m_read_pointer - m_buffer);
    abort();
    return NULL;
}

// Deserialize a given object from a buffer of known size
void Buffer_deserializer::deserialize(
    Serializable* serializable,				// deserialize to here
    const Uint8* buffer,				// buffer storing the serialized data
    size_t buffer_size)					// size of the given buffer
{
    m_buffer = buffer;
    m_buffer_size = buffer_size;
    m_read_pointer = buffer;

    serializable->deserialize(this);

    if (m_read_pointer == m_buffer + buffer_size)
    {
    clear_shared_objects();
    return;
    }

    //fprintf(stderr, "Buffer underflow while deserializing object\n");
    abort();
}

// Read back various value types
void Buffer_deserializer::read_impl(
    char* buffer,					// destination for writing data
    size_t size)					// number of bytes to read
{
    m_valid = ensure_size(size);
    if (!m_valid)
    {
        // Note: this should really be fatal because there is no good way to recover from this.
        // However, a few code paths which use this function support a user defined callback that
        // is invoked for deserialization errors.
        LOG::mod_log->error(M_SERIAL, LOG::Mod_log::C_MISC, "Buffer deserializer: "
                            "reading beyond end of buffer");
        return;
    }

    memcpy(buffer, m_read_pointer, size);
    m_read_pointer += size;

}
bool Buffer_deserializer::is_valid() const
{
    return m_valid;
}

} // namespace DB

} // namespace MI

