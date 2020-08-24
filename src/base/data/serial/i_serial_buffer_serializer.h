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

#ifndef BASE_DATA_SERIAL_I_SERIAL_BUFFER_SERIALIZER_H
#define BASE_DATA_SERIAL_I_SERIAL_BUFFER_SERIALIZER_H

#include "serial.h"

namespace MI
{

namespace SERIAL
{

// The Serializer will abstract from the concrete serialization target and will free the
// Serializable classes from having to write out class id etc.
class Buffer_serializer : public Serializer_impl
{
public:
    // Constructor
    Buffer_serializer();

    // Destructor
    ~Buffer_serializer();

    // Reset it so that it can be reused
    void reset();

    // Get the buffer holding the serialized data
    Uint8* get_buffer();

    // Get the buffer, detaching it from the serializer
    Uint8* takeover_buffer();

    // Get the size of the buffer holding the serialized data
    size_t get_buffer_size();

    using Serializer_impl::write;

    // Give a hint to the serializer that the given number of bytes
    // are written to the serializer soon.
    void reserve(
	size_t size);

protected:
    // Write out various value types
    void write_impl(
	const char* buffer,				// read data from here
	size_t size);					// write this amount of data

private:
    Uint8* m_buffer;					// the buffer where to write data to
    size_t m_buffer_size;				// the size of the buffer
    size_t m_written_size;				// which part of buffer is already used

    // ensure that the buffer has the needed number of bytes free
    void ensure_size(
	size_t needed_size);				// the needed size
};

// The Deserializer will abstract from the concrete deserialization source.
class Buffer_deserializer : public Deserializer_impl
{
public:
    using Deserializer_impl::deserialize;

    // This deserializer's read() and write() functions work without having a
    // Deserialization_manager, but the deserialize() method does not because it needs to look up
    // the class's constructor function. The global manager instance can be obtained from
    // Mod_data::get_deserialization_manager().
    explicit Buffer_deserializer(
	Deserialization_manager* manager = NULL);	// the set of registered classes

    // Destructor
    ~Buffer_deserializer();

    // Set the deserializer to use the given buffer for input
    void reset(
	const Uint8* buffer,				// the buffer
	size_t buffer_size);				// the size of the buffer

    // Deserialize from a buffer of known size
    Serializable* deserialize(
	const Uint8* buffer,				// buffer storing the serialized data
	size_t buffer_size);				// size of the given buffer

    // Deserialize a given object from a buffer of known size
    void deserialize(
    	Serializable* serializable,			// deserialize to here
	const Uint8* buffer,				// buffer storing the serialized data
	size_t buffer_size);				// size of the given buffer

    using Deserializer_impl::read;

    // ensure that the buffer has the needed number of bytes free
    bool ensure_size(
	size_t needed_size);				// the needed size

    bool is_valid() const;

protected:

    // Read back various value types
    void read_impl(
	char* buffer,					// destination for writing data
	size_t size);					// number of bytes to read


private:
    const Uint8* m_buffer;			// read the data from here
    const Uint8* m_read_pointer;			// pointer to next byte to be read
    size_t m_buffer_size;				// size of all data
    bool m_valid;
};

} // namespace SERIAL

} // namespace MI

#endif // BASE_DATA_SERIAL_I_SERIAL_BUFFER_SERIALIZER_H
