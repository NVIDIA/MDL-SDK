/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief This declares a simple disk serializer/deserializer pair.
///
///  This implements serializing to and from files.

#ifndef BASE_DATA_SERIAL_I_SERIAL_BUFFERED_FILE_SERIALIZER_H
#define BASE_DATA_SERIAL_I_SERIAL_BUFFERED_FILE_SERIALIZER_H

#include <base/data/serial/serial.h>
#include <base/hal/disk/i_disk_buffered_reader.h>

namespace MI {

namespace SERIAL {
    
/// Serializer to write out a serializable to a file on disk
///
/// Marking Serializer_impl inheritance as private because only
/// #serialize() will be called.
class Buffered_file_serializer: private SERIAL::Serializer_impl
{
public:
    /// Do the serialization of a serializable to the given file. It is assumed that a seek
    /// happened on the given file, so that the file is at the correct position for writing.
    /// \param The thing to serialize
    /// \param File to write to
    /// \return Return true if okay, false if an error occurred.
    bool serialize(const SERIAL::Serializable* serializable, DISK::IFile* file)
    {
	m_file = file;
	m_error = false;
	Serializer_impl::serialize(serializable);
	/// The serialization might have registered shared objects. Now write them out.
	clear_shared_objects();
	return !m_error;
    }

private:
    DISK::IFile* m_file; ///< file to write to
    bool m_error; ///< error occurred?

    // Implementation of the virtual interface
    void write_impl(const char* buffer, size_t size)
    { if (size == 0) return; m_error |= (m_file->write(buffer, size) < 0); }
};


/// Deserializer to read a serializable from a file on disk
///
/// Marking Deserializer_impl inheritance as private because only
/// #serialize() will be called.
class Buffered_file_deserializer : private SERIAL::Deserializer_impl
{
public:
    /// Constructor
    /// \param deserialization_manager For constructing objects
    Buffered_file_deserializer(SERIAL::Deserialization_manager* deserialization_manager, 
    	DISK::IFile* file) : Deserializer_impl(deserialization_manager), m_file(file)  { }

    /// Do the deserialization of a serializable from the given file. It is assumed that a seek
    /// happened on the given file, so that the file is at the correct position for reading.
    /// \param The file to read from
    /// \return A newly constructed serializable
    SERIAL::Serializable* deserialize()
    {
	m_error = false;
	SERIAL::Serializable *serializable = Deserializer_impl::deserialize();
	if (m_error) 
	{
	    delete serializable;
	    return NULL;
	}
    
	// The deserialize call may have registered open references to shared
	// objects. Now resolve them.
	clear_shared_objects();
	return serializable;
    }

private:
    DISK::Buffered_reader m_file; ///< file to read from
    bool m_error; ///< error occurred?

    // Implementation of the virtual interface
    void read_impl(char* buffer, size_t size) 
    { if (size == 0) return; m_error |= !(m_file.read(buffer, size)); }

    bool is_valid() const { return m_error == false; }
};

} // namespace SERIAL
} // namespace MI

#endif // BASE_DATA_SERIAL_I_SERIAL_BUFFERED_FILE_SERIALIZER_H
