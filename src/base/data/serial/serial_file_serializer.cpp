/******************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief file serializer
/// Provide a serializer/deserializer pair which serializes to and
/// deserializes from file. This can be used for different purposes
/// including implementing importer/exporter. Based on
/// base/data/serial/i_serial_buffer_serializer.h. The following
/// example explains how to use it:
/// \code
///  	Some_serializable serializable(...);
///  	File_serializer serializer;
///  	serializable.serialize(serializer);
///  	File_deserializer deserializer;
///  	Some_serializable deserialized;
///  	deserializer.deserialize(&deserialized, serializer->get_buffer(),
/// 	    serializer->get_buffer_size());
/// \endcode
///
/// TODO: Current implementation do not care about endian.
/// If endian is a problem, please implement each read/write

#include "pch.h"

#include "i_serial_file_serializer.h"

#include <base/hal/disk/disk.h>

namespace MI
{
namespace SERIAL
{
//----------------------------------------------------------------------
// Constructor
File_serializer::File_serializer()
    :
    m_p_file(0),
    m_is_valid(false)
{
    // empty
}

//----------------------------------------------------------------------
// Destructor
File_serializer::~File_serializer()
{
    // empty
}

//----------------------------------------------------------------------
// set output file
void File_serializer::set_output_file(
    MI::DISK::IFile * p_file
    )
{
    m_p_file   = p_file;
    this->set_valid(false);

    if(this->is_file_valid(p_file)){
        this->set_valid(true);
    }
}

//----------------------------------------------------------------------
// peek current file
MI::DISK::IFile * File_serializer::peek_output_file() const
{
    return m_p_file;
}

//----------------------------------------------------------------------
// is this serializer ok?
bool File_serializer::is_valid() const
{
    return m_is_valid;
}

//----------------------------------------------------------------------
// Write out various value types
void File_serializer::write_impl(const char* buffer, size_t size)
{
    ASSERT(M_DB, this->is_file_valid(m_p_file));

    if (m_p_file->write(buffer, size) != Sint64(size)){
	this->set_valid(false);
    }
}

//----------------------------------------------------------------------
// set serializer status.
void File_serializer::set_valid(
    bool is_valid
    )
{
    m_is_valid = is_valid;
}

//----------------------------------------------------------------------
// check the file can write ready.
bool File_serializer::is_file_valid(
    MI::DISK::IFile * p_file
    ) const
{
    if((p_file != 0) && (p_file->is_open()) && !(p_file->eof())){
        return true;
    }
    return false;
}

//----------------------------------------------------------------------
//======================================================================
//----------------------------------------------------------------------
// This deserializer's read() and write() functions work without
// having a Deserialization_manager, but the deserialize() method does
// not because it needs to look up the class's constructor function.
File_deserializer::File_deserializer(
    Deserialization_manager* p_manager // the deserialization manager
    )
    :
    Deserializer_impl(p_manager),
    m_p_file(0),
    m_is_valid(false)
{
    // empty
}

//----------------------------------------------------------------------
// Destructor
File_deserializer::~File_deserializer()
{
    // empty
}

//----------------------------------------------------------------------
// set input file
void File_deserializer::set_input_file(
    MI::DISK::IFile * p_file
    )
{
    m_p_file = p_file;
    this->set_valid(false);

    if(this->is_file_valid(p_file)){
        this->set_valid(true);
    }
}

//----------------------------------------------------------------------
// peek current file
MI::DISK::IFile * File_deserializer::peek_input_file() const
{
    return m_p_file;
}

//----------------------------------------------------------------------
// is this serializer ok?
bool File_deserializer::is_valid() const
{
    return m_is_valid;
}

//----------------------------------------------------------------------
// Deserialize from a buffer of known size
// Note: The reason why this method name is not deserialize, see
// header's comment.
Serializable* File_deserializer::deserialize_file()
{
    if(!this->is_file_valid(m_p_file)){
        // fprintf(stderr, "Can not read file while deserializing an object\n");
        m_is_valid = false;
        return 0;
    }
    else{
        m_is_valid  = true;
    }

    Serializable* p_serializable = Deserializer_impl::deserialize();
    return p_serializable;
}

//----------------------------------------------------------------------
// Read back various value types
void File_deserializer::read_impl(
    char* buffer,               // destination for writing data
    size_t size)                // number of bytes to read
{
    ASSERT(M_DB, this->is_file_valid(m_p_file));

    if(m_p_file->read(buffer, size) != Sint64(size)){
	m_is_valid = false;
    }
}

//----------------------------------------------------------------------
// set serializer status.
void File_deserializer::set_valid(
    bool is_valid
    )
{
    m_is_valid = is_valid;
}

//----------------------------------------------------------------------
// check the file can read ready.
bool File_deserializer::is_file_valid(
    MI::DISK::IFile * p_file
    ) const
{
    if((p_file != 0) && (p_file->is_open()) && !(p_file->eof())){
        return true;
    }
    return false;
}

//----------------------------------------------------------------------
} // namespace DB
} // namespace MI

