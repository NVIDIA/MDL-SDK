/******************************************************************************
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
 *****************************************************************************/

#include "pch.h"

#include "compilercore_positions.h"
#include "compilercore_serializer.h"

namespace mi {
namespace mdl {

// Get the start line.
int Position_impl::get_start_line() const
{
    return m_line;
}

// Set the start line.
void Position_impl::set_start_line(int line)
{
    m_line = line;
}


// Get the start column.
int Position_impl::get_start_column() const
{
    return m_start_column;
}


// Set the start column.
void Position_impl::set_start_column(int column)
{
    m_start_column = column;
}

// Get the end line.
int Position_impl::get_end_line() const
{
    return m_end_line;
}


// Set the end line.
void Position_impl::set_end_line(int line)
{
    m_end_line = line;
}

// Get the end column.
int Position_impl::get_end_column() const
{
    return m_end_column;
}

// Set the end column.
void Position_impl::set_end_column(int column)
{
    m_end_column = column;
}

// Get the filename id.
size_t Position_impl::get_filename_id() const
{
    return m_filename_id;
}

// set the filename id.
void Position_impl::set_filename_id(size_t id)
{
    m_filename_id = id;
}

// Serialize this position.
void Position_impl::serialize(Entity_serializer &serialiazer) const
{
   serialiazer.write_int(m_line);
   serialiazer.write_int(m_end_line);
   serialiazer.write_int(m_start_column);
   serialiazer.write_int(m_end_column);
   serialiazer.write_encoded_tag(m_filename_id);
}

// Deserializing constructor
Position_impl::Position_impl(Entity_deserializer &deserialiazer)
: Base()
, m_line(deserialiazer.read_int())
, m_end_line(deserialiazer.read_int())
, m_start_column(deserialiazer.read_int())
, m_end_column(deserialiazer.read_int())
, m_filename_id(deserialiazer.read_encoded_tag())
{
}

// Constructor.
Position_impl::Position_impl(int start_line, int start_column, int end_line, int end_column)
: Base()
, m_line(start_line)
, m_end_line(end_line)
, m_start_column(start_column)
, m_end_column(end_column)
, m_filename_id(OWNER_FILE_ID)
{
}

// Constructor.
Position_impl::Position_impl(Position const *other)
: Base()
, m_line(0)
, m_end_line(0)
, m_start_column(0)
, m_end_column(0)
, m_filename_id(OWNER_FILE_ID)
{
    if (other != NULL) {
        m_line         = other->get_start_line();
        m_end_line     = other->get_end_line();
        m_start_column = other->get_start_column();
        m_end_column   = other->get_end_column();
        m_filename_id  = other->get_filename_id();
    }
}

}  // mdl
}  // mi
