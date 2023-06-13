/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_LOCATIONS_H
#define MDLTLC_LOCATIONS_H 1

#include <mi/mdl/mdl_iowned.h>

/// A source code location.
class Location : public mi::mdl::Interface_owned
{
    typedef mi::mdl::Interface_owned Base;
public:
    enum File_id { OWNER_FILE_IDX = 0 };

    /// Get the file ID.
    size_t get_file_id() const { return m_file_id; }

    /// Get the line.
    unsigned get_line() const { return m_line; }

    /// Get the column:
    unsigned get_column() const { return m_column; }

public:
    /// Constructor.
    Location(
        size_t   file_id,
        unsigned line,
        unsigned column)
    : m_file_id(file_id)
    , m_line(line)
    , m_column(column)
    {
    }

    /// Copy-Constructor.
    Location(Location const &other)
    : m_file_id(other.m_file_id)
    , m_line(other.m_line)
    , m_column(other.m_column)
    {
    }

private:
    // No assignment operator.
    Location &operator=(Location const &other);

private:
    /// The file ID of this location.
    size_t const m_file_id;
    /// The line of this location.
    unsigned const m_line;
    /// The column of this location.
    unsigned const m_column;
};

#endif
