/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_LOCATIONS_H
#define MDL_COMPILER_HLSL_LOCATIONS_H 1

#include "compiler_hlsl_cc_conf.h"

#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {
namespace hlsl {

/// A source code location.
class Location : public Interface_owned
{
    typedef Interface_owned Base;
public:
    enum File_id { OWNER_FILE_IDX = 0 };

    /// Get the line.
    unsigned get_line() const { return m_line; }

    /// Get the column.
    unsigned get_column() const { return m_column; }

    /// Get the file ID.
    unsigned get_file_id() const { return m_file_id; }

public:
    /// Constructor.
    Location(
        unsigned line,
        unsigned column,
        unsigned file_id = OWNER_FILE_IDX)
    : m_line(line)
    , m_column(column)
    , m_file_id(file_id)
    {
    }

private:
    /// The line of this location.
    unsigned m_line;
    /// The column of this location.
    unsigned m_column;
    /// The file ID of this location.
    unsigned m_file_id;
};

}  // hlsl
}  // mdl
}  // mi

#endif
