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
/// \file mi/mdl/mdl_positions.h
/// \brief Interfaces for describing source positions
#ifndef MDL_POSITIONS_H
#define MDL_POSITIONS_H 1

#include <cstddef>

#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

/// A source code position.
///
/// A source code position is a quadrupel of start line, start column, end line, and end column.
/// Additionally, a file identifier is stored. The file identifier (ID) is an index into
/// the file table, which is stored at the owner of the position.
class Position : public Interface_owned {
public:
    enum File_id {
        OWNER_FILE_ID = 0    ///< A predefined constant meaning "owner filename"
    };

    /// Get the start line.
    virtual int get_start_line() const = 0;

    /// Set the start line.
    ///
    /// \param line  the new start line
    virtual void set_start_line(int line) = 0;

    /// Get the start column.
    virtual int get_start_column() const = 0;

    /// Set the start column.
    ///
    /// \param column  the new start column
    virtual void set_start_column(int column) = 0;

    /// Get the end line.
    virtual int get_end_line() const = 0;

    /// Set the end line.
    ///
    /// \param line  the new end line
    virtual void set_end_line(int line) = 0;

    /// Get the end column.
    virtual int get_end_column() const = 0;

    /// Set the end column.
    ///
    /// \param column  the new end column
    virtual void set_end_column(int column) = 0;

    /// Get the filename id.
    virtual size_t get_filename_id() const = 0;

    /// Set the filename id.
    ///
    /// \param id  the file identifier
    virtual void set_filename_id(size_t id) = 0;
};

}  // mdl
}  // mi

#endif
