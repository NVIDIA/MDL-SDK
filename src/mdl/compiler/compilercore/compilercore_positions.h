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

#ifndef MDL_COMPILERCORE_POSITIONS_H
#define MDL_COMPILERCORE_POSITIONS_H 1

#include <mi/mdl/mdl_positions.h>
#include "compilercore_cc_conf.h"
#include "compilercore_memory_arena.h"

namespace mi {
namespace mdl {

class Entity_serializer;
class Entity_deserializer;

/// Implementation of the Position interface.
class Position_impl : public Position {
    typedef Position Base;
    template<typename T> friend class Expr_base;
    template<typename T> friend class Stmt_base;
    template<typename T> friend class Decl_base;
    template<typename T> friend class Parameter_base;
    template<typename T> friend class Argument_base;
    friend class Annotation;
    friend class Annotation_block;
    friend class Simple_name;
    friend class Qualified_name;
    friend class Type_name;
    friend class Arena_builder;
    friend class Sema_analysis;
    friend class Message;
    friend class Messages_impl;
    friend class Syntax_error;
    friend class NT_analysis;
    friend class GLSL_code_generator;
public:

    /// Get the start line.
    int get_start_line() const MDL_FINAL;

    /// Set the start line.
    void set_start_line(int line) MDL_FINAL;

    /// Get the start column.
    int get_start_column() const MDL_FINAL;

    /// Set the start column.
    void set_start_column(int column) MDL_FINAL;

    /// Get the end line.
    int get_end_line() const MDL_FINAL;

    /// Set the end line.
    void set_end_line(int line) MDL_FINAL;

    /// Get the end column.
    int get_end_column() const MDL_FINAL;

    /// Set the end column.
    void set_end_column(int column) MDL_FINAL;

    /// Get the filename id.
    size_t get_filename_id() const MDL_FINAL;

    /// set the filename id.
    void set_filename_id(size_t id) MDL_FINAL;

public:
    /// Constructor.
    explicit Position_impl(int start_line, int start_column, int end_line, int end_column);

private:
    /// Serialize this position.
    ///
    /// \param serializer  an entity serializer
    void serialize(Entity_serializer &serialiazer) const;

    /// Create a position from another one.
    ///
    /// \param other  the other position, if NULL, the ZERO position will be created
    explicit Position_impl(Position const *other);

    /// Deserializing constructor.
    ///
    /// \param deserializer  an entity deserializer
    explicit Position_impl(Entity_deserializer &deserialiazer);

private:
    /// The (start) line number of this position.
    int m_line;

    /// The end line number of this position if provided.
    int m_end_line;

    /// The start column of this position if provided.
    int m_start_column;

    /// The end column of this position if provided.
    int m_end_column;

    /// The id of the module describing
    size_t m_filename_id;
};

}  // mdl
}  // mi

#endif
