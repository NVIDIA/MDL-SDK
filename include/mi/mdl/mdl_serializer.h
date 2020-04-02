/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_serializer.h
/// \brief Interfaces for serialization of MDL entities
#ifndef MDL_SERIALIZER_H
#define MDL_SERIALIZER_H 1

#include <cstddef>

namespace mi {
namespace mdl {

#define TAG_FOURCC(a, b, c, d)  ((a) | ((b) << 8) | ((c) << 16) | ((d) << 24))

/// This namespace contains helper types and Interfaces for the serialization of MDL entities.
namespace Serializer {

/// Tags used by the MDL serializer to mark sections.
enum Serializer_tags {
    /// The binary start tag. Mark the stream as a complete binary.
    ST_BINARY_START = TAG_FOURCC('m', 'd', 'l', 'S'),

    /// The binary end tag.
    ST_BINARY_END   = TAG_FOURCC('m', 'd', 'l', 'E'),

    /// Mark the start of a new module.
    ST_MODULE_START = TAG_FOURCC('m', 'o', 'd', 'S'),

    /// Mark the end of the current module.
    ST_MODULE_END   = TAG_FOURCC('m', 'o', 'd', 'E'),

    /// Mark the start of a symbol table.
    ST_SYMBOL_TABLE = TAG_FOURCC('s', 'y', 'm', 'T'),

    /// Mark the start of a type table.
    ST_TYPE_TABLE   = TAG_FOURCC('t', 'y', 'p', 'T'),

    /// Mark the start of a value table.
    ST_VALUE_TABLE  = TAG_FOURCC('v', 'a', 'l', 'T'),

    /// Mark the start of a definition table.
    ST_DEF_TABLE    = TAG_FOURCC('d', 'e', 'f', 'T'),

    /// Mark the start of a scope.
    ST_SCOPE_START  = TAG_FOURCC('s', 'c', 'p', 'S'),

    /// Mark the end of a scope.
    ST_SCOPE_END    = TAG_FOURCC('s', 'c', 'p', 'E'),

    /// Mark the start of a definition.
    ST_DEFINITION   = TAG_FOURCC('d', 'f', 'n', 't'),

    /// Mark the start of the AST.
    ST_AST          = TAG_FOURCC('a', 's', 't', '_'),

    /// The code DAG start tag.
    ST_DAG_START    = TAG_FOURCC('d', 'a', 'g', 'S'),

    /// The code DAG end tag.
    ST_DAG_END      = TAG_FOURCC('d', 'a', 'g', 'E'),

    /// Mark the start of a lambda function.
    ST_LAMBDA_START = TAG_FOURCC('l', 'm', 'd', 'S'),

    /// Mark the end of a lambda function.
    ST_LAMBDA_END   = TAG_FOURCC('l', 'm', 'd', 'E'),
};

}  // Serializer

/// Serializer interface for MDL objects.
///
/// This interface must be implemented by user application to serialize MDL content.
class ISerializer {
public:
    typedef unsigned char Byte;

    /// Write a byte.
    ///
    /// \param b  the byte to write
    virtual void write(Byte b) = 0;

    /// Write an int.
    ///
    /// \param v  the integer to write
    virtual void write_int(int v) = 0;

    /// Write a float.
    ///
    /// \param v  the float to write
    virtual void write_float(float v) = 0;

    /// Write a double.
    ///
    /// \param v  the double to write
    virtual void write_double(double v) = 0;

    /// Write an MDL section tag.
    ///
    /// \param tag  the MDL section tag to write.
    virtual void write_section_tag(Serializer::Serializer_tags tag) = 0;

    /// Write a (general) tag, assuming small values.
    ///
    /// \param tag  the tag to write
    virtual void write_encoded_tag(size_t tag) = 0;

    /// Write a c-string, supports NULL pointer.
    ///
    /// \param s  the string
    virtual void write_cstring(char const *s) = 0;

    /// Write a DB::Tag.
    ///
    /// \param tag  the DB::Tag encoded as 32bit
    virtual void write_db_tag(unsigned tag) = 0;
};

/// Deserializer interface for MDL objects.
///
/// This interface must be implemented by user application to deserialize MDL content.
class IDeserializer {
public:
    typedef unsigned char Byte;

    /// Read a byte.
    virtual Byte read() = 0;

    /// Read an int.
    virtual int read_int() = 0;

    /// Read a float.
    virtual float read_float() = 0;

    /// Read a double.
    virtual double read_double() = 0;

    /// Read an MDL section tag.
    virtual Serializer::Serializer_tags read_section_tag() = 0;

    /// Read a (general) tag, assuming small values.
    virtual size_t read_encoded_tag() = 0;

    /// Read a c-string, supports NULL pointer.
    virtual char const *read_cstring() = 0;

    /// Reads a DB::Tag 32bit encoding.
    virtual unsigned read_db_tag() = 0;
};

}  // mdl
}  // mi

#endif
