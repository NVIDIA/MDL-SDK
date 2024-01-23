/******************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_SYMBOLS_H
#define MDL_COMPILER_GLSL_SYMBOLS_H 1

#include <cstring>
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_cstring_hash.h"

namespace mi {
namespace mdl {
namespace glsl {

///
/// A GLSL symbol.
///
/// A symbol is used to represent "a string" in a GLSL program.
/// For fast lookup, every symbol is unique and has an Id (a number) associated.
/// Two symbols of the same symbol table are equal if and only if there addresses are
/// equal and different otherwise.
///
class Symbol {
    friend class Arena_builder;

public:
    ///
    /// Symbol Id's of predefined symbols.
    ///
    /// Note: partial order is important, must be:
    /// - scalar, vector2, vector3, vector4
    /// - MatrixNx2, MatrixNx3, MatrixNx4
    ///
    enum Predefined_id {
        SYM_ERROR,                  ///< special error symbol
        SYM_OPERATOR,               ///< the Id of ALL operators
        SYM_USER_TYPE_NAME,         ///< set of all user type names

        // types
        SYM_TYPE_FIRST,
        SYM_TYPE_VOID = SYM_TYPE_FIRST,
        SYM_TYPE_BOOL,
        SYM_TYPE_BVEC2,
        SYM_TYPE_BVEC3,
        SYM_TYPE_BVEC4,
        SYM_TYPE_INT,
        SYM_TYPE_IVEC2,
        SYM_TYPE_IVEC3,
        SYM_TYPE_IVEC4,
        SYM_TYPE_UINT,
        SYM_TYPE_UVEC2,
        SYM_TYPE_UVEC3,
        SYM_TYPE_UVEC4,
        SYM_TYPE_INT8_T,
        SYM_TYPE_I8VEC2,
        SYM_TYPE_I8VEC3,
        SYM_TYPE_I8VEC4,
        SYM_TYPE_UINT8_T,
        SYM_TYPE_U8VEC2,
        SYM_TYPE_U8VEC3,
        SYM_TYPE_U8VEC4,
        SYM_TYPE_INT16_T,
        SYM_TYPE_I16VEC2,
        SYM_TYPE_I16VEC3,
        SYM_TYPE_I16VEC4,
        SYM_TYPE_UINT16_T,
        SYM_TYPE_U16VEC2,
        SYM_TYPE_U16VEC3,
        SYM_TYPE_U16VEC4,
        SYM_TYPE_INT32_T,
        SYM_TYPE_I32VEC2,
        SYM_TYPE_I32VEC3,
        SYM_TYPE_I32VEC4,
        SYM_TYPE_UINT32_T,
        SYM_TYPE_U32VEC2,
        SYM_TYPE_U32VEC3,
        SYM_TYPE_U32VEC4,
        SYM_TYPE_INT64_T,
        SYM_TYPE_I64VEC2,
        SYM_TYPE_I64VEC3,
        SYM_TYPE_I64VEC4,
        SYM_TYPE_UINT64_T,
        SYM_TYPE_U64VEC2,
        SYM_TYPE_U64VEC3,
        SYM_TYPE_U64VEC4,
        SYM_TYPE_FLOAT16_T,
        SYM_TYPE_FLOAT,
        SYM_TYPE_VEC2,
        SYM_TYPE_VEC3,
        SYM_TYPE_VEC4,
        SYM_TYPE_DOUBLE,
        SYM_TYPE_DVEC2,
        SYM_TYPE_DVEC3,
        SYM_TYPE_DVEC4,
        SYM_TYPE_MAT2,
        SYM_TYPE_MAT2X2 = SYM_TYPE_MAT2,
        SYM_TYPE_MAT2X3,
        SYM_TYPE_MAT2X4,
        SYM_TYPE_MAT3X2,
        SYM_TYPE_MAT3,
        SYM_TYPE_MAT3X3 = SYM_TYPE_MAT3,
        SYM_TYPE_MAT3X4,
        SYM_TYPE_MAT4X2,
        SYM_TYPE_MAT4X3,
        SYM_TYPE_MAT4,
        SYM_TYPE_MAT4X4 = SYM_TYPE_MAT4,
        SYM_TYPE_DMAT2,
        SYM_TYPE_DMAT2X2 = SYM_TYPE_DMAT2,
        SYM_TYPE_DMAT2X3,
        SYM_TYPE_DMAT2X4,
        SYM_TYPE_DMAT3X2,
        SYM_TYPE_DMAT3,
        SYM_TYPE_DMAT3X3 = SYM_TYPE_DMAT3,
        SYM_TYPE_DMAT3X4,
        SYM_TYPE_DMAT4X2,
        SYM_TYPE_DMAT4X3,
        SYM_TYPE_DMAT4,
        SYM_TYPE_DMAT4X4 = SYM_TYPE_DMAT4,
        SYM_TYPE_ATOMIC_UINT,
        SYM_TYPE_SAMPLER1D,
        SYM_TYPE_SAMPLER2D,
        SYM_TYPE_SAMPLER3D,
        SYM_TYPE_SAMPLERCUBE,
        SYM_TYPE_SAMPLER1DSHADOW,
        SYM_TYPE_SAMPLER2DSHADOW,
        SYM_TYPE_SAMPLERCUBESHADOW,
        SYM_TYPE_SAMPLER1DARRAY,
        SYM_TYPE_SAMPLER2DARRAY,
        SYM_TYPE_SAMPLER1DARRAYSHADOW,
        SYM_TYPE_SAMPLER2DARRAYSHADOW,
        SYM_TYPE_SAMPLERCUBEARRAY,
        SYM_TYPE_SAMPLERCUBEARRAYSHADOW,
        SYM_TYPE_ISAMPLER1D,
        SYM_TYPE_ISAMPLER2D,
        SYM_TYPE_ISAMPLER3D,
        SYM_TYPE_ISAMPLERCUBE,
        SYM_TYPE_ISAMPLER1DARRAY,
        SYM_TYPE_ISAMPLER2DARRAY,
        SYM_TYPE_ISAMPLERCUBEARRAY,
        SYM_TYPE_USAMPLER1D,
        SYM_TYPE_USAMPLER2D,
        SYM_TYPE_USAMPLER3D,
        SYM_TYPE_USAMPLERCUBE,
        SYM_TYPE_USAMPLER1DARRAY,
        SYM_TYPE_USAMPLER2DARRAY,
        SYM_TYPE_USAMPLERCUBEARRAY,
        SYM_TYPE_SAMPLER2DRECT,
        SYM_TYPE_SAMPLER2DRECTSHADOW,
        SYM_TYPE_ISAMPLER2DRECT,
        SYM_TYPE_USAMPLER2DRECT,
        SYM_TYPE_SAMPLERBUFFER,
        SYM_TYPE_ISAMPLERBUFFER,
        SYM_TYPE_USAMPLERBUFFER,
        SYM_TYPE_SAMPLER2DMS,
        SYM_TYPE_ISAMPLER2DMS,
        SYM_TYPE_USAMPLER2DMS,
        SYM_TYPE_SAMPLER2DMSARRAY,
        SYM_TYPE_ISAMPLER2DMSARRAY,
        SYM_TYPE_USAMPLER2DMSARRAY,
        SYM_TYPE_IMAGE1D,
        SYM_TYPE_IIMAGE1D,
        SYM_TYPE_UIMAGE1D,
        SYM_TYPE_IMAGE2D,
        SYM_TYPE_IIMAGE2D,
        SYM_TYPE_UIMAGE2D,
        SYM_TYPE_IMAGE3D,
        SYM_TYPE_IIMAGE3D,
        SYM_TYPE_UIMAGE3D,
        SYM_TYPE_IMAGE2DRECT,
        SYM_TYPE_IIMAGE2DRECT,
        SYM_TYPE_UIMAGE2DRECT,
        SYM_TYPE_IMAGECUBE,
        SYM_TYPE_IIMAGECUBE,
        SYM_TYPE_UIMAGECUBE,
        SYM_TYPE_IMAGEBUFFER ,
        SYM_TYPE_IIMAGEBUFFER,
        SYM_TYPE_UIMAGEBUFFER,
        SYM_TYPE_IMAGE1DARRAY,
        SYM_TYPE_IIMAGE1DARRAY,
        SYM_TYPE_UIMAGE1DARRAY,
        SYM_TYPE_IMAGE2DARRAY,
        SYM_TYPE_IIMAGE2DARRAY,
        SYM_TYPE_UIMAGE2DARRAY,
        SYM_TYPE_IMAGECUBEARRAY,
        SYM_TYPE_IIMAGECUBEARRAY,
        SYM_TYPE_UIMAGECUBEARRAY,
        SYM_TYPE_IMAGE2DMS,
        SYM_TYPE_IIMAGE2DMS,
        SYM_TYPE_UIMAGE2DMS,
        SYM_TYPE_IMAGE2DMSARRAY ,
        SYM_TYPE_IIMAGE2DMSARRAY,
        SYM_TYPE_UIMAGE2DMSARRAY,
        // GL_OES_EGL_image_external extension
        SYM_TYPE_SAMPLEREXTERNALOES,
        SYM_TYPE_LAST = SYM_TYPE_SAMPLEREXTERNALOES,

        // constants
        SYM_CNST_FIRST,
        SYM_CNST_TRUE = SYM_CNST_FIRST,
        SYM_CNST_FALSE,
        SYM_CNST_LAST = SYM_CNST_FALSE,

        // These be last in the given order.
        SYM_USER
    };


public:
    /// Get the name of the symbol.
    char const *get_name() const { return m_name; }

    /// Get the id of the symbol.
    size_t get_id() const { return m_id; }

    /// Returns true if this symbol is predefined.
    bool is_predefined() const { return m_id < SYM_USER; }

public:
    /// Constructor for an operator symbol.
    ///
    /// \param name  the operator name
    explicit Symbol(char const *name);

    /// Constructor for a syntax element symbol.
    ///
    /// \param id    the ID of this symbol
    /// \param name  the name of the symbol
    explicit Symbol(size_t id, char const *name);

private:
    /// The id of this symbol.
    size_t const m_id;

    /// The name of this symbol.
    char const * const m_name;
};

/// The GLSL symbol table.
class Symbol_table
{
    friend class Compilation_unit;
public:
    /// Get an existing Symbol for the given name.
    ///
    /// \param name  the name to lookup
    ///
    /// \return the symbol for this name or NULL
    Symbol *lookup_symbol(char const *name) const;

    /// Get or create a new Symbol for the given name.
    ///
    /// \param name  the name to lookup
    ///
    /// \return the symbol for this name, creates one if not exists
    Symbol *get_symbol(char const *name);

    /// Return the Symbol for a given id or NULL.
    ///
    /// \param id  the id to lookup
    ///
    /// \return the symbol for this id of NULL.
    Symbol *get_symbol_for_id(size_t id) const;

    /// Return the error symbol.
    static Symbol *get_error_symbol();

    /// Return the symbol for an operator.
    ///
    /// \param op  the operator kind
    //Symbol *get_operator_symbol(Expression::Operator op) const;

    /// Return a predefined symbol.
    ///
    /// \param id   the predefined ID
    static Symbol *get_predefined_symbol(Symbol::Predefined_id id);

    /// Get or create a new Symbol for the given user type name.
    ///
    /// \param name  the user type name to lookup
    ///
    /// \return the symbol for this user type name, creates one if not exists
    Symbol *get_user_type_symbol(char const *name);

    /// Create a new anonymous symbol.
    ///
    /// a new unique anonymous symbol
    Symbol *get_anonymous_symbol();

private:
    /// Constructor.
    ///
    /// \param arena  the memory arena that will be used for the symbol table
    explicit Symbol_table(Memory_arena &arena);

private:
    /// Create all predefined symbols.
    void create_predefined();

    /// Create a new symbol for the given Id.
    ///
    /// \param ident  the identifier (text) of this symbol
    /// \param id     the id for this symbol
    Symbol *enter_symbol(char const *ident, size_t id);

    /// Enter a predefined symbol into the table.
    ///
    /// \param sym  the symbol to enter
    void enter_predefined(Symbol *sym);

    /// Create a save copy of a string by putting it into the memory arena.
    ///
    /// \param s  the C-string to internalize
    ///
    /// \return a copy of s that is allocated on the memory arena of this table
    char const *internalize(char const *s);

private:
    /// The builder for symbols.
    Arena_builder m_builder;

    /// Memory arena for internalized strings.
    Memory_arena m_string_arena;

    typedef Arena_hash_map<
        const char *,
        Symbol *,
        cstring_hash,
        cstring_equal_to
    >::Type Symbol_map;

    /// Maps strings to symbols.
    Symbol_map m_symbol_map;

    /// Maps id's to symbols.
    Arena_vector<Symbol *>::Type m_symbols;

    /// Next id.
    size_t m_next_id;

    /// Next anonymous symbol id.
    size_t m_anon_id;
};

}  // glsl
}  // mdl
}  // mi

#endif // MDL_COMPILER_GLSL_SYMBOLS_H
