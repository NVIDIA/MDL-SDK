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

#ifndef MDL_COMPILER_HLSL_SYMBOLS_H
#define MDL_COMPILER_HLSL_SYMBOLS_H 1

#include <cstring>
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_cstring_hash.h"

namespace mi {
namespace mdl {
namespace hlsl {

#define MAKE_VECTOR(SCALAR) \
    SCALAR ## 1, \
    SCALAR ## 2, \
    SCALAR ## 3, \
    SCALAR ## 4

#define MAKE_MATRIX(SCALAR) \
    SCALAR ## 1X1, \
    SCALAR ## 1X2, \
    SCALAR ## 1X3, \
    SCALAR ## 1X4, \
    SCALAR ## 2X1, \
    SCALAR ## 2X2, \
    SCALAR ## 2X3, \
    SCALAR ## 2X4, \
    SCALAR ## 3X1, \
    SCALAR ## 3X2, \
    SCALAR ## 3X3, \
    SCALAR ## 3X4, \
    SCALAR ## 4X1, \
    SCALAR ## 4X2, \
    SCALAR ## 4X3, \
    SCALAR ## 4X4

#define MAKE_ALL(SCALAR) \
    SCALAR,              \
    MAKE_VECTOR(SCALAR), \
    MAKE_MATRIX(SCALAR)

///
/// A HLSL symbol.
///
/// A symbol is used to represent "a string" in a HLSL program.
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
    /// - scalar, vector1, vector2, vector3, vector4
    /// - MatrixNx2, MatrixNx3, MatrixNx4
    ///
    enum Predefined_id {
        SYM_ERROR,                  ///< special error symbol
        SYM_OPERATOR,               ///< the Id of ALL operators
        SYM_KEYWORD,                ///< set of all keywords
        SYM_RESERVED,               ///< reserved words
        SYM_USER_TYPE_NAME,         ///< set of all user type names

        // types
        SYM_TYPE_FIRST,
        SYM_TYPE_VOID = SYM_TYPE_FIRST,
        MAKE_ALL(SYM_TYPE_BOOL),
        MAKE_ALL(SYM_TYPE_INT),
        MAKE_ALL(SYM_TYPE_UINT),
        MAKE_ALL(SYM_TYPE_DWORD),
        MAKE_ALL(SYM_TYPE_HALF),
        MAKE_ALL(SYM_TYPE_FLOAT),
        MAKE_ALL(SYM_TYPE_DOUBLE),
        MAKE_ALL(SYM_TYPE_MIN12INT),
        MAKE_ALL(SYM_TYPE_MIN16INT),
        MAKE_ALL(SYM_TYPE_MIN16UINT),
        MAKE_ALL(SYM_TYPE_MIN10FLOAT),
        MAKE_ALL(SYM_TYPE_MIN16FLOAT),

        SYM_TYPE_SAMPLER,

        SYM_TYPE_TEXTURE,
        SYM_TYPE_TEXTURE1D,
        SYM_TYPE_TEXTURE1DARRAY,
        SYM_TYPE_TEXTURE2D,
        SYM_TYPE_TEXTURE2DARRAY,
        SYM_TYPE_TEXTURE3D,
        SYM_TYPE_TEXTURECUBE,

        SYM_TYPE_LAST = SYM_TYPE_TEXTURECUBE,

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

/// The HLSL symbol table.
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

#undef MAKE_ALL
#undef MAKE_MATRIX
#undef MAKE_VECTOR

}  // hlsl
}  // mdl
}  // mi

#endif // MDL_COMPILER_HLSL_SYMBOLS_H
