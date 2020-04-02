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

#ifndef MDL_COMPILERCORE_SYMBOLS_H
#define MDL_COMPILERCORE_SYMBOLS_H 1

#include <cstring>

#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_expressions.h>

#include "compilercore_cc_conf.h"
#include "compilercore_memory_arena.h"
#include "compilercore_cstring_hash.h"

namespace mi {
namespace mdl {

class Factory_serializer;
class Factory_deserializer;

/// Implementation of a Symbol.
class Symbol : public ISymbol
{
    typedef ISymbol Base;
    friend class Arena_builder;
public:

    /// Get the name of the symbol.
    char const *get_name() const MDL_FINAL;

    /// Get the id of the symbol.
    size_t get_id() const MDL_FINAL;

    /// Returns true if this symbol is predefined.
    bool is_predefined() const;

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

/// Implementation of the Symbol table.
class Symbol_table : public ISymbol_table
{
    friend class Module;
    friend class Generated_code_dag;
    friend class Lambda_function;
public:
    /// Create a symbol.
    ///
    /// \param  name        The name of the symbol.
    ///
    /// \returns            The symbol.
    ISymbol const *create_symbol(char const *name) MDL_FINAL;

    /// Create a symbol for the given user type name.
    ///
    /// \param name         The user type name.
    ///
    /// \returns            The symbol.
    ///
    /// \note Use this symbols only for the name of user types!
    ///       Currently this method is exactly the same as \c create_shared_symbol()
    ISymbol const *create_user_type_symbol(char const *name) MDL_FINAL;

    /// Create a shared symbol (no unique ID).
    ///
    /// \param name         The shared symbol name.
    ///
    /// \returns            The symbol.
    ISymbol const *create_shared_symbol(char const *name) MDL_FINAL;

    // --------------------------- not interface methods ---------------------------

    /// Get an existing Symbol for the given name.
    ///
    /// \param name  the name to lookup
    ///
    /// \return the symbol for this name or NULL
    ISymbol const *lookup_symbol(char const *name) const;

    /// Get an existing operator Symbol for the given name.
    ///
    /// \param name      the operator name to lookup
    /// \param n_params  the number of operator parameters
    ///
    /// \return the symbol for this name or NULL
    ISymbol const *lookup_operator_symbol(
        char const *name,
        size_t     n_params) const;

    /// Get or create a new Symbol for the given name.
    ///
    /// \param name  the name to lookup
    ///
    /// \return the symbol for this name, creates one if not exists
    ISymbol const *get_symbol(char const *name);

    /// Return the Symbol for a given id or NULL.
    ///
    /// \param id  the id to lookup
    ///
    /// \return the symbol for this id of NULL.
    ISymbol const *get_symbol_for_id(size_t id) const;

    /// Return the error symbol.
    ISymbol const *get_error_symbol() const;

    /// Find the equal symbol of this symbol table if it exists.
    ///
    /// \param other  another symbol, typically from another symbol table
    ///
    /// \return the equal symbol of this symbol table if exists or NULL
    ISymbol const *find_equal_symbol(ISymbol const *other) const;

    /// Return the symbol for an operator.
    ///
    /// \param op  the operator kind
    ISymbol const *get_operator_symbol(IExpression::Operator op) const;

    /// Return a predefined symbol.
    ///
    /// \param id   the predefined ID
    static ISymbol const *get_predefined_symbol(ISymbol::Predefined_id id);

    /// Get or create a new Symbol for the given user type name.
    ///
    /// \param name  the user type name to lookup
    ///
    /// \return the symbol for this user type name, creates one if not exists
    ///
    /// \note Currently the same as get_shared_symbol().
    ISymbol const *get_user_type_symbol(char const *name);

    /// Get or create a new shared Symbol for the given name.
    ///
    /// \param name  the shared symbol name to lookup
    ///
    /// \return the symbol for this  name, creates one if not exists
    ISymbol const *get_shared_symbol(char const *name);

public:
    /// Constructor.
    ///
    /// \param arena  the memory arena that will be used for the symbol table
    explicit Symbol_table(Memory_arena &arena);

private:
    // non copyable
    Symbol_table(Symbol_table const &) MDL_DELETED_FUNCTION;
    Symbol_table &operator=(Symbol_table const &) MDL_DELETED_FUNCTION;

private:
    /// Create all predefined symbols.
    void create_predefined();

    /// Register all predefined symbols in the serializer.
    ///
    /// \param serializer  the factory serializer
    void register_predefined(Factory_serializer &serializer) const;

    /// Register all predefined symbols in the deserializer.
    ///
    /// \param deserializer  the factory deserializer
    void register_predefined(Factory_deserializer &deserializer) const;

    /// Create a new symbol for the given Id.
    ///
    /// \param ident  the identifier (text) of this symbol
    /// \param id     the id for this symbol
    ISymbol const *enter_symbol(char const *ident, size_t id);

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

    /// Serialize the symbol table.
    ///
    /// \param serializer  the module serializer
    void serialize(Factory_serializer &serializer) const;

    /// Deserialize the symbol table.
    ///
    /// \param deserializer  the module deserializer
    void deserialize(Factory_deserializer &deserializer);

private:
    /// The builder for symbols.
    Arena_builder m_builder;

    /// Memory arena for internalized strings.
    Memory_arena m_string_arena;

    typedef Arena_hash_map<
        char const *,
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
};

}  // mdl
}  // mi

#endif
