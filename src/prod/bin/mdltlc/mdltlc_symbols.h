/******************************************************************************
 * Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDLTLC_SYMBOLS_H
#define MDLTLC_SYMBOLS_H 1

#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_cstring_hash.h>

#include "mdltlc_exprs.h"

///
/// A mdltlc symbol.
///
/// A symbol is used to represent "a string" in an mdltl file.
/// For fast lookup, every symbol is unique.  Two symbols of the same
/// symbol table are equal if and only if there addresses are equal
/// and different otherwise.
///
class Symbol {
    friend class mi::mdl::Arena_builder;

public:

public:
    /// Get the name of the symbol.
    char const *get_name() const { return m_name; }

public:
    /// Constructor for a syntax element symbol.
    ///
    /// \param id    the ID of this symbol
    /// \param name  the name of the symbol
    explicit Symbol(char const *name);

private:
    /// The name of this symbol.
    char const * const m_name;
};

/// The mdltlc symbol table.
class Symbol_table
{
    friend class Compiler;
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

    /// Get the allocator.
    mi::mdl::IAllocator *get_allocator() const { return m_string_arena.get_allocator(); }

private:
    /// Constructor.
    ///
    /// \param arena  the memory arena that will be used for the symbol table
    explicit Symbol_table(mi::mdl::Memory_arena &arena);

private:
    /// Create a new symbol for the given Id.
    ///
    /// \param ident  the identifier (text) of this symbol
    /// \param id     the id for this symbol
    Symbol *enter_symbol(char const *ident);

    /// Create a save copy of a string by putting it into the memory arena.
    ///
    /// \param s  the C-string to internalize
    ///
    /// \return a copy of s that is allocated on the memory arena of this table
    char const *internalize(char const *s);

private:
    /// The builder for symbols.
    mi::mdl::Arena_builder m_builder;

    /// Memory arena for internalized strings.
    mi::mdl::Memory_arena m_string_arena;

    typedef mi::mdl::Arena_hash_map<
        const char *,
        Symbol *,
        mi::mdl::cstring_hash,
        mi::mdl::cstring_equal_to
    >::Type Symbol_map;

    /// Maps strings to symbols.
    Symbol_map m_symbol_map;

    // Maps id's to symbols.
    mi::mdl::Arena_vector<Symbol *>::Type m_symbols;
};

#endif // MDLTLC_SYMBOLS_H
