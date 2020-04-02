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

#ifndef MDL_COMPILERCORE_NAMES_H
#define MDL_COMPILERCORE_NAMES_H 1

#include <mi/mdl/mdl_names.h>

#include "compilercore_cc_conf.h"
#include "compilercore_memory_arena.h"

namespace mi {
namespace mdl {

class Symbol_table;

/// A factory for MDL AST names.
class Name_factory : public IName_factory
{
    friend class Module;
public:
    /// Create a new Symbol.
    ISymbol const *create_symbol(char const *name) MDL_FINAL;

    /// Creates a simple name.
    ///
    /// \param sym           the Symbol of the name
    /// \param start_line    the start line of this name in the input
    /// \param start_column  the start column of this name in the input
    /// \param end_line      the end line of this name in the input
    /// \param end_column    the end column of this name in the input
    ISimple_name *create_simple_name(
        ISymbol const *sym,
        int           start_line = 0,
        int           start_column = 0,
        int           end_line = 0,
        int           end_column = 0) MDL_FINAL;

    /// Creates a new (empty) qualified name.
    ///
    /// \param start_line    the start line of this name in the input
    /// \param start_column  the start column of this name in the input
    /// \param end_line      the end line of this name in the input
    /// \param end_column    the end column of this name in the input
    IQualified_name *create_qualified_name(
        int start_line = 0,
        int start_column = 0,
        int end_line = 0,
        int end_column = 0) MDL_FINAL;

    /// Creates a new type name.
    ///
    /// \param qualified_name  the qualified name of the type name
    /// \param start_line      the start line of this name in the input
    /// \param start_column    the start column of this name in the input
    /// \param end_line        the end line of this name in the input
    /// \param end_column      the end column of this name in the input
    IType_name *create_type_name(
        IQualified_name *qualified_name,
        int             start_line = 0,
        int             start_column = 0,
        int             end_line = 0,
        int             end_column = 0) MDL_FINAL;

private:
    /// Constructor.
    ///
    /// \param symtab  the symbol table for creating symbols
    /// \param arena   a memory arena that is used to allocate objects on
    explicit Name_factory(
        Symbol_table &symtab,
        Memory_arena &arena);

private:
    // non copyable
    Name_factory(Name_factory const &) MDL_DELETED_FUNCTION;
    Name_factory &operator=(Name_factory const &) MDL_DELETED_FUNCTION;

private:
    /// The symbol tabel for creating symbols;
    Symbol_table &m_sym_tab;

    /// Builder for names.
    Arena_builder m_builder;
};

}  // mdl
}  // mi

#endif
