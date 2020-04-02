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

#ifndef MDL_COMPILER_HLSL_COMPILATION_UNIT_H
#define MDL_COMPILER_HLSL_COMPILATION_UNIT_H 1

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_memory_arena.h"

#include "compiler_hlsl_symbols.h"
#include "compiler_hlsl_types.h"
#include "compiler_hlsl_values.h"
#include "compiler_hlsl_declarations.h"
#include "compiler_hlsl_definitions.h"
#include "compiler_hlsl_exprs.h"
#include "compiler_hlsl_stmts.h"
#include "compiler_hlsl_messages.h"

namespace mi {
namespace mdl {
namespace hlsl {

class ICompiler;

/// Interface of the HLSL compilation unit.
class ICompilation_unit : public
    mi::base::Interface_declare<0x2a4d1ef3,0x019a,0x4da7,0x84,0xea,0x6c,0x8e,0xd4,0x90,0xb8,0x90,
    mi::base::IInterface>
{
public:
    /// Get the absolute name of the file from which this unit was loaded.
    ///
    /// \returns    The absolute name of the file from which the module was loaded,
    ///             or null if no such file exists.
    virtual char const *get_filename() const = 0;

    /// Get the file name of a referenced file by its ID.
    ///
    /// \param file_id   the ID of the referenced file
    ///
    /// \returns  the referenced file name or NULL if the ID is invalid.
    ///
    /// \note the result for ID == 0 is always the same as get_filename()
    virtual char const *get_filename_by_id(unsigned file_id) const = 0;

    /// Get the number of referenced files.
    virtual unsigned get_filename_id_count() const = 0;

    /// Analyze the unit.
    ///
    /// \param compiler  the HLSL compiler
    ///
    /// \returns      True if the module is valid and false otherwise.
    virtual bool analyze(
        ICompiler &compiler) = 0;

    /// Check if the module has been analyzed.
    virtual bool is_analyzed() const = 0;

    /// Check if the module contents are valid.
    virtual bool is_valid() const = 0;

    /// Get the Declaration_factory factory.
    virtual Decl_factory &get_declaration_factory() = 0;

    /// Get the expression factory.
    virtual Expr_factory &get_expression_factory() = 0;

    /// Get the statement factory.
    virtual Stmt_factory &get_statement_factory() = 0;

    /// Get the type factory.
    virtual Type_factory &get_type_factory() = 0;

    /// Get the value factory.
    virtual Value_factory &get_value_factory() = 0;

    /// Get the symbol table of this module.
    virtual Symbol_table &get_symbol_table() = 0;

    /// Get the definition table of this module.
    virtual Definition_table &get_definition_table() = 0;

    /// Get the compiler messages of this compilation unit.
    virtual Messages const &get_messages() const = 0;
};

/// Implementation of a compilation unit.
class Compilation_unit : public Allocator_interface_implement<ICompilation_unit>
{
    typedef Allocator_interface_implement<ICompilation_unit> Base;
    friend class mi::mdl::Allocator_builder;
    friend class Compiler;
public:
    typedef Declaration_list::iterator       iterator;
    typedef Declaration_list::const_iterator const_iterator;

    /// Get the absolute name of the file from which this unit was loaded.
    ///
    /// \returns    The absolute name of the file from which the module was loaded,
    ///             or null if no such file exists.
    char const *get_filename() const HLSL_FINAL;

    /// Get the file name of a referenced file by its ID.
    ///
    /// \param file_id   the ID of the referenced file
    ///
    /// \returns  the referenced file name or NULL if the ID is invalid.
    ///
    /// \note the result for ID == 0 is always the same as get_filename()
    char const *get_filename_by_id(unsigned file_id) const HLSL_FINAL;

    /// Get the number of referenced files.
    unsigned get_filename_id_count() const HLSL_FINAL;

    /// Analyze the unit.
    ///
    /// \param compiler  the HLSL compiler
    ///
    /// \returns      True if the module is valid and false otherwise.
    bool analyze(
        ICompiler &compiler) HLSL_FINAL;

    /// Check if the module has been analyzed.
    bool is_analyzed() const HLSL_FINAL;

    /// Check if the module contents are valid.
    bool is_valid() const HLSL_FINAL;

    /// Get the Declaration_factory factory.
    Decl_factory &get_declaration_factory() HLSL_FINAL;

    /// Get the expression factory.
    Expr_factory &get_expression_factory() HLSL_FINAL;

    /// Get the statement factory.
    Stmt_factory &get_statement_factory() HLSL_FINAL;

    /// Get the type factory.
    Type_factory &get_type_factory() HLSL_FINAL;

    /// Get the value factory.
    Value_factory &get_value_factory() HLSL_FINAL;

    /// Get the symbol table of this module.
    Symbol_table &get_symbol_table() HLSL_FINAL;

    /// Get the definition table of this module.
    Definition_table &get_definition_table() HLSL_FINAL;

    /// Get the compiler messages of this compilation unit.
    Messages const &get_messages() const HLSL_FINAL;

    // --------------------------- non interface methods ---------------------------

    /// Get the compiler messages of this compilation unit.
    Messages &get_messages() { return m_msgs; }

    /// Add a new declaration to the end of the declaration list.
    ///
    /// \param decl  the declaration to add
    void add_decl(Declaration *decl);

    /// Get the first declaration.
    iterator decl_begin() { return m_decls.begin(); }

    /// Get the end declaration.
    iterator decl_end() { return m_decls.end(); }

    /// Get the first declaration.
    const_iterator decl_begin() const { return m_decls.begin(); }

    /// Get the end declaration.
    const_iterator decl_end() const { return m_decls.end(); }

    /// Register a new file name.
    unsigned register_filename(char const *fname);

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_arena.get_allocator(); }

private:
    /// Constructor.
    ///
    /// \param alloc        the allocator
    /// \param file_name    the file name of the module
    explicit Compilation_unit(
        IAllocator      *alloc,
        char const      *file_name);

private:
    /// The memory arena of this module, use to allocate all elements of this module.
    Memory_arena m_arena;

    /// The name of the file from which the module was loaded.
    char const *m_filename;

    /// Set if this module is valid.
    bool m_is_valid;

    /// The symbol table of this module.
    Symbol_table m_sym_tab;

    /// The type factory of this module.
    Type_factory m_type_factory;

    /// The value factory of this module.
    Value_factory m_value_factory;

    /// The declaration factory of this module.
    Decl_factory m_decl_factory;

    /// The expression factory of this module.
    Expr_factory m_expr_factory;

    /// The statement factory of this module.
    Stmt_factory m_stmt_factory;

    /// The definition table of this module;
    Definition_table m_def_tab;

    /// Compiler messages.
    Messages m_msgs;

    /// The AST root.
    Declaration_list m_decls;

    /// Referenced files.
    vector<char const *>::Type m_ref_files;
};

}  // hlsl
}  // mdl
}  // mi

#endif
