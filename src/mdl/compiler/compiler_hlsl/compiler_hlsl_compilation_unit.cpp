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

#include "pch.h"

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"

#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_optimizer.h"
#include "compiler_hlsl_tools.h"

namespace mi {
namespace mdl {
namespace hlsl {

// Constructor.
Compilation_unit::Compilation_unit(
    IAllocator      *alloc,
    char const      *file_name)
: Base(alloc)
, m_arena(alloc)
, m_filename(file_name != NULL ? Arena_strdup(m_arena, file_name) : NULL)
, m_is_valid(false)
, m_sym_tab(m_arena)
, m_type_factory(m_arena, m_sym_tab)
, m_value_factory(m_arena, m_type_factory)
, m_decl_factory(m_arena)
, m_expr_factory(m_arena, m_value_factory)
, m_stmt_factory(m_arena)
, m_def_tab(*this)
, m_msgs(alloc, m_filename)
, m_decls()
, m_ref_files(alloc)
{
}

// Get the absolute name of the file from which this unit was loaded.
char const *Compilation_unit::get_filename() const
{
    return m_filename;
}

// Get the file name of a referenced file by its ID.
char const *Compilation_unit::get_filename_by_id(unsigned file_id) const
{
    if (file_id == 0)
        return get_filename();
    if (size_t(file_id - 1u) < m_ref_files.size())
        return m_ref_files[file_id - 1u];
    return NULL;
}

// Get the number of referenced files.
unsigned Compilation_unit::get_filename_id_count() const
{
    return 1 + unsigned(m_ref_files.size());
}

// Analyze the unit.
bool Compilation_unit::analyze(
    ICompiler &compiler)
{
    // FIXME: no semantic analysis yet, assume it IS valid
    m_is_valid = true;

    if (is_valid()) {
        Optimizer::run(get_allocator(), impl_cast<Compiler>(compiler), *this, 3);
    }

    return false;
}

// Check if the module has been analyzed.
bool Compilation_unit::is_analyzed() const {
    return false;
}

// Check if the module contents are valid.
bool Compilation_unit::is_valid() const
{
    return m_is_valid;
}

// Get the Declaration_factory factory.
Decl_factory &Compilation_unit::get_declaration_factory()
{
    return m_decl_factory;
}

// Get the expression factory.
Expr_factory &Compilation_unit::get_expression_factory()
{
    return m_expr_factory;
}

// Get the statement factory.
Stmt_factory &Compilation_unit::get_statement_factory()
{
    return m_stmt_factory;
}

// Get the type factory.
Type_factory &Compilation_unit::get_type_factory()
{
    return m_type_factory;
}

// Get the value factory.
Value_factory &Compilation_unit::get_value_factory()
{
    return m_value_factory;
}

// Get the symbol table of this module.
Symbol_table &Compilation_unit::get_symbol_table()
{
    return m_sym_tab;
}

// Get the definition table of this module.
Definition_table &Compilation_unit::get_definition_table()
{
    return m_def_tab;
}

// Get the compiler messages of this compilation unit.
Messages const &Compilation_unit::get_messages() const
{
    return m_msgs;
}

// Add a new declaration to the end of the declaration list.
void Compilation_unit::add_decl(Declaration *decl)
{
    m_decls.push(decl);
}

// Register a new file name.
unsigned Compilation_unit::register_filename(char const *fname)
{
    char const *p = Arena_strdup(m_arena, fname);
    m_ref_files.push_back(p);
    return unsigned(m_ref_files.size());
}

}  // hlsl
}  // mdl
}  // mi
