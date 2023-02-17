/******************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_GLSL_ANALYSIS_H
#define MDL_COMPILER_GLSL_ANALYSIS_H 1

#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_streams.h"

#include "compiler_glsl_cc_conf.h"
#include "compiler_glsl_definitions.h"
#include "compiler_glsl_errors.h"
#include "compiler_glsl_locations.h"
#include "compiler_glsl_symbols.h"
#include "compiler_glsl_tools.h"
#include "compiler_glsl_type_cache.h"
#include "compiler_glsl_visitor.h"

namespace mi {
namespace mdl {
namespace glsl {

// forward
class Compiler;
class Compilation_unit;

///
/// A base class for all semantic analysis passes.
///
class Analysis : protected CUnit_visitor {
public:
    /// Get the allocator of this analysis path.
    IAllocator *get_allocator() const { return m_builder.get_allocator(); }

protected:
    /// Creates a new error.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  error message parameter inserts
    void error(
        GLSL_compiler_error code,
        Err_location const  &loc,
        Error_params const  &params);

    /// Creates a new warning.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  warning message parameter inserts
    void warning(
        GLSL_compiler_error code,
        Err_location const  &loc,
        Error_params const  &params);


    /// Add a note to the last error/warning.
    ///
    /// \param code    the error code
    /// \param loc     location of the error
    /// \param params  note message parameter inserts
    void add_note(
        GLSL_compiler_error code,
        Err_location const  &loc,
        Error_params const  &params);

    /// Add a compiler note for a previous definition.
    ///
    /// \param prev_def  the previous definition
    void add_prev_definition_note(
        Definition const *prev_def);

    /// Issue an error for some previously defined entity.
    ///
    /// \param kind    the kind of the current entity
    /// \param def     the previous definition
    /// \param loc     location of the current definition
    /// \param err     the error to issue
    void err_redeclaration(
        Definition::Kind    kind,
        Definition const    *def,
        Err_location const  &loc,
        GLSL_compiler_error err);

    /// Issue a warning if some previously defined entity is shadowed.
    ///
    /// \param def     the previous definition
    /// \param loc     location of the current definition
    /// \param warn    the warning to issue
    void warn_shadow(
        Definition const    *def,
        Location const      &loc,
        GLSL_compiler_error warn);

    /// Get the one and only "error definition" of the processed compilation unit.
    Definition *get_error_definition() const;

    // Find the definition for a name.
    Definition *find_definition_for_name(Name *name);

    /// Apply a type qualifier to a given type.
    ///
    /// \param qual  the qualifier to apply
    /// \param type  the unqualified type
    ///
    /// \return the qualified type
    Type *qualify_type(Type_qualifier *qual, Type *type);

    /// Apply array specifiers to a type.
    ///
    /// \param type  the type
    /// \param as    the array specifiers list
    Type *apply_array_specifiers(Type *type, Array_specifiers const &as);

protected:
    /// Constructor.
    ///
    /// \param compiler  the GLSL compiler
    /// \param unit      the current unit that is analyzed
    Analysis(
        Compiler         &compiler,
        Compilation_unit &unit);

private:
    // non copyable
    Analysis(Analysis const &) GLSL_DELETED_FUNCTION;
    Analysis &operator=(Analysis const &) GLSL_DELETED_FUNCTION;

private:
    /// Format a message.
    ///
    /// \param code    the GLSL compilation error code
    /// \param params  error parameters
    string format_msg(
        GLSL_compiler_error code,
        Error_params const  &params);

protected:
    /// The builder.
    Allocator_builder m_builder;

    /// The GLSL compiler.
    Compiler &m_compiler;

    /// The current compilation unit.
    Compilation_unit &m_unit;

    /// The symbol table, retrieved from the compilation unit.
    Symbol_table &m_st;

    /// The type cache.
    Type_cache m_tc;

    /// The definition table for entities.
    Definition_table &m_def_tab;

    /// The index of the last generated error message.
    size_t m_last_msg_idx;

    /// A string buffer used for error messages.
    mi::base::Handle<Buffer_output_stream> m_string_buf;

    /// Printer for error messages.
    mi::base::Handle<IPrinter> m_printer;

    /// Bitset containing disabled warnings.
    Raw_bitset<MAX_ERROR_NUM + 1> m_disabled_warnings;

    /// Bitset containing warnings treated as errors.
    Raw_bitset<MAX_ERROR_NUM + 1> m_warnings_are_errors;

    /// If true, all warnings are errors.
    bool m_all_warnings_are_errors;

    /// If true, compile in strict mode.
    bool m_strict;
};

///
/// The combined name and type analysis.
///
class NT_analysis GLSL_FINAL : public Analysis {
    typedef Analysis Base;
public:
    /// Run the name and type analysis on the current unit.
    ///
    /// \param compiler  the GLSL compiler
    /// \param unit      the current unit that is analyzed
    static void run(
        Compiler         &compiler,
        Compilation_unit &unit);

private:
    /// Returns the definition of a symbol at the at a given scope.
    ///
    /// \param sym    the symbol
    /// \param scope  the scope
    Definition *get_definition_at_scope(
        Symbol *sym,
        Scope  *scope) const;

    /// Returns the definition of a symbol at the current scope only.
    ///
    /// \param sym  the symbol
    Definition *get_definition_at_scope(Symbol *sym) const;

    /// Return the Type from a Type_name, handling errors if the Type_name
    /// does not name a type.
    ///
    /// \param type_name  an type name
    ///
    /// \return the type, if type_name does not name a type returns the error type
    ///         and enters it into type_name
    Type *as_type(Type_name *type_name);


    /// end of a name
    void post_visit(Name *n) GLSL_FINAL;

    /// start of a type name
    bool pre_visit(Type_name *tn) GLSL_FINAL;

    /// start of compound statement
    bool pre_visit(Stmt_compound *block) GLSL_FINAL;

    /// end of compound statement
    void post_visit(Stmt_compound *block) GLSL_FINAL;

    /// start of a for statement
    bool pre_visit(Stmt_for *for_stmt) GLSL_FINAL;

    // end of a for statement
    void post_visit(Stmt_for *for_stmt) GLSL_FINAL;

    /// End of a invalid declaration.
    void post_visit(Declaration_invalid *decl) GLSL_FINAL;

    // Start of a struct declaration.
    bool pre_visit(Declaration_struct *sdecl) GLSL_FINAL;

    /// Start of a variable declaration.
    bool pre_visit(Declaration_variable *vdecl) GLSL_FINAL;

    /// Start of a function.
    bool pre_visit(Declaration_function *fkt_decl) GLSL_FINAL;

private:
    /// Constructor.
    ///
    /// \param compiler  the GLSL compiler
    /// \param unit      the current unit that is analyzed
    NT_analysis(
        Compiler         &compiler,
        Compilation_unit &unit);

private:
};

/// Returns true for error definitions.
///
/// \param def  the definition to check
extern inline bool is_error(Definition const *def)
{
    Type *type = def->get_type();
    if (is<Type_error>(type))
        return true;
    return def->get_symbol()->get_id() == Symbol::SYM_ERROR;
}

/// Returns true for error expressions.
///
/// \param expr  the expression to check
extern inline bool is_error(Expr const *expr)
{
    Type *type = expr->get_type();
    return is<Type_error>(type);
}

/// Returns true for error names.
extern inline bool is_error(Name const *name)
{
    return name->get_symbol()->get_id() == Symbol::SYM_ERROR;
}

/// Debug helper: Dump the AST of a compilation unit.
///
/// \param unit  the unit to dump
void dump_ast(ICompilation_unit const *unit);

/// Debug helper: Dump the AST of a declaration.
///
/// \param decl  the declaration to dump
void dump_ast(Declaration const *decl);

/// Debug helper: Dump the AST of an expression.
///
/// \param expr  the expression to dump
void dump_expr(Expr const *expr);

/// Debug helper: Dump a definition.
///
/// \param def  the definition to dump
void dump_def(Definition const *def);

}  // glsl
}  // mdl
}  // mi

#endif
