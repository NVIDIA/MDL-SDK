/******************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_CHECKER_H
#define MDL_COMPILERCORE_CHECKER_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_printers.h>
#include "compilercore_allocator.h"
#include "compilercore_visitor.h"

namespace mi {
namespace mdl {

class IDefinition;
class IModule;
class IType;
class IMDL;
class Definition_table;
class Generated_code_dag;
class Module;
class Type_factory;
class Value_factory;

/// Base class for code checkers.
class Code_checker
{
protected:
    /// Constructor.
    ///
    /// \param verbose  if true, write a verbose output to stderr
    /// \param printer  the printer for writing error messages, takes ownership
    explicit Code_checker(bool verbose, IPrinter *printer);

    /// Check a value factory for soundness.
    ///
    /// \param factory  the value factory to check
    void check_value_factory(
        Value_factory const *factory);

    /// Check a given value for soundness.
    ///
    /// \param owner  the owner factory of the value to check
    /// \param value  the value to check
    void check_value(
        Value_factory const *owner,
        IValue const        *value);

    /// Check a given type for soundness.
    ///
    /// \param owner  the owner factory of the type to check
    /// \param type   the type to check
    void check_type(
        Type_factory const *owner,
        IType const        *type);

    /// Check a given definition for soundness.
    ///
    /// \param owner  the owner definition table of the definition to check
    /// \param def   the definition to check
    void check_definition(
        Definition_table const *owner,
        IDefinition const      *def);

    /// Report an error.
    ///
    /// \param msg  the error message
    void report(
        char const *msg);

    /// Get the error count.
    size_t get_error_count() const { return m_error_count; }

protected:
    /// If true, verbose mode is activated
    bool const m_verbose;

private:
    /// The current error count.
    size_t m_error_count;

protected:
    /// The printer for error reports.
    mi::base::Handle<IPrinter> m_printer;
};

/// A checker for Modules.
class Module_checker : protected Code_checker, private Module_visitor
{
    typedef Code_checker Base;
public:
    /// Check a module.
    ///
    /// \param compiler  the MDL compiler (that owns the module)
    /// \param module    the module to check
    /// \param verbose   if true, write a verbose output the stderr
    ///
    /// \return true on success
    static bool check(
        IMDL const    *compiler,
        IModule const *module,
        bool          verbose);

private:
    /// Default post visitor for expressions.
    ///
    /// \param expr  the expression
    IExpression *post_visit(IExpression *expr) MDL_FINAL;

    // Post visitor for literal expressions.
    ///
    /// \param expr  the expression
    IExpression *post_visit(IExpression_literal *expr) MDL_FINAL;

    // Post visitor for reference expressions.
    ///
    /// \param expr  the expression
    IExpression *post_visit(IExpression_reference *expr) MDL_FINAL;

private:
    /// Constructor.
    ///
    /// \param module    the module to check
    /// \param verbose  if true, write a verbose output to stderr
    /// \param printer  if non-NULL, the printer for writing error messages, takes ownership
    explicit Module_checker(
        Module const *module,
        bool         verbose,
        IPrinter     *printer);

private:
    /// The value factory of the module to check.
    Value_factory const *m_vf;

    /// The type factory of the module to check.
    Type_factory const *m_tf;

    /// The definition table of the module to check.
    Definition_table const *m_deftab;
};

/// Helper class to check that our input is really a Tree, and not a DAG.
/// Note: technically, it is safe to share constants ans any sub-ASTs that do NOT
// contain definitions, but this is much harder to detect, so for now we check that the
// complete AST is a Tree.
class Tree_checker : public Code_checker, private Module_visitor {
public:
    /// Checker.
    ///
    /// \param compiler  the MDL compiler (that owns the module)
    /// \param module    the module to check
    /// \param verbose   if true, write a verbose output the stderr
    ///
    /// \return true on success
    static bool check(
        IMDL const    *compiler,
        IModule const *module,
        bool          verbose);

private:
    /// Constructor.
    ///
    /// \param alloc   the allocator
    Tree_checker(
        IAllocator *alloc,
        IPrinter   *printer,
        bool       verbose);

private:
    void post_visit(ISimple_name *sname) MDL_FINAL;

    void post_visit(IQualified_name *qname) MDL_FINAL;

    void post_visit(IType_name *tname) MDL_FINAL;

    IExpression *post_visit(IExpression *expr) MDL_FINAL;

    void post_visit(IStatement *stmt) MDL_FINAL;

    void post_visit(IDeclaration *decl) MDL_FINAL;

private:
    typedef ptr_hash_set<void const>::Type Ptr_set;

    Ptr_set m_ast_set;
};

}  // mdl
}  // mi

#endif  // MDL_COMPILERCORE_CHECKER_H
