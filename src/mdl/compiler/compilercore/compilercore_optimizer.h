/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_OPTIMIZER_H
#define MDL_COMPILERCORE_OPTIMIZER_H 1

#include <mi/mdl/mdl_expressions.h>
#include "compilercore_stmt_info.h"

namespace mi {
namespace mdl {

class MDL;
class Module;
class Call_graph;
class Statement_factory;
class Expression_factory;
class Value_factory;
class NT_analysis;
class Position;
class Thread_context;

class Optimizer {
public:
    /// Run the optimizer on this module.
    ///
    /// \param compiler        the current compiler
    /// \param module          the module to analyze
    /// \param ctx             the current thread context
    /// \param nt_ana          the name and type analysis pass
    /// \param stmt_info_data  analysis data for statements
    static void run(
        MDL                  *compiler,
        Module               &module,
        Thread_context       &ctx,
        NT_analysis          &nt_ana,
        Stmt_info_data const &stmt_info_data);

private:
    // Creates an unary expression.
    IExpression_unary *create_unary(
        IExpression_unary::Operator op,
        IExpression const *arg,
        Position const    &pos);

    // Creates a binary expression.
    IExpression_binary *create_binary(
        IExpression_binary::Operator op,
        IExpression const *left,
        IExpression const *right,
        Position const    &pos);

    /// Execute a function on the bodies of all MDL functions.
    void run_on_function(void (Optimizer::* func)(IStatement *));

    /// Execute a function on the bodies of all MDL functions and one on the default initializers.
    ///
    /// \param body_func  function to execute on the body all a MDL function
    /// \param expr_func  function to execute on the default initializers of
    ///                   a MDL function's parameter
    void run_on_function(
        IStatement const *(Optimizer::* body_func)(IStatement const *),
        IExpression const *(Optimizer::* expr_func)(IExpression const *));

    /// Remove unused functions from the AST.
    void remove_unused_functions();

    /// Remove dead code from functions.
    void remove_dead_code();

    /// Remove dead child-statements.
    void remove_dead_code(IStatement *stmt);

    /// Promote a result to the given type (explicitly).
    IExpression const *promote(
        IExpression const *expr,
        IType const       *type);

    /// Run local optimizations.
    void local_opt();

    /// Run local optimizations.
    IDeclaration const *local_opt(IDeclaration const *decl);

    /// Run local optimizations.
    IStatement const *local_opt(IStatement const *stmt);

    /// Run local optimizations.
    IExpression const *local_opt(IExpression const *expr);

    /// Inline the given call.
    ///
    /// \param call  the call
    ///
    /// \return NULL if call is to complex, else the new expression
    IExpression const *do_inline(IExpression_call *call);

private:
    /// Constructor.
    Optimizer(
        MDL                  *compiler,
        Module               &module,
        NT_analysis          &nt_ana,
        Stmt_info_data const &stmt_info_data,
        int                  opt_level);

private:
    /// The MDL compiler.
    MDL *m_compiler;

    /// The current module.
    Module &m_module;

    // The name and type analysis pass
    NT_analysis &m_nt_ana;

    /// The statement factory of the current module.
    Statement_factory &m_stmt_factory;

    /// The Expression factory of the current module.
    Expression_factory &m_expr_factory;

    // The value factory of the current module.
    Value_factory &m_value_factory;

    /// Statement analysis data.
    Stmt_info_data const &m_stmt_info_data;

    /// Current optimizer level.
    int m_opt_level;
};

}  // mdl
}  // mi

#endif  // MDL_COMPILERCORE_OPTIMIZER_H
