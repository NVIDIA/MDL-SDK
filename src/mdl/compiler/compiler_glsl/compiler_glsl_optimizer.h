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

#ifndef MDL_COMPILER_GLSL_OPTIMIZER_H
#define MDL_COMPILER_GLSL_OPTIMIZER_H 1

#include <mdl/compiler/compilercore/compilercore_allocator.h>

#include "compiler_glsl_cc_conf.h"
#include "compiler_glsl_exprs.h"

namespace mi {
namespace mdl {
namespace glsl {

// forward
class Compiler;
class Compilation_unit;
class Declaration;
class Expr_factory;
class Stmt;
class Stmt_factory;
class Value_factory;

/// Very simple GLSL optimizer.
class Optimizer {
public:
    /// Run the optimizer on this compilation unit.
    ///
    /// \param alloc           the allocator to be used
    /// \param compiler        the current compiler
    /// \param unit            the compilation unit to optimize
    /// \param opt_level       optimization level
    static void run(
        IAllocator       *alloc,
        Compiler         &compiler,
        Compilation_unit &unit,
        int              opt_level);

    /// Checks if two given expressions are semantically the same.
    bool same_expr(
        Expr *a,
        Expr *b) const;

    /// Creates an unary expression.
    Expr *create_unary(
        Expr_unary::Operator op,
        Expr                 *arg,
        Location const       &pos);

    /// Optimize vector constructors.
    Expr *optimize_vector_constructor(
        Expr_call *constr);

    /// Optimize calls.
    Expr *optimize_call(
        Expr_call *call);

    /// Execute a function on the bodies of all GLSL functions.
    ///
    /// \param body_func  execute on all statements of a function body
    /// \param expr_func  execute on all expressions of a function body
    void run_on_function(
        Stmt *(Optimizer:: *body_func)(Stmt *),
        Expr *(Optimizer:: *expr_func)(Expr *));

    /// Run local optimizations.
    Declaration *local_opt(
        Declaration *decl);

    /// Run local optimizations.
    Stmt *local_opt(
        Stmt *stmt);

    /// Run local optimizations.
    Expr *local_opt(
        Expr *expr);

    /// Run local optimizations.
    void local_opt();

    /// Check if the given binary expression is a vector swizzle.
    ///
    /// \param binary  a binary expression
    /// \param lhs     the left argument of binary
    /// \param rhs     the right argument of binary
    static bool is_swizzle(
        Expr_binary *binary,
        Expr *lhs,
        Expr *rhs);

    /// Optimize a swizzle lhs.rhs.
    ///
    /// \param res_type  the result type
    /// \param lhs       left side of the swizzle
    /// \param rhs       the swizzle
    Expr *optimize_swizzle(
        Type *res_type,
        Expr *lhs,
        Expr *rhs);

private:
    /// Constructor.
    ///
    /// \param alloc           the allocator to be used
    /// \param compiler        the current compiler
    /// \param unit            the compilation unit to optimize
    /// \param opt_level       optimization level
    Optimizer(
        IAllocator *alloc,
        Compiler &compiler,
        Compilation_unit &unit,
        int              opt_level);

private:
    // no copy constructor.
    Optimizer(Optimizer const &) GLSL_DELETED_FUNCTION;
    // no assignment.
    Optimizer &operator=(Optimizer const &) GLSL_DELETED_FUNCTION;

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// The GLSL compiler.
    Compiler &m_compiler;

    /// The current compilation unit.
    Compilation_unit &m_unit;

    /// The statement factory of the current unit.
    Stmt_factory &m_sf;

    /// The Expression factory of the current unit.
    Expr_factory &m_ef;

    /// The Declaration factory of the current unit.
    Decl_factory &m_df;

    /// The value factory of the current unit.
    Value_factory &m_value_factory;

    /// The symbol table of the current unit.
    Symbol_table &m_symtab;

    /// Current optimizer level.
    int m_opt_level;
};

}  // glsl
}  // mdl
}  // mi

#endif
