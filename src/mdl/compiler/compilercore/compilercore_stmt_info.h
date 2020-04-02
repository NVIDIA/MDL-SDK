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

#ifndef MDL_COMPILERCORE_STMT_INFO_H
#define MDL_COMPILERCORE_STMT_INFO_H 1

#include "compilercore_allocator.h"

namespace mi {
namespace mdl {

/// Analysis info for a statement.
struct Stmt_info {
    unsigned m_reachable_start : 1;  ///< reachable at start
    unsigned m_reachable_exit  : 1;  ///< reachable at end
    unsigned m_case_has_break  : 1;  ///< switch case has a break
    unsigned m_loop_has_break  : 1;  ///< loop has a break

    Stmt_info()
        : m_reachable_start(0)
        , m_reachable_exit(0)
        , m_case_has_break(0)
        , m_loop_has_break(0)
    {}
};

/// Analysis info for an expression.
struct Expr_info {
    unsigned m_has_effect : 1; ///< expression has an effect

    Expr_info()
        : m_has_effect(1)
    {
    }
};


///
/// Contains analysis informations for statements.
///
class Stmt_info_data
{
private:
    /// A Stmt_map contains analysis info for every statement.
    typedef ptr_hash_map<IStatement const, Stmt_info>::Type Stmt_map;

    /// An Expr_map contains analysis info for every expression.
    typedef ptr_hash_map<IExpression const, Expr_info>::Type Expr_map;

public:
    explicit Stmt_info_data(IAllocator *alloc)
    : m_stmt_map(0, Stmt_map::hasher(), Stmt_map::key_equal(), alloc)
    , m_expr_map(0, Expr_map::hasher(), Expr_map::key_equal(), alloc)
    , m_stmt_dummy()
    , m_expr_dummy()
    {
    }

    /// Get the analysis info for a statement.
    ///
    /// \param stmt  the statement
    Stmt_info &get_stmt_info(IStatement const *stmt);

    /// Get the analysis info for a statement.
    ///
    /// \param stmt  the statement
    Stmt_info const &get_stmt_info(IStatement const *stmt) const;

    /// Get the analysis info for an expression.
    ///
    /// \param expr  the expression
    Expr_info &get_expr_info(IExpression const *expr);

    /// Get the analysis info for an expression.
    ///
    /// \param expr  the expression
    Expr_info const &get_expr_info(IExpression const *expr) const;

private:
    /// Stores the statement analysis info.
    Stmt_map m_stmt_map;

    /// Stores the expression analysis info.
    Expr_map m_expr_map;

    /// Dummy entry.
    Stmt_info const m_stmt_dummy;

    /// Dummy entry.
    Expr_info const m_expr_dummy;
};

}  // mdl
}  // mi

#endif
