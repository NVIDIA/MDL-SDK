/******************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_OPTIMIZER_H
#define MDL_COMPILER_HLSL_OPTIMIZER_H 1

#include "compiler_hlsl_cc_conf.h"

namespace mi {
namespace mdl {
namespace hlsl {

class Compiler;
class Compilation_unit;
class Expr_factory;
class Stmt_factory;
class Value_factory;

/// This class implements the AST optimizer for the HLSL compiler.
class Optimizer {
public:
    /// Run the optimizer on this compilation unit.
    ///
    /// \param compiler        the current compiler
    /// \param unit            the compilation unit to optimize
    /// \param opt_level       optimization level
    static void run(
        Compiler          &compiler,
        Compilation_unit  &unit,
        int              opt_level);

private:
    /// Constructor.
    ///
    /// \param compiler        the current compiler
    /// \param unit            the compilation unit to optimize
    /// \param opt_level       optimization level
    Optimizer(
        Compiler         &compiler,
        Compilation_unit &unit,
        int              opt_level);

private:
    // no copy constructor.
    Optimizer(Optimizer const &) HLSL_DELETED_FUNCTION;
    // no assignment.
    Optimizer &operator=(Optimizer const &) HLSL_DELETED_FUNCTION;

private:
    /// The HLSL compiler.
    Compiler &m_compiler;

    /// The current compilation unit.
    Compilation_unit &m_unit;

    /// The statement factory of the current unit.
    Stmt_factory &m_stmt_factory;

    /// The Expression factory of the current unit.
    Expr_factory &m_expr_factory;

    /// The value factory of the current unit.
    Value_factory &m_value_factory;

    /// Current optimizer level.
    int m_opt_level;
};

}  // hlsl
}  // mdl
}  // mi

#endif
