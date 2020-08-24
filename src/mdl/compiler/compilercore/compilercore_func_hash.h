/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_FUNC_HASH_H
#define MDL_COMPILERCORE_FUNC_HASH_H 1

#include "compilercore_cc_conf.h"
#include "compilercore_visitor.h"
#include "compilercore_call_graph.h"

namespace mi {
namespace mdl {

class Module;
class MDL;

///
/// Helper class to computer semantic hashes for function definitions
///
class Sema_hasher : protected Module_visitor {
public:
    /// Compute the hashes.
    ///
    /// \param alloc     the allocator
    /// \param mod       the module
    /// \param compiler  the MDL compiler
    static void run(
        IAllocator *alloc,
        Module     *mod,
        MDL        *compiler);

private:
    /// Compute the semantic hashes for all exported functions.
    void compute_hashes();

    /// Post visit a call expression.
    IExpression *post_visit(IExpression_call *expr) MDL_FINAL;

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_cg.get_allocator(); }

private:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param mod       the module
    /// \param compiler  the MDL compiler
    Sema_hasher(
        IAllocator *alloc,
        Module     *mod,
        MDL        *compiler);

private:
    /// The current module.
    Module &m_mod;

    /// The call graph.
    Call_graph m_cg;

    /// The MDL compiler.
    MDL *m_compiler;

    /// The definition of the currently processed function.
    Definition *m_curr_def;

    struct Hash_info {
        unsigned x;
    };

    typedef ptr_hash_map<Definition const, Hash_info *>::Type  Sema_hash_map;

    /// Maps definitions to sema hashes.
    Sema_hash_map m_hashes;

    typedef queue<Definition *>::Type Wait_queue;

    /// Wait queue.
    Wait_queue m_wq;


};

}  // mdl
}  // mi

#endif
