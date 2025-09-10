/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_CODE_TOOLS_H
#define MDL_GENERATOR_CODE_TOOLS_H 1

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_function_instance.h"

namespace mi {
namespace mdl {

class Module;

/// RAII helper class to handle a stack of MDL modules.
template<typename Generator>
class Module_scope {
public:
    /// Constructor.
    ///
    /// \param generator   the code generator that manages the stack by (push|pop)_module()
    /// \param mod         the module that should be put on stack while the scope exists
    Module_scope(Generator &generator, Module const *mod)
    : m_generator(generator)
    {
        generator.push_module(mod);
    }

    /// Destructor.
    ~Module_scope() { m_generator.pop_module(); }

private:
    /// The generator containing the stack.
    Generator &m_generator;
};

/// RAII-like break destination scope handler.
template <typename Ctx, typename Target>
class Break_scope {
public:
    /// Constructor.
    ///
    /// \param ctx   the function context
    /// \param dest  the basic block that would be the destination of a break statement
    Break_scope(Ctx &ctx, Target dest)
    : m_ctx(ctx)
    {
        ctx.push_break(dest);
    }

    /// Destructor.
    ~Break_scope() {
        m_ctx.pop_break();
    }

private:
    /// The function context.
    Ctx &m_ctx;
};

/// RAII-like continue destination scope handler.
template <typename Ctx, typename Target>
class Continue_scope {
public:
    /// Constructor.
    ///
    /// \param ctx   the function context
    /// \param dest  the basic block that would be the destination of a continue statement
    Continue_scope(Ctx &ctx, Target dest)
    : m_ctx(ctx)
    {
        ctx.push_continue(dest);
    }

    /// Destructor.
    ~Continue_scope() {
        m_ctx.pop_continue();
    }

private:
    /// The function context.
    Ctx &m_ctx;
};

/// Value class to handle owner and function instances pairs.
///
/// \param owner   the MDL module that owns the function instance
/// \param inst    the function instance
class Owner_func_inst_pair {
public:
    /// Constructor.
    Owner_func_inst_pair(Module const *owner, Function_instance const &inst)
    : m_owner(owner)
    , m_inst(inst)
    {
    }

    /// Get the owner module.
    Module const *get_owner() const { return m_owner; }

    /// The function instance.
    Function_instance const &get_instance() const { return m_inst; }

private:
    /// The owner module.
    Module const *m_owner;

    /// The function instance.
    Function_instance      m_inst;
};

/// A wait queue of function instances.
typedef mi::mdl::queue<Owner_func_inst_pair>::Type Function_wait_queue;

/// Helper struct to map (local) AST file IDs to (global) DAG file IDs.
struct File_id_table {
    size_t module_id;   ///< the ID of the module that "produces" this table
    size_t len;         ///< number of entries in this table
    size_t map[1];      ///< the table itself

public:
    /// Constructor.
    ///
    /// \param module_id   the global unique module ID of the module that produces this table
    /// \param len         number of entries in this table
    File_id_table(
        size_t module_id,
        size_t len)
    : module_id(module_id)
    , len(len)
    {
        memset(map, 0, sizeof(map[0]) * len);
    }

    /// Get DAG global file id for an AST local file id.
    ///
    /// \param ast_id  the local file ID inside an AST module
    ///
    /// \return the global DAG file ID
    size_t get_dag_id(size_t ast_id) const {
        if (ast_id < len) {
            return map[ast_id];
        }
        MDL_ASSERT(!"cannot map unknown file ID");
        return 0;
    }

    /// Set a DAG global file id for an AST local file id.
    ///
    /// \param ast_id   the local AST module file ID
    /// \param dag_id   the global DAG file ID
    void set_dag_id(size_t ast_id, size_t dag_id)
    {
        if (ast_id < len) {
            map[ast_id] = dag_id;
            return;
        } else {
            MDL_ASSERT(!"cannot map unknown file ID");
        }
    }
};


}  // mdl
}  // mi

#endif // MDL_GENERATOR_CODE_TOOLS_H
