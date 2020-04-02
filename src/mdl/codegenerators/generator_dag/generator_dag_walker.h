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

#ifndef MDL_GENERATOR_DAG_WALKER_H
#define MDL_GENERATOR_DAG_WALKER_H

#include <mi/mdl/mdl_generated_dag.h>
#include <mdl/compiler/compilercore/compilercore_memory_arena.h>

#include "generator_dag_generated_dag.h"

namespace mi {
namespace mdl {

class MD5_hasher;

class IDAG_ir_visitor {
public:
    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    virtual void visit(DAG_constant *cnst) = 0;

    /// Post-visit a Temporary.
    ///
    /// \param tmp  the temporary that is visited
    virtual void visit(DAG_temporary *tmp) = 0;

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    virtual void visit(DAG_call *call) = 0;

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    virtual void visit(DAG_parameter *param) = 0;

    /// Post-visit a temporary initializer.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    virtual void visit(int index, DAG_node *init) = 0;
};

/// A walker for DAG IR nodes.
class DAG_ir_walker {
public:
    /// Constructor.
    ///
    /// \param alloc    an allocator for temporary memory
    /// \param as_tree  if true, walk the DAG as a Tree, NOT as a DAG, i.e. a node that has several
    ///                 users will be visited more than one.
    explicit DAG_ir_walker(
        IAllocator *alloc,
        bool       as_tree);

    /// Walk the IR nodes of a material, including temporaries.
    ///
    /// \param dag        the code DAG that will be visited
    /// \param mat_index  the index of the material to walk
    /// \param visitor    the visitor
    void walk_material(
        Generated_code_dag *dag,
        int                mat_index,
        IDAG_ir_visitor    *visitor);

    /// Walk the IR nodes of an instance, including temporaries.
    ///
    /// \param instance   the instance that will be visited
    /// \param visitor    the visitor
    void walk_instance(
        Generated_code_dag::Material_instance *instance,
        IDAG_ir_visitor                       *visitor);

    /// Walk the IR nodes of an instance material slot, including temporaries.
    ///
    /// \param instance   the instance that will be visited
    /// \param slot       the material slot
    /// \param visitor    the visitor
    void walk_instance_slot(
        Generated_code_dag::Material_instance       *instance,
        Generated_code_dag::Material_instance::Slot slot,
        IDAG_ir_visitor                             *visitor);

    /// Walk the IR nodes of a function, including temporaries.
    ///
    /// \param dag         the code DAG that will be visited
    /// \param func_index  the index of the function to walk
    /// \param visitor     the visitor
    void walk_function(
        Generated_code_dag *dag,
        int                func_index,
        IDAG_ir_visitor    *visitor);

    /// Walk a DAG IR node.
    ///
    /// \param node       the DAG IR node that will be visited
    /// \param visitor    the visitor
    void walk_node(
        DAG_node        *node,
        IDAG_ir_visitor *visitor);

private:
    typedef Arena_ptr_hash_set<DAG_node>::Type Visited_node_set;
    typedef list<int>::Type                    Temp_queue;

    /// Walk an DAG IR node.
    ///
    /// \param marker      the marker set
    /// \param temp_queue  the queue of unprocessed temporaries
    /// \param node        the root node to traverse
    /// \param visitor     the visitor interface
    void do_walk_node(
        Visited_node_set &marker,
        Temp_queue       &temp_queue,
        DAG_node         *node,
        IDAG_ir_visitor  *visitor);

private:
    /// The allocator.
    IAllocator *m_alloc;

    /// If true, walk as a Tree.
    bool m_as_tree;
};

/// Helper class: hashes a DAG.
class Dag_hasher : public IDAG_ir_visitor
{
public:
    /// Constructor.
    ///
    /// \param hasher  the stream hasher to feed
    explicit Dag_hasher(MD5_hasher &hasher);

    /// Post-visit a Constant.
    void visit(DAG_constant *cnst) MDL_FINAL;

    /// Post-visit a variable (temporary).
    void visit(DAG_temporary *tmp) MDL_FINAL;

    /// Post-visit a call.
    void visit(DAG_call *call) MDL_FINAL;

    /// Post-visit a Parameter.
    void visit(DAG_parameter *param) MDL_FINAL;

    /// Post-visit a Temporary.
    void visit(int index, DAG_node *init) MDL_FINAL;

    /// Hash a parameter.
    void hash_parameter(char const *name, IType const *type);

private:
    /// Hash a type.
    void hash(IType const *tp);

    /// Hash a value.
    void hash(IValue const *v);

private:
    /// The hasher used.
    MD5_hasher &m_hasher;
};

} // mdl
} // mi

#endif
