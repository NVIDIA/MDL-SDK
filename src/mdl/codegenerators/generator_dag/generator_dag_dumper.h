/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_DUMPER
#define MDL_GENERATOR_DAG_DUMPER 1

#include <mi/base/handle.h>

#include "mdl/compiler//compilercore/compilercore_cc_conf.h"
#include "generator_dag_walker.h"

namespace mi {
namespace mdl {

///
/// Helper base class to dump an DAG expressions into a dot file.
///
class DAG_dumper : public IDAG_ir_visitor {
protected:

    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param out    an output stream, the dot file is written to
    DAG_dumper(
        IAllocator     *alloc,
        IOutput_stream *out);

protected:
    /// Get a new unique ID for a node.
    size_t get_unique_id();

    /// Get the ID of a DAG IR node.
    size_t get_id(DAG_node const *node);

    /// Print the name of a dependence graph node.
    ///
    /// \param node  the DAG IR node
    void node_name(
        DAG_node const *node);

    /// Print the name of a dependence graph node.
    ///
    /// \param type   the node's type: 't for temporary, 'a' for argument
    /// \param index  the node's index
    void node_name(
        char   type,
        size_t index);

    /// Print a DAG IR node.
    ///
    /// \param node   the DAG IR node
    /// \param color  the color of the node, NULL for default
    void node(
        DAG_node const *node,
        char const     *color = NULL);

    /// Print a DAG IR temporary.
    ///
    /// \param index  the temporary node's index
    /// \param color  the color of the node, NULL for default
    void temporary(
        int            index,
        char const     *color = NULL);

    /// Print a DAG IR argument.
    ///
    /// \param name   the argument node's name
    /// \param index  the argument node's index
    /// \param color  the color of the node, NULL for default
    void argument(
        char const *name,
        size_t     index,
        char const *color = NULL);

    /// Print a dependence edge.
    ///
    /// \param src    the source node of the edge
    /// \param dst    the destination node of the edge
    /// \param label  the edge label, NULL for none
    /// \param color  the color of the edge, NULL for default
    void edge(
        DAG_node const *src,
        DAG_node const *dst,
        char const     *label,
        char const     *color = NULL);

    /// Print a dependence edge.
    ///
    /// \param type   the type of the source node,'t' for temporary, 'a' for argument
    /// \param src    the source node of the edge
    /// \param dst    the destination node of the edge
    /// \param label  the edge label, NULL for none
    /// \param color  the color of the edge, NULL for default
    void edge(
        char           type,
        size_t         src,
        DAG_node const *dst,
        char const     *label,
        char const     *color = NULL);

    /// Post-visit a Constant.
    ///
    /// \param cnst  the constant that is visited
    void visit(DAG_constant *cnst) MDL_FINAL;

    /// Post-visit a temporary.
    ///
    /// \param tmp  the temporary that is visited
    void visit(DAG_temporary *tmp) MDL_FINAL;

    /// Post-visit a call.
    ///
    /// \param call  the call that is visited
    void visit(DAG_call *call) MDL_FINAL;

    /// Post-visit a Parameter.
    ///
    /// \param param  the parameter that is visited
    void visit(DAG_parameter *param) MDL_FINAL;

    /// Post-visit a Temporary.
    ///
    /// \param index  the index of the temporary
    /// \param init   the initializer expression of this temporary
    void visit(int index, DAG_node *init) MDL_FINAL;

    /// Get the parameter name for the given index if any.
    ///
    /// \param index  the index of the parameter
    virtual char const *get_parameter_name(int index) = 0;

protected:

    /// The next expression ID.
    size_t                    m_next_node_id;

    /// The walker for DAG IR nodes.
    DAG_ir_walker             m_walker;

    /// A printer, use to print into the output stream.
    mi::base::Handle<Printer> m_printer;

    typedef ptr_hash_map<DAG_node const, size_t>::Type Node_to_id_map;

    /// A map assigning ID's to IR nodes.
    Node_to_id_map            m_node_to_is_map;
};

} // mdl
} // mi

#endif // MDL_GENERATOR_DAG_DUMPER
