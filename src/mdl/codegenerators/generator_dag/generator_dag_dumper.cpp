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

#include "pch.h"

#include <mi/base/handle.h>

#include <mdl/compiler/compilercore/compilercore_printers.h>

#include "generator_dag_dumper.h"
#include "generator_dag_tools.h"

#include <cstdio>

namespace mi {
namespace mdl {


// Constructor.
DAG_dumper::DAG_dumper(
    IAllocator           *alloc,
    IOutput_stream       *out)
: m_next_node_id(0)
, m_walker(alloc, /*as_tree=*/false)
, m_printer()
, m_node_to_is_map(0, Node_to_id_map::hasher(), Node_to_id_map::key_equal(), alloc)
{
    Allocator_builder builder(alloc);

    m_printer = mi::base::make_handle(builder.create<Printer>(alloc, out));
    // Avoid syntax error in .gv labels.
    m_printer->set_string_quote("'");
}

// Get a new unique ID for a node.
size_t DAG_dumper::get_unique_id()
{
    return m_next_node_id++;
}

// Get the ID of a DAG IR node.
size_t DAG_dumper::get_id(DAG_node const *node)
{
    Node_to_id_map::iterator it = m_node_to_is_map.find(node);
    if (it == m_node_to_is_map.end()) {
        it = m_node_to_is_map.insert(std::make_pair(node, get_unique_id())).first;
    }
    return it->second;
}

// Print the name of a dependence graph node.
void DAG_dumper::node_name(DAG_node const *node)
{
    char buf[32];

    snprintf(buf, sizeof(buf), "n%ld", (long)get_id(node));
    buf[sizeof(buf) - 1] = '\0';
    m_printer->print(buf);
}

// Print the name of a dependence graph node.
void DAG_dumper::node_name(char type, size_t index)
{
    char buf[32];

    snprintf(buf, sizeof(buf), "%c%u", type, (unsigned)index);
    buf[sizeof(buf) - 1] = '\0';
    m_printer->print(buf);
}

// Print a DAG IR node.
void DAG_dumper::node(DAG_node const *node, char const *color)
{
    bool use_box_shape = true;

    m_printer->print("  ");
    node_name(node);
    m_printer->print(" [label=\"");
    
    switch (node->get_kind()) {
    case DAG_node::EK_CONSTANT:
        {
            DAG_constant const *c = cast<DAG_constant>(node);
            m_printer->print("Const ");
            m_printer->print(c->get_value());
        }
        break;
    case DAG_node::EK_TEMPORARY:
        {
            DAG_temporary const *t = cast<DAG_temporary>(node);
            m_printer->print("Temp _");
            m_printer->print((long)t->get_index());
        }
        break;
    case DAG_node::EK_PARAMETER:
        {
            DAG_parameter const *p = cast<DAG_parameter>(node);
            int index = p->get_index();
            m_printer->print("Parm ");
            m_printer->print((long)index);

            if (char const *name = get_parameter_name(index)) {
                m_printer->print(": ");
                m_printer->print(name);
            }
            use_box_shape = false;
        }
        break;
    case DAG_node::EK_CALL:
        {
            DAG_call const *c = cast<DAG_call>(node);
            m_printer->print(c->get_name());
        }
        break;
    }
    m_printer->print("\"");

    if (color != NULL) {
        m_printer->print(" color=");
        m_printer->print(color);
    }

    if (use_box_shape)
        m_printer->print(" shape=box");
    else
        m_printer->print(" shape=ellipse");

    m_printer->print("];\n");
}

// Print a DAG IR temporary.
void DAG_dumper::temporary(int index, char const *color)
{
    bool use_box_shape = false;

    m_printer->print("  ");
    node_name('t', index);
    m_printer->print(" [label=\"Temp _");

    m_printer->print((long)index);
    m_printer->print("\"");

    if (color != NULL) {
        m_printer->print(" color=");
        m_printer->print(color);
    }

    if (use_box_shape)
        m_printer->print(" shape=box");

    m_printer->print("];\n");
}

// Print a DAG IR argument.
void DAG_dumper::argument(
    char const *name,
    size_t     index,
    char const *color)
{
    m_printer->print("  ");
    node_name('a', index);
    m_printer->print(" [label=\"Parm ");
    m_printer->print((long)index);

    m_printer->print(": ");
    m_printer->print(name);
    m_printer->print("\"");

    if (color != NULL) {
        m_printer->print(" color=");
        m_printer->print(color);
    }

    m_printer->print(" shape=ellipse");

    m_printer->print("];\n");
}

// Print a dependency edge.
void DAG_dumper::edge(
    DAG_node const *src,
    DAG_node const *dst,
    char const     *label,
    char const     *color)
{
    m_printer->print("  ");
    node_name(src);

    m_printer->print(" -> ");
    node_name(dst);

    if (color != NULL) {
        m_printer->print(" [color=");
        m_printer->print(color);
        m_printer->print("]");
    }

    if (label != NULL) {
        m_printer->print(" [label=\"");
        m_printer->print(label);
        m_printer->print("\"]");
    }

    m_printer->print(";\n");
}

// Print a dependency edge.
void DAG_dumper::edge(
    char           type,
    size_t         src,
    DAG_node const *dst,
    char const     *label,
    char const     *color)
{
    m_printer->print("  ");
    node_name(type, src);

    m_printer->print(" -> ");
    node_name(dst);

    if (color != NULL) {
        m_printer->print(" [color=");
        m_printer->print(color);
        m_printer->print("]");
    }

    if (label != NULL) {
        m_printer->print(" [label=\"");
        m_printer->print(label);
        m_printer->print("\"]");
    }

    m_printer->print(";\n");
}

// Post-visit a Constant.
void DAG_dumper::visit(DAG_constant *cnst)
{
    node(cnst, NULL);
}

// Post-visit a Temporary.
void DAG_dumper::visit(DAG_temporary *tmp)
{
    node(tmp, NULL);
}

// Post-visit a Parameter.
void DAG_dumper::visit(DAG_parameter *param)
{
    node(param, NULL);
}

// Post-visit a call.
void DAG_dumper::visit(DAG_call *call)
{
    node(call, NULL);
    for (int i = 0, n = call->get_argument_count(); i < n; ++i) {
        DAG_node const *arg = call->get_argument(i);
        char const *label = call->get_parameter_name(i);
        char const *color = NULL;

        edge(call, arg, label, color);
    }
}

// Post-visit a Temporary.
void DAG_dumper::visit(int index, DAG_node *init)
{
    temporary(index, NULL);
    edge('t', index, init, "init", NULL);
}

} // mdl
} // mi
