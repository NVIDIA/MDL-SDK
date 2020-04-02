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

#include "pch.h"

#include <mi/mdl/mdl_types.h>

#include "compilercore_cc_conf.h"
#include "compilercore_dependency_graph.h"
#include "compilercore_def_table.h"
#include "compilercore_streams.h"
#include "compilercore_printers.h"
#include "compilercore_assert.h"

namespace mi {
namespace mdl {

namespace {
template<typename T>
static T min(T a, T b) { return a <= b ? a : b; }
}

/// Get the combined type modifiers of a type.
static IType::Modifiers get_type_modifiers(IType const *type)
{
    IType::Modifiers res = type->get_type_modifiers();

    if (IType_array const *a_tp = as<IType_array>(type)) {
        IType const *e_tp = a_tp->get_element_type();
        res |= get_type_modifiers(e_tp);
    }
    return res;
}

// Constructor of a node.
Dependence_graph::Node::Node(
    Memory_arena     *arena,
    Kind             kind,
    size_t           id,
    IType const      *type,
    Definition       *def,
    IStatement const *stmt,
    Id_type          ctrl_boundary)
: m_kind(kind)
, m_id(id)
, m_visited(0)
, m_dfs_num(0)
, m_low(0)
, m_scc_id(0)
, m_on_stack(false)
, m_control_boundary(ctrl_boundary)
, m_auto_type(AT_TOP)
, m_type(type)
, m_def(def)
, m_stmt(stmt)
, m_ctrl_uplink(0)
, m_dep_edges(arena)
, m_loop_exit_edges(arena)
{
    if (kind == NK_VARYING_CALL) {
        // the varying call is already at the bottom ...
        m_auto_type = AT_VARYING;
    } else {
        IType::Modifiers mod = get_type_modifiers(type);

        if (mod & IType::MK_VARYING)
            m_auto_type = AT_VARYING;
        else if (mod & IType::MK_UNIFORM)
            m_auto_type = AT_UNIFORM;
    }
}

// Check if this node is already visited.
bool Dependence_graph::Node::check_visited(size_t visited) const
{
    if (m_visited < visited) {
        m_visited = visited;
        return false;
    }
    return true;
}

// Set the control uplink.
void Dependence_graph::Node::set_control_uplink(Id_type uplink)
{
    MDL_ASSERT(m_kind == NK_CONTROL && "Only control nodes have an uplink");
    MDL_ASSERT(m_id != uplink && "Unlink cannot be self-loop");
    m_ctrl_uplink = uplink;
}

// Constructor of a dependence graph.
Dependence_graph::Dependence_graph(
    Memory_arena *arena,
    char const   *name)
: m_builder(*arena)
, m_nodes(arena)
, m_node_map(0, Node_map::hasher(), Node_map::key_equal(), arena->get_allocator())
, m_stack(Node_stack::container_type(arena->get_allocator()))
, m_visited(0)
, m_next_dfs_num(0)
, m_next_scc_id(0)
, m_name(name != NULL ? Arena_strdup(*arena, name) : NULL)
{
}

// Checks whether a node depends directly on another node.
bool Dependence_graph::depends_directly(Id_type src, Id_type dst) const
{
    Node *n = m_nodes[src];

    Node::Edge_iterator it = find(n->edges_begin(), n->edges_end(), dst);
    return it != n->edges_end();
}

// Checks whether a loop node has the given exit node.
bool Dependence_graph::has_exit(Id_type loop, Id_type exit) const
{
    Node *n = m_nodes[loop];

    Node::Edge_iterator it = find(n->loop_exits_begin(), n->loop_exits_end(), exit);
    return it != n->loop_exits_end();
}

// Add a dependency from the src node to the dst node if not already exists.
bool Dependence_graph::add_dependency(
    Id_type src,
    Id_type dst)
{
    if (src == dst) {
        // ignore self loops
        return false;
    }

    if (!depends_directly(src, dst)) {
        Node *n = m_nodes[src];
        n->add_edge(dst);
        return true;
    }
    return false;
}

// Add a loop exit from the loop node to the exit node if not already exists, i.e. exit
bool Dependence_graph::add_loop_exit(Id_type loop, Id_type exit)
{
    if (!has_exit(loop, exit)) {
        Node *n = m_nodes[loop];
        n->add_loop_exit_edge(exit);
        return true;
    }
    return false;
}

// Walker helper.
void Dependence_graph::do_walk(Node *root, IDependence_visitor *visitor)
{
    visitor->visit(root, IDependence_visitor::PRE_ORDER);

    for (Node::Edge_iterator it(root->edges_begin()), end(root->edges_end()); it != end; ++it) {
        Id_type prev_id = *it;
        Node    *prev   = m_nodes[prev_id];

        if (!prev->check_visited(m_visited)) {
            do_walk(prev, visitor);
        }
    }

    visitor->visit(root, IDependence_visitor::POST_ORDER);
}

// Visit all nodes of the call graph in post order.
void Dependence_graph::walk(IDependence_visitor *visitor)
{
    ++m_visited;

    for (size_t i = 0, n = m_nodes.size(); i < n; ++i) {
        Node *root = m_nodes[i];

        if (!root->check_visited(m_visited)) {
            // ignore the varying call node if it's the root node
            if (root->get_kind() != Node::NK_VARYING_CALL)
                do_walk(root, visitor);
        }
    }
}

// Implements the lattice supremum of two elements.
Dependence_graph::Auto_type Dependence_graph::supremum(
    Auto_type a, Auto_type b)
{
    int va = a;
    int vb = b;

    int r = va > vb ? va : vb;
    return Auto_type(r);
}

// Creates a new auxiliary node.
Dependence_graph::Node *Dependence_graph::create_aux_node(
    Node::Kind       kind,
    IType const      *type)
{
    Id_type id = m_nodes.size();
    Node *n = m_builder.create<Node>(
        m_builder.get_arena(), kind, id, type, (Definition *)NULL, (IStatement *)NULL, 0);

    m_nodes.push_back(n);
    return n;
}

// Creates a new parameter node.
Dependence_graph::Node *Dependence_graph::create_param_node(
    Definition *def)
{
    Id_type     id    = m_nodes.size();
    IType const *type = def->get_type();

    Node *n = m_builder.create<Node>(
        m_builder.get_arena(), Dependence_graph::Node::NK_PARAM,
        id, type, def, (IStatement *)NULL, 0);

    // set the auto type of parameters to "parameter-dependent"
    // if not set explicitly
    if (n->get_auto_type() == Dependence_graph::AT_TOP)
        n->set_auto_type(Dependence_graph::AT_PARAM);

    m_nodes.push_back(n);

    // remember it for fast lookup
    m_node_map[def] = id;

    return n;
}

// Creates a new local node.
Dependence_graph::Node *Dependence_graph::create_local_node(
    Definition *def,
    Id_type    ctrl_boundary)
{
    Id_type     id    = m_nodes.size();
    IType const *type = def->get_type();

    Node *n = m_builder.create<Node>(
        m_builder.get_arena(), Dependence_graph::Node::NK_VARIABLE,
        id, type, def, (IStatement *)NULL, ctrl_boundary);

    m_nodes.push_back(n);

    // remember it for fast lookup
    m_node_map[def] = id;

    return n;
}

// Creates a new control node.
Dependence_graph::Node *Dependence_graph::create_control_node(
    IType const      *type,
    IStatement const *stmt)
{
    Id_type id = m_nodes.size();
    Node *n = m_builder.create<Node>(
        m_builder.get_arena(), Dependence_graph::Node::NK_CONTROL,
        id, type, (Definition *)NULL, stmt, 0);

    m_nodes.push_back(n);
    return n;
}

// Get the node for the given Id.
Dependence_graph::Node *Dependence_graph::get_node(Id_type id)
{
    if (id < m_nodes.size())
        return m_nodes[id];
    return NULL;
}

// Get the node for the given (parameter or local) definition.
Dependence_graph::Node *Dependence_graph::get_node(IDefinition const *def)
{
    Node_map::iterator it = m_node_map.find(def);
    return it == m_node_map.end() ? NULL : m_nodes[it->second];
}

// Calculate the strongly coupled components.
void Dependence_graph::do_dfs(Node *node)
{
    node->m_visited = m_visited;

    node->m_dfs_num = m_next_dfs_num++;
    node->m_low     = node->m_dfs_num;

    m_stack.push(node);
    node->m_on_stack = true;

    for (Node::Edge_iterator it(node->edges_begin()), end(node->edges_end()); it != end; ++it) {
        Node *pred = get_node(*it);

        if (pred->m_visited < m_visited) {
            do_dfs(pred);
            node->m_low = min(node->m_low, pred->m_low);
        }
        if (pred->m_dfs_num < node->m_dfs_num && pred->m_on_stack) {
            node->m_low = min(pred->m_dfs_num, node->m_low);
        }
    }

    if (node->m_low == node->m_dfs_num) {
        Node *x;
        // found SCC
        Scc_vector scc(m_builder.get_arena()->get_allocator());
        ++m_next_scc_id;
        do {
            x = m_stack.top();
            x->m_scc_id = m_next_scc_id;
            x->m_on_stack = false;
            m_stack.pop();
            scc.push_back(x);
        } while (x != node);

        process_scc(scc);
    }
}

// Process a SCC.
void Dependence_graph::process_scc(Scc_vector const &scc)
{
    Auto_type at = AT_TOP;

    size_t scc_id = scc[0]->m_scc_id;

    // calculate the type of the whole SCC , t = SUP{n} {sup(n.type, SUP{n->edges} type)
    for (Scc_vector::const_iterator it(scc.begin()), end(scc.end()); it != end; ++it) {
        Node *n = *it;

        at = supremum(at, n->get_auto_type());

        for (Node::Edge_iterator e(n->edges_begin()), ee(n->edges_end()); e != ee; ++e) {
            Node *p = get_node(*e);

            if (p->m_scc_id != scc_id) {
                MDL_ASSERT(p->m_scc_id != 0 && p->m_scc_id < scc_id && "Tarjan finder failed");

                at = supremum(at, p->get_auto_type());
            }
        }
    }

    // propagate the type
    for (Scc_vector::const_iterator it(scc.begin()), end(scc.end()); it != end; ++it) {
        Node *n = *it;
    
        n->set_auto_type(at);
    }
}

// Calculate the auto types of the dependence graph.
void Dependence_graph::calc_auto_types()
{
    ++m_visited;
    m_next_dfs_num = 0;

    for (size_t i = 0, n = m_nodes.size(); i < n; ++i) {
        Node *root = m_nodes[i];

        if (root->m_visited < m_visited)
            do_dfs(root);
    }
}

namespace {

///
/// Helper class to dump a dependence graph as a dot file.
///
class Dumper : public IDependence_visitor {
public:
    /// Constructor.
    ///
    /// \param alloc              the allocator
    /// \param out                an output stream, the dot file is written to
    /// \param dg                 the dependence graph to dump
    /// \param only_dependencies  if true, only dependency edges will be dumped, uplink
    ///                           and loop exit edges will be suppressed
    explicit Dumper(
        IAllocator       *alloc,
        IOutput_stream   *out,
        Dependence_graph &dg,
        bool             only_dependencies);

    /// Dump the dependence graph to the output stream.
    void dump();

private:
    /// Print the name of a dependence graph node.
    ///
    /// \param n  the node
    void node_name(
        Dependence_graph::Node const *n);

    /// Print a dependence graph node.
    ///
    /// \param n      the node
    /// \param color  the color of the node, NULL for default
    void node(
        Dependence_graph::Node const *n,
        char const                   *color = NULL);

    /// Print a dependence edge.
    ///
    /// \param src    the source node of the edge
    /// \param dst    the destination node of the edge
    /// \param color  the color of the edge, NULL for default
    void edge(
        Dependence_graph::Node const *src,
        Dependence_graph::Node const *dst,
        char const                   *color = NULL);

    /// Dependence graph node visitor.
    ///
    /// \param node   the currently visited node
    /// \param order  PRE or POST order
    void visit(Dependence_graph::Node *node, Order order) MDL_FINAL;

private:
    /// The dependence graph.
    Dependence_graph          &m_dg;

    /// A printer, use to print into the output stream.
    mi::base::Handle<Printer> m_printer;

    /// Current graph depth from the root.
    size_t                    m_depth;

    /// If true, only dependency edges will be dumped, else uplink and 
    /// loop exits will be added additionally
    bool m_only_dependencies;
};

// Constructor.
Dumper::Dumper(
    IAllocator       *alloc,
    IOutput_stream   *out,
    Dependence_graph &dg,
    bool             only_dependencies)
: m_dg(dg)
, m_printer()
, m_depth(0)
, m_only_dependencies(only_dependencies)
{
    Allocator_builder builder(alloc);

    m_printer = mi::base::make_handle(builder.create<Printer>(alloc, out));
}

// Dump the dependence graph to the output stream.
void Dumper::dump()
{
    m_printer->print("digraph \"");
    char const *name = m_dg.get_name();
    if (name == NULL || name[0] == '\0')
        name = "dependency_graph";
    m_printer->print(name);
    m_printer->print("\" {\n");
    m_dg.walk(this);
    m_printer->print("}\n");
}

// Dump the dependence graph to the output stream.
void Dumper::node_name(Dependence_graph::Node const *n)
{
    char buf[32];

    snprintf(buf, sizeof(buf), "n%ld", (long)n->get_id());
    buf[sizeof(buf) - 1] = '\0';
    m_printer->print(buf);
}

void Dumper::node(Dependence_graph::Node const *n, char const *color)
{
    bool use_box_shape = false;

    m_printer->print("  ");
    node_name(n);
    m_printer->print(" [label=\"");
    
    switch (n->get_kind()) {
    case Dependence_graph::Node::NK_PARAM:
    case Dependence_graph::Node::NK_VARIABLE:
        {
            Definition const *def = n->get_definition();
            m_printer->print(def->get_sym());
            use_box_shape = true;
        }
        break;
    case Dependence_graph::Node::NK_RETURN_VALUE:
        m_printer->print("ReturnValue");
        break;
    case Dependence_graph::Node::NK_VARYING_CALL:
        m_printer->print("VaryingCall");
        break;
    case Dependence_graph::Node::NK_CONTROL:
        {
            char buf[32];

            IStatement const *stmt = n->get_statement();
            char const *s = "control";
            switch (stmt->get_kind()) {
            case IStatement::SK_IF:       s = "if"; break;
            case IStatement::SK_WHILE:    s = "while"; break;
            case IStatement::SK_DO_WHILE: s = "do while"; break;
            case IStatement::SK_FOR:      s = "for"; break;
            case IStatement::SK_SWITCH:   s = "switch"; break;
            default:
                break;
            }
            snprintf(buf, sizeof(buf), "%s line %d", s, stmt->access_position().get_start_line());
            buf[sizeof(buf) - 1] = '\0';
            m_printer->print(buf);
        }
        break;
    }
    m_printer->print(": ");

    char const *s = "T";
    switch (n->get_auto_type()) {
    case Dependence_graph::AT_TOP:                    break;
    case Dependence_graph::AT_UNIFORM: s = "uniform"; break;
    case Dependence_graph::AT_PARAM:   s = "param";   break;
    case Dependence_graph::AT_VARYING: s = "varying"; break;
    }
    m_printer->print(s);

    m_printer->print("\"");

    if (color != NULL) {
        m_printer->print(" color=");
        m_printer->print(color);
    }

    if (use_box_shape)
        m_printer->print(" shape=box");

    m_printer->print("];\n");
}

// Print a dependence edge.
void Dumper::edge(
    Dependence_graph::Node const *src,
    Dependence_graph::Node const *dst,
    char const                   *color)
{
    m_printer->print("  ");
    if (src != NULL) {
        node_name(src);
    } else {
        m_printer->print("root");
    }
    m_printer->print(" -> ");
    if (dst != NULL) {
        node_name(dst);
    } else {
        m_printer->print("root");
    }

    if (color != NULL) {
        m_printer->print(" [color=");
        m_printer->print(color);
        m_printer->print("]");
    }

    m_printer->print(";\n");
}

// Dependence graph node visitor.
void Dumper::visit(Dependence_graph::Node *n, Order order)
{
    if (order == IDependence_visitor::PRE_ORDER) {
        // make roots green
        char const *color = m_depth > 0 ? NULL : "green";
        node(n, color);
        ++m_depth;
    } else {
        for (Dependence_graph::Node::Edge_iterator it(n->edges_begin()), end(n->edges_end());
             it != end;
             ++it)
        {
            Dependence_graph::Node const *prev = m_dg.get_node(*it);
            char const *color = NULL;
            if (prev->get_kind() == Dependence_graph::Node::NK_CONTROL) {
                // make control dependence blue
                color = "blue";
            }

            edge(n, prev, color);
        }

        if (!m_only_dependencies) {
            Dependence_graph::Id_type up_id = n->get_control_uplink();
            if (up_id != 0) {
                Dependence_graph::Node const *up = m_dg.get_node(up_id);

                // make uplinks red
                edge(n, up, "red");
            }

            for (Dependence_graph::Node::Edge_iterator
                it(n->loop_exits_begin()), end(n->loop_exits_end());
                it != end;
                ++it)
            {
                Dependence_graph::Node const *prev = m_dg.get_node(*it);

                // make loop exists purple
                edge(n, prev, "purple");
            }
        }

        --m_depth;
    }
}

}  // anonymous


// Dump the dependency graph.
void Dependence_graph::dump(IOutput_stream *out)
{
    Dumper dumper(m_builder.get_arena()->get_allocator(), out, *this, /*only_dependencies=*/true);

    dumper.dump();
}

} // mdl
} // mi

