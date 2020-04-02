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

#ifndef MDL_COMPILERCORE_DEPENDENCY_GRAPH_H
#define MDL_COMPILERCORE_DEPENDENCY_GRAPH_H 1

#include "compilercore_allocator.h"
#include "compilercore_bitset.h"
#include "compilercore_memory_arena.h"

namespace mi {
namespace mdl {

class IType;
class IOutput_stream;
class IDependence_visitor;
class IDefinition;
class Definition;
class IStatement;

///
/// A dependence graph.
///
class Dependence_graph {
    friend class Arena_builder;
public:
    typedef size_t Id_type;

    /// The possible Auto type values.
    enum Auto_type {
        AT_TOP,     //< The TOP type.
        AT_UNIFORM, //< This type is uniform.
        AT_PARAM,   //< This type depends on the parameter type.
        AT_VARYING, //< This type is varying (and BOTTOM).
    };

    ///
    /// A base class for all nodes of the Dependence graph.
    ///
    class Node {
        friend class Arena_builder;
        friend class Dependence_graph;
    public:
        typedef Arena_list<Id_type>::Type Edge_list;
        typedef Edge_list::const_iterator Edge_iterator;

        /// The node kind.
        enum Kind {
            NK_PARAM,         ///< a parameter node
            NK_VARIABLE,      ///< a local variable node
            NK_RETURN_VALUE,  ///< the return value of the function
            NK_VARYING_CALL,  ///< the one and only varying call node
            NK_CONTROL,       ///< a control dependence node
        };

        /// Get the node kind.
        Kind get_kind() const { return m_kind; }

        /// Get the id of this node.
        Id_type get_id() const { return m_id; }

        /// Get the auto type of this node.
        Auto_type get_auto_type() const { return m_auto_type; }

        /// Set the auto type of this node.
        void set_auto_type(Auto_type type) { m_auto_type = type; }

        /// Add an dependency edge from this node to the dst node.
        ///
        /// \param dst   the ID of the destination node
        void add_edge(Id_type dst) { m_dep_edges.push_back(dst); }

        /// Get the iterator to the edges start.
        Edge_iterator edges_begin() const { return m_dep_edges.begin(); }

        /// Get the iterator to the edges end.
        Edge_iterator edges_end() const { return m_dep_edges.end(); }

        /// Check if this node is already visited.
        ///
        /// \param visited  the visit count of the graph
        ///
        /// \return false  if this node was not yet visited,
        ///         true   if it was already visited
        bool check_visited(size_t visited) const;

        /// Return the definition of this one if one exists.
        Definition *get_definition() const { return m_def; }

        /// Return the statement of this one if one exists.
        IStatement const *get_statement() const { return m_stmt; }

        /// Get the control boundary of this node.
        Id_type get_control_boundary() const { return m_control_boundary; }

        /// Get the control uplink.
        Id_type get_control_uplink() const { return m_ctrl_uplink; }

        /// Set the control uplink.
        void set_control_uplink(Id_type uplink);

        /// Add a loop exit edge.
        ///
        /// \param dst   the ID of the exit node
        void add_loop_exit_edge(Id_type dst) { m_loop_exit_edges.push_back(dst); }

        /// Get the iterator to the loop exits start.
        Edge_iterator loop_exits_begin() const { return m_loop_exit_edges.begin(); }

        /// Get the iterator to the loop exits end.
        Edge_iterator loop_exits_end() const { return m_loop_exit_edges.end(); }

    protected:
        /// Constructor.
        ///
        /// \param arena          a memory arena to allocate from
        /// \param kind           the node kind
        /// \param id             the unique id of this node
        /// \param type           the initial type of this node
        /// \param def            the definition of this node if one exists
        /// \param stmt           the statement of this node if one exists
        /// \param ctrl_boundary  the control boundary
        explicit Node(
            Memory_arena     *arena,
            Kind             kind,
            Id_type          id,
            IType const      *type,
            Definition       *def,
            IStatement const *stmt,
            Id_type          ctrl_boundary);

    protected:
        /// The kind of this node.
        Kind const m_kind;

        /// The Id of this node.
        Id_type const m_id;

        /// The visited count.
        mutable size_t m_visited;

        /// DFS number for Tarjan's SCC finder.
        size_t m_dfs_num;

        /// low link number for Tarjan's SCC finder.
        size_t m_low;

        /// The Id of the SCC this node belongs too, > 0 once assigned.
        size_t m_scc_id;

        /// Stack marker for Tarjan's SCC finder.
        bool m_on_stack;

        /// If non-zero the ID of the control boundary for this node.
        Id_type const m_control_boundary;

        /// The auto type of this node.
        Auto_type m_auto_type;

        /// The initial type of this node.
        IType const *m_type;

        /// The definition of this node if one exists.
        Definition * const m_def;

        /// The statement of this node if one exists.
        IStatement const * const m_stmt;

        /// Control uplink dependency.
        Id_type m_ctrl_uplink;

        /// The dependency edges.
        Arena_list<Id_type>::Type m_dep_edges;

        /// The loop exit edges.
        Arena_list<Id_type>::Type m_loop_exit_edges;
    };

    typedef vector<Node *>::Type Scc_vector;

public:
    /// Creates a new auxiliary node.
    ///
    /// \param kind   the node kind
    /// \param type   the initial type of this node
    ///
    /// \return the created dependency graph node
    Node *create_aux_node(
        Node::Kind       kind,
        IType const      *type);

    /// Creates a new parameter node.
    ///
    /// \param def  the definition of this node
    ///
    /// \return the created dependency graph node
    Node *create_param_node(
        Definition *def);

    /// Creates a new local node.
    ///
    /// \param def            the definition of this node
    /// \param ctrl_boundary  the control boundary of this node
    ///
    /// \return the created dependency graph node
    Node *create_local_node(
        Definition *def,
        Id_type    ctrl_boundary);

    /// Creates a new control node.
    ///
    /// \param type   the initial type of this node
    /// \param stmt   the statement of this node
    ///
    /// \return the created dependency graph node
    Node *create_control_node(
        IType const      *type,
        IStatement const *stmt);

    /// Add a dependency from the src node to the dst node if not already exists, i.e. src
    /// depends on dst.
    ///
    /// \param src  the ID of the source node
    /// \param dst  the ID of the destination node
    ///
    /// \return true if a new edge (src->dst) was added, false if the edge already exists.
    bool add_dependency(Id_type src, Id_type dst);

    /// Add a loop exit from the loop node to the exit node if not already exists, i.e. exit
    /// exits loop.
    ///
    /// \param loop  the ID of the loop node
    /// \param exit  the ID of the exit node
    ///
    /// \return true if a new edge (loop->exit) was added, false if the edge already exists.
    bool add_loop_exit(Id_type loop, Id_type exit);

    /// Visit all nodes of the call graph in post order.
    void walk(IDependence_visitor *visitor);

    /// Get the name of this dependency graph.
    char const *get_name() const { return m_name; }

    /// Get the node for the given Id.
    Node *get_node(Id_type id);

    /// Get the node for the given (parameter or local) definition.
    Node *get_node(IDefinition const *def);

    /// Return the number of nodes in the dependence graph.
    size_t get_nodes_count() const { return m_nodes.size(); }

    /// Calculate the auto types of the dependence graph.
    void calc_auto_types();

    /// Dump the dependency graph as a dot file.
    ///
    /// \param out  an output stream
    void dump(IOutput_stream *out);

private:
    /// Constructor of a dependence graph.
    ///
    /// \param arena      the memory arena to allocate from
    /// \param name       name of this graph
    explicit Dependence_graph(
        Memory_arena *arena,
        char const   *name);

    /// Implements the lattice supremum of two elements.
    ///
    /// \param a  left operand
    /// \param b  right operand
    ///
    /// \return the sup(a, b)
    static Auto_type supremum(Auto_type a, Auto_type b);

    /// Checks whether a node depends directly on another node.
    ///
    /// \param src  the ID of the source node
    /// \param dst  the ID of the destination node
    ///
    /// \return true if from directly depends on to.
    bool depends_directly(Id_type src, Id_type dst) const;

    /// Checks whether a loop node has the given exit node.
    ///
    /// \param loop  the ID of the loop node
    /// \param exit  the ID of the exit node
    ///
    /// \return true if from directly depends on to.
    bool has_exit(Id_type loop, Id_type exit) const;

    /// Walker helper.
    ///
    /// \param root     the root node to start the walk on
    /// \param visitor  the visitor I/F
    void do_walk(Node *root, IDependence_visitor *visitor);

    /// Calculate the strongly coupled components using Tarjan's SCC finder.
    ///
    /// \param node  the root node
    void do_dfs(Node *node);

    /// Process a SCC.
    ///
    /// \param scc  one scc, either a single node or a loop of the dependence tree
    void process_scc(Scc_vector const &scc);

private:
    /// The builder for objects on this graph.
    Arena_builder m_builder;

    /// All nodes in the graph.
    Arena_vector<Node *>::Type m_nodes;

    typedef ptr_hash_map<IDefinition const, size_t>::Type Node_map;

    /// Maps definitions to nodes.
    Node_map m_node_map;

    typedef stack<Node *>::Type Node_stack;

    /// Node stack for Tarjan's SCC finder.
    Node_stack m_stack;

    /// Current visit count of the graph.
    size_t m_visited;

    /// Next DFS number for Tarjan's SCC finder.
    size_t m_next_dfs_num;

    /// The next ID for an SCC.
    size_t m_next_scc_id;

    /// The name of this graph (mostly for debugging).
    char const *m_name;
};

///
/// Visitor interface.
///
class IDependence_visitor {
public:
    enum Order {
        PRE_ORDER,
        POST_ORDER
    };

    /// Visit.
    virtual void visit(Dependence_graph::Node *node, Order order) = 0;
};


}  // mdl
}  // mi

#endif // MDL_COMPILERCORE_DEPENDENCY_GRAPH_H
