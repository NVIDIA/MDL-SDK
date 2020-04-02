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

#ifndef MDL_COMPILERCORE_CALL_GRAPH_H
#define MDL_COMPILERCORE_CALL_GRAPH_H 1

#include <mi/mdl/mdl_iowned.h>

#include "compilercore_cc_conf.h"
#include "compilercore_allocator.h"
#include "compilercore_memory_arena.h"

namespace mi {
namespace mdl {

class Definition;
class Call_node;
class IOutput_stream;

typedef vector<Call_node *>::Type       Call_node_vec;

typedef Arena_list<Call_node *>::Type   Arena_call_node_list;
typedef Arena_call_node_list::iterator  Callee_iterator;

///
/// An interface for the call graph walker.
///
class ICallgraph_visitor {
public:
    enum Order {
        PRE_ORDER,
        POST_ORDER
    };

    /// Visit a node of the call graph.
    virtual void visit_cg_node(Call_node *node, ICallgraph_visitor::Order order) = 0;
};

///
/// An interface for the call graph finisher.
///
class ICallgraph_scc_visitor {
public:
    /// Process a strongly coupled component inside of a call graph.
    virtual void process_scc(Call_node_vec const &scc) = 0;
};

///
/// An immutable node in the call graph.
///
class Call_node : public Interface_owned {
    friend class Call_graph;
    friend class Call_graph_walker;
    friend class Arena_builder;
public:
    enum Flag {
        FL_UNREACHABLE  = 0,       ///< node is not reachable
        FL_REACHABLE    = 1U << 0, ///< node is reachable from shader main
    };

public:
    /// Callee iterator start.
    Callee_iterator callee_begin() { return m_call_sites.begin(); }

    /// Callee iterator end.
    Callee_iterator callee_end() { return m_call_sites.end(); }

    /// Get the definition of this call node.
    Definition *get_definition() const { return m_def; }

    /// Check for flag.
    bool has_reachability_flag(Flag flag) const { return (m_reachability & flag) != 0; }

    /// Set reachability flag.
    void set_reachability_flag(Flag flag) { m_reachability |= flag; }

private:
    /// private Constructor.
    explicit Call_node(Memory_arena *arena, Definition *def);

    // non copyable
    Call_node(Call_node const &) MDL_DELETED_FUNCTION;
    Call_node &operator=(Call_node const &) MDL_DELETED_FUNCTION;

private:
    /// Add a callee.
    void add_callee(Call_node *callee);

private:

    /// Callees of this node.
    Arena_call_node_list m_call_sites;

    /// The Definition of this node.
    Definition *m_def;

    /// visit count for graph visitor.
    size_t m_visit_count;

    /// used for scc
    size_t m_dfs_num;
    size_t m_low;
    bool   m_on_stack;

    /// The reachability flags.
    unsigned m_reachability;
};

///
/// Represents the call graph.
///
class Call_graph {
    friend class Call_graph_walker;

public:
    struct Def_line_less {
        // compare Definitions by its unique id
        bool operator() (Definition const *d1, Definition const *d2) const;
    };

    typedef set<Definition *, Def_line_less>::Type Definition_set;

private:
    /// Depth first search.
    void do_dfs(Call_node *node);

    /// Called to process a non-trivial scc.
    void process_scc(Call_node_vec const &scc) const
    {
        if (m_scc_visitor != NULL)
            m_scc_visitor->process_scc(scc);
    }

    /// replace declarations by definitions in the call graph.
    void skip_declarations();

    /// Creates the initial root set.
    void create_root_set();

    /// Distribute the reachability in the call graph.
    void distribute_reachability();

    /// Calculate the strongly coupled components.
    ///
    /// \param node     the current call graph node
    /// \param visitor  if non-NULL, visit SCCs
    void calc_scc(Call_node *node, ICallgraph_scc_visitor *visitor);

public:

    /// Default constructor.
    ///
    /// \param alloc      the allocator
    /// \param name       name of the call graph (for debugging)
    /// \param is_stdlib  if true, this is a standard library module
    explicit Call_graph(
        IAllocator *alloc,
        char const *name,
        bool       is_stdlib);

    /// Return the call node for a definition.
    ///
    /// \param def  the definition to look up
    Call_node *get_call_node(Definition *def);

    /// Return the root call node for a definition.
    ///
    /// \param def  the definition to look up
    Call_node *get_root_call_node(Definition *def);

    /// Add a definition to the call graph.
    ///
    /// \param def  the definition to add
    void add_node(Definition *def);

    /// Remove a node from the call graph.
    ///
    /// \param def  the definition to remove
    void remove_node(Definition const *def);

    /// Add a call from caller to callee to the call graph.
    ///
    /// \param caller  the definition of the caller
    /// \param callee  the definition of the callee
    void add_call(Definition *caller, Definition *callee);

    /// Return the root set of the call graph.
    Call_node_vec const &get_root_set() const { return m_root_set; }

    /// Return the call graph for a material.
    ///
    /// \param mat  the material
    Call_node *get_call_graph_for_material(Definition *mat);

    /// Finalize the call graph and check for recursions.
    ///
    /// \param visitor  if non-NULL, the visitor for the strongly coupled components found
    void finalize(ICallgraph_scc_visitor *visitor = NULL);

    /// Return a set reference containing all function/method definitions.
    Definition_set const &get_definite_defs() const { return m_definition_set; }

    /// Get the call graph name.
    char const *get_name() const { return m_name; }

    /// Dump the call graph as a dot file.
    ///
    /// \param out  an output stream
    void dump(IOutput_stream *out) const;

    /// Get the allocator.
    IAllocator *get_allocator() const { return m_arena.get_allocator(); }

private:
    /// The memory arena for all allocated nodes.
    Memory_arena m_arena;

    /// The builder for call graph nodes.
    Arena_builder m_builder;

    /// The root set.
    Call_node_vec m_root_set;

    typedef ptr_hash_map<Definition, Call_node *>::Type  Definition_call_map;

    /// hash map mapping definitions to call nodes.
    Definition_call_map m_call_nodes;

    /// current visit count
    mutable size_t m_visit_count;

    /// used for scc
    size_t m_next_dfs_num;

    /// Callgraph visitor interface used inside process_scc() and walk()
    ICallgraph_visitor *m_visitor;

    /// Callgraph visitor interface used inside process_scc() and walk()
    ICallgraph_scc_visitor *m_scc_visitor;

    /// Stack for scc.
    typedef stack<Call_node *>::Type Call_node_stack;
    Call_node_stack m_stack;

    /// Set of all definite definitions inside this call graph.
    Definition_set m_definition_set;

    /// Name of the graph.
    char const *m_name;

    /// If true, this is a call graph of a standard library module.
    bool m_is_stdlib;
};

/// A Call graph walker.
class Call_graph_walker
{
public:
    /// SWalk over the graph using the root set
    ///
    /// \param cg       the call graph
    /// \param visitor  the visitor, may be NULL
    static void walk(
        Call_graph const   &cg,
        ICallgraph_visitor *visitor)
    {
        Call_graph_walker(cg, visitor).walk();
    }

    /// Walk the graph starting at a given root.
    ///
    /// \param root     the root
    void walk(Call_node *root);

    /// Walk the graph starting at the root set.
    void walk();

private:
    /// Walker.
    void do_walk(Call_node *node);

public:
    /// Constructor.
    ///
    /// \param cg       the call graph
    /// \param visitor  the visitor, may be NULL
    Call_graph_walker(
        Call_graph const   &cg,
        ICallgraph_visitor *visitor)
    : m_cg(cg)
    , m_visitor(visitor)
    {
    }

private:
    /// The call graph that is visited.
    Call_graph const &m_cg;

    /// The visitor used.
    ICallgraph_visitor *m_visitor;
};

}  // mdl
}  // mi

#endif
