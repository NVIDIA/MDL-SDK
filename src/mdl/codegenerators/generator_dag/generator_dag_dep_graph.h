/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_GENERATOR_DAG_DEP_GRAPH_H
#define MDL_GENERATOR_DAG_DEP_GRAPH_H 1

#include <mi/mdl/mdl_mdl.h>

#include "mdl/compiler/compilercore/compilercore_memory_arena.h"
#include "mdl/compiler/compilercore/compilercore_cstring_hash.h"
#include "mdl/compiler/compilercore/compilercore_array_ref.h"

namespace mi {
namespace mdl {

class DAG_builder;
class Dependence_node;
class Generated_code_dag;
class IDefinition;
class Indexable;

/// A dependence edge.
class Dependence_edge {
    friend class Arena_builder;
    friend class Dependence_node;
    friend class DG_walker;

public:
    /// Get the destination of this edge.
    Dependence_node *get_dst() const { return m_dst; }

    /// Returns true if this is a true dependence edge.
    bool is_true_dependence() const { return m_true_dep; }

    /// Make the dependence true.
    void make_true_dependence() { m_true_dep = true; }

    /// Compare two edges
    bool operator==(Dependence_edge const &other) const {
        // Two edges are identical, if the destination is.
        return m_dst == other.m_dst;
    }

    /// Compare two edges
    bool operator!=(Dependence_edge const &other) const {
        return !operator==(other);
    }

private:
    /// Constructor.
    Dependence_edge(
        Dependence_node *dst,
        bool            true_dep)
    : m_dst(dst)
    , m_true_dep(true_dep)
    {
    }

private:
    /// The destination of the edge.
    Dependence_node *m_dst;

    /// If true, this is a true dependency, else a dependency through default parameter
    bool m_true_dep;
};

typedef Arena_list<Dependence_node *>::Type Node_list;
typedef Arena_list<Dependence_edge *>::Type Edge_list;
typedef Edge_list::const_iterator           Edge_iterator;
typedef vector<Dependence_node *>::Type     Node_vec;

///
/// A node in the dependence graph. It represents either a real functions, or a DAG
/// intrinsic.
///
class Dependence_node {
    friend class Arena_builder;

public:
    typedef Edge_list::const_iterator Edge_iterator;

    /// Possible flags.
    enum Flags {
        FL_IS_BUILTIN  = 1 << 0,  ///< This node is a built-in entity.
        FL_IS_IMPORTED = 1 << 1,  ///< This node is imported.
        FL_IS_EXPORTED = 1 << 2,  ///< This node is exported.
    };

    /// A value class for handling DAG intrinsic parameters.
    class Parameter{
        friend class Dependence_node;
    public:
        /// Constructor.
        Parameter(
            IType const *type,
            char const  *name)
        : m_type(type)
        , m_name(name)
        {
        }

    private:
        IType const *m_type;
        char const  *m_name;
    };

public:
    /// Get the next node.
    Dependence_node *get_next() { return m_next; }

    /// Set the next node.
    void set_next(Dependence_node *next) { m_next = next; }

    /// Get the definition of this node.
    IDefinition const *get_definition() const { return m_def; }

    /// Add a dependency edge.
    ///
    /// \param arena        the arena to allocate an edge on
    /// \param callee       the current node depends on callee
    /// \param is_def_arg   true, if this dependency comes through a default argument
    void add_edge(
        Memory_arena    &arena,
        Dependence_node *callee,
        bool            is_def_arg);

    /// Get the callee edges list.
    Edge_list const &get_callee_edges() const { return m_edgess; }

    /// Get the first edge.
    Edge_iterator edges_begin() const { return m_edgess.begin(); }

    /// Get the end edge.
    Edge_iterator edges_end() const { return m_edgess.end(); }

    /// Returns true if this node is a root node.
    bool is_root() const { return m_is_root; }

    /// Returns true if this node was not yet visited for the given visit count.
    ///
    /// \param visit_count  the visit count
    bool mark_visited(size_t visit_count);

    /// get the DFS num.
    size_t get_dfs_num() const { return m_dfs_num; }

    /// Set the DFS num.
    void set_dfs_num(size_t num);

    /// Get the low value.
    size_t get_low() const { return m_low; }

    /// Set the low value.
    void set_low(size_t low) { m_low = low; }

    /// Get the id of this node.
    size_t get_id() const { return m_id; }

    /// Returns true, if this node is a local entity (i.e. not imported).
    bool is_local() const;

    /// Returns true, if this node is a local export.
    bool is_local_export() const;

    /// Returns true, if this node is a (local or re-) export.
    bool is_export() const;

    /// Get the DAG name of this node.
    char const *get_dag_name() const { return m_dag_name; }

    /// Get the DAG simple name of this node.
    char const *get_dag_simple_name() const { return m_dag_simple_name; }

    /// Get the DAG alias name of this node if any.
    char const *get_dag_alias_name() const { return m_dag_alias_name; }

    /// Get the DAG preset name of this node if any.
    char const *get_dag_preset_name() const { return m_dag_preset_name; }

    /// Get the parameter count of this node.
    size_t get_parameter_count() const { return m_n_params; }

    /// Get the parameter type and name of this node at the given index.
    ///
    /// \param      index  the parameter index
    /// \param[out] type   the type of the parameter
    /// \param[out] name   the name of the parameter
    void get_parameter(
        size_t      index,
        IType const *&type,
        char const  *&name) const;

    /// Get the return type of this node.
    IType const *get_return_type() const { return m_ret_type; }

    /// Get the semantics of this node.
    IDefinition::Semantics get_semantics() const { return m_sema; }

    /// Check if the node is on the stack.
    bool is_on_stack() const { return m_on_stack; }

    /// Mark the node to by on the stack.
    void mark_on_stack(bool flag) { m_on_stack = flag; }

private:
    /// Constructor from a definition.
    ///
    /// \param arena           the arena to allocate from
    /// \param id              the id of this node
    /// \param def             the definition
    /// \param dag_name        the DAG (mangled) name of this node
    /// \param dag_simple_name the DAG simple name of this node
    /// \param dag_alias_name  the DAG (mangled) alias name of this node
    /// \param dag_preset_name the DAG (mangled) preset name of this node
    /// \param flags           node flags for this node
    /// \param next            the next node in the topological sort
    Dependence_node(
        Memory_arena      *arena,
        size_t            id,
        IDefinition const *def,
        char const        *dag_name,
        char const        *dag_simple_name,
        char const        *dag_alias_name,
        char const        *dag_preset_name,
        unsigned          flags,
        Dependence_node   *next);

    /// Constructor from a name + semantic.
    ///
    /// \param arena           the arena to allocate from
    /// \param id              the id of this node
    /// \param sema            the semantic
    /// \param ret_type        the return type of this node
    /// \param params          the parameters of this node
    /// \param dag_name        the DAG (mangled) name of this node
    /// \param dag_simple_name the DAG simple name of this node
    /// \param dag_alias_name  the DAG (mangled) alias name of this node
    /// \param dag_preset_name the DAG (mangled) preset name of this node
    /// \param flags           node flags for this node
    /// \param next            the next node in the topological sort
    Dependence_node(
        Memory_arena               *arena,
        size_t                     id,
        IDefinition::Semantics     sema,
        IType const                *ret_type,
        Array_ref<Parameter> const &params,
        char const                 *dag_name,
        char const                 *dag_simple_name,
        char const                 *dag_alias_name,
        char const                 *dag_preset_name,
        unsigned                   flags,
        Dependence_node            *next);

private:
    /// The owner definition of this node if any.
    IDefinition const *m_def;

    /// The semantic of this node.
    IDefinition::Semantics m_sema;

    /// The return type of this node.
    IType const  *m_ret_type;

    /// The parameters of this node.
    Parameter const *m_params;

    /// The number of parameters.
    size_t m_n_params;

    /// Bitset of Node flags,
    unsigned m_flags;

    /// The DAG name of this node.
    char const *m_dag_name;

    /// The DAG simple name of this node.
    char const *m_dag_simple_name;

    /// The DAG alias name of this node.
    char const *m_dag_alias_name;

    /// The DAG preset name of this node.
    char const *m_dag_preset_name;

    /// Points to the next node in the topological sort.
    Dependence_node *m_next;

    /// The list of callees.
    Edge_list m_edgess;

    // The ID of this node.
    size_t const m_id;

    /// The current visit count of this node.
    size_t m_visit_count;

    /// used for scc
    size_t m_dfs_num;
    size_t m_low;
    bool m_on_stack;

    /// True if this node is a root node.
    bool m_is_root;
};

/// Visitor interface for the dependence graph walker.
class IDG_visitor {
public:
    enum Order { PRE_ORDER, POST_ORDER };

    // Visit a node.
    virtual void visit(Dependence_node const *node, Order order) = 0;
};

/// The dependence-graph for DAGs.
///
/// This is basically a function call-graph, but contains also edges from functions to
/// callees inside the default arguments. Due to this, it might contain loops.
///
class DAG_dependence_graph {
public:
    typedef Arena_list<Dependence_node *>::Type                      Node_list;
    typedef Arena_vector<IDefinition const *>::Type                  Definition_vec;
    typedef ptr_hash_map<IDefinition const, Dependence_node *>::Type Def_node_map;

    friend class DG_walker;
    friend class DG_creator;

    /// Constructor.
    ///
    /// \param alloc           the allocator
    /// \param dag             the DAG to report errors
    /// \param dag_builder     the DAG builder this graph is constructed from.
    /// \param invisible_sym   the invisible symbol of the current module
    /// \param include_locals  if true, include local entities in the dependence graph
    DAG_dependence_graph(
        IAllocator           *alloc,
        Generated_code_dag   &dag,
        DAG_builder          &dag_builder,
        ISymbol const        *invisible_sym,
        bool                 include_locals);

    /// Get the exported nodes of the given module.
    Node_list const &get_exported_module_nodes();

    /// Get the entity nodes of the given module in topological order.
    ///
    /// \param[out] has_loops  if true, no topological sort possible
    Node_list const &get_module_entities(bool &has_loops);

    /// Get the dependency node for the given definition.
    ///
    /// \param def  the definition of the node
    Dependence_node *get_node(IDefinition const *def);

    /// Get the DAG dependency node for the given entity.
    ///
    /// \param dag_name        the DAG name of this node
    /// \param dag_simple_name the DAG simple name of this node
    /// \param dag_alias_name  the DAG alias name of this node
    /// \param dag_preset_name the DAG preset name of this node
    /// \param sema            the semantics of the node
    /// \param ret_type        the return type of this node
    /// \param params          the parameter descriptions of this node
    /// \param flags           the node flags
    Dependence_node *get_node(
        char const                                  *dag_name,
        char const                                  *dag_simple_name,
        char const                                  *dag_alias_name,
        char const                                  *dag_preset_name,
        IDefinition::Semantics                      sema,
        IType const                                 *ret_type,
        Array_ref<Dependence_node::Parameter> const &params,
        unsigned                                    flags);

    /// Check if the given definition already exists in the dependency graph.
    bool node_exists(IDefinition const *def) const;

    /// Walk over the dependence graph.
    ///
    /// \param visitor  the visitor
    void walk(IDG_visitor &visitor);

    /// Dump the dependency graph.
    ///
    /// \param file_name  file name for dump
    ///
    /// \return true on success, false of file error
    bool dump(char const *file_name);

    /// Returns true for MDL definition that should not be visible in the DAG backend.
    ///
    /// \param def  the definition
    static bool skip_definition(IDefinition const *def);

    /// Called, if a dependence loop was detected.
    ///
    /// \param nodes  the nodes that form the loop.
    void has_dependence_loop(Node_vec const &nodes);

private:
    /// Create all necessary nodes for an exported definition.
    ///
    /// \param module   the current module
    /// \param exp_def  an exported definition
    void create_exported_nodes(
        IModule const     *module,
        IDefinition const *exp_def);

    /// Create one-and-only index operator node.
    ///
    /// \param indexable_type  the template type for all indexable types, (<0>[])
    /// \param int_type        the MDL integer type
    void create_dag_index_function(
        IType const *indexable_type,
        IType const *int_type);

    /// Create one-and-only array length operator node.
    ///
    /// \param indexable_type  the template type for all indexable types, (<0>[])
    /// \param int_type        the MDL integer type
    Dependence_node *create_dag_array_len_operator(
        IType const *indexable_type,
        IType const *int_type);

    /// Create the one-and-only ternary operator node.
    ///
    /// \param type           the type of the return type, true, and false expressions
    /// \param bool_type      the MDL boolean type
    void create_dag_ternary_operator(
        IType const *type,
        IType const *bool_type);

    /// Create the one-and-only array constructor node.
    ///
    /// \param int_type  the any type
    void create_dag_array_constructor(IType const *any_type);

    /// Create the one-and-only cast operator.
    ///
    /// \param int_type  the any type
    void create_dag_cast_operator(IType const *any_type);

private:
    /// The memory arena for all dependency nodes.
    Memory_arena m_arena;

    /// The arena builder, used to create nodes.
    Arena_builder m_builder;

    /// The DAG to report errors.
    Generated_code_dag &m_dag;

    /// The DAG builder described by this graph.
    DAG_builder &m_dag_builder;

    /// The invisible symbol of the current module.
    ISymbol const *m_invisible_sym;

    /// List of exported nodes.
    Node_list m_exported_nodes;

    /// The topologically ordered list of nodes in this module.
    Node_list m_list;

    /// The map of already created nodes from definitions.
    Def_node_map m_def_node_map;

    typedef hash_map<
        char const *,
        Dependence_node *,
        cstring_hash,
        cstring_equal_to>::Type Name_node_map;

    /// The map of already created nodes from (DAG) names.
    Name_node_map m_name_node_map;

    /// Points to the last created node.
    Dependence_node *m_last;

    /// Next node id.
    size_t m_next_id;

    /// The visit count for walker on this dependence graph.
    mutable size_t m_visit_count;

    /// if true, local entities are included in the dependence graph, else only exported once.
    bool m_include_locals;

    /// True if we build det DG for the builtins module.
    bool m_is_builtins;

    /// if true, the dependence graph has loops.
    bool m_has_loops;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_DAG_DEP_GRAPH_H
