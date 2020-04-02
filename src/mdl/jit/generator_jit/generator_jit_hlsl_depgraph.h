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

#ifndef MDL_GENERATOR_JIT_HLSL_DEPGRAPH_H
#define MDL_GENERATOR_JIT_HLSL_DEPGRAPH_H 1

#include <mdl/compiler/compilercore/compilercore_cc_conf.h>
#include <mdl/compiler/compilercore/compilercore_memory_arena.h>
#include <mdl/compiler/compilercore/compilercore_assert.h>
#include <mdl/compiler/compiler_hlsl/compiler_hlsl_definitions.h>

namespace mi {
namespace mdl {
namespace hlsl {

class Declaration;

/// A dependency graph node.
class DG_node {
    friend class mi::mdl::Arena_builder;
    friend class Dep_graph;

    typedef Arena_list<DG_node *>::Type Edge_list;
public:
    typedef Edge_list::const_iterator Edge_iterator;

public:
    /// Get the definition.
    Definition const *get_definition() const {
        return m_def;
    }

    /// Get the edge start iterator.
    Edge_iterator begin() const { return m_edge_list.begin(); }

    /// Get the edge end iterator.
    Edge_iterator end() const { return m_edge_list.end(); }

    /// returns true if this is a root node.
    bool is_root_node() const { return m_is_root; }

private:
    /// Add a new edge.
    void add_edge(DG_node *dst) {
        m_edge_list.push_back(dst);
    }

    /// Mark this node as non-root.
    void mark_non_root() { m_is_root = false; }

    /// Get the next created DG_node.
    DG_node const *get_next() const { return m_next; }

    /// Set the next created DG_node.
    void set_next(DG_node const *next) { m_next = next; }

private:
    /// Constructor from a function instance.
    ///
    /// \param arena  the memory arena to allocate on
    /// \param def    the function definition of the entity
    explicit DG_node(
        Memory_arena     *arena,
        Definition const *def)
    : m_def(def)
    , m_edge_list(arena)
    , m_next(NULL)
    , m_is_root(true)
    {
    }

    // no copy constructor
    DG_node(DG_node const &) MDL_DELETED_FUNCTION;

    // no assignment operator
    DG_node const &operator=(DG_node const &) MDL_DELETED_FUNCTION;

private:
    /// The definition of the entity.
    Definition const *m_def;

    /// The edge list, contains only unique entries.
    Edge_list m_edge_list;

    /// The previous created DG_node, to iterate in a deterministic order.
    DG_node const *m_next;

    /// True if this is a root node (no edge is pointing to this node).
    bool m_is_root;
};

/// Visitor interface for dependency graph walk.
class DG_visitor {
public:
    /// Pre-visit a node.
    virtual void pre_visit(DG_node const *node) = 0;

    /// Post-visit a node.
    virtual void post_visit(DG_node const *node) = 0;
};


/// The dependency graph.
class Dep_graph {
    typedef ptr_hash_map<
        Definition const,
        DG_node *>::Type Node_map;

    /// Helper class to lookup if an edge already exists
    class Edge {
    public:
        /// Constructor.
        Edge(Definition const *src, Definition const *dst)
        : m_src(src), m_dst(dst)
        {
        }

        /// Hash this entry.
        size_t hash() const {
            Hash_ptr<Definition> hasher;
            return (hasher(m_src) * 7) ^ (hasher(m_dst) * 3);
        }

        /// Compare two entries.
        bool operator==(Edge const &o) const {
            return m_dst == o.m_dst && m_src == o.m_src;
        }

    private:
        /// The source of this edge.
        Definition const *m_src;
        /// The destination of this edge.
        Definition const *m_dst;
    };

    class Hasher {
    public:
        size_t operator()(Edge const &p) const {
            return p.hash();
        }
    };

    class Equal {
    public:
        bool operator()(Edge const &p, Edge const &q) const {
            return p == q;
        }
    };

    typedef hash_set<Edge, Hasher, Equal>::Type Edge_set;
    typedef ptr_hash_set<DG_node const>::Type   DG_node_set;

public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    explicit Dep_graph(IAllocator *alloc)
    : m_arena(alloc)
    , m_node_map(0, Node_map::hasher(), Node_map::key_equal(), alloc)
    , m_edge_set(0, Edge_set::hasher(), Edge_set::key_equal(), alloc)
    , m_first_node(NULL)
    , m_last_node(NULL)
    {
    }

    /// Destructor.
    ~Dep_graph() {
        // the nodes are allocated on the arena, buf they contain Function_instances that must be
        // destructed
        for (DG_node *p = m_first_node; p != NULL;) {
            DG_node *n = const_cast<DG_node *>(p->get_next());
            p->~DG_node();
            p = n;
        }
        m_first_node = m_last_node = NULL;
    }

    /// Get the node for a definition, create one if necessary.
    ///
    /// \param def  a function definition
    DG_node *get_node(Definition const *def) {
        Node_map::const_iterator it = m_node_map.find(def);
        if (it != m_node_map.end()) {
            return it->second;
        }
        Arena_builder builder(m_arena);

        DG_node *res = builder.create<DG_node>(&m_arena, def);
        m_node_map.insert(Node_map::value_type(def, res));

        if (m_first_node == NULL) {
            m_first_node = res;
        } else {
            m_last_node->set_next(res);
        }
        m_last_node = res;

        return res;
    }

    /// Add a new edge if it does not yet exists.
    ///
    /// \param src  the source node (the caller) of the edge
    /// \param dst  the destination node (the callee) of the edge
    bool add_edge(DG_node *src, DG_node *dst) {
        Edge ee(src->get_definition(), dst->get_definition());

        if (m_edge_set.insert(ee).second) {
            // new edge
            src->add_edge(dst);
            // if dst is called from src, it cannot be a root
            dst->mark_non_root();
            return true;
        }
        // already exists
        return false;
    }

    /// Walk over the dependency graph.
    ///
    /// \param visitor  the node visitor interface
    void walk(DG_visitor &visitor)
    {
        IAllocator *alloc = m_arena.get_allocator();
        DG_node_set marker(0, DG_node_set::hasher(), DG_node_set::key_equal(), alloc);

        for (DG_node const *p = m_first_node; p != NULL; p = p->get_next()) {
            if (p->is_root_node()) {
                do_walk(p, visitor, marker);
            }
        }
    }

private:
    /// Helper, walk over a DAG, pre/post visit nodes.
    ///
    /// \param node     the currently visited node
    /// \param visitor  the visitor interface
    /// \param marker   the set of all already visited nodes
    static void do_walk(DG_node const *node, DG_visitor &visitor, DG_node_set &marker)
    {
        if (!marker.insert(node).second) {
            // already visited
            return;
        }
        visitor.pre_visit(node);

        for (DG_node::Edge_iterator it(node->begin()), end(node->end()); it != end; ++it) {
            DG_node const *child = *it;
            do_walk(child, visitor, marker);
        }
        visitor.post_visit(node);
    }

private:
    /// The memory arena where nodes are allocated on.
    Memory_arena m_arena;

    /// The node map.
    Node_map m_node_map;

    /// The set of all edges for fast test if an edges already exists.
    Edge_set m_edge_set;

    /// First created DG node.
    DG_node *m_first_node;

    /// Last created DG node.
    DG_node *m_last_node;
};

}  // hlsl
}  // mdl
}  // mi

#endif // MDL_GENERATOR_JIT_HLSL_DEPGRAPH_H
