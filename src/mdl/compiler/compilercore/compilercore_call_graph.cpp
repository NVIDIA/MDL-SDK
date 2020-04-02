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

#include <mi/mdl/mdl_streams.h>

#include "compilercore_cc_conf.h"

#include "compilercore_call_graph.h"
#include "compilercore_def_table.h"
#include "compilercore_assert.h"
#include "compilercore_allocator.h"
#include "compilercore_printers.h"
#include "compilercore_tools.h"

#include <cstdio>

namespace mi {
namespace mdl {

namespace {

template<typename T>
MDL_CONSTEXPR T min(T a, T b) { return a <= b ? a : b; }
}

bool Call_graph::Def_line_less::operator() (Definition const *d1, Definition const *d2) const {
    return d1->get_unique_id() < d2->get_unique_id();
}

// private Constructor.
Call_node::Call_node(Memory_arena *arena, Definition *def)
 : m_call_sites(arena)
 , m_def(def)
 , m_visit_count(0)
 , m_dfs_num(0)
 , m_low(0)
 , m_on_stack(false)
 , m_reachability(FL_UNREACHABLE)
{
}

// Add a callee.
void Call_node::add_callee(Call_node *callee)
{
    // check if we hasn't set it already
    for (Callee_iterator it = m_call_sites.begin(), end = m_call_sites.end(); it != end; ++it) {
        if (*it == callee) {
            // already here
            return;
        }
    }
    m_call_sites.push_back(callee);
}

// Default constructor.
Call_graph::Call_graph(
    IAllocator *alloc,
    char const *name,
    bool       is_stdlib)
: m_arena(alloc)
, m_builder(m_arena)
, m_root_set(alloc)
, m_call_nodes(0, Definition_call_map::hasher(), Definition_call_map::key_equal(), alloc)
, m_visit_count(0)
, m_next_dfs_num(0)
, m_visitor(NULL)
, m_scc_visitor(NULL)
, m_stack(Call_node_stack::container_type(alloc))
, m_definition_set(Definition_set::key_compare(), alloc)
, m_name(Arena_strdup(m_arena, name != NULL ? name : "call_graph"))
, m_is_stdlib(is_stdlib)
{
}

// Return the call node for a definition.
Call_node *Call_graph::get_call_node(Definition *def)
{
    Definition_call_map::iterator it = m_call_nodes.find(def);
    if (it != m_call_nodes.end())
        return it->second;
    Call_node *node = m_builder.create<Call_node>(&m_arena, def);
    m_call_nodes[def] = node;
    return node;
}

// Return the root call node for a definition.
Call_node *Call_graph::get_root_call_node(Definition *def)
{
    Definition_call_map::iterator it = m_call_nodes.find(def);
    if (it != m_call_nodes.end())
        return it->second;
    Call_node *node = m_builder.create<Call_node>(&m_arena, def);
    m_call_nodes[def] = node;
    m_root_set.push_back(node);
    return node;
}

// Add a definition to the call graph.
void Call_graph::add_node(Definition *def)
{
    MDL_ASSERT(def != NULL);
    (void)get_call_node(def);
}

// Remove a node from the call graph.
void Call_graph::remove_node(Definition const *def)
{
    MDL_ASSERT(def != NULL);

    Definition_call_map::iterator it = m_call_nodes.find(const_cast<Definition *>(def));
    if (it != m_call_nodes.end()) {
        Call_node *node = it->second;
        m_call_nodes.erase(it);

        Call_node_vec::iterator rit = std::find(m_root_set.begin(), m_root_set.end(), node);
        if (rit != m_root_set.end()) {
            m_root_set.erase(rit);
        }
        // the node cannot be freed, but we could add it to a pool so save some space
    }
}

// Add a call from caller to callee to the call graph.
void Call_graph::add_call(Definition *caller, Definition *callee)
{
    MDL_ASSERT(caller != NULL && caller != NULL);
    Call_node *caller_node = get_call_node(caller);
    Call_node *callee_node = get_call_node(callee);
    caller_node->add_callee(callee_node);
}

// Return the call graph for a material.
Call_node *Call_graph::get_call_graph_for_material(Definition *mat)
{
    Call_node *node = get_call_node(mat);
    for (size_t i = 0, n = m_root_set.size(); i < n; ++i) {
        if (m_root_set[i] == node)
            return node;
    }
    return NULL;
}

// Calculate the strongly coupled components.
void Call_graph::do_dfs(Call_node *node)
{
    node->m_visit_count = m_visit_count;

    node->m_dfs_num = m_next_dfs_num++;
    node->m_low     = node->m_dfs_num;

    m_stack.push(node);
    node->m_on_stack = true;

    for (Callee_iterator it = node->callee_begin(); it != node->callee_end(); ++it) {
        Call_node *pred = *it;

        if (pred->m_visit_count < m_visit_count) {
            do_dfs(pred);
            node->m_low = min(node->m_low, pred->m_low);
        }
        if (pred->m_dfs_num < node->m_dfs_num && pred->m_on_stack) {
            node->m_low = min(pred->m_dfs_num, node->m_low);
        }
    }

    if (node->m_low == node->m_dfs_num) {
        Call_node *x;
        // found SCC
        Call_node_vec scc(m_arena.get_allocator());
        do {
            x = m_stack.top();
            x->m_on_stack = false;
            m_stack.pop();
            scc.push_back(x);
        } while (x != node);

        size_t size = scc.size();
        if (size == 1) {
            // single entry scc
            x = scc[0];

            // check for self recursion
            for (Callee_iterator it = x->callee_begin(); it != x->callee_end(); ++it) {
                if (x == *it) {
                    Call_node_vec scc(m_arena.get_allocator());
                    scc.push_back(x);
                    process_scc(scc);
                    break;
                }
            }
        } else {
            // recursion through more than one function
            process_scc(scc);
        }
    }
}

// Calculate the strongly coupled components.
void Call_graph::calc_scc(Call_node *node, ICallgraph_scc_visitor *visitor)
{
    Store<ICallgraph_scc_visitor *> store(m_scc_visitor, visitor);

    ++m_visit_count;
    m_next_dfs_num = 0;
    do_dfs(node);
}

// Finalize the call graph and check for recursions.
void Call_graph::finalize(ICallgraph_scc_visitor *visitor)
{
    skip_declarations();
    create_root_set();
    distribute_reachability();

    for (size_t i = 0, n = m_root_set.size(); i < n; ++i)
        calc_scc(m_root_set[i], visitor);

    // fill the definite definitions map
    m_definition_set.clear();
    Definition_call_map::iterator it, end;
    for (it = m_call_nodes.begin(), end = m_call_nodes.end(); it != end; ++it) {
        Call_node  *node = it->second;
        Definition *def  = node->get_definition();

        def = def->get_definite_definition();
        if (def)
            m_definition_set.insert(def);
    }
}

// Create the root set of the call graph.
void Call_graph::create_root_set()
{
    size_t curr_visited_count = ++m_visit_count;

    Call_graph_walker walker(*this, NULL);

    // create the initial root set: shader main, constructor, destructor
    for (Definition_call_map::iterator it = m_call_nodes.begin(), end = m_call_nodes.end();
        it != end;
        ++it)
    {
        Definition const *def  = it->first;
        Call_node        *node = it->second;

        MDL_ASSERT(def == node->get_definition());

        // ignore declarations, they are useless roots but beware of standard library functions,
        // because those do not have a definition
        if (!def->has_flag(Definition::DEF_IS_DECL_ONLY) || m_is_stdlib) {
            // exported functions are roots
            if (def->has_flag(Definition::DEF_IS_EXPORTED) &&
                def->get_kind() == Definition::DK_FUNCTION) {
                m_root_set.push_back(node);
            }
        }
    }

    for (size_t i = 0, n = m_root_set.size(); i < n; ++i) {
        Call_node *root = m_root_set[i];

        // set reachability
        root->set_reachability_flag(Call_node::FL_REACHABLE);
        walker.walk(root);
    }

    Call_node_vec all_nodes(m_arena.get_allocator());
    for (Definition_call_map::iterator it = m_call_nodes.begin(), end = m_call_nodes.end();
         it != end;
         ++it)
    {
        Call_node *node = it->second;

        if (node->m_visit_count < curr_visited_count)
            all_nodes.push_back(node);
    }

    // all non-visited nodes are unreachable: mark them and put them into the root set
    for (size_t i = 0, n = all_nodes.size(); i < n; ++i) {
        Call_node *node = all_nodes[i];

        if (node->m_visit_count < curr_visited_count) {
            // found an unreachable node
            Definition *def = node->get_definition();

            // ignore declarations, they are useless roots
            if (!def->has_flag(Definition::DEF_IS_DECL_ONLY)) {
                // add a new (unreachable) root
                m_root_set.push_back(node);
                walker.walk(node);
            }
        }
    }
}

// Replace all declarations by definite definitions in the call-graph.
void Call_graph::skip_declarations()
{
    for (Definition_call_map::iterator it = m_call_nodes.begin(), end = m_call_nodes.end();
        it != end;
        ++it)
    {
        Call_node *node = it->second;

        for (Callee_iterator it = node->callee_begin(), end = node->callee_end(); it != end; ++it)
        {
            Call_node  *callee      = *it;
            Definition *callee_def  = callee->get_definition();
            Definition *definite    = callee_def->get_definite_definition();

            if (definite && callee_def != definite) {
                // callee_def is a declaration only, definite is its definition, replace it.
                // Note that declarations cannot have callee's, so this
                // operation is simple.
                MDL_ASSERT(callee->callee_begin() == callee->callee_end());

                *it = get_call_node(definite);
            }
        }
    }
}

// Distribute the reachability flags through the callgraph.
void Call_graph::distribute_reachability()
{
    // Note: we do it this way because every node might be reached more than once ...
    typedef mi::mdl::queue<Call_node *>::Type Queue;
    
    Queue queue(Queue::container_type(m_arena.get_allocator()));

    // push all reachable roots
    for (size_t i = 0, n = m_root_set.size(); i < n; ++i) {
        Call_node *root = m_root_set[i];

        if (root->m_reachability != Call_node::FL_UNREACHABLE)
            queue.push(root);
    }

    while (! queue.empty()) {
        Call_node *node = queue.front();
        queue.pop();

        unsigned reachability = node->m_reachability;
        for (Callee_iterator it = node->callee_begin(), end = node->callee_end();
            it != end;
            ++it)
        {
            Call_node *callee = *it;
            unsigned  c_reachability = callee->m_reachability;

            if ((c_reachability | reachability) != c_reachability) {
                callee->m_reachability |= reachability;
                queue.push(callee);
            }
        }
    }

    // write the distributed info back into the definitions
    Definition_call_map::iterator it, end;
    for (it = m_call_nodes.begin(), end = m_call_nodes.end(); it != end; ++it) {
        Call_node  *node = it->second;
        Definition *def  = node->get_definition();
        unsigned   reachability = node->m_reachability;

        if (reachability & Call_node::FL_REACHABLE) {
            // this function/method is reachable from shader main
            def->set_flag(Definition::DEF_IS_REACHABLE);
        }
    }
}

namespace {

/// Helper class to dump a call graph into a VCG file.
class Dumper : public ICallgraph_visitor {
public:
    /// Constructor.
    explicit Dumper(
        IAllocator       *alloc,
        IOutput_stream   *out,
        Call_graph const &cg);

    /// Dump the graph.
    void dump();

private:
    void node_name(Call_node const *n);
    void node(Call_node const *n, char const *color = NULL);
    void edge(Call_node const *src, Call_node const *dst, char const *color = NULL);

    void visit_cg_node(Call_node *node, ICallgraph_visitor::Order order) MDL_FINAL;

private:
    Call_graph const          &m_cg;
    mi::base::Handle<Printer> m_printer;
    size_t                    m_depth;
};

Dumper::Dumper(
    IAllocator       *alloc,
    IOutput_stream   *out,
    Call_graph const &cg)
: m_cg(cg)
, m_printer()
, m_depth(0)
{
    Allocator_builder builder(alloc);

    m_printer = mi::base::make_handle(builder.create<Printer>(alloc, out));
}

void Dumper::dump()
{
    m_printer->print("digraph \"");
    char const *name = m_cg.get_name();
    if (name == NULL || name[0] == '\0')
        name = "call_graph";
    m_printer->print(name);
    m_printer->print("\" {\n");
    m_printer->print("root [label=\"RootSet\"];\n");

    Call_graph_walker::walk(m_cg, this);

    // create the "virtual" root
    Call_node_vec const &root_set = m_cg.get_root_set();
    for (size_t i = 0, n = root_set.size(); i < n; ++i) {
        Call_node *root = root_set[i];

        edge(NULL, root, "blue");
    }
    m_printer->print("}\n");
}

void Dumper::node_name(Call_node const *n)
{
    char buf[32];

    snprintf(buf, sizeof(buf), "n%p", n);
    buf[sizeof(buf) - 1] = '\0';
    m_printer->print(buf);
}

void Dumper::node(Call_node const *n, char const *color)
{
    bool use_box_shape = false;

    m_printer->print("  ");
    node_name(n);
    m_printer->print(" [label=\"");
    Definition const *def = n->get_definition();

    Definition::Kind kind = def->get_kind();
    if (kind != Definition::DK_FUNCTION && kind != Definition::DK_CONSTRUCTOR) {
        m_printer->print(def->get_sym());
    } else {
        IType_function const *func_type = cast<IType_function>(def->get_type());

        if (kind == Definition::DK_FUNCTION) {
            IType const *ret_type = func_type->get_return_type();

            IType_struct const *s_type = as<IType_struct>(ret_type);
            if (s_type != NULL && s_type->get_predefined_id() == IType_struct::SID_MATERIAL)
                use_box_shape = true;

            m_printer->print(ret_type);
            m_printer->print(" ");
        }

        m_printer->print(def->get_sym());
        m_printer->print("(");
        for (int i = 0, n = func_type->get_parameter_count(); i < n; ++i) {
            IType const   *p_type;
            ISymbol const *p_sym;
            func_type->get_parameter(i, p_type, p_sym);

            if (i > 0)
                m_printer->print(", ");
            m_printer->print(p_type);
        }
    }
    m_printer->print(")\"");

    if (color != NULL) {
        m_printer->print(" color=");
        m_printer->print(color);
    }

    if (use_box_shape)
        m_printer->print(" shape=box");

    m_printer->print("];\n");
}

void Dumper::edge(Call_node const *src, Call_node const *dst, char const *color)
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

void Dumper::visit_cg_node(Call_node *n, ICallgraph_visitor::Order order)
{
    if (order == ICallgraph_visitor::PRE_ORDER) {
        char const *color = m_depth > 0 ? NULL : "blue";
        node(n, color);
        ++m_depth;
    } else {
        for (Callee_iterator it = n->callee_begin(), end = n->callee_end();
             it != end;
             ++it)
        {
            Call_node *callee = *it;

            edge(n, callee);
        }
        --m_depth;
    }
}

}  // anonymous

// Dump the call graph as a dot file.
void Call_graph::dump(IOutput_stream *out) const
{
    Dumper dumper(m_builder.get_arena()->get_allocator(), out, *this);

    dumper.dump();
}

// ------------------------------- visitor -------------------------------

// Calculate the strongly coupled components.
void Call_graph_walker::do_walk(Call_node *node)
{
    node->m_visit_count = m_cg.m_visit_count;

    if (m_visitor != NULL)
        m_visitor->visit_cg_node(node, ICallgraph_visitor::PRE_ORDER);
    for (Callee_iterator it = node->callee_begin(); it != node->callee_end(); ++it) {
        Call_node *callee = *it;
        if (callee->m_visit_count < m_cg.m_visit_count) {
            do_walk(callee);
        }
    }
    if (m_visitor != NULL)
        m_visitor->visit_cg_node(node, ICallgraph_visitor::POST_ORDER);
}

// Walk the graph, starting at a given root.
void Call_graph_walker::walk(Call_node *root)
{
    ++m_cg.m_visit_count;
    do_walk(root);
}

// Walk the graph starting at the root set.
void Call_graph_walker::walk()
{
    ++m_cg.m_visit_count;
    for (Call_node_vec::const_iterator it(m_cg.m_root_set.begin()), end(m_cg.m_root_set.end());
        it != end;
        ++it)
    {
        Call_node *root = *it;
        if (root->m_visit_count < m_cg.m_visit_count) {
            do_walk(root);
        }
    }
}


}  // mdl
}  // mi

