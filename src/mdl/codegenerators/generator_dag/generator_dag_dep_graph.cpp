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

#include "pch.h"

#include <cstdio>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_modules.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_printers.h"
#include "mdl/compiler/compilercore/compilercore_streams.h"
#include "mdl/compiler/compilercore/compilercore_visitor.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"

#include "generator_dag_dep_graph.h"
#include "generator_dag_builder.h"
#include "generator_dag_generated_dag.h"
#include "generator_dag_tools.h"

namespace mi {
namespace mdl {

typedef DAG_dependence_graph::Def_node_map Def_node_map;
typedef Store<bool>                        Flag_store;

/// A walker over the dependency graph.
class DG_walker {
public:
    /// Constructor.
    ///
    /// \param alloc  the allocator
    /// \param graph  the graph that will be visited
    DG_walker(
        IAllocator           *alloc,
        DAG_dependence_graph &graph)
    : m_alloc(alloc)
    , m_dg(graph)
    , m_last(NULL)
    , m_first(NULL)
    , m_visit_count(graph.m_visit_count)
    , m_next_dfs_num(0)
    , m_stack(Dep_node_stack::container_type(alloc))
    , m_has_loops(false)
    {
    }

    /// Walk the graph and link them together in topological order.
    Dependence_node *link_nodes_in_topo_order()
    {
        ++m_visit_count;
        m_last = NULL;
        do_link_nodes(m_dg.get_exported_module_nodes());
        if (m_last != NULL)
            m_last->set_next(NULL);
        m_last = NULL;

        Dependence_node *res = m_first;
        m_first = NULL;
        return res;
    }

    /// Walk the graphs.
    ///
    /// \param visitor  the visitor interface
    void walk(IDG_visitor &visitor)
    {
        ++m_visit_count;
        do_walk(m_dg.get_exported_module_nodes(), visitor);
    }

    /// Start the scc run.
    void start_scc() {
        ++m_visit_count;
    }

    /// Calculate the strongly coupled components and process any found loop.
    ///
    /// \param edge  one edge
    bool calc_scc(Dependence_node *node)
    {
        m_next_dfs_num = 0;

        node->mark_visited(m_visit_count);
        do_dfs(node);

        return m_has_loops;
    }

private:
    /// Depth first search.
    ///
    /// \param n  the node to follow
    void do_dfs(Dependence_node *n)
    {
        n->set_dfs_num(m_next_dfs_num++);

        m_stack.push(n);
        n->mark_on_stack(true);

        Edge_list const &edges = n->get_callee_edges();
        for (Edge_list::const_iterator it(edges.begin()), end(edges.end()); it != end; ++it) {
            Dependence_node *pred = (*it)->get_dst();

            if (pred->mark_visited(m_visit_count)) {
                do_dfs(pred);
                n->set_low(min(n->get_low(), pred->get_low()));
            }
            if (pred->get_dfs_num() < n->get_dfs_num() && pred->is_on_stack()) {
                n->set_low(min(pred->get_dfs_num(), n->get_low()));
            }
        }

        if (n->get_low() == n->get_dfs_num()) {
            // found SCC
            Dependence_node *x;
            Node_vec scc(m_alloc);
            do {
                x = m_stack.top();
                x->mark_on_stack(false);
                m_stack.pop();
                scc.push_back(x);
            } while (x != n);

            size_t size = scc.size();
            if (size == 1) {
                // single entry scc
                x = scc[0];

                // check for self recursion
                Edge_list const &callees = x->get_callee_edges();
                for (Edge_list::const_iterator it(callees.begin()), end(callees.end());
                    it != end;
                    ++it)
                {
                    Dependence_edge const *edge = *it;
                    if (x == edge->get_dst()) {
                        process_scc(scc);
                        m_has_loops = true;
                        break;
                    }
                }
            } else {
                // recursion through more than one function
                process_scc(scc);
            }
        }
    }

    /// Process a found loop.
    ///
    /// \param scc  the nodes of the loop, might by only one if self-loop
    void process_scc(Node_vec const &scc)
    {
        m_dg.has_dependence_loop(scc);
    }

    /// Walk a node and link them together in topological order.
    ///
    /// \param n  the node
    void do_link_nodes(Dependence_node *n)
    {
        if (n->mark_visited(m_visit_count)) {
            do_link_nodes(n->get_callee_edges());

            if (n->is_local() || n->is_export()) {
                // we are only interested in local or exported nodes
                if (m_first == NULL) {
                    m_first = m_last = n;
                } else {
                    m_last->set_next(n);
                    m_last = n;
                }
            }
        }
    }

    /// Walk a list of edges and link them together in topological order.
    ///
    /// \param edges  the edges
    void do_link_nodes(Edge_list const &edges)
    {
        for (Edge_list::const_iterator it(edges.begin()), end(edges.end()); it != end; ++it) {
            Dependence_edge const *edge = *it;
            do_link_nodes(edge->get_dst());
        }
    }

    /// Walk a list of root nodes and link them together in topological order.
    ///
    /// \param roots  the roots
    void do_link_nodes(Node_list const &roots)
    {
        for (Node_list::const_iterator it(roots.begin()), end(roots.end()); it != end; ++it)
            do_link_nodes(*it);
    }

    /// Walk a node.
    ///
    /// \param n        the node
    /// \param visitor  the visitor
    void do_walk(Dependence_node *n, IDG_visitor &visitor)
    {
        if (n->mark_visited(m_visit_count)) {
            visitor.visit(n, IDG_visitor::PRE_ORDER);
            do_walk(n->get_callee_edges(), visitor);
            visitor.visit(n, IDG_visitor::POST_ORDER);
        }
    }

    /// Walk a list of root nodes and link them together in topological order.
    ///
    /// \param roots    the roots
    /// \param visitor  the visitor
    void do_walk(Node_list const &roots, IDG_visitor &visitor)
    {
        for (Node_list::const_iterator it(roots.begin()), end(roots.end()); it != end; ++it)
            do_walk(*it, visitor);
    }

    /// Walk a list of edges and link them together in topological order.
    ///
    /// \param edges    the edges
    /// \param visitor  the visitor
    void do_walk(Edge_list const &edges, IDG_visitor &visitor)
    {
        for (Edge_list::const_iterator it(edges.begin()), end(edges.end()); it != end; ++it) {
            Dependence_edge const *edge = *it;
            do_walk(edge->get_dst(), visitor);
        }
    }

    static size_t min(size_t a, size_t b) { return a < b ? a : b; }

private:
    /// The allocator.
    IAllocator *m_alloc;

    // The graph that is visited
    DAG_dependence_graph &m_dg;

    /// Last node in the generated list.
    Dependence_node *m_last;

    /// First node in the generated list.
    Dependence_node *m_first;

    /// The visit count for this walker taken from the dependence graph.
    size_t &m_visit_count;

    /// used for scc
    size_t m_next_dfs_num;

    typedef stack<Dependence_node *>::Type Dep_node_stack;

    /// Stack for scc.
    Dep_node_stack m_stack;

    /// If set, dependence loop were detected.
    bool m_has_loops;
};

/// Helper class to create the dependency graph.
class DG_creator : public Module_visitor {
    typedef ptr_hash_map<ISymbol const, IDefinition const *>::Type Symbol_to_param_map;
public:
    /// Constructor.
    ///
    /// \param alloc       the memory arena
    /// \param dg          the dependency graph that is created
    /// \param map         the node map containing existing nodes
    /// \param restricted  if true, only already existing nodes from \c map are used
    DG_creator(
        Memory_arena         &arena,
        DAG_dependence_graph &dg,
        Def_node_map const   &map,
        bool                 restricted)
    : m_arena(arena)
    , m_dg(dg)
    , m_dag_builder(dg.m_dag_builder)
    , m_curr(NULL)
    , m_module(dg.m_dag_builder.tos_module())
    , m_known_callees(map)
    , m_wait_q(Def_wait_queue::container_type(arena.get_allocator()))
    , m_marker(0, Def_set::hasher(), Def_set::key_equal(), arena.get_allocator())
    , m_curr_parameters(
        0, Symbol_to_param_map::hasher(), Symbol_to_param_map::key_equal(), arena.get_allocator())
    , m_restricted(restricted)
    , m_inside_parameter(false)
    , m_inside_preset(false)
    {
    }

    /// Create the dependency tree.
    ///
    /// \param list  the list of all nodes created so far
    void create(Dependence_node *list)
    {
        for (Dependence_node *n = list; n != NULL; n = n->get_next()) {
            IDefinition const *def = n->get_definition();

            if (def == NULL) {
                // no declaration without def ... a DAG-BE generated node
                continue;
            }

            IDeclaration const *decl = def->get_declaration();

            // the dependency graph is restricted to one module, so stop at imports;
            // for compiler generated entities decl can be NULL
            if (!def->get_property(IDefinition::DP_IS_IMPORTED) && decl != NULL) {
                m_marker.insert(def);
                m_wait_q.push(n);
            }
        }

        while (!m_wait_q.empty()) {
            m_curr_parameters.clear();

            m_curr = m_wait_q.front();

            IDefinition const *def = m_curr->get_definition();
            IDefinition::Kind kind = def->get_kind();

            switch (kind) {
            case IDefinition::DK_ANNOTATION:
                {
                    IDeclaration_annotation const *decl =
                        cast<IDeclaration_annotation>(def->get_declaration());

                    // first collect all deferred sized parameters, so we can handle array
                    // length operators
                    collect_deferred_sized_parameters(decl);

                    // now visit the declaration
                    visit(decl);
                }
                break;
            case IDefinition::DK_FUNCTION:
                {
                    IDeclaration const *proto_decl = def->get_prototype_declaration();

                    if (proto_decl != NULL) {
                        // default parameters are visible on the prototype
                        visit(proto_decl);
                    }

                    IDeclaration_function const *decl =
                        cast<IDeclaration_function>(def->get_declaration());

                    Flag_store inside_preset(m_inside_preset, decl->is_preset());

                    // first collect all deferred sized parameters, so we can handle array
                    // length operators
                    collect_deferred_sized_parameters(decl);

                    // now visit the declaration
                    visit(decl);
                }
                break;
            case IDefinition::DK_CONSTRUCTOR:
                {
                    IDeclaration_type_struct const *s_decl =
                        as<IDeclaration_type_struct>(def->get_declaration());

                    if (s_decl != NULL) {
                        // a struct declaration, add dependency to default initializers

                        IDefinition::Semantics sema = def->get_semantics();
                        if (sema == IDefinition::DS_ELEM_CONSTRUCTOR ||
                            sema == IDefinition::DS_DEFAULT_STRUCT_CONSTRUCTOR)
                        {
                            Flag_store inside_preset(m_inside_parameter, true);

                            for (int i = 0, n = s_decl->get_field_count(); i < n; ++i) {
                                IExpression const *init = s_decl->get_field_init(i);

                                if (init != NULL)
                                    visit(init);
                            }
                        }
                    }
                }
                break;
            default:
                MDL_ASSERT(!"unexpected definition kind");
                break;
            }

            m_wait_q.pop();
        }
    }

private:

    /// Pre-visit a parameter.
    bool pre_visit(IParameter *param) MDL_FINAL
    {
        m_inside_parameter = true;
        // visit children
        return true;
    }

    /// Post-visit a parameter.
    void post_visit(IParameter *param) MDL_FINAL
    {
        m_inside_parameter = false;
    }

    /// Post visit an annotation.
    void post_visit(IAnnotation *anno) MDL_FINAL
    {
        IQualified_name const *name = anno->get_name();
        IDefinition const *callee = name->get_definition();

        bool is_imported = callee->get_property(IDefinition::DP_IS_IMPORTED);

        Dependence_node *node = m_dg.get_node(callee);

        if (m_marker.insert(callee).second) {
            // new node discovered
            IDeclaration const *decl = callee->get_declaration();

            // the dependency graph is restricted to one module, so stop at imports;
            // for compiler generated entities decl can be NULL
            if (!is_imported && decl != NULL) {
                m_wait_q.push(node);
            }
        }

        m_curr->add_edge(m_arena, node, m_inside_parameter);
    }

    /// Post visit a call.
    IExpression *post_visit(IExpression_call *call) MDL_FINAL
    {
        IExpression_reference const *ref = as<IExpression_reference>(call->get_reference());
        if (ref == NULL) {
            return call;
        }
        if (ref->is_array_constructor()) {
            // add a reference to the DAG array constructor here
            IType_factory   *fact = m_module->get_type_factory();
            Dependence_node *node = m_dg.get_node(
                get_array_constructor_signature(),
                get_array_constructor_signature_without_suffix(),
                /*dag_alias_name=*/NULL,
                /*dag_preset_name=*/NULL,
                IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
                fact->create_int(),
                Array_ref<Dependence_node::Parameter>(),
                Dependence_node::FL_IS_BUILTIN);

            m_curr->add_edge(m_arena, node, m_inside_parameter);

            return call;
        }

        IDefinition const *callee = ref->get_definition();

        IDefinition::Semantics sema = callee->get_semantics();
        if (sema == IDefinition::DS_INVALID_REF_CONSTRUCTOR) {
            // invalid ref is always folded in the DAG representation
            return call;
        }

        Dependence_node *node = NULL;

        if (m_restricted) {
            Def_node_map::const_iterator it = m_known_callees.find(callee);
            if (it == m_known_callees.end()) {
                return call;
            }

            node = it->second;
        } else {
            node = m_dg.get_node(callee);

            if (m_marker.insert(callee).second) {
                // new node discovered
                IDeclaration const *decl = callee->get_declaration();

                // the dependency graph is restricted to one module, so stop at imports;
                // for compiler generated entities decl can be NULL
                if (decl != NULL && !callee->get_property(IDefinition::DP_IS_IMPORTED)) {
                    m_wait_q.push(node);
                }
            }
        }

        m_curr->add_edge(m_arena, node, m_inside_parameter);
        return call;
    }

    /// Post-visit a binary expression.
    IExpression *post_visit(IExpression_binary *expr) MDL_FINAL
    {
        if (m_restricted) {
            // all operators are already created
            return expr;
        }

        IExpression_binary::Operator op = expr->get_operator();

        if (op == IExpression_binary::OK_ASSIGN || op == IExpression_binary::OK_SEQUENCE) {
            // these operators are NEVER used inside DAG, ignore them
            return expr;
        }

        string name(m_dag_builder.get_binary_name(expr));
        string simple_name(m_dag_builder.get_binary_name(expr, false));
        unsigned flags = is_local_operator(name) ? 0 : Dependence_node::FL_IS_BUILTIN;

        if (flags == 0 && m_inside_preset) {
            // all operators inside a preset must be visible
            flags = Dependence_node::FL_IS_EXPORTED;
        }

        IExpression const *left = expr->get_left_argument();
        IType const       *l_tp = left->get_type()->skip_type_alias();
        Dependence_node   *node = NULL;

        if (op == IExpression_binary::OK_SELECT) {
            // the binary select operator is mapped to the unary DS_INTRINSIC_DAG_FIELD_ACCESS
            // function in the DAG world

            Dependence_node::Parameter params[] = {
                Dependence_node::Parameter(l_tp, "s")
            };
            node = m_dg.get_node(
                name.c_str(),
                simple_name.c_str(),
                /*dag_alias_name=*/NULL,
                /*dag_preset_name=*/NULL,
                IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS,
                expr->get_type()->skip_type_alias(),
                params,
                flags);
        } else if (op == IExpression_binary::OK_ARRAY_INDEX) {
            IExpression const *right = expr->get_right_argument();
            IType const       *r_tp = right->get_type()->skip_type_alias();

            Dependence_node::Parameter params[] = {
                Dependence_node::Parameter(l_tp, "a"),
                Dependence_node::Parameter(r_tp, "i")
            };
            node = m_dg.get_node(
                name.c_str(),
                simple_name.c_str(),
                /*dag_alias_name=*/NULL,
                /*dag_preset_name=*/NULL,
                operator_to_semantic(op),
                expr->get_type()->skip_type_alias(),
                params,
                flags);
        } else {
            // default case: binary operator
            IExpression const *right = expr->get_right_argument();
            IType const       *r_tp = right->get_type()->skip_type_alias();

            // we "known" that the parameter names are x, y here
            Dependence_node::Parameter params[] = {
                Dependence_node::Parameter(l_tp, "x"),
                Dependence_node::Parameter(r_tp, "y")
            };

            node = m_dg.get_node(
                name.c_str(),
                simple_name.c_str(),
                /*dag_alias_name=*/NULL,
                /*dag_preset_name=*/NULL,
                operator_to_semantic(op),
                expr->get_type()->skip_type_alias(),
                params,
                flags);
        }

        m_curr->add_edge(m_arena, node, m_inside_parameter);
        return expr;
    }

    /// Post-visit an unary expression.
    IExpression *post_visit(IExpression_unary *expr) MDL_FINAL
    {
        if (m_restricted) {
            // all operators are already created
            return expr;
        }

        IExpression_unary::Operator op = expr->get_operator();

        switch (op) {
        case IExpression_unary::OK_BITWISE_COMPLEMENT:
        case IExpression_unary::OK_LOGICAL_NOT:
        case IExpression_unary::OK_POSITIVE:
        case IExpression_unary::OK_NEGATIVE:
            break;

        case IExpression_unary::OK_PRE_INCREMENT:
        case IExpression_unary::OK_PRE_DECREMENT:
        case IExpression_unary::OK_POST_INCREMENT:
        case IExpression_unary::OK_POST_DECREMENT:
            // these operators are not supported in the DAG representation
            return expr;

        case IExpression_unary::OK_CAST:
            break;
        }

        string name(m_dag_builder.get_unary_name(expr));
        string simple_name(m_dag_builder.get_unary_name(expr, false));
        unsigned flags = is_local_operator(name) ? 0 : Dependence_node::FL_IS_BUILTIN;

        IExpression const *arg = expr->get_argument();
        IType const *arg_tp = arg->get_type()->skip_type_alias();

        // we "known" that the parameter name is x here
        Dependence_node::Parameter param(arg_tp, "x");

        Dependence_node *node = m_dg.get_node(
            name.c_str(),
            simple_name.c_str(),
            /*dag_alias_name=*/NULL,
            /*dag_preset_name=*/NULL,
            operator_to_semantic(op),
            expr->get_type()->skip_type_alias(),
            param,
            flags);

        m_curr->add_edge(m_arena, node, m_inside_parameter);
        return expr;
    }

    /// Post-visit a conditional expression.
    IExpression *post_visit(IExpression_conditional *expr) MDL_FINAL
    {
        if (m_restricted) {
            // all operators are already created
            return expr;
        }

        IExpression const *cond = expr->get_condition();
        IExpression const *t_expr = expr->get_true();
        IExpression const *f_expr = expr->get_false();

        IType const *c_tp = cond->get_type()->skip_type_alias();
        IType const *t_tp = t_expr->get_type()->skip_type_alias();
        IType const *f_tp = f_expr->get_type()->skip_type_alias();

        // Note: there is no def for the ?: operator in MDL itself
        Dependence_node::Parameter params[] = {
            Dependence_node::Parameter(c_tp, "cond"),
            Dependence_node::Parameter(t_tp, "true_exp"),
            Dependence_node::Parameter(f_tp, "false_exp")
        };

        Dependence_node *node = m_dg.get_node(
            get_ternary_operator_signature(),
            get_ternary_operator_signature_without_suffix(),
            /*dag_alias_name=*/NULL,
            /*dag_preset_name=*/NULL,
            operator_to_semantic(IExpression::OK_TERNARY),
            expr->get_type()->skip_type_alias(),
            params,
            Dependence_node::FL_IS_BUILTIN);

        m_curr->add_edge(m_arena, node, m_inside_parameter);
        return expr;
    }

    /// Check if the name of a given operator refers to a "local" one.
    bool is_local_operator(string const &name)
    {
        if (name[0] == ':' && name[1] == ':') {
            // check if it has the module name as prefix.

            IModule const *module   = m_dag_builder.tos_module();
            char const    *abs_name = module->get_name();
            size_t        len       = strlen(abs_name);

            if (len >= name.size()) {
                // mismatch
                return false;
            }

            for (size_t i = 0; i < len; ++i) {
                if (name[i] != abs_name[i]) {
                    // mismatch
                    return false;
                }
            }

            // skip "::" after module name
            if (len + 2 >= name.size()) {
                // "::" missing
            }
            if (name[len] == ':' && name[len + 1] == ':')
                len += 2;

            for (size_t i = len, n = name.size(); i < n; ++i) {
                char c = name[i];

                if (c == '(') {
                    // good, found signature, is a local operator
                    return true;
                } else if (c == ':') {
                    // another scope, this module's name is just a prefix, not a local operator
                    return false;
                }
            }
            // name without type signature, but local
            return true;
        }
        // non-absolute name, a builtin operator
        return false;
    }

    /// Handle array sizes.
    IExpression *post_visit(IExpression_reference *ref_expr) MDL_FINAL
    {
        IDefinition const *def = ref_expr->get_definition();
        if (def == NULL) {
            // array constructor
            return ref_expr;
        }

        if (def->get_kind() != IDefinition::DK_ARRAY_SIZE) {
            // not an array size symbol
            return ref_expr;
        }

        // The only source of array sizes here should be parameters (and temporaries?).
        // Find the parameter.
        MDL_ASSERT(
            find_parameter_for_size(def->get_symbol()) != NULL &&
            "could not find parameter for array size");
        return ref_expr;
    }

    // Find a parameter for a given array size.
    IDefinition const *find_parameter_for_size(ISymbol const *sym) const
    {
        Symbol_to_param_map::const_iterator it(m_curr_parameters.find(sym));
        if (it != m_curr_parameters.end())
            return it->second;
        return NULL;
    }

    // Enter a parameter.
    void enter_param(IDefinition const *param)
    {
        MDL_ASSERT(param->get_kind() == IDefinition::DK_PARAMETER);

        IType_array const *a_type = as<IType_array>(param->get_type());
        if (a_type == NULL) {
            // not array typed
            return;
        }

        if (a_type->is_immediate_sized()) {
            // immediate sized
            return;
        }

        IType_array_size const *size     = a_type->get_deferred_size();
        ISymbol const          *size_sym = size->get_size_symbol();

        // use insert here, so the first parameter win
        m_curr_parameters.insert(Symbol_to_param_map::value_type(size_sym, param));
    }

    /// Collect all deferred sized parameters from a function declaration.
    ///
    /// \param fkt_decl  the function decl
    void collect_deferred_sized_parameters(IDeclaration_function const *fkt_decl)
    {
        // enter all parameters
        for (int i = 0, n = fkt_decl->get_parameter_count(); i < n; ++i) {
            IParameter const  *param = fkt_decl->get_parameter(i);
            IDefinition const *def   = param->get_name()->get_definition();

            enter_param(def);
        }
    }

    /// Collect all deferred sized parameters from an annotation declaration.
    ///
    /// \param anno_decl  the annotation decl
    void collect_deferred_sized_parameters(IDeclaration_annotation const *anno_decl)
    {
        // enter all parameters
        for (int i = 0, n = anno_decl->get_parameter_count(); i < n; ++i) {
            IParameter const  *param = anno_decl->get_parameter(i);
            IDefinition const *def = param->get_name()->get_definition();

            enter_param(def);
        }
    }

private:
    /// The memory arena to allocate on.
    Memory_arena         &m_arena;

    /// The dependence graph that is built.
    DAG_dependence_graph &m_dg;

    /// The DAG builder of the dependence graph.
    DAG_builder const &m_dag_builder;

    /// The current node.
    Dependence_node *m_curr;

    /// The current module.
    IModule const *m_module;

    /// The node map of known callees.
    Def_node_map const &m_known_callees;

    typedef queue<Dependence_node *>::Type Def_wait_queue;
    typedef ptr_hash_set<IDefinition const>::Type Def_set;

    /// The wait queue of nodes that must be processed.
    Def_wait_queue                 m_wait_q;

    /// Marker set for already processed nodes.
    Def_set                        m_marker;

    /// The array-typed parameters of the currently processed function
    Symbol_to_param_map            m_curr_parameters;

    /// If true, only nodes from the m_known_callees are used.
    bool m_restricted;

    /// if true, we are inside a parameter.
    bool m_inside_parameter;

    /// if true, we are inside a material/function preset.
    bool m_inside_preset;
};


namespace {

///
/// Helper class to dump a dependence graph as a dot file.
///
class Dumper : public IDG_visitor {
public:
    /// Constructor.
    ///
    /// \param alloc   the allocator
    /// \param out     an output stream, the dot file is written to
    /// \param dg      the dependence graph to dump
    explicit Dumper(
        IAllocator       *alloc,
        IOutput_stream   *out,
        DAG_dependence_graph &dg);

    /// Dump the dependence graph to the output stream.
    void dump();

private:
    /// Print the name of a dependence graph node.
    ///
    /// \param n  the node
    void node_name(
        Dependence_node const *n);

    /// Print a dependence graph node.
    ///
    /// \param n      the node
    /// \param color  the color of the node, NULL for default
    void node(
        Dependence_node const *n,
        char const            *color = NULL);

    /// Print a dependence edge.
    ///
    /// \param src    the source node of the edge
    /// \param dst    the destination node of the edge
    /// \param color  the color of the edge, NULL for default
    void edge(
        Dependence_node const *src,
        Dependence_node const *dst,
        char const            *color = NULL);

    /// Dependence graph node visitor.
    ///
    /// \param node   the currently visited node
    /// \param order  PRE or POST order
    void visit(Dependence_node const *node, Order order) MDL_FINAL;

private:
    /// The dependence graph.
    DAG_dependence_graph          &m_dg;

    /// A printer, use to print into the output stream.
    mi::base::Handle<Printer> m_printer;

    /// Current graph depth from the root.
    size_t                    m_depth;
};

// Constructor.
Dumper::Dumper(
    IAllocator       *alloc,
    IOutput_stream   *out,
    DAG_dependence_graph &dg)
: m_dg(dg)
, m_printer()
, m_depth(0)
{
    Allocator_builder builder(alloc);

    m_printer = mi::base::make_handle(builder.create<Printer>(alloc, out));
}

// Dump the dependence graph to the output stream.
void Dumper::dump()
{
    m_printer->print("digraph \"");
    char const *name = "dependence_graph";
    m_printer->print(name);
    m_printer->print("\" {\n");

    m_dg.walk(*this);
    m_printer->print("}\n");
}

// Dump the dependence graph to the output stream.
void Dumper::node_name(Dependence_node const *n)
{
    char buf[32];

    snprintf(buf, sizeof(buf), "n%ld", (long)n->get_id());
    buf[sizeof(buf) - 1] = '\0';
    m_printer->print(buf);
}

// Print a dependence graph node.
void Dumper::node(Dependence_node const *n, char const *color)
{
    bool use_box_shape = n->is_local();

    m_printer->print("  ");
    node_name(n);
    m_printer->print(" [label=\"");    
    m_printer->print(n->get_dag_name());
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
    Dependence_node const *src,
    Dependence_node const *dst,
    char const            *style)
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

    if (style != NULL) {
        m_printer->print(" [style=");
        m_printer->print(style);
        m_printer->print("]");
    }

    m_printer->print(";\n");
}

// Dependence graph node visitor.
void Dumper::visit(Dependence_node const *n, Order order)
{
    if (order == PRE_ORDER) {
        char const *color = NULL;
        // make exported nodes green
        if (n->is_local_export())
            color = "green";
        node(n, color);
        ++m_depth;
    } else {
        for (Dependence_node::Edge_iterator it(n->edges_begin()), end(n->edges_end());
             it != end;
             ++it)
        {
            Dependence_edge const *e    = *it;
            Dependence_node const *prev = e->get_dst();

            char const *style = e->is_true_dependence() ? "solid" : "dashed";

            edge(n, prev, style);
        }
        --m_depth;
    }
}

}  // anonymous

// Constructor from a definition.
Dependence_node::Dependence_node(
    Memory_arena      *arena,
    size_t            id,
    IDefinition const *def,
    char const        *dag_name,
    char const        *dag_simple_name,
    char const        *dag_alias_name,
    char const        *dag_preset_name,
    unsigned          flags,
    Dependence_node   *next)
: m_def(def)
, m_sema(def->get_semantics())
, m_ret_type(NULL)
, m_params(NULL)
, m_n_params(0)
, m_flags(flags)
, m_dag_name(dag_name)
, m_dag_simple_name(dag_simple_name)
, m_dag_alias_name(dag_alias_name)
, m_dag_preset_name(dag_preset_name)
, m_next(next)
, m_edgess(arena)
, m_id(id)
, m_visit_count(0)
, m_dfs_num(0)
, m_low(0)
, m_on_stack(false)
, m_is_root(true)
{
    IType_function const *f_type = cast<IType_function>(def->get_type());

    m_ret_type = f_type->get_return_type();
    int n_params = f_type->get_parameter_count();

    if (n_params > 0) {
        Parameter *params = reinterpret_cast<Parameter *>(
            arena->allocate(n_params * sizeof(Parameter)));

        for (int i = 0; i < n_params; ++i) {
            Parameter *p = &params[i];

            IType const *t;
            ISymbol const *s;

            f_type->get_parameter(i, t, s);
            
            new (p) Parameter(t, s->get_name());
        }

        m_params   = params;
        m_n_params = n_params;
    }
}

// Constructor from a name + semantic.
Dependence_node::Dependence_node(
    Memory_arena               *arena,
    size_t                     id,
    IDefinition::Semantics     sema,
    IType const                *ret_type,
    Array_ref<Parameter> const &c_params,
    char const                 *dag_name,
    char const                 *dag_simple_name,
    char const                 *dag_alias_name,
    char const                 *dag_preset_name,
    unsigned                   flags,
    Dependence_node            *next)
: m_def(NULL)
, m_sema(sema)
, m_ret_type(ret_type)
, m_params(NULL)
, m_n_params(0)
, m_flags(flags)
, m_dag_name(dag_name)
, m_dag_simple_name(dag_simple_name)
, m_dag_alias_name(dag_alias_name)
, m_dag_preset_name(dag_preset_name)
, m_next(next)
, m_edgess(arena)
, m_id(id)
, m_visit_count(0)
, m_dfs_num(0)
, m_low(0)
, m_on_stack(false)
, m_is_root(true)
{
    size_t n_params = c_params.size();
    if (n_params > 0) {
        Parameter *params = reinterpret_cast<Parameter *>(
            arena->allocate(n_params * sizeof(Parameter)));

        for (int i = 0; i < n_params; ++i) {
            Parameter &p = params[i];

            p = c_params[i];
        }
        m_params   = params;
        m_n_params = n_params;
    }
}

namespace {

/// Predicate class for comparing dependence edge destinations.
class Pred {
public:
    /// Constructor.
    Pred(Dependence_node const *key) : m_key(key) {}

    /// Returns true iff the destination of the given edge is the node given at construction.
    bool operator()(Dependence_edge const *edge) { return edge->get_dst() == m_key; }

private:
    Dependence_node const *m_key;
};

}  // anonymous

// Add a dependency edge.
void Dependence_node::add_edge(
    Memory_arena    &arena,
    Dependence_node *callee,
    bool            is_def_arg)
{
    // unused yet
    (void)is_def_arg;

    // callee cannot be a root
    callee->m_is_root = false;

    Pred predicate(callee);

    // check if it's already in our list
    // FIXME: this is O(n^2)
    Edge_iterator it = std::find_if(m_edgess.begin(), m_edgess.end(), predicate);
    if (it == m_edgess.end()) {
        Arena_builder builder(arena);

        Dependence_edge *edge = builder.create<Dependence_edge>(callee, !is_def_arg);
        m_edgess.push_back(edge);
    } else if (!is_def_arg) {
        // make it a true dependence edge
        Dependence_edge *edge = *it;
        edge->make_true_dependence();
    }
}

// Returns true if this node was not yet visited for the given visit count.
bool Dependence_node::mark_visited(size_t visited)
{
    bool res = m_visit_count < visited;
    if (res)
        m_visit_count = visited;
    return res;
}

// Set the DFS num.
void Dependence_node::set_dfs_num(size_t num)
{
    m_dfs_num = num;
    set_low(num);
}

// Returns true, if this node is a local entity (i.e. not imported).
bool Dependence_node::is_local() const
{
    return (m_flags & (FL_IS_BUILTIN | FL_IS_IMPORTED)) == 0;
}

// Returns true, if this node is a local export.
bool Dependence_node::is_local_export() const
{
    return is_local() && ((m_flags & FL_IS_EXPORTED) != 0);
}

// Returns true, if this node is a (local or re-) export.
bool Dependence_node::is_export() const
{
    return (m_flags & FL_IS_EXPORTED) != 0;
}

// Get the parameter type and name.
void Dependence_node::get_parameter(
    size_t      index,
    IType const *&type,
    char const  *&name) const
{
    if (index < m_n_params) {
        type = m_params[index].m_type;
        name = m_params[index].m_name;
    } else {
        type = NULL;
        name = NULL;
    }
}

// Constructor.
DAG_dependence_graph::DAG_dependence_graph(
    IAllocator           *alloc,
    Generated_code_dag   &dag,
    DAG_builder          &dag_builder,
    ISymbol const        *invisible_sym,
    bool                 include_locals)
: m_arena(alloc)
, m_builder(m_arena)
, m_dag(dag)
, m_dag_builder(dag_builder)
, m_invisible_sym(invisible_sym)
, m_exported_nodes(&m_arena)
, m_list(&m_arena)
, m_def_node_map(0, Def_node_map::hasher(), Def_node_map::key_equal(), alloc)
, m_name_node_map(0, Name_node_map::hasher(), Name_node_map::key_equal(), alloc)
, m_last(NULL)
, m_next_id(0)
, m_visit_count(0)
, m_include_locals(include_locals)
, m_is_builtins(false)
, m_has_loops(false)
{
    IModule const *module = m_dag_builder.tos_module();
    m_is_builtins = module->is_builtins();

    int def_count     = module->get_exported_definition_count();
    int builtin_count = m_is_builtins ? module->get_builtin_definition_count() : 0;

    if (m_is_builtins) {
        // add builtins
        for (int i = 0; i < builtin_count; ++i) {
            IDefinition const *def = module->get_builtin_definition(i);

            create_exported_nodes(module, def);
        }
    }

    // add normal exports
    for (int i = 0; i < def_count; ++i) {
        IDefinition const *def = module->get_exported_definition(i);

        create_exported_nodes(module, def);
    }

    // generate DAG intrinsic functions first, so they are available for all
    // functions of the module
    if (module->is_builtins()) {
        IType_factory *fact           = module->get_type_factory();
        IType const   *int_type       = fact->create_int();
        IType const   *bool_type      = fact->create_bool();
        IType const   *indexable_type = int_type;  // FIXME
        IType const   *any_type       = int_type;  // FIXME
        IType const   *array_type     = int_type;  // FIXME

        // Generate the array constructor.
        create_dag_array_constructor(array_type);
        // Generate the cast operator.
        create_dag_cast_operator(any_type);
        // Generate the ternary operator
        create_dag_ternary_operator(any_type, bool_type);
        // Generate the index operator
        create_dag_index_function(indexable_type, int_type);
        // Generate the array length operator
        create_dag_array_len_operator(array_type, int_type);
    }

    DG_creator creator(m_arena, *this, m_def_node_map, !m_include_locals);

    creator.create(m_last);

    DG_walker walker(m_arena.get_allocator(), *this);

    // now check if there are loops in the dependency graph. This *is* possible,
    // because it contains edges to the default arguments ...

    // iterate over ALL entry points here
    walker.start_scc();
    for (Node_list::const_iterator it(m_exported_nodes.begin()), end(m_exported_nodes.end());
         it != end;
         ++it)
    {
        if (walker.calc_scc(*it)) {
            m_has_loops = true;
        }
    }

    // walk over the graph and create a linear list
    Dependence_node *p = walker.link_nodes_in_topo_order();

    for (; p != NULL; p = p->get_next()) {
        m_list.push_back(p);
    }
}

// Get the exported definitions of the given module.
DAG_dependence_graph::Node_list const &DAG_dependence_graph::get_exported_module_nodes()
{
    return m_exported_nodes;
}

// Get the entity nodes of the given module in topological order.
DAG_dependence_graph::Node_list const &DAG_dependence_graph::get_module_entities(
    bool &has_loops)
{
    has_loops = m_has_loops;
    return m_list;
}

// Get the dependency node for the given definition.
Dependence_node *DAG_dependence_graph::get_node(IDefinition const *idef)
{
    Definition const *def = impl_cast<Definition>(idef);

    if (Definition const *def_def = def->get_definite_definition())
        def = def_def;

    Def_node_map::iterator it = m_def_node_map.find(def);
    if (it != m_def_node_map.end()) {
        // already known
        return it->second;
    }

    // create a new one
    char const *dag_name                     = NULL;
    char const *dag_simple_name              = NULL;
    char const *dag_alias_name               = NULL;
    char const *dag_preset_name              = NULL;

    unsigned flags = 0;

    IModule const *module = m_dag_builder.tos_module();

    if (def->get_property(IDefinition::DP_IS_IMPORTED)) {
        // imported
        flags |= Dependence_node::FL_IS_IMPORTED;

        mi::base::Handle<IModule const> owner(module->get_owner_module(def));
        if (def->get_property(IDefinition::DP_IS_EXPORTED)) {
            // an exported import
            flags |= Dependence_node::FL_IS_EXPORTED;

            dag_name = Arena_strdup(m_arena, m_dag_builder.def_to_name(def, module, true).c_str());
            dag_simple_name
                = Arena_strdup(m_arena, m_dag_builder.def_to_name(def, (char const *)NULL, false).c_str());

            dag_alias_name =
                Arena_strdup(m_arena, m_dag_builder.def_to_name(def, owner.get()).c_str());
        } else {
            // only imported
            dag_name = Arena_strdup(m_arena, m_dag_builder.def_to_name(def, owner.get(), true).c_str());
            dag_simple_name
                = Arena_strdup(m_arena, m_dag_builder.def_to_name(def, (char const *)NULL, false).c_str());
        }

        // check for preset
        idef = module->get_original_definition(def);
        IDefinition const *preset_def = skip_presets(idef, owner);
        if (preset_def != idef) {
            dag_preset_name =
                Arena_strdup(m_arena, m_dag_builder.def_to_name(preset_def, owner.get()).c_str());
        }
    } else {
        // from this module

        // entities from the builtin module are always exported
        if (m_is_builtins || def->get_property(IDefinition::DP_IS_EXPORTED))
            flags |= Dependence_node::FL_IS_EXPORTED;

        if (def->get_kind() == IDefinition::DK_CONSTRUCTOR &&
            def->get_declaration() == NULL)
        {
            // can only be a built-in constructor: put it to builtins, for the DAG
            // backend this comes from the "builtin" module
            dag_name = Arena_strdup(
                m_arena, m_dag_builder.def_to_name(def, (char const *)NULL, true).c_str());
            dag_simple_name = Arena_strdup(
                m_arena, m_dag_builder.def_to_name(def, (char const *)NULL, false).c_str());
            flags |= m_is_builtins ? 0 : Dependence_node::FL_IS_BUILTIN;
        } else {
            dag_name = Arena_strdup(m_arena, m_dag_builder.def_to_name(def, module, true).c_str());
            dag_simple_name
                = Arena_strdup(m_arena, m_dag_builder.def_to_name(def, (char const *)NULL, false).c_str());
        }

        // check for preset
        mi::base::Handle<IModule const> owner= mi::base::make_handle_dup(module);
        IDefinition const *preset_def = skip_presets(idef, owner);
        if (preset_def != idef) {
            dag_preset_name =
                Arena_strdup(m_arena, m_dag_builder.def_to_name(preset_def, owner.get()).c_str());
        }
    }

    Dependence_node *n = m_builder.create<Dependence_node>(
        &m_arena,
        m_next_id++,
        def,
        dag_name,
        dag_simple_name,
        dag_alias_name,
        dag_preset_name,
        flags,
        m_last);

    m_def_node_map[def] = n;
    m_last = n;

    return n;
}

// Get the DAG dependency node for the given entity.
Dependence_node *DAG_dependence_graph::get_node(
    char const                                  *dag_name,
    char const                                  *dag_simple_name,
    char const                                  *dag_alias_name,
    char const                                  *dag_preset_name,
    IDefinition::Semantics                      sema,
    IType const                                 *ret_type,
    Array_ref<Dependence_node::Parameter> const &params,
    unsigned                                    flags)
{
    MDL_ASSERT(sema != operator_to_semantic(IExpression::OK_SELECT));

    Name_node_map::iterator it = m_name_node_map.find(dag_name);

    if (it != m_name_node_map.end()) {
        // already known
        return it->second;
    }

    // store the name
    dag_name                     = Arena_strdup(m_arena, dag_name);
    dag_simple_name              = Arena_strdup(m_arena, dag_simple_name);

    if (dag_alias_name != NULL)
        dag_alias_name = Arena_strdup(m_arena, dag_alias_name);
    if (dag_preset_name != NULL)
        dag_preset_name = Arena_strdup(m_arena, dag_preset_name);

    Dependence_node *n = m_builder.create<Dependence_node>(
        &m_arena,
        m_next_id++,
        sema,
        ret_type,
        params,
        dag_name,
        dag_simple_name,
        dag_alias_name,
        dag_preset_name,
        flags,
        m_last);

    m_name_node_map[dag_name] = n;
    m_last = n;

    return n;
}

// Check if the given definition already exists in the dependency graph.
bool DAG_dependence_graph::node_exists(IDefinition const *def) const
{
    return m_def_node_map.find(def) != m_def_node_map.end();
}

// Walk over the dependence graph.
void DAG_dependence_graph::walk(IDG_visitor &visitor)
{
    DG_walker walker(m_arena.get_allocator(), *this);

    walker.walk(visitor);
}

// Dump the dependency graph.
bool DAG_dependence_graph::dump(char const *file_name)
{
    FILE *f = fopen(file_name, "w");
    if (f == NULL)
        return false;

    IAllocator *alloc = m_arena.get_allocator();
    Allocator_builder builder(alloc);

    mi::base::Handle<File_Output_stream> out(
        builder.create<File_Output_stream>(alloc, f, /*close_on_destroy=*/true));

    Dumper dumper(m_builder.get_arena()->get_allocator(), out.get(), *this);

    dumper.dump();

    return true;
}

// Returns true for MDL definition that should not be visible in the DAG backend.
bool DAG_dependence_graph::skip_definition(IDefinition const *def)
{
    switch (def->get_kind()) {
    case IDefinition::DK_TYPE:          // handle types,
    case IDefinition::DK_FUNCTION:      // functions,
    case IDefinition::DK_ANNOTATION:    // annotations,
    case IDefinition::DK_CONSTRUCTOR:   // and constructors
        return false;
    case IDefinition::DK_OPERATOR:      // handle operators.
        if (def->get_property(IDefinition::DP_NEED_REFERENCE)) {
            // do not "export" operators with reference parameters, we cannot
            // represent these in neuray
            return true;
        }
        // handle others
        return false;
    default:
        return true;
    }
}

// Called, if a dependence loop was detected.
void DAG_dependence_graph::has_dependence_loop(Node_vec const &nodes)
{
    // because we check that every module is recursion free, a dependence cycle can only
    // happen through at least one "fake" dependency through a default parameter.
    size_t n = nodes.size();

    bool found_error = false;

    for (size_t i = 0; i < n; ++i) {
        Dependence_node *s = nodes[i];

        size_t j = i + 1;
        if (j == n)
            j = 0;

        Dependence_node *d = nodes[j];

        // find the edge from s to d
        Edge_list const &edges = s->get_callee_edges();
        for (Edge_iterator it(edges.begin()), end(edges.end()); it != end; ++it) {
            Dependence_edge const *e = *it;

            if (e->get_dst() == d && !e->is_true_dependence()) {
                if (IDefinition const *def = s->get_definition()) {

                    string msg(m_arena.get_allocator());

                    msg += "Default arguments of '";
                    msg += def->get_symbol()->get_name();
                    msg += "' form a dependence cycle, this is not supported in this context";
                    msg += "(inside ";
                    msg += m_dag.m_renderer_context_name;
                    msg += ")";

                    m_dag.error(
                        Generated_code_dag::DEPENDENCE_GRAPH_HAS_LOOPS,
                        *def->get_position(),
                        msg.c_str());

                    found_error = true;
                }
            }
        }
    }

    if (!found_error) {
        // bad
    }
}

// Create a node.
void DAG_dependence_graph::create_exported_nodes(
    IModule const     *module,
    IDefinition const *def)
{
    if (skip_definition(def))
        return;

    IType const *type = def->get_type()->skip_type_alias();
    switch (type->get_kind()) {
    case IType::TK_FUNCTION:
        {
            Dependence_node *n = get_node(def);
            m_exported_nodes.push_back(n);

            IDefinition::Semantics sema = def->get_semantics();

            if (sema == IDefinition::DS_ELEM_CONSTRUCTOR) {
                // create field access functions here
                IType_function const *f_type   = cast<IType_function>(type);
                IType const          *ret_type = f_type->get_return_type()->skip_type_alias();

                if (is<IType_vector>(ret_type)) {
                    string name = m_dag_builder.type_to_name(ret_type);

                    // create nodes for field access here
                    for (int l = 0, n_params = f_type->get_parameter_count(); l < n_params; ++l) {
                        IType const   *parameter_type;
                        ISymbol const *parameter_symbol;
                        f_type->get_parameter(l, parameter_type, parameter_symbol);

                        // name and simple name are identical for builtin types
                        string dag_simple_name(name);
                        dag_simple_name += '.';
                        dag_simple_name += parameter_symbol->get_name();

                        string dag_name(dag_simple_name);
                        dag_name += '(';
                        dag_name += m_dag_builder.type_to_name(ret_type);
                        dag_name += ')';

                        // we don't have the symbol here
                        Dependence_node::Parameter param(ret_type, "s");

                        n = get_node(
                            dag_name.c_str(),
                            dag_simple_name.c_str(),
                            /*dag_alias_name=*/NULL,
                            /*dag_preset_name=*/NULL,
                            IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS,
                            parameter_type,
                            param,
                            Dependence_node::FL_IS_EXPORTED);
                        m_exported_nodes.push_back(n);
                    }
                }
            }
        }
        break;
    case IType::TK_STRUCT:
        {
            IType_struct const *s_type = cast<IType_struct>(type);
            int ctor_count = module->get_type_constructor_count(s_type);

            // add all constructors of the struct type
            for (int k = 0; k < ctor_count; ++k) {
                IDefinition const *ctor_def = module->get_type_constructor(s_type, k);

                Dependence_node *n = get_node(ctor_def);
                m_exported_nodes.push_back(n);

                IDefinition::Semantics sema = ctor_def->get_semantics();
                if (sema == IDefinition::DS_ELEM_CONSTRUCTOR) {
                    // create field access nodes here
                    string name        = m_dag_builder.type_to_name(s_type);
                    string simple_name(m_arena.get_allocator());

                    size_t pos = name.find_last_of(':');
                    if (pos != string::npos) {
                        simple_name = name.substr(pos + 1);
                    } else {
                        simple_name = name;
                    }

                    string alias(m_arena.get_allocator());

                    if (def->get_property(IDefinition::DP_IS_IMPORTED)) {
                        // remap the type name here, it is imported, we need an alias
                        alias = name;
                        name = module->get_name();
                        name += "::";
                        name += simple_name;
                    }

                    for (int l = 0, n_fields = s_type->get_field_count(); l < n_fields; ++l) {
                        IType const   *field_type;
                        ISymbol const *field_symbol;

                        s_type->get_field(l, field_type, field_symbol);

                        string field_access(m_arena.get_allocator());
                        field_access += '.';
                        field_access += field_symbol->get_name();

                        string signature_suffix(m_arena.get_allocator());
                        signature_suffix += '(';
                        signature_suffix += m_dag_builder.type_to_name(s_type);
                        signature_suffix += ')';

                        string dag_name        = name + field_access + signature_suffix;
                        string dag_simple_name = simple_name + field_access;
                        string dag_alias_name(m_arena.get_allocator());
                        if (!alias.empty())
                            dag_alias_name = alias + field_access + signature_suffix;

                        // we don't have the symbol here
                        Dependence_node::Parameter param(s_type, "s");

                        n = get_node(
                            dag_name.c_str(),
                            dag_simple_name.c_str(),
                            dag_alias_name.empty() ? NULL : dag_alias_name.c_str(),
                            /*dag_preset_name=*/NULL,
                            IDefinition::DS_INTRINSIC_DAG_FIELD_ACCESS,
                            field_type,
                            param,
                            Dependence_node::FL_IS_EXPORTED);
                        m_exported_nodes.push_back(n);
                    }
                }
            }
        }
        break;
    case IType::TK_ENUM:
        {
            IType_enum const *enum_type = cast<IType_enum>(type);

            int ctor_count = module->get_type_constructor_count(enum_type);
            for (int k = 0; k < ctor_count; ++k) {
                IDefinition const *ctor_def = module->get_type_constructor(enum_type, k);

                Dependence_node *n = get_node(ctor_def);
                m_exported_nodes.push_back(n);
            }

            // enum types have conversion operators
            int op_count = module->get_conversion_operator_count(enum_type);
            for (int k = 0; k < op_count; ++k) {
                IDefinition const *op_def = module->get_conversion_operator(enum_type, k);

                Dependence_node *n = get_node(op_def);
                m_exported_nodes.push_back(n);
            }
        }
        break;
    default:
        break;
    }
}

// Create a DAG intrinsic index function node.
void DAG_dependence_graph::create_dag_index_function(
    IType const *indexable_type,
    IType const *int_type)
{
    IType const *tmpl_type   = int_type; // FIXME
    IType const *return_type = int_type; // FIXME

    Dependence_node::Parameter params[2] = {
        Dependence_node::Parameter(tmpl_type, "a"),
        Dependence_node::Parameter(int_type,  "i"),
    };

    Dependence_node *n = get_node(
        "operator[](<0>[],int)",
        "operator[]",
        /*dag_alias_name=*/NULL,
        /*dag_preset_name=*/NULL,
        operator_to_semantic(IExpression::OK_ARRAY_INDEX),
        return_type,
        params,
        Dependence_node::FL_IS_EXPORTED);

    m_exported_nodes.push_back(n);
}

// Create a DAG intrinsic array length function.
Dependence_node *DAG_dependence_graph::create_dag_array_len_operator(
    IType const *int_type,
    IType const *type)
{
    Dependence_node::Parameter param(type, "a");

    Dependence_node *n = get_node(
        "operator_len(<0>[])",
        "operator_len",
        /*dag_alias_name=*/NULL,
        /*dag_preset_name=*/NULL,
        IDefinition::DS_INTRINSIC_DAG_ARRAY_LENGTH,
        int_type,
        param,
        Dependence_node::FL_IS_EXPORTED);

    m_exported_nodes.push_back(n);

    return n;
}

// Create a DAG intrinsic ternary operator function node.
void DAG_dependence_graph::create_dag_ternary_operator(
    IType const *type,
    IType const *bool_type)
{
    MDL_ASSERT(m_is_builtins);

    Dependence_node::Parameter params[] = {
        Dependence_node::Parameter(bool_type, "cond"),
        Dependence_node::Parameter(type, "true_exp"),
        Dependence_node::Parameter(type, "false_exp")
    };

    Dependence_node *n = get_node(
        get_ternary_operator_signature(),
        get_ternary_operator_signature_without_suffix(),
        /*dag_alias_name=*/NULL,
        /*dag_preset_name=*/NULL,
        operator_to_semantic(IExpression::OK_TERNARY),
        type,
        params,
        Dependence_node::FL_IS_EXPORTED);
    m_exported_nodes.push_back(n);
}

// Create the one-and-only array constructor.
void DAG_dependence_graph::create_dag_array_constructor(
    IType const *any_type)
{
    MDL_ASSERT(m_is_builtins);

    // Create the magic array constructor. There is only one now.
    Dependence_node *n = get_node(
        get_array_constructor_signature(),
        get_array_constructor_signature_without_suffix(),
        /*dag_alias_name=*/NULL,
        /*dag_preset_name=*/NULL,
        IDefinition::DS_INTRINSIC_DAG_ARRAY_CONSTRUCTOR,
        // Arg, no way to express "Any array type" here, but we need one return type.
        any_type,
        Array_ref<Dependence_node::Parameter>(),
        Dependence_node::FL_IS_EXPORTED);
    m_exported_nodes.push_back(n);
}

// Create the one-and-only cast operator.
void DAG_dependence_graph::create_dag_cast_operator(IType const *any_type)
{
    MDL_ASSERT(m_is_builtins);

    Dependence_node::Parameter params[] = {
        Dependence_node::Parameter(any_type, "cast")
    };

    // Create the magic cast operator. There is only one.
    Dependence_node *n = get_node(
        "operator_cast(<0>)",
        "operator_cast",
        /*dag_alias_name=*/NULL,
        /*dag_preset_name=*/NULL,
        operator_to_semantic(IExpression::OK_CAST),
        any_type,
        params,
        Dependence_node::FL_IS_EXPORTED);
    m_exported_nodes.push_back(n);
}

} // mdl
} // mi
