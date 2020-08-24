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

#include <mi/base/iallocator.h>

#include "compilercore_cc_conf.h"
#include "compilercore_analysis.h"
#include "compilercore_modules.h"
#include "compilercore_def_table.h"
#include "compilercore_errors.h"
#include "compilercore_mdl.h"
#include "compilercore_tools.h"
#include "compilercore_assert.h"
#include "compilercore_positions.h"

namespace mi {
namespace mdl {

typedef Store<Definition const *>  Definition_store;

// Get the analysis info for a statement.
Stmt_info &Stmt_info_data::get_stmt_info(IStatement const *stmt)
{
    Stmt_map::iterator it = m_stmt_map.find(stmt);
    if (it == m_stmt_map.end()) {
        it = m_stmt_map.insert(std::make_pair(stmt, Stmt_info())).first;
    }
    return it->second;
}

// Get the analysis info for a statement.
Stmt_info const &Stmt_info_data::get_stmt_info(IStatement const *stmt) const 
{
    Stmt_map::const_iterator it = m_stmt_map.find(stmt);
    if (it == m_stmt_map.end()) {
        MDL_ASSERT(!"unknown statement in statement analysis info");
        return m_stmt_dummy;
    }
    return it->second;
}

// Get the analysis info for an expression.
Expr_info &Stmt_info_data::get_expr_info(IExpression const *expr)
{
    Expr_map::iterator it = m_expr_map.find(expr);
    if (it == m_expr_map.end()) {
        it = m_expr_map.insert(std::make_pair(expr, Expr_info())).first;
    }
    return it->second;
}

// Get the analysis info for an expression.
Expr_info const &Stmt_info_data::get_expr_info(IExpression const *expr) const
{
    Expr_map::const_iterator it = m_expr_map.find(expr);
    if (it == m_expr_map.end()) {
        MDL_ASSERT(!"unknown expression in statement analysis info");
        return m_expr_dummy;
    }
    return it->second;
}

// --------------------------- Semantic analysis ----------------------- //

// Run the semantic analysis on a module.
void Sema_analysis::run()
{
    visit(&m_module);
    report_unused_entities();
    check_exported_completeness();
}

// Constructor.
Sema_analysis::Sema_analysis(
    MDL            *compiler,
    Module         &module,
    Thread_context &ctx)
: Analysis(compiler, module, ctx)
, m_next_stmt_is_reachable(false)
, m_last_stmt_is_reachable(false)
, m_has_side_effect(true)
, m_has_call(true)
, m_inside_single_expr_body(false)
, m_curr_assigned_def(NULL)
, m_curr_entity_decl(NULL)
, m_context_stack(module.get_allocator())
, m_context_depth(0)
, m_loop_depth(0)
, m_switch_depth(0)
, m_expr_depth(0)
, m_stmt_info_data(module.get_allocator())
, m_curr_funcname(module.get_allocator())
{
}

namespace {

/// Helper class, reports non-exported but used entities.
class Default_checker : public Module_visitor
{
public:
    /// Constructor
    ///
    /// \param ana  the analysis
    explicit Default_checker(Analysis &ana, Definition const *entity)
    : m_ana(ana)
    , m_ent(entity)
    {
    }

private:
    /// Post-visitor for reference expressions.
    ///
    /// \param expr  the visited expression
    IExpression *post_visit(IExpression_reference *expr) MDL_FINAL
    {
        if (expr->is_array_constructor()) {
            // array constructors are predefined, no error
            return expr;
        }

        Definition const *def = impl_cast<Definition>(expr->get_definition());
        if (is_error(def)) {
            return expr;
        }

        IDefinition::Kind kind = def->get_kind();
        if (kind == IDefinition::DK_PARAMETER || kind == IDefinition::DK_MEMBER) {
            // access to other parameter/member is ok
            return expr;
        }

        if (def->has_flag(Definition::DEF_IS_PREDEFINED)) {
            // no need to export it
            return expr;
        }
        if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
            // there is someone that exports it
            return expr;
        }

        if (!def->has_flag(Definition::DEF_IS_EXPORTED)) {
            m_ana.error(
                DEFAULT_NOT_EXPORTED,
                expr->access_position(),
                Error_params(m_ana)
                    .add_signature(def)
                    .add_signature(m_ent));
        }
        return expr;
    }

private:
    /// The analysis.
    Analysis &m_ana;
    /// The exported entity.
    Definition const *m_ent;
};

}  // anonymous

// Check that the given type is exported.
bool Sema_analysis::check_exported_type(
    Definition const *user,
    IType const      *type,
    Bad_type_set     &bad_types)
{
    type = type->skip_type_alias();
    if (is<IType_error>(type)) {
        // the error type must not be exported
        return true;
    }

    if (IType_array const *a_type = as<IType_array>(type))
        type = a_type->get_element_type()->skip_type_alias();

    Scope *scope = m_def_tab->get_type_scope(type);
    MDL_ASSERT(scope != NULL);

    Definition const *type_def = scope->get_owner_definition();
    MDL_ASSERT(type_def != NULL);

    if (type_def->has_flag(Definition::DEF_IS_PREDEFINED)) {
        // predefined types must not (and cannot) be exported
        return true;
    }
    if (type_def->has_flag(Definition::DEF_IS_IMPORTED)) {
        // there is someone that exports it
        return true;
    }

    if (!type_def->has_flag(Definition::DEF_IS_EXPORTED)) {
        Bad_type_set::iterator it = bad_types.find(type);
        if (it != bad_types.end()) {
            // already reported
            return false;
        }
        bad_types.insert(type);

        error(
            USED_TYPE_NOT_EXPORTED,
            user,
            Error_params(*this)
            .add_signature(user)
            .add(type)
            );
        return false;
    }
    return true;
}

// Check that all referenced entities of an exported function are exported.
void Sema_analysis::check_exported_function_completeness(
    Definition const            *func_def,
    IDeclaration_function const *func_decl)
{
    Default_checker checker(*this, func_def);

    Bad_type_set bad_types(
        0, Bad_type_set::hasher(), Bad_type_set::key_equal(), m_module.get_allocator());

    IType_name const *rt_name = func_decl->get_return_type_name();
    IType const      *r_type  = rt_name->get_type();

    bool is_valid = check_exported_type(func_def, r_type, bad_types);

    for (int i = 0, n = func_decl->get_parameter_count(); i < n; ++i) {
        IParameter const  *param = func_decl->get_parameter(i);
        IExpression const *init  = param->get_init_expr();

        if (init != NULL) {
            (void)checker.visit(init);
        }

        IType_name const *t_name = param->get_type_name();
        IType const      *p_type = t_name->get_type();

        is_valid &= check_exported_type(func_def, p_type, bad_types);
    }

    if (!is_valid) {
        // We found type errors: Replace the function type by the error type, so
        // we would not have problems in modules importing this function.
        Definition *bad_def = const_cast<Definition *>(func_def);
        bad_def->set_type(m_tc.error_type);

        // this function cannot be found anymore in function overloading
        bad_def->set_flag(Definition::DEF_IGNORE_OVERLOAD);
    }
}

// Check that all referenced entities of an exported struct type are exported.
void Sema_analysis::check_exported_struct_completeness(
    Definition const               *struct_def,
    IDeclaration_type_struct const *struct_decl)
{
    Default_checker checker(*this, struct_def);

    Bad_type_set bad_types(
        0, Bad_type_set::hasher(), Bad_type_set::key_equal(), m_module.get_allocator());

    bool is_valid = true;
    for (int i = 0, n = struct_decl->get_field_count(); i < n; ++i) {
        IExpression const *init = struct_decl->get_field_init(i);

        if (init != NULL) {
            (void)checker.visit(init);
        }

        IType_name const *t_name = struct_decl->get_field_type_name(i);
        IType const      *p_type = t_name->get_type();

        is_valid &= check_exported_type(struct_def, p_type, bad_types);
    }

    if (!is_valid) {
        // We found type errors: Replace the struct type by the error type, so
        // we would not have problems in modules importing this function.
        Definition *bad_def = const_cast<Definition *>(struct_def);
        bad_def->set_type(m_tc.error_type);
    }
}

// Check that all referenced entities of an exported annotations are exported.
void Sema_analysis::check_exported_annotation_completeness(
    Definition const              *anno_def,
    IDeclaration_annotation const *anno_decl)
{
    Default_checker checker(*this, anno_def);

    Bad_type_set bad_types(
        0, Bad_type_set::hasher(), Bad_type_set::key_equal(), m_module.get_allocator());

    bool is_valid = true;
    for (int i = 0, n = anno_decl->get_parameter_count(); i < n; ++i) {
        IParameter const  *param = anno_decl->get_parameter(i);
        IExpression const *init  = param->get_init_expr();

        if (init != NULL) {
            (void)checker.visit(init);
        }

        IType_name const *t_name = param->get_type_name();
        IType const      *p_type = t_name->get_type();

        is_valid &= check_exported_type(anno_def, p_type, bad_types);
    }

    if (!is_valid) {
        // We found type errors: Replace the annotation type by the error type, so
        // we would not have problems in modules importing this function.
        Definition *bad_def = const_cast<Definition *>(anno_def);
        bad_def->set_type(m_tc.error_type);
    }
}

// Check that all referenced entities of an exported constant are exported.
void Sema_analysis::check_exported_constant_completeness(
    Definition const            *const_def,
    IDeclaration_constant const *const_decl)
{
    Bad_type_set bad_types(
        0, Bad_type_set::hasher(), Bad_type_set::key_equal(), m_module.get_allocator());

    IType_name const *c_name = const_decl->get_type_name();
    IType const      *c_type = c_name->get_type();

    if (!check_exported_type(const_def, c_type, bad_types)) {
        // We found a type error: Replace the constant type by the error type, so
        // we would not have problems in modules importing this function.
        Definition *bad_def = const_cast<Definition *>(const_def);
        bad_def->set_type(m_tc.error_type);
    }
}

// Check that all referenced types and all referenced entities
// (in a default initializer) of an exported entity are exported too.
void Sema_analysis::check_exported_completeness()
{
    for (int i = 0, n = m_module.get_exported_definition_count(); i < n; ++i) {
        Definition const *def = m_module.get_exported_definition(i);

        if (def->has_flag(Definition::DEF_IS_IMPORTED)) {
            // do not re-check imported definitions
            continue;
        }

        IDeclaration const *decl = def->get_declaration();
        if (decl == NULL) {
            // compiler generated
            continue;
        }

        switch (decl->get_kind()) {
        case IDeclaration::DK_FUNCTION:
            check_exported_function_completeness(def, cast<IDeclaration_function>(decl));
            break;
        case IDeclaration::DK_TYPE_STRUCT:
            check_exported_struct_completeness(def, cast<IDeclaration_type_struct>(decl));
            break;
        case IDeclaration::DK_ANNOTATION:
            check_exported_annotation_completeness(def, cast<IDeclaration_annotation>(decl));
            break;
        case IDeclaration::DK_CONSTANT:
            check_exported_constant_completeness(def, cast<IDeclaration_constant>(decl));
            break;
        default:
            break;
        }
    }
}

// Finds a parameter for the given array size if any.
Definition const *Sema_analysis::find_parameter_for_array_size(Definition const *arr_size)
{
    if (m_curr_entity_decl == NULL) {
        return NULL;
    }

    for (int i = 0, n = m_curr_entity_decl->get_parameter_count(); i < n; ++i) {
        IParameter const  *param  = m_curr_entity_decl->get_parameter(i);
        IDefinition const *p_def  = param->get_name()->get_definition();
        IType_array const *p_type = as<IType_array>(p_def->get_type());

        if (p_type == NULL || p_type->is_immediate_sized()) {
            continue;
        }

        IType_array_size const *size = p_type->get_deferred_size();
        if (size->get_size_symbol() == arr_size->get_sym())
            return impl_cast<Definition>(p_def);
    }
    return NULL;
}

// Mark the given entity as used and check for deprecation.
void Sema_analysis::mark_used(Definition const *def, Position const &pos)
{
    const_cast<Definition *>(def)->set_flag(Definition::DEF_IS_USED);

    if (def->has_flag(Definition::DEF_IS_DEPRECATED)) {
        // don't report deprecated warning, if the current entity is already deprecated
        if (m_curr_entity_def && m_curr_entity_def->has_flag(Definition::DEF_IS_DEPRECATED)) {
            return;
        }

        IValue_string const *msg = m_module.get_deprecated_message(def);

        warning(
            DEPRECATED_ENTITY,
            pos,
            Error_params(*this)
                .add(def->get_sym())
                .add_opt_message(msg));
    }
}

// Report unused entities.
void Sema_analysis::report_unused_entities()
{
    class Reporter : public IDefinition_visitor {
    public:
        Reporter(Analysis &ana) : m_ana(ana) {}

        void visit(Definition const *def) const MDL_FINAL
        {
            if (is_error(def)) {
                return;
            }
            if (def->has_flag(Definition::DEF_IS_USED)) {
                if (def->has_flag(Definition::DEF_IS_UNUSED)) {
                    // used entity marked as unused
                    int wcode = 0;
                    switch (def->get_kind()) {
                    case IDefinition::DK_VARIABLE:
                        wcode = USED_VARIABLE_MARKED_UNUSED;
                        break;
                    case IDefinition::DK_CONSTANT:
                        wcode = USED_CONSTANT_MARKED_UNUSED;
                        break;
                    case IDefinition::DK_PARAMETER:
                        wcode = USED_PARAMETER_MARKED_UNUSED;
                        break;
                    default:
                        return;
                    }

                    m_ana.warning(
                        wcode,
                        def,
                        Error_params(m_ana).add_signature(def));
                }
            } else {
                // unused entity
                if (def->has_flag(Definition::DEF_IS_EXPORTED)) {
                    // exported entities are never unused
                    return;
                }
                if (def->has_flag(Definition::DEF_IS_COMPILER_GEN)) {
                    // compiler generated entities are never unused
                    return;
                }
                if (def->has_flag(Definition::DEF_IS_UNUSED) ||
                    def->has_flag(Definition::DEF_IS_IMPORTED)) {
                    // suppress warning
                    return;
                }

                Position const *pos = def->get_position();
                if (pos == NULL) {
                    // only built-in entities have no position, these are never unused
                    return;
                }

                int wcode = 0;
                switch (def->get_kind()) {
                case IDefinition::DK_VARIABLE:
                    wcode = UNUSED_VARIABLE;
                    break;
                case IDefinition::DK_CONSTANT:
                    wcode = UNUSED_CONSTANT;
                    break;
                case IDefinition::DK_PARAMETER:
                    wcode = UNUSED_PARAMETER;
                    if (def->has_flag(Definition::DEF_IS_WRITTEN)) {
                        // a parameter that is only written is not strictly "unused"
                        wcode = PARAMETER_ONLY_WRITTEN;
                    }
                    break;
                default:
                    return;
                }

                m_ana.warning(
                    wcode,
                    def,
                    Error_params(m_ana).add_signature(def));
                if (wcode != PARAMETER_ONLY_WRITTEN && def->has_flag(Definition::DEF_IS_WRITTEN)) {
                    m_ana.add_note(
                        WRITTEN_BUT_NOT_READ,
                        def,
                        Error_params(m_ana).add_signature(def));
                }
            }
        }

    private:
        Analysis &m_ana;
    };

    Reporter reporter(*this);

    m_def_tab->walk(&reporter);
}

// Get the analysis info for a statement.
Stmt_info &Sema_analysis::get_stmt_info(IStatement const *stmt)
{
    return m_stmt_info_data.get_stmt_info(stmt);
}

// Enter a statement and put it into the context stack.
void Sema_analysis::enter_statement(IStatement const *stmt)
{
    if (m_context_depth < m_context_stack.size()) {
        m_context_stack[m_context_depth] = stmt;
    } else {
        m_context_stack.push_back(stmt);
    }
    ++m_context_depth;
}

// Leave a statement and remove it from the context stack.
void Sema_analysis::leave_statement(IStatement const *stmt)
{
    MDL_ASSERT(stmt == get_context_stmt(0));
    --m_context_depth;
}

// Enter a loop into the context stack.
void Sema_analysis::enter_loop(IStatement_loop const *loop)
{
    ++m_loop_depth;
    enter_statement(loop);
}

// Leave a loop and remove it from the context stack.
void Sema_analysis::leave_loop(IStatement_loop const *loop)
{
    leave_statement(loop);
    --m_loop_depth;
}

// Enter a switch and put it onto the context stack.
void Sema_analysis::enter_switch(IStatement_switch const *stmt)
{
    ++m_switch_depth;
    enter_statement(stmt);
}

// Leave a switch and remove it from the context stack.
void Sema_analysis::leave_switch(IStatement_switch const *stmt)
{
    leave_statement(stmt);
    --m_switch_depth;
}

// Enter a case and put it onto the context stack.
void Sema_analysis::enter_case(IStatement_case const *stmt)
{
    IStatement const *switch_stmt = get_context_stmt(0);

    // the grammar guarantees this, but check to be sure
    MDL_ASSERT(is<IStatement_switch>(switch_stmt));

    Stmt_info &switch_info = get_stmt_info(switch_stmt);

    // a case label is reachable if the switch is reachable
    Stmt_info &case_info = get_stmt_info(stmt);
    case_info.m_reachable_start = m_next_stmt_is_reachable = switch_info.m_reachable_start;

    enter_statement(stmt);
}

// Leave a case and remove it from the context stack.
void Sema_analysis::leave_case(IStatement_case const *stmt)
{
    leave_statement(stmt);
}

// Return the top context stack statement.
IStatement const *Sema_analysis::get_context_stmt(size_t depth) const
{
    if (depth < m_context_depth) {
        return m_context_stack[m_context_depth - 1 - depth];
    }
    return NULL;
}

// Return true if a switch has one case that can left the switch.
bool Sema_analysis::has_one_case_reachable_exit(
    IStatement_switch const *switch_stmt,
    bool                    &has_error)
{
    has_error = false;

    IStatement_case const *def_node = NULL;

    typedef hash_set<int>::Type Case_set;
    Case_set case_set(0, Case_set::hasher(), Case_set::key_equal(), m_module.get_allocator());

    int n = switch_stmt->get_case_count();
    for (int i = 0; i < n; ++i) {
        IStatement_case const *case_stmt = as<IStatement_case>(switch_stmt->get_case(i));

        if (case_stmt == NULL) {
            // ignore error
            has_error = true;
            continue;
        }

        if (IExpression const *label = case_stmt->get_label()) {
            if (IExpression_literal const *lit = as<IExpression_literal>(label)) {
                Stmt_info const &info = get_stmt_info(case_stmt);
                if (info.m_case_has_break) {
                    // case has a reachable break
                    return true;
                }

                if (IValue_enum const *val = as<IValue_enum>(lit->get_value())) {
                    // remember this case
                    case_set.insert(val->get_value());
                }
            } else {
                // should not happen here, a previous error
                has_error = true;
            }
        } else {
            if (def_node != NULL)
                has_error = true;
            def_node = case_stmt;
        }
    }

    // we have not found a reachable break so far, check if the last
    // case has a reachable fall though
    if (n > 0) {
        Stmt_info const &info = get_stmt_info(switch_stmt->get_case(n - 1));

        // reachable if we could fall through the last case
        if (info.m_reachable_exit) {
            return true;
        }
    }

    IExpression const *cond        = switch_stmt->get_condition();
    IType const       *switch_type = cond->get_type();

    // check, if we have a implicit default
    if (def_node == NULL) {
        if (IType_enum const *e_type = as<IType_enum>(switch_type)) {
            // Enum switch: check if all cases are handled ...
            for (int i = 0, n = e_type->get_value_count(); i < n; ++i) {
                ISymbol const *e_sym;
                int           e_code;

                e_type->get_value(i, e_sym, e_code);

                Case_set::const_iterator it = case_set.find(e_code);
                if (it == case_set.end()) {
                    // we found a case that is not handled, so we can fall through the switch
                    return true;
                }
            }
            // all cases are handled, so NO implicit fall through exists
            return false;
        } else {
            // Int switch: we have a silent default if NO explicit default was specified
            return true;
        }
    } else {
        // we have an explicit default, check if it is dead
        bool is_dead = true;

        if (IType_enum const *e_type = as<IType_enum>(switch_type)) {
            // Enum switch: check if all cases are handled ...
            for (int i = 0, n = e_type->get_value_count(); i < n; ++i) {
                ISymbol const *e_sym;
                int           e_code;

                e_type->get_value(i, e_sym, e_code);

                Case_set::const_iterator it = case_set.find(e_code);
                if (it == case_set.end()) {
                    // we found a case that is not handled, so the default is not dead
                    is_dead = false;
                    break;
                }
            }
        } else {
            // Int switch: default is never dead
            is_dead = false;
        }

        if (!is_dead) {
            Stmt_info const &info = get_stmt_info(def_node);

            // we can exit the switch if the default case has a reachable
            // break or is reachable at exit
            if (info.m_case_has_break || info.m_reachable_exit) {
                return true;
            }
        }
    }
    return false;
}

// Returns true if the parent statement is an if (i.e. we are inside a the or an else)
bool Sema_analysis::inside_if() const
{
  if (m_context_depth == 0) {
      return false;
  }
  return is<IStatement_if>(get_context_stmt(0));
}

// Check an if-cascade for identical conditions.
void Sema_analysis::check_if_cascade(IStatement_if const *stmt)
{
    size_t depth = 0;

    for (IStatement_if const *top = stmt; top != NULL; ++depth) {
        IStatement const *t = top->get_else_statement();
        top = t != NULL ? as<IStatement_if>(t) : NULL;
    }

    if (depth <= 1) {
        return;
    }

    VLA <IExpression const *> conds(m_module.get_allocator(), depth);
    VLA <size_t>              marker(m_module.get_allocator(), depth);

    depth = 0;
    for (IStatement_if const *top = stmt; top != NULL; ++depth) {
        conds[depth]  = top->get_condition();
        marker[depth] = 0;

        IStatement const *t = top->get_else_statement();
        top = t != NULL ? as<IStatement_if>(t) : NULL;
    }

    size_t bads = 0;
    for (size_t i = depth - 1; i > 0; --i) {
        bool found = false;
        for (size_t j = 0; j < i; ++j) {
            if (marker[j] != 0)
                continue;
            if (identical_expressions(conds[j], conds[i])) {
                marker[j] = i;
                found = true;
            }
        }
        if (found) {
            marker[i] = i;
            ++bads;
        }
    }

    if (bads > 0) {
        // found identical conditions
        size_t start = 0;
        for (size_t i = 1; i <= bads; ++i) {
            size_t j, m = 0;
            for (j = start; j < depth; ++j) {
                if (marker[j] != 0) {
                    m = marker[j];
                    start = j + 1;
                    break;
                }
            }
            warning(
                IDENTICAL_IF_CONDITION,
                conds[j]->access_position(),
                Error_params(*this));
            for (++j; j < depth; ++j) {
                if (marker[j] == m) {
                    add_note(
                        SAME_CONDITION,
                        conds[j]->access_position(),
                        Error_params(*this));
                }
            }
        }
    }
}

// Begin of a statement.
Stmt_info &Sema_analysis::start_statement(IStatement const *stmt)
{
    Stmt_info &info = get_stmt_info(stmt);

    info.m_reachable_start = m_next_stmt_is_reachable;
    info.m_reachable_exit  = info.m_reachable_start;

    if (m_last_stmt_is_reachable && !m_next_stmt_is_reachable) {
        // this statement is the first non-reachable one
        warning(
            UNREACHABLE_STATEMENT,
            stmt->access_position(),
            Error_params(*this));

        // Note: we must issue the warning BEFORE any children are evaluated,
        // but than have to ensure the it is not issued for the children again,
        // as the m_last_stmt_is_reachable is set only at function end
        m_last_stmt_is_reachable = false;
    }

    return info;
}

// End of a statement.
void Sema_analysis::end_statement(IStatement const *stmt)
{
    Stmt_info &info = get_stmt_info(stmt);

    m_last_stmt_is_reachable = info.m_reachable_start;
    info.m_reachable_exit    = m_next_stmt_is_reachable;
}

// End of a statement.
void Sema_analysis::end_statement(Stmt_info &info)
{
    m_last_stmt_is_reachable = info.m_reachable_start;
    info.m_reachable_exit    = m_next_stmt_is_reachable;
}

// generic start
bool Sema_analysis::pre_visit(IStatement *stmt)
{
    start_statement(stmt);
    return true;
}

// generic end
void Sema_analysis::post_visit(IStatement *stmt)
{
    end_statement(stmt);
}

// compound start
bool Sema_analysis::pre_visit(IStatement_compound *stmt)
{
    enter_statement(stmt);
    start_statement(stmt);
    return true;
}

// compound end
void Sema_analysis::post_visit(IStatement_compound *stmt)
{
    end_statement(stmt);
    leave_statement(stmt);
}

// handle while
bool Sema_analysis::pre_visit(IStatement_while *stmt)
{
    enter_loop(stmt);

    Stmt_info &info = start_statement(stmt);

    // check for useless expression statements
    m_has_side_effect = m_has_call = false;

    IExpression const *cond   = stmt->get_condition();
    IExpression const *n_cond = visit(cond);
    if (n_cond != cond) {
        stmt->set_condition(n_cond);
    }

    IStatement const *body = stmt->get_body();

    if (!m_has_side_effect && is<IStatement_expression>(body)) {
        IStatement_expression const *expr_stmt = cast<IStatement_expression>(body);
        if (expr_stmt->get_expression() == NULL) {
            // while (...);
            // we have no volatile in MDL, so this construct is probably an error
            warning(
                EMPTY_CONTROL_STMT,
                stmt->access_position(),
                Error_params(*this).add("while"));
            if (m_has_call) {
                add_note(
                    FUNCTION_CALLS_ARE_STATE_INDEPENDENT,
                    stmt->access_position(),
                    Error_params(*this));
            }
        }
    }
    visit(body);

    end_statement(info);

    // do not visit children anymore
    return false;
}

// start do-while
bool Sema_analysis::pre_visit(IStatement_do_while *stmt)
{
    enter_loop(stmt);

    start_statement(stmt);
    return true;
}

// start for
bool Sema_analysis::pre_visit(IStatement_for *stmt)
{
    enter_loop(stmt);

    start_statement(stmt);
    return true;
}

// start switch
bool Sema_analysis::pre_visit(IStatement_switch *stmt)
{
    enter_switch(stmt);

    start_statement(stmt);
    return true;
}

// start case
bool Sema_analysis::pre_visit(IStatement_case *stmt)
{
    enter_case(stmt);

    start_statement(stmt);
    return true;
}

// handle if statement
bool Sema_analysis::pre_visit(IStatement_if *stmt)
{
    enter_statement(stmt);

    Stmt_info &info = start_statement(stmt);

    IExpression const *expr           = stmt->get_condition();
    IStatement const  *then_statement = stmt->get_then_statement();
    IStatement const  *else_statement = stmt->get_else_statement();

    if (else_statement == NULL && is<IStatement_expression>(then_statement)) {
        IStatement_expression const *expr_stmt = cast<IStatement_expression>(then_statement);
        if (expr_stmt->get_expression() == NULL) {
            // found an if (...);
            warning(
                EMPTY_CONTROL_STMT,
                stmt->access_position(),
                Error_params(*this).add("if"));
        }
    }
    IExpression const *n_expr = visit(expr);
    if (n_expr != expr) {
        stmt->set_condition(n_expr);
        expr = n_expr;
    }
    visit(then_statement);

    if (else_statement == NULL) {
        // the next statement after an if without else is reachable if
        // the if was reachable at start
        m_next_stmt_is_reachable = info.m_reachable_start;
    } else {
        // the else statement is reachable at start if the if was
        m_next_stmt_is_reachable = info.m_reachable_start;
        visit(else_statement);

        if (info.m_reachable_start) {
            // the next statement after an if is reachable if
            // the if or the else case is reachable at exit
            Stmt_info const &last_info = get_stmt_info(then_statement);

            bool after_last = last_info.m_reachable_exit;

            // we have not check for the last child of then, this
            // is m_next_statement ...
            m_next_stmt_is_reachable |= after_last;
        }
    }

    if (else_statement != NULL) {
        if (identical_statements(then_statement, else_statement)) {
            warning(
                IDENTICAL_THEN_ELSE,
                stmt->access_position(),
                Error_params(*this));
        }
    }

    end_statement(info);

    leave_statement(stmt);

    if (!inside_if()) {
        check_if_cascade(stmt);
    }

    // do not visit children
    return false;
}

bool Sema_analysis::pre_visit(IStatement_expression *stmt)
{
    start_statement(stmt);

    // check for useless expression statements
    m_has_side_effect = m_has_call = false;

    return true;
}

// Set the current function name from a (valid) function definition.
void Sema_analysis::set_curr_funcname(
    IDefinition const *def,
    bool              is_material)
{
    m_string_buf->clear();

    m_printer->print(def->get_symbol());
    if (!is_material && !is_error(def)) {
        IType_function const *f_type = cast<IType_function>(def->get_type());

        m_printer->print('(');
        for (int i = 0, n = f_type->get_parameter_count(); i < n; ++i) {
            IType const   *p_tp;
            ISymbol const *p_sym;

            f_type->get_parameter(i, p_tp, p_sym);
            if (i > 0) {
                m_printer->print(", ");
            }
            m_printer->print(p_tp);
        }
        m_printer->print(')');
    }
    m_curr_funcname = m_string_buf->get_data();
}

// start of function/material
bool Sema_analysis::pre_visit(IDeclaration_function *decl)
{
    // first statement will be reachable
    m_next_stmt_is_reachable = true;
    m_last_stmt_is_reachable = true;

    m_inside_single_expr_body = false;

    IDefinition const *def = decl->get_definition();
    if (!is_error(def)) {
        m_curr_entity_def = impl_cast<Definition>(def);

        IType_function const *ftype = cast<IType_function>(def->get_type());
        IType const          *rtype = ftype->get_return_type();

        if (rtype == m_tc.material_type) {
            m_inside_single_expr_body = true;
        }
    }

    set_curr_funcname(def, m_inside_single_expr_body);

    IType_name const *rname = decl->get_return_type_name();
    visit(rname);

    if (IAnnotation_block const *ret_anno = decl->get_return_annotations()) {
        visit(ret_anno);
    }

    ISimple_name const *sname = decl->get_name();
    visit(sname);

    for (size_t i = 0, n = decl->get_parameter_count(); i < n; ++i) {
        IParameter const *param = decl->get_parameter(i);
        visit(param);
    }

    if (IStatement const *body = decl->get_body()) {
        if (m_module.get_mdl_version() >= IMDL::MDL_VERSION_1_6 &&
            is<IStatement_expression>(body))
        {
            // check for single expression functions
            m_inside_single_expr_body = true;
        }

        // set the current decl only INSIDE the body
        if (!is_error(def)) {
            m_curr_entity_decl = decl;
        }
        visit(body);
        m_curr_entity_decl = NULL;
    }

    if (IAnnotation_block const *anno = decl->get_annotations()) {
        visit(anno);
    }

    m_curr_entity_def = NULL;

    // do not visit children, all done already
    return false;
}

// end while
void Sema_analysis::post_visit(IStatement_while *stmt)
{
    leave_loop(stmt);

    Stmt_info &info = get_stmt_info(stmt);

    // the next statement is reachable at start if the while loop was
    m_next_stmt_is_reachable = info.m_reachable_start;

    end_statement(info);
}

// end of for
void Sema_analysis::post_visit(IStatement_for *stmt)
{
    leave_loop(stmt);

    Stmt_info &info = get_stmt_info(stmt);

    // the next statement is reachable if the loop was reachable
    m_next_stmt_is_reachable = info.m_reachable_start;

    end_statement(info);
}

// end of do-while
void Sema_analysis::post_visit(IStatement_do_while *stmt)
{
    Stmt_info &info = get_stmt_info(stmt);

    leave_loop(stmt);
    if (info.m_reachable_start) {
        // the next statement is reachable if the last statement was reachable
        // or this reachable loop had a reachable break
        if (info.m_loop_has_break) {
            m_next_stmt_is_reachable = true;
        }
    }
    end_statement(info);
}

// end of switch
void Sema_analysis::post_visit(IStatement_switch *stmt)
{
    Stmt_info &info = get_stmt_info(stmt);

    leave_switch(stmt);

    if (info.m_reachable_start) {
        // fix simple fall-through labels
        int n = stmt->get_case_count();
        for (int i = 0; i < n; ++i) {
            IStatement_case const *case_stmt = as<IStatement_case>(stmt->get_case(i));

            if (case_stmt == NULL) {
                // ignore error
                continue;
            }

            if (case_stmt->get_statement_count() == 0) {
                // this is just a label, the statements are in the next case
                IStatement_case const *next = NULL;
                int j = i + 1;
                for (; j < n; ++j) {
                    next = as<IStatement_case>(stmt->get_case(j));
                    if (next == NULL) {
                        // error, what now?
                        break;
                    }
                    if (next->get_statement_count() != 0) {
                        // found real code
                        break;
                    }
                }
                if (next != NULL) {
                    Stmt_info const &n_info = get_stmt_info(next);

                    // fill up all case statement until j, all are simple labels
                    for (; i < j; ++i) {
                        IStatement_case const *case_stmt = as<IStatement_case>(stmt->get_case(i));

                        if (case_stmt == NULL) {
                            // ignore error
                            continue;
                        }

                        Stmt_info &info = get_stmt_info(case_stmt);
                        info = n_info;
                    }
                }
            }
        }

        // the next statement is reachable if we have a silent default or
        // one of our cases is ends reachable
        bool has_error;
        m_next_stmt_is_reachable = has_one_case_reachable_exit(stmt, has_error);

        if (has_error) {
            // assume reachable ???
            m_next_stmt_is_reachable = true;
        }
    }
    end_statement(info);
}

// end of a case
void Sema_analysis::post_visit(IStatement_case *stmt)
{
    leave_case(stmt);
    end_statement(stmt);
}

// end of a break
void Sema_analysis::post_visit(IStatement_break *stmt)
{
    Stmt_info &info = get_stmt_info(stmt);

    if (!inside_loop() && !inside_switch()) {
        error(
            BREAK_OUTSIDE_LOOP_OR_SWITCH,
            stmt->access_position(),
            Error_params(*this));
    } else {
        if (info.m_reachable_start) {
            // reachable break
            bool walk = true;
            bool is_conditional = false;

            for (size_t depth = 0; walk; ++depth) {
                IStatement const *context_stmt = get_context_stmt(depth);
                MDL_ASSERT(context_stmt != NULL && "context stack unexpected empty");

                switch (context_stmt->get_kind()) {
                case IStatement::SK_CASE:
                    // case or a default statement
                    {
                        Stmt_info &context_info = get_stmt_info(context_stmt);
                        context_info.m_case_has_break = true;
                        walk = false;
                    }
                    break;
                case IStatement::SK_DO_WHILE:
                case IStatement::SK_WHILE:
                case IStatement::SK_FOR:
                    // a loop
                    {
                        Stmt_info &context_info = get_stmt_info(context_stmt);
                        walk = false;
                        context_info.m_loop_has_break = true;

                        if (!is_conditional) {
                            warning(
                                UNCONDITIONAL_EXIT_IN_LOOP,
                                stmt->access_position(),
                                Error_params(*this).add("break"));
                        }
                    }
                    break;
                case IStatement::SK_IF:
                    is_conditional = true;
                    break;
                default:
                    // search upwards
                    break;
                }
            }
        }
        // if the break is valid, the next statement will not be reachable
        m_next_stmt_is_reachable = false;
    }
    end_statement(info);
}

// end of a continue
void Sema_analysis::post_visit(IStatement_continue *stmt)
{
    if (!inside_loop()) {
        error(
            CONTINUE_OUTSIDE_LOOP,
            stmt->access_position(),
            Error_params(*this));
    } else {
        // if the continue is valid, the next statement will not be reachable
        m_next_stmt_is_reachable = false;
    }
    end_statement(stmt);
}

// end of a return
void Sema_analysis::post_visit(IStatement_return *stmt)
{
    // the next statement will not be reachable
    m_next_stmt_is_reachable = false;

    if (is_loop_unconditional()) {
        warning(
            UNCONDITIONAL_EXIT_IN_LOOP,
            stmt->access_position(),
            Error_params(*this).add("return"));
    }
    end_statement(stmt);
}

// end of statement expression
void Sema_analysis::post_visit(IStatement_expression *stmt)
{
    // check for useless expression statements
    if (!m_inside_single_expr_body) {
        if (!m_has_side_effect) {
            IExpression const *expr = stmt->get_expression();
            if (expr != NULL) {
                if (is<IType_error>(expr->get_type())) {
                    // suppress warnings for erroneous expressions
                    goto end;
                }

                bool warned = false;
                switch (expr->get_kind()) {
                case IExpression::EK_BINARY:
                    {
                        IExpression_binary const *bin_expr = cast<IExpression_binary>(expr);
                        IExpression_binary::Operator op = bin_expr->get_operator();
                        switch (op) {
                        case IExpression_binary::OK_EQUAL:
                            warning(
                                OPERATOR_EQUAL_WITHOUT_EFFECT,
                                expr->access_position(),
                                Error_params(*this).add(op));
                            warned = true;
                            break;
                        case IExpression_binary::OK_SELECT:
                        case IExpression_binary::OK_ARRAY_INDEX:
                        case IExpression_binary::OK_SEQUENCE:
                            // for those the "operator has no effect" warning looks strange
                            break;
                        default:
                            warning(
                                OPERATOR_WITHOUT_EFFECT,
                                expr->access_position(),
                                Error_params(*this).add(op));
                            warned = true;
                            break;
                        }
                    }
                    break;
                case IExpression::EK_UNARY:
                    {
                        IExpression_unary const *un_expr = cast<IExpression_unary>(expr);
                        IExpression_unary::Operator op = un_expr->get_operator();
                        warning(
                            OPERATOR_WITHOUT_EFFECT,
                            expr->access_position(),
                            Error_params(*this).add(op));
                        warned = true;
                    }
                    break;
                default:
                    break;
                }

                if (!warned) {
                    warning(
                        EFFECTLESS_STATEMENT,
                        stmt->access_position(),
                        Error_params(*this));
                    if (m_has_call) {
                        add_note(
                            FUNCTION_CALLS_ARE_STATE_INDEPENDENT,
                            stmt->access_position(),
                            Error_params(*this));
                    }
                }
            }
        }
    }
end:
    end_statement(stmt);
}

// end function/material
void Sema_analysis::post_visit(IDeclaration_function *decl)
{
    MDL_ASSERT(m_context_depth == 0 && "Context stack not empty at end of function");

    IDefinition const *def = decl->get_definition();
    if (!is_error(def)) {
        IType_function const *ftype = cast<IType_function>(def->get_type());
        IType const          *rtype = ftype->get_return_type();

        if (rtype != m_tc.material_type) {
            // real function
            if (IStatement const *body = decl->get_body()) {
                if (IStatement_compound const *block = as<IStatement_compound>(body)) {
                    if (m_next_stmt_is_reachable) {
                        // we could fall off a non-void function, a return is missing
                        Position_impl end_pos(&block->access_position());
                        end_pos.set_start_line(end_pos.get_end_line());
                        end_pos.set_start_column(end_pos.get_end_column());

                        error(
                            FUNCTION_MUST_RETURN_VALUE,
                            end_pos,
                            Error_params(*this).add(def->get_symbol()));
                    }
                }
            }
        }
    }
    if (m_inside_single_expr_body) {
        if (IStatement_expression const *body = as<IStatement_expression>(decl->get_body())) {
            if (IExpression const *expr = body->get_expression()) {
                // ensure that the material body expression is treated as "has effect", but skip
                // all sequences
                IExpression const *orig_expr = expr;
                while (IExpression_binary const *bin_expr = as<IExpression_binary>(expr)) {
                    if (bin_expr->get_operator() != IExpression_binary::OK_SEQUENCE) {
                        break;
                    }

                    // the left one NEVER has an effect
                    IExpression const *left = bin_expr->get_left_argument();
                    Expr_info &info = m_stmt_info_data.get_expr_info(left);
                    info.m_has_effect = false;

                    expr = bin_expr->get_right_argument();
                }
                if (orig_expr != expr)
                    const_cast<IStatement_expression *>(body)->set_expression(expr);

                Expr_info &info = m_stmt_info_data.get_expr_info(expr);
                info.m_has_effect = true;
            }
        }
    }
    m_inside_single_expr_body = false;

    m_curr_funcname.clear();
}

// Start of an expression.
void Sema_analysis::start_expression()
{
    ++m_expr_depth;
}

// End of an expression.
void Sema_analysis::end_expression(IExpression const *expr)
{
    Expr_info &info = m_stmt_info_data.get_expr_info(expr);
    info.m_has_effect = m_has_side_effect;

    MDL_ASSERT(m_expr_depth > 0);
    --m_expr_depth;
}

// Check if two expressions are syntactically identical.
bool Sema_analysis::identical_expressions(
    IExpression const *lhs,
    IExpression const *rhs) const
{
    IExpression::Kind kind = lhs->get_kind();
    if (kind != rhs->get_kind())
        return false;

    switch (kind) {
    case IExpression::EK_INVALID:
        // invalid expressions are never identical
        return false;

    case IExpression::EK_LITERAL:
        {
            IExpression_literal const *l_lhs = cast<IExpression_literal>(lhs);
            IExpression_literal const *l_rhs = cast<IExpression_literal>(rhs);

            return l_lhs->get_value() == l_rhs->get_value();
        }

    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *l_lhs = cast<IExpression_reference>(lhs);
            IExpression_reference const *l_rhs = cast<IExpression_reference>(rhs);

            bool is_arr_constr = l_lhs->is_array_constructor();
            if (is_arr_constr != l_rhs->is_array_constructor()) {
                return false;
            }

            return identical_type_names(l_lhs->get_name(), l_rhs->get_name());
        }

    case IExpression::EK_UNARY:
        {
            IExpression_unary const *u_lhs = cast<IExpression_unary>(lhs);
            IExpression_unary const *u_rhs = cast<IExpression_unary>(rhs);

            if (u_lhs->get_operator() != u_rhs->get_operator()) {
                return false;
            }
            return identical_expressions(u_lhs->get_argument(), u_rhs->get_argument());
        }

    case IExpression::EK_BINARY:
        {
            IExpression_binary const *b_lhs = cast<IExpression_binary>(lhs);
            IExpression_binary const *b_rhs = cast<IExpression_binary>(rhs);

            if (b_lhs->get_operator() != b_rhs->get_operator()) {
                return false;
            }
            if (!identical_expressions(b_lhs->get_left_argument(), b_rhs->get_left_argument())) {
                return false;
            }
            return identical_expressions(b_lhs->get_right_argument(), b_rhs->get_right_argument());
        }

    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *c_lhs = cast<IExpression_conditional>(lhs);
            IExpression_conditional const *c_rhs = cast<IExpression_conditional>(rhs);

            if (!identical_expressions(c_lhs->get_condition(), c_rhs->get_condition())) {
                return false;
            }
            if (!identical_expressions(c_lhs->get_true(), c_rhs->get_true())) {
                return false;
            }
            return identical_expressions(c_lhs->get_false(), c_rhs->get_false());
        }

    case IExpression::EK_CALL:
        {
            IExpression_call const *c_lhs = cast<IExpression_call>(lhs);
            IExpression_call const *c_rhs = cast<IExpression_call>(rhs);

            if (!identical_expressions(c_lhs->get_reference(), c_rhs->get_reference())) {
                return false;
            }
            int arg_count = c_lhs->get_argument_count();
            if (arg_count != c_rhs->get_argument_count()) {
                return false;
            }
            for (int i = 0; i < arg_count; ++i) {
                IArgument const *l_arg = c_lhs->get_argument(i);
                IArgument const *r_arg = c_rhs->get_argument(i);

                IArgument::Kind kind = l_arg->get_kind();
                if (kind != r_arg->get_kind()) {
                    return false;
                }
                if (kind == IArgument::AK_NAMED) {
                    IArgument_named const *l_n_arg = cast<IArgument_named>(l_arg);
                    IArgument_named const *r_n_arg = cast<IArgument_named>(r_arg);

                    ISimple_name const *l_name = l_n_arg->get_parameter_name();
                    ISimple_name const *r_name = r_n_arg->get_parameter_name();

                    if (l_name->get_symbol() != r_name->get_symbol()) {
                        return false;
                    }
                }
                if (!identical_expressions(
                    l_arg->get_argument_expr(), r_arg->get_argument_expr()))
                {
                    return false;
                }
            }
            return true;
        }

    case IExpression::EK_LET:
        {
            IExpression_let const *l_lhs = cast<IExpression_let>(lhs);
            IExpression_let const *l_rhs = cast<IExpression_let>(rhs);

            return identical_expressions(l_lhs->get_expression(), l_rhs->get_expression());
        }
    }
    MDL_ASSERT(!"Unknown expression kind");
    return false;
}

// Check if two statements are syntactically identical.
bool Sema_analysis::identical_statements(
    IStatement const *lhs,
    IStatement const *rhs) const
{
    IStatement::Kind kind = lhs->get_kind();
    if (kind != rhs->get_kind()) {
        return false;
    }

    switch (kind) {
    case IStatement::SK_INVALID:
        // invalid statements are never identical
        return false;

    case IStatement::SK_COMPOUND:
        {
            IStatement_compound const *c_lhs = cast<IStatement_compound>(lhs);
            IStatement_compound const *c_rhs = cast<IStatement_compound>(rhs);

            int n_stmts = c_lhs->get_statement_count();
            if (n_stmts != c_rhs->get_statement_count()) {
                return false;
            }
            for (int i = 0; i < n_stmts; ++i) {
                IStatement const *l_sub = c_lhs->get_statement(i);
                IStatement const *r_sub = c_rhs->get_statement(i);

                if (!identical_statements(l_sub, r_sub)) {
                    return false;
                }
            }
            return true;
        }
    case IStatement::SK_DECLARATION:
        {
            IStatement_declaration const *d_lhs = cast<IStatement_declaration>(lhs);
            IStatement_declaration const *d_rhs = cast<IStatement_declaration>(rhs);

            IDeclaration const *l_decl = d_lhs->get_declaration();
            IDeclaration const *r_decl = d_rhs->get_declaration();

            return identical_declarations(l_decl, r_decl);
        }
    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression const *e_lhs = cast<IStatement_expression>(lhs);
            IStatement_expression const *e_rhs = cast<IStatement_expression>(rhs);

            return identical_expressions(e_lhs->get_expression(), e_rhs->get_expression());
        }
    case IStatement::SK_IF:
        {
            IStatement_if const *i_lhs = cast<IStatement_if>(lhs);
            IStatement_if const *i_rhs = cast<IStatement_if>(rhs);

            if (!identical_expressions(i_lhs->get_condition(), i_rhs->get_condition())) {
                return false;
            }

            if (!identical_statements(i_lhs->get_then_statement(), i_rhs->get_then_statement())) {
                return false;
            }

            IStatement const *e_lhs = i_lhs->get_then_statement();
            IStatement const *e_rhs = i_rhs->get_then_statement();

            if (e_lhs == e_rhs) {
                return true;
            }
            if (e_lhs == NULL || e_rhs == NULL) {
                return false;
            }
            return identical_statements(e_lhs, e_rhs);
        }
    case IStatement::SK_CASE:
        {
            IStatement_case const *c_lhs = cast<IStatement_case>(lhs);
            IStatement_case const *c_rhs = cast<IStatement_case>(rhs);

            IExpression const *l_expr = c_lhs->get_label();
            IExpression const *r_expr = c_rhs->get_label();

            if (l_expr == NULL || r_expr == NULL) {
                if (l_expr != r_expr) {
                    // only one is a default case
                    return false;
                }
            } else {
                if (!identical_expressions(l_expr, r_expr)) {
                    return false;
                }
            }

            int n_stmts = c_lhs->get_statement_count();
            if (n_stmts != c_rhs->get_statement_count()) {
                return false;
            }
            for (int i = 0; i < n_stmts; ++i) {
                IStatement const *l_sub = c_lhs->get_statement(i);
                IStatement const *r_sub = c_rhs->get_statement(i);

                if (!identical_statements(l_sub, r_sub)) {
                    return false;
                }
            }
            return true;
        }
    case IStatement::SK_SWITCH:
        {
            IStatement_switch const *s_lhs = cast<IStatement_switch>(lhs);
            IStatement_switch const *s_rhs = cast<IStatement_switch>(rhs);

            if (!identical_expressions(s_lhs->get_condition(), s_rhs->get_condition())) {
                return false;
            }
            int n_cases = s_lhs->get_case_count();
            if (n_cases != s_rhs->get_case_count()) {
                return false;
            }

            for (int i = 0; i < n_cases; ++i) {
                IStatement const *l_stmt = s_lhs->get_case(i);
                IStatement const *r_stmt = s_rhs->get_case(i);

                if (!identical_statements(l_stmt, r_stmt)) {
                    return false;
                }
            }
            return true;
        }
    case IStatement::SK_WHILE:
        {
            IStatement_while const *w_lhs = cast<IStatement_while>(lhs);
            IStatement_while const *w_rhs = cast<IStatement_while>(rhs);

            if (!identical_expressions(w_lhs->get_condition(), w_rhs->get_condition())) {
                return false;
            }
            return identical_statements(w_lhs->get_body(), w_rhs->get_body());
        }
    case IStatement::SK_DO_WHILE:
        {
            IStatement_do_while const *w_lhs = cast<IStatement_do_while>(lhs);
            IStatement_do_while const *w_rhs = cast<IStatement_do_while>(rhs);

            if (!identical_expressions(w_lhs->get_condition(), w_rhs->get_condition())) {
                return false;
            }
            return identical_statements(w_lhs->get_body(), w_rhs->get_body());
        }
    case IStatement::SK_FOR:
        {
            IStatement_for const *f_lhs = cast<IStatement_for>(lhs);
            IStatement_for const *f_rhs = cast<IStatement_for>(rhs);

            {
                IStatement const *i_lhs = f_lhs->get_init();
                IStatement const *i_rhs = f_rhs->get_init();

                if (i_lhs == NULL || i_rhs == NULL) {
                    if (i_lhs != i_rhs) {
                        // only one has an initializer
                        return false;
                    }
                } else {
                    if (!identical_statements(i_lhs, i_rhs)) {
                        return false;
                    }
                }
            }
            {
                IExpression const *e_lhs = f_lhs->get_condition();
                IExpression const *e_rhs = f_rhs->get_condition();

                if (e_lhs == NULL || e_rhs == NULL) {
                    if (e_lhs != e_rhs) {
                        // only one has a condition
                        return false;
                    }
                } else {
                    if (!identical_expressions(e_lhs, e_rhs)) {
                        return false;
                    }
                }
            }
            {
                IExpression const *n_lhs = f_lhs->get_update();
                IExpression const *n_rhs = f_rhs->get_update();

                if (n_lhs == NULL || n_rhs == NULL) {
                    if (n_lhs != n_rhs) {
                        // only one has an update
                        return false;
                    }
                } else {
                    if (!identical_expressions(n_lhs, n_rhs)) {
                        return false;
                    }
                }
            }
            return identical_statements(f_lhs->get_body(), f_rhs->get_body());
        }
    case IStatement::SK_BREAK:
        return true;
    case IStatement::SK_CONTINUE:
        return true;
    case IStatement::SK_RETURN:
        {
            IStatement_return const *l_lhs = cast<IStatement_return>(lhs);
            IStatement_return const *r_rhs = cast<IStatement_return>(rhs);

            IExpression const *l_expr = l_lhs->get_expression();
            IExpression const *r_expr = r_rhs->get_expression();

            if (l_expr == NULL || r_expr == NULL) {
                if (l_expr != r_expr) {
                    // only one is a return;
                    return false;
                }
            }
            return identical_expressions(l_expr, r_expr);
        }
    }

    MDL_ASSERT(!"Unknown statement kind");
    return false;
}

// Check if two declarations are syntactically identical.
bool Sema_analysis::identical_declarations(
    IDeclaration const *lhs,
    IDeclaration const *rhs) const
{
    IDeclaration::Kind kind = lhs->get_kind();
    if (kind != rhs->get_kind())
        return false;

    switch (kind) {
    case IDeclaration::DK_INVALID:
        // invalid declarations are never identical
        return false;

    case IDeclaration::DK_IMPORT:
    case IDeclaration::DK_ANNOTATION:
    case IDeclaration::DK_CONSTANT:
    case IDeclaration::DK_FUNCTION:
    case IDeclaration::DK_MODULE:
    case IDeclaration::DK_NAMESPACE_ALIAS:
        // only allowed at global scope, ignore
        return false;
    case IDeclaration::DK_TYPE_ALIAS:
    case IDeclaration::DK_TYPE_STRUCT:
    case IDeclaration::DK_TYPE_ENUM:
        // TODO: implement
        return false;
    case IDeclaration::DK_VARIABLE:
        {
            IDeclaration_variable const *v_lhs = cast<IDeclaration_variable>(lhs);
            IDeclaration_variable const *v_rhs = cast<IDeclaration_variable>(rhs);

            IType_name const *l_tp_name = v_lhs->get_type_name();
            IType_name const *r_tp_name = v_rhs->get_type_name();

            if (!identical_type_names(l_tp_name, r_tp_name)) {
                return false;
            }

            int n_vars = v_lhs->get_variable_count();
            if (n_vars != v_rhs->get_variable_count()) {
                return false;
            }

            for (int i = 0; i < n_vars; ++i) {
                ISimple_name const *l_name = v_lhs->get_variable_name(i);
                ISimple_name const *r_name = v_rhs->get_variable_name(i);

                if (l_name->get_symbol() != r_name->get_symbol()) {
                    return false;
                }

                IExpression const *l_init = v_lhs->get_variable_init(i);
                IExpression const *r_init = v_rhs->get_variable_init(i);

                if (l_init != NULL && r_init != NULL) {
                    if (!identical_expressions(l_init, r_init)) {
                        return false;
                    }
                } else {
                    if (l_init != r_init) {
                        // only one has a initializer
                        return false;
                    }
                }

                IAnnotation_block const *l_annos = v_lhs->get_annotations(i);
                IAnnotation_block const *r_annos = v_rhs->get_annotations(i);

                if (!identical_anno_blocks(l_annos, r_annos)) {
                    return false;
                }
            }
            return true;
        }
    }

    MDL_ASSERT(!"Unknown statement kind");
    return false;
}

// Check if two type names are syntactically identical.
bool Sema_analysis::identical_type_names(
    IType_name const *lhs,
    IType_name const *rhs) const
{
    if (lhs->is_absolute() != rhs->is_absolute()) {
        return false;
    }
    if (lhs->get_qualifier() != rhs->get_qualifier()) {
        return false;
    }

    IQualified_name const *q_lhs = lhs->get_qualified_name();
    IQualified_name const *q_rhs = rhs->get_qualified_name();

    if (!identical_qnames(q_lhs, q_rhs)) {
        return false;
    }

    bool is_arr = lhs->is_array();
    if (is_arr != rhs->is_array()) {
        return false;
    }

    if (is_arr) {
        bool is_concrete = lhs->is_concrete_array();
        if (is_concrete != rhs->is_concrete_array())
            return false;

        bool is_incomplete = lhs->is_incomplete_array();
        if (is_incomplete != rhs->is_incomplete_array()) {
            return false;
        }
        if (is_concrete) {
            if (!is_incomplete &&
                !identical_expressions(lhs->get_array_size(), rhs->get_array_size()))
            {
                return false;
            }
        } else {
            if (lhs->get_size_name()->get_symbol() != rhs->get_size_name()->get_symbol()) {
                return false;
            }
        }
    }
    return true;
}

// Check if two annotation blocks are syntactically identical.
bool Sema_analysis::identical_anno_blocks(
    IAnnotation_block const *lhs,
    IAnnotation_block const *rhs) const
{
    if (lhs == NULL || rhs == NULL) {
        return lhs == rhs;
    }

    int n = lhs->get_annotation_count();
    if (n != rhs->get_annotation_count()) {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        IAnnotation const *l = lhs->get_annotation(i);
        IAnnotation const *r = rhs->get_annotation(i);

        IQualified_name const *l_name = l->get_name();
        IQualified_name const *r_name = r->get_name();

        if (!identical_qnames(l_name, r_name)) {
            return false;
        }

        int arg_count = l->get_argument_count();
        if (arg_count != r->get_argument_count()) {
            return false;
        }

        for (int j = 0; j < arg_count; ++j) {
            IArgument const *l_arg = l->get_argument(j);
            IArgument const *r_arg = r->get_argument(j);

            IArgument::Kind kind = l_arg->get_kind();
            if (kind != r_arg->get_kind())
                return false;

            if (kind == IArgument::AK_NAMED) {
                IArgument_named const *n_l_arg = cast<IArgument_named>(l_arg);
                IArgument_named const *n_r_arg = cast<IArgument_named>(r_arg);

                ISimple_name const *l_name = n_l_arg->get_parameter_name();
                ISimple_name const *r_name = n_r_arg->get_parameter_name();

                if (l_name->get_symbol() != r_name->get_symbol()) {
                    return false;
                }
            }
            if (!identical_expressions(l_arg->get_argument_expr(), r_arg->get_argument_expr())) {
                return false;
            }
        }
    }
    return true;
}

// Check if two qualified names are syntactically identical.
bool Sema_analysis::identical_qnames(
    IQualified_name const *lhs,
    IQualified_name const *rhs) const
{
    if (lhs->is_absolute() != rhs->is_absolute()) {
        return false;
    }
    int n_comp = lhs->get_component_count();
    if (n_comp != rhs->get_component_count()) {
        return false;
    }

    for (int i = 0; i < n_comp; ++i) {
        ISimple_name const *l_name = lhs->get_component(i);
        ISimple_name const *r_name = rhs->get_component(i);

        if (l_name->get_symbol() != r_name->get_symbol()) {
            return false;
        }
    }
    return true;
}

// Returns true if the current statement is unconditional inside a loop.
bool Sema_analysis::is_loop_unconditional() const
{
    if (!inside_loop()) {
        return false;
    }

    for (size_t depth = 0; ; ++depth) {
        IStatement const *ctx_stmt = get_context_stmt(depth);

        MDL_ASSERT(ctx_stmt != NULL && "context stack unexpected empty");
        switch (ctx_stmt->get_kind()) {
        case IStatement::SK_IF:
            // inside an if, conditional
            return false;
        case IStatement::SK_CASE:
            // inside a case, not regarding the loop
            return false;
        case IStatement::SK_FOR:
        case IStatement::SK_WHILE:
        case IStatement::SK_DO_WHILE:
            // reached the loop
            return true;
        default:
            // continue search
            break;
        }
    }
}

// Replace the third and fourth parameter of assert by the
// current module name and this expression's line.
void Sema_analysis::insert_assert_params(IExpression_call *expr)
{
    Expression_factory &expr_fact = *m_module.get_expression_factory();
    Value_factory      &val_fact  = *m_module.get_value_factory();

    IValue_string const       *f        = val_fact.create_string(m_curr_funcname.c_str());
    IExpression_literal const *lit_func = expr_fact.create_literal(f);

    IArgument *arg_func = const_cast<IArgument *>(expr->get_argument(2));
    arg_func->set_argument_expr(lit_func);

    char const                *name     = m_module.get_name();
    IValue_string const       *s        = val_fact.create_string(name);
    IExpression_literal const *lit_file = expr_fact.create_literal(s);

    IArgument *arg_file = const_cast<IArgument *>(expr->get_argument(3));
    arg_file->set_argument_expr(lit_file);

    IValue_int const          *line     = val_fact.create_int(
        expr->access_position().get_start_line());
    IExpression_literal const *lit_line = expr_fact.create_literal(line);

    IArgument *arg_line = const_cast<IArgument *>(expr->get_argument(4));
    arg_line->set_argument_expr(lit_line);
}

// start of an expression
bool Sema_analysis::pre_visit(IExpression *)
{
    start_expression();
    // visit children
    return true;
}

// end of an expression
IExpression *Sema_analysis::post_visit(IExpression *expr)
{
    end_expression(expr);
    return expr;
}

// start of binary expression
bool Sema_analysis::pre_visit(IExpression_binary *expr)
{
    start_expression();

    bool is_assignment = false;
    bool is_read = false;
    IExpression_binary::Operator op = expr->get_operator();
    switch (op) {
    case IExpression_binary::OK_MULTIPLY_ASSIGN:
    case IExpression_binary::OK_DIVIDE_ASSIGN:
    case IExpression_binary::OK_MODULO_ASSIGN:
    case IExpression_binary::OK_PLUS_ASSIGN:
    case IExpression_binary::OK_MINUS_ASSIGN:
    case IExpression_binary::OK_SHIFT_LEFT_ASSIGN:
    case IExpression_binary::OK_SHIFT_RIGHT_ASSIGN:
    case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT_ASSIGN:
    case IExpression_binary::OK_BITWISE_AND_ASSIGN:
    case IExpression_binary::OK_BITWISE_XOR_ASSIGN:
    case IExpression_binary::OK_BITWISE_OR_ASSIGN:
        is_read = true;
        // fall through
    case IExpression_binary::OK_ASSIGN:
        is_assignment = true;
        break;
    case IExpression_binary::OK_LOGICAL_AND:
    case IExpression_binary::OK_LOGICAL_OR:
        if (identical_expressions(expr->get_left_argument(), expr->get_right_argument())) {
            warning(
                IDENTICAL_SUBEXPRESSION,
                expr->access_position(),
                Error_params(*this).add(op));
        }
        if (m_inside_single_expr_body) {
            // short-cut operator is allowed in materials, but work strict here
            warning(
                OPERATOR_IS_STRICT_IN_MATERIAL,
                expr->access_position(),
                Error_params(*this)
                    .add(op));
        }
        break;
    case IExpression_binary::OK_BITWISE_AND:
    case IExpression_binary::OK_BITWISE_OR:
    case IExpression_binary::OK_BITWISE_XOR:
        if (identical_expressions(expr->get_left_argument(), expr->get_right_argument())) {
            warning(
                IDENTICAL_SUBEXPRESSION,
                expr->access_position(),
                Error_params(*this).add(op));
        }
        break;
    case IExpression_binary::OK_SEQUENCE:
        if (m_inside_single_expr_body) {
            warning(
                SEQUENCE_WITHOUT_EFFECT_INSIDE_MATERIAL,
                expr->access_position(),
                Error_params(*this));
        }
        break;
    default:
        break;
    }

    {
        IExpression const *lhs = expr->get_left_argument();
        Definition const  *def = NULL;

        if (is_assignment) {
            IDefinition const *idef = get_lvalue_base(lhs);
            def = impl_cast<Definition>(idef);

            if (def != NULL) {
                const_cast<Definition *>(def)->set_flag(Definition::DEF_IS_WRITTEN);

                if (is_read || inside_nested_expression()) {
                    // we are either in a combined op= operator or
                    // in a nested expression, hence the value of the assigned
                    // variable is not lost
                    mark_used(def, lhs->access_position());
                }
            }
        }

        Definition_store current_assigned(m_curr_assigned_def, def);
        visit(lhs);
    }
    {
        IExpression const *rhs = expr->get_right_argument();
        visit(rhs);
    }

    if (is_assignment) {
        // assignments always have an effect
        m_has_side_effect = true;
    }

    // do not visit children
    return false;
}

// start of unary expression
bool Sema_analysis::pre_visit(IExpression_unary *expr)
{
    start_expression();

    bool is_assignment = false;
    switch (expr->get_operator()) {
    case IExpression_unary::OK_PRE_INCREMENT:
    case IExpression_unary::OK_PRE_DECREMENT:
    case IExpression_unary::OK_POST_INCREMENT:
    case IExpression_unary::OK_POST_DECREMENT:
        // always contains an assignment
        is_assignment = true;
        break;
    default:
        break;
    }

    {
        IExpression const *arg = expr->get_argument();
        Definition const  *def = NULL;

        if (is_assignment) {
            IDefinition const *idef = get_lvalue_base(arg);
            def = impl_cast<Definition>(idef);

            if (def != NULL) {
                // The increment/decrement operators do a READ and WRITE.
                const_cast<Definition *>(def)->set_flag(Definition::DEF_IS_WRITTEN);

                mark_used(def, arg->access_position());
            }
        }

        Definition_store current_assigned(m_curr_assigned_def, def);
        visit(arg);
    }

    if (is_assignment) {
        // assignments always have an effect
        m_has_side_effect = true;
    }

    // do not visit children
    return false;
}

// end of function call
IExpression *Sema_analysis::post_visit(IExpression_call *expr)
{
    // record the call
    m_has_call = true;

    IType const *res_type = expr->get_type();
    bool has_effect = false;

    if (is<IType_error>(res_type)) {
        has_effect = true;
    } else {
        // check for functions from the debug module, do NOT optimize them away, so
        // declare these have a side effect
        if (IExpression_reference const *ref = as<IExpression_reference>(expr->get_reference())) {
            if (IDefinition const *def = ref->get_definition()) {
                IDefinition::Semantics sema = def->get_semantics();
                if (is_debug_semantic(sema)) {
                    has_effect = true;

                    if (sema == IDefinition::DS_INTRINSIC_DEBUG_ASSERT) {
                        insert_assert_params(expr);
                    }
                }
            }
        }
        // Only function returning non-uniform results treated as having an effect here.
        // As there is no way to WRITE the state we could even ignore those ...
        IType::Modifiers mod = res_type->get_type_modifiers();
        if (!(mod & IType::MK_UNIFORM))
            has_effect = true;
    }

    if (has_effect) {
        m_has_side_effect = true;
    }

    if (IExpression_reference const *ref = as<IExpression_reference>(expr->get_reference())) {
        if (!ref->is_array_constructor()) {
            Definition const *def = impl_cast<Definition>(ref->get_definition());

            if (!is_error(def)) {
                if (def->has_flag(Definition::DEF_LITERAL_PARAM)) {
                    IArgument const   *arg  = expr->get_argument(0);
                    IExpression const *expr = arg->get_argument_expr();

                    if (!is<IExpression_literal>(expr)) {
                        // the first argument should be a literal: try to const-fold
                        bool is_invalid = false;
                        if (is_const_expression(expr, is_invalid)) {
                            IValue const *val =
                                expr->fold(&m_module, m_module.get_value_factory(), NULL);

                            if (!is<IValue_bad>(val)) {
                                Position const *pos = &expr->access_position();
                                expr = m_module.create_literal(val, pos);
                                const_cast<IArgument *>(arg)->set_argument_expr(expr);
                            } else {
                                MDL_ASSERT(is_invalid && "Const fold failed for valid const_expr");
                            }
                        } else {
                            error(
                                CONST_EXPR_ARGUMENT_REQUIRED,
                                expr->access_position(),
                                Error_params(*this)
                                .add_signature(def));
                        }
                    }
                }
            }
        }
    }

    end_expression(expr);
    return expr;
}

// end of reference
IExpression *Sema_analysis::post_visit(IExpression_reference *expr)
{
    // get the definition from the type name
    IType_name const      *type_name = expr->get_name();
    IQualified_name const *qual_name = type_name->get_qualified_name();
    Definition const      *def = impl_cast<Definition>(qual_name->get_definition());

    if (!is_error(def)) {
        if (!def->has_flag(Definition::DEF_IS_INCOMPLETE)) {
            if (def != m_curr_assigned_def) {
                // We are referencing this entity, so it is used.
                // The usage of the current assigned entity is handled in assignments.
                mark_used(def, expr->access_position());

                if (def->get_kind() == Definition::DK_ARRAY_SIZE) {
                    // If we are referencing an array size symbol of an parameter,
                    // mark the parameter as used. This avoids strange unused warnings on
                    // 'length()' implementations
                    Definition const *p_def = find_parameter_for_array_size(def);
                    if (p_def != NULL) {
                        // FIXME: Should we check for deprecation here?
                        const_cast<Definition *>(p_def)->set_flag(Definition::DEF_IS_USED);
                    }
                }
            }
        }
    }
    end_expression(expr);
    return expr;
}

// end of a ternary operator
IExpression *Sema_analysis::post_visit(IExpression_conditional *expr)
{
    if (identical_expressions(expr->get_true(), expr->get_false())) {
        warning(
            IDENTICAL_SUBEXPRESSION,
            expr->access_position(),
            Error_params(*this).add("?:"));
    }
    if (m_inside_single_expr_body) {
        // ternary operator is allowed in materials, but work strict here, not
        // lazy
        warning(
            OPERATOR_IS_STRICT_IN_MATERIAL,
            expr->access_position(),
            Error_params(*this).add(IExpression::OK_TERNARY));
    }
    end_expression(expr);
    return expr;
}

}  // mdl
}  // mi
