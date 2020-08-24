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

#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_statements.h>

#include "compilercore_optimizer.h"
#include "compilercore_analysis.h"
#include "compilercore_mdl.h"
#include "compilercore_def_table.h"
#include "compilercore_modules.h"
#include "compilercore_call_graph.h"
#include "compilercore_allocator.h"
#include "compilercore_tools.h"

namespace mi {
namespace mdl {

#define POS(pos) \
    (pos).get_start_line(), (pos).get_start_column(), \
    (pos).get_end_line(), (pos).get_end_column()

/// Inverse a binary relation, i.e. return R^(-1)
static bool inverse_relation(
    IExpression_binary::Operator op,
    IExpression_binary::Operator &inv_op)
{
    switch (op) {
    case IExpression_binary::OK_LESS:
        // < --> >
        inv_op = IExpression_binary::OK_GREATER;
        return true;
    case IExpression_binary::OK_LESS_OR_EQUAL:
        // <= --> >=
        inv_op = IExpression_binary::OK_GREATER_OR_EQUAL;
        return true;
    case IExpression_binary::OK_GREATER_OR_EQUAL:
        // >= --> <=
        inv_op = IExpression_binary::OK_LESS_OR_EQUAL;
        return true;
    case IExpression_binary::OK_GREATER:
        // > --> <
        inv_op = IExpression_binary::OK_LESS;
        return true;
    case IExpression_binary::OK_EQUAL:
        // == --> ==
        inv_op = IExpression_binary::OK_EQUAL;
        return true;
    case IExpression_binary::OK_NOT_EQUAL:
        // != --> !=
        inv_op = IExpression_binary::OK_NOT_EQUAL;
        return true;
    default:
        break;
    }
    // not a relation at all
    inv_op = op;
    return false;
}

/// Negate a binary relation, i.e. !R(a,b)
static bool negate_relation(
    IExpression_binary::Operator op,
    IType const                  *op_type,
    IExpression_binary::Operator &neg_op)
{
    op_type = op_type->skip_type_alias();

    IType::Kind kind = op_type->get_kind();

    bool unrestricted = true;
    if (kind == IType::TK_VECTOR) {
        IType_vector const *v_type = cast<IType_vector>(op_type);
        kind = v_type->get_element_type()->get_kind();
        unrestricted = false;
    } else if (kind == IType::TK_ARRAY) {
        IType_array const *a_type = cast<IType_array>(op_type);
        kind = a_type->get_element_type()->skip_type_alias()->get_kind();
        unrestricted = false;
    }

    switch (kind) {
    case IType::TK_BOOL:
    case IType::TK_INT:
        // all relations
        switch (op) {
        case IExpression_binary::OK_LESS:
            // < --> >=
            neg_op = IExpression_binary::OK_GREATER_OR_EQUAL;
            return unrestricted;
        case IExpression_binary::OK_LESS_OR_EQUAL:
            // <= --> >
            neg_op = IExpression_binary::OK_GREATER;
            return unrestricted;
        case IExpression_binary::OK_GREATER_OR_EQUAL:
            // >= --> <
            neg_op = IExpression_binary::OK_LESS;
            return unrestricted;
        case IExpression_binary::OK_GREATER:
            // > --> <=
            neg_op = IExpression_binary::OK_LESS_OR_EQUAL;
            return unrestricted;
        case IExpression_binary::OK_EQUAL:
            // == --> !=
            neg_op = IExpression_binary::OK_NOT_EQUAL;
            return true;
        case IExpression_binary::OK_NOT_EQUAL:
            // != --> ==
            neg_op = IExpression_binary::OK_EQUAL;
            return true;
        default:
            break;
        }
        break;
    case IType::TK_STRING:
    case IType::TK_ENUM:
        // restricted only
        switch (op) {
        case IExpression_binary::OK_EQUAL:
            // == --> !=
            neg_op = IExpression_binary::OK_NOT_EQUAL;
            return true;
        case IExpression_binary::OK_NOT_EQUAL:
            // != --> ==
            neg_op = IExpression_binary::OK_EQUAL;
            return true;
        default:
            break;
        }
        break;
    default:
        break;
    }
    neg_op = op;
    return false;
}

// Run the optimizer on this module.
void Optimizer::run(
    MDL                  *compiler,
    Module               &module,
    Thread_context       &ctx,
    NT_analysis          &nt_ana,
    Stmt_info_data const &stmt_info_data)
{
    if (module.access_messages().get_error_message_count() > 0)
        return;

    int opt_level = compiler->get_compiler_int_option(&ctx, MDL::option_opt_level, 2);

    if (opt_level == 0) {
        // all optimizations are switched off
        return;
    }

    Optimizer opt(compiler, module, nt_ana, stmt_info_data, opt_level);

    opt.remove_unused_functions();
    opt.remove_dead_code();
    opt.local_opt();
}

// Creates an unary expression.
IExpression_unary *Optimizer::create_unary(
    IExpression_unary::Operator op,
    IExpression const *arg,
    Position const    &pos)
{
    IExpression_unary *res = m_expr_factory.create_unary(op, arg, POS(pos));

    IType const *arg_types[] = { arg->get_type() };
    Definition const *def = m_nt_ana.find_operator(
        res, IExpression::Operator(op), arg_types, dimension_of(arg_types));

    MDL_ASSERT(def != NULL && "Failed to create unary operation");
    IType const *res_type = Analysis::get_result_type(def);
    res->set_type(res_type);

    return res;
}

// Creates a binary expression.
IExpression_binary *Optimizer::create_binary(
    IExpression_binary::Operator op,
    IExpression const *left,
    IExpression const *right,
    Position const    &pos)
{
    IExpression_binary *res = m_expr_factory.create_binary(
        op, left, right, POS(pos));

    IType const *arg_types[] = { left->get_type(), right->get_type() };
    Definition const *def = m_nt_ana.find_operator(
        res, IExpression::Operator(op), arg_types, dimension_of(arg_types));

    MDL_ASSERT(def != NULL && "Failed to create binary operation");
    IType const *res_type = Analysis::get_result_type(def);
    res->set_type(res_type);

    return res;
}

// Execute a function on the bodies of all MDL functions.
void Optimizer::run_on_function(void (Optimizer::* func)(IStatement *))
{
    int n = m_module.get_declaration_count();

    for (int i = 0; i < n; ++i) {
        IDeclaration const *decl = m_module.get_declaration(i);

        if (IDeclaration_function const *fdecl = as<IDeclaration_function>(decl)) {
            if (IStatement *body = const_cast<IStatement *>(fdecl->get_body()))
                (this->*func)(body);
        }
    }
}

// Execute a function on the bodies of all MDL functions.
void Optimizer::run_on_function(
    IStatement const *(Optimizer::* body_func)(IStatement const *),
    IExpression const *(Optimizer::* expr_func)(IExpression const *))
{
    int n = m_module.get_declaration_count();

    for (int i = 0; i < n; ++i) {
        IDeclaration *decl = const_cast<IDeclaration *>(m_module.get_declaration(i));

        if (IDeclaration_function *fdecl = as<IDeclaration_function>(decl)) {
            // optimize default initializers
            Definition const *def = impl_cast<Definition>(fdecl->get_definition());

            // should be of this module
            MDL_ASSERT(def->get_original_import_idx() == 0);

            for (int i = 0, n = fdecl->get_parameter_count(); i < n; ++i) {
                IParameter const *param = fdecl->get_parameter(i);
                if (IExpression const *init = def->get_default_param_initializer(i)) {
                    IExpression const *n_init = (this->*expr_func)(init);
                    if (n_init != NULL) {
                        const_cast<Definition *>(def)->set_default_param_initializer(i, n_init);
                        if (def->get_prototype_declaration() == NULL) {
                            // only reflect this change in the AST if this def has no prototype,
                            // which means it IS either a prototype itself or it is a declaration
                            // without a prototype
                            const_cast<IParameter *>(param)->set_init_expr(n_init);
                        }
                    }
                }
                // optimize enable_if expressions
                if (IAnnotation_block const *block = param->get_annotations()) {
                    for (int j = 0, m = block->get_annotation_count(); j < m; ++j) {
                        if (IAnnotation_enable_if const *ei =
                            as<IAnnotation_enable_if>(block->get_annotation(j))) {
                            IExpression const *expr = ei->get_expression();
                            IExpression const *n_expr = (this->*expr_func)(expr);
                            if (n_expr != NULL) {
                                const_cast<IAnnotation_enable_if *>(ei)->set_expression(n_expr);
                            }
                        }
                    }
                }
            }

            // optimize the function body
            if (IStatement const *body = fdecl->get_body()) {
                IStatement const *n_body = (this->*body_func)(body);
                if (n_body != NULL) {
                    fdecl->set_body(n_body);
                } else {
                    // The body would be empty, but this is not supported.
                    // Note that the body is a compound statement for functions
                    // and it is cleared in that case, so do nothing here.
                }
            }
        }
    }
}

// Remove unused functions from the AST.
void Optimizer::remove_unused_functions()
{
    int n = m_module.get_declaration_count();

    VLA<IDeclaration const *>n_decls(m_module.get_allocator(), n);

    int j = 0;
    for (int i = 0; i < n; ++i) {
        IDeclaration const *decl = m_module.get_declaration(i);

        if (IDeclaration_function const *fdecl = as<IDeclaration_function>(decl)) {
            Definition const *def     = impl_cast<Definition>(fdecl->get_definition());
            Definition const *def_def = def->get_definite_definition();
            if (def_def != NULL)
                def = def_def;

            if (!def->has_flag(Definition::DEF_IS_USED)) {
                // remove the reference
                const_cast<Definition*>(def)->set_declaration(NULL);
                continue;
            }
        }
        n_decls[j++] = decl;
    }
    if (j != n)
        m_module.replace_declarations(n_decls.data(), j);
}

// Remove dead code from functions.
void Optimizer::remove_dead_code()
{
    run_on_function(&Optimizer::remove_dead_code);
}

// Remove dead child-statements.
void Optimizer::remove_dead_code(IStatement *stmt)
{
    switch (stmt->get_kind()) {
    case IStatement::SK_INVALID:
        // should not happen
        break;
    case IStatement::SK_COMPOUND:
    case IStatement::SK_CASE:
        {
            IStatement_compound *c_smtm = cast<IStatement_compound>(stmt);

            int n = c_smtm->get_statement_count();
            int j = 0;
            for (int i = 0; i < n; ++i) {
                IStatement const *s = c_smtm->get_statement(i);

                Stmt_info const &info = m_stmt_info_data.get_stmt_info(s);

                if (info.m_reachable_start) {
                    ++j;
                    remove_dead_code(const_cast<IStatement *>(s));
                } else {
                    // we are inside a compound, so all further statements are dead as well
                    break;
                }
            }
            if (j != n) {
                // dead code found
                c_smtm->drop_statements_after(j);
            }
        }
        return;
    case IStatement::SK_IF:
        {
            IStatement_if *if_stmt = cast<IStatement_if>(stmt);
            IStatement *then_stmt = const_cast<IStatement *>(if_stmt->get_then_statement());
            remove_dead_code(then_stmt);
            IStatement *else_stmt = const_cast<IStatement *>(if_stmt->get_else_statement());
            if (else_stmt)
                remove_dead_code(else_stmt);
        }
        return;

    case IStatement::SK_SWITCH:
        {
            IStatement_switch *switch_stmt = cast<IStatement_switch>(stmt);
            for (int i = 0, n = switch_stmt->get_case_count(); i < n; ++i) {
                IStatement *s = const_cast<IStatement *>(switch_stmt->get_case(i));
                remove_dead_code(s);
            }
        }
        return;

    case IStatement::SK_WHILE:
    case IStatement::SK_DO_WHILE:
    case IStatement::SK_FOR:
        {
            IStatement_loop *loop_stmt = cast<IStatement_loop>(stmt);
            IStatement      *body      = const_cast<IStatement *>(loop_stmt->get_body());
            remove_dead_code(body);
        }
        return;

    case IStatement::SK_DECLARATION:
    case IStatement::SK_EXPRESSION:
    case IStatement::SK_BREAK:
    case IStatement::SK_CONTINUE:
    case IStatement::SK_RETURN:
        // these have no sub-statements
        return;
    }
    MDL_ASSERT(!"invalid statement found");
}

// Run local optimizations.
void Optimizer::local_opt()
{
    run_on_function(&Optimizer::local_opt, &Optimizer::local_opt);
}

// Run local optimizations
IDeclaration const *Optimizer::local_opt(IDeclaration const *c_decl)
{
    IDeclaration *decl = const_cast<IDeclaration *>(c_decl);

    switch (decl->get_kind()) {
    case IDeclaration::DK_INVALID:
        // should not happen
        break;
    case IDeclaration::DK_IMPORT:
    case IDeclaration::DK_ANNOTATION:
    case IDeclaration::DK_CONSTANT:
    case IDeclaration::DK_TYPE_ALIAS:
    case IDeclaration::DK_TYPE_STRUCT:
    case IDeclaration::DK_TYPE_ENUM:
    case IDeclaration::DK_FUNCTION:
    case IDeclaration::DK_MODULE:
    case IDeclaration::DK_NAMESPACE_ALIAS:
        // do nothing
        return decl;
    case IDeclaration::DK_VARIABLE:
        {
            IDeclaration_variable *v_decl = cast<IDeclaration_variable>(decl);

            for (int i = 0, n = v_decl->get_variable_count(); i < n; ++i) {
                if (IExpression const *init = v_decl->get_variable_init(i)) {
                    init = local_opt(init);
                    v_decl->set_variable_init(i, init);
                }
            }
            return v_decl;
        }
    }
    MDL_ASSERT(!"Unsupported declaraton kind");
    return decl;
}

// Run local optimizations
IStatement const *Optimizer::local_opt(IStatement const *c_stmt)
{
    IStatement *stmt = const_cast<IStatement *>(c_stmt);

    IStatement::Kind kind = stmt->get_kind();
    switch (kind) {
    case IStatement::SK_INVALID:
        // should not happen
        break;
    case IStatement::SK_COMPOUND:
    case IStatement::SK_CASE:
        {
            IStatement_compound *c_smtm = cast<IStatement_compound>(stmt);

            int n = c_smtm->get_statement_count();
            if (n == 0 && kind == IStatement::SK_COMPOUND) {
                // drop the empty block
                return NULL;
            }

            VLA <IStatement const *> n_stmts(m_module.get_allocator(), n);

            size_t j = 0;
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                IStatement const *s = c_smtm->get_statement(i);
                IStatement const *n = local_opt(s);

                changed |= n != s;
                if (n != NULL)
                    n_stmts[j++] = n;
            }
            if (changed) {
                c_smtm->replace_statements(n_stmts.data(), j);

                if (j == 0 && kind == IStatement::SK_COMPOUND) {
                    // Drop the empty block. Do this AFTER all
                    // sub statements are removed, so the block
                    // can be retained by the caller if necessary.
                    return NULL;
                }
            }
            return c_smtm;
        }

    case IStatement::SK_IF:
        {
            IStatement_if     *if_stmt   = cast<IStatement_if>(stmt);
            IExpression const *cond      = if_stmt->get_condition();
            IStatement const  *then_stmt = if_stmt->get_then_statement();
            IStatement const  *else_stmt = if_stmt->get_else_statement();

            IExpression const *n_cond = local_opt(cond);
            if (n_cond != cond)
                if_stmt->set_condition(n_cond);

            if (IExpression_literal const *lit = as<IExpression_literal>(n_cond)) {
                IValue_bool const *val = cast<IValue_bool>(lit->get_value());

                if (val->get_value())
                    return local_opt(then_stmt);
                else
                    return else_stmt != NULL ? local_opt(else_stmt) : NULL;
            }
            IStatement const *n_then = local_opt(then_stmt);
            IStatement const *n_else = else_stmt != NULL ? local_opt(else_stmt) : NULL;

            if (n_then == NULL && n_else == NULL) {
                // both branches are empty, preserve just the condition
                Position const &pos = if_stmt->access_position();
                IStatement *n_stmt = m_stmt_factory.create_expression(n_cond, POS(pos));

                return local_opt(n_stmt);
            }
            if (n_then == NULL) {
                // no then but else
                Position const &e_pos = n_cond->access_position();
                IExpression const *neg = create_unary(
                    IExpression_unary::OK_LOGICAL_NOT,
                    n_cond,
                    e_pos);

                neg = local_opt(neg);

                if_stmt->set_condition(neg);
                n_then = n_else;
                n_else = NULL;
            }
            if_stmt->set_then_statement(n_then);
            if_stmt->set_else_statement(n_else);

            return if_stmt;
        }

    case IStatement::SK_SWITCH:
        {
            IStatement_switch *switch_stmt = cast<IStatement_switch>(stmt);
            for (int i = 0, n = switch_stmt->get_case_count(); i < n; ++i) {
                IStatement const *s = switch_stmt->get_case(i);
                local_opt(s);
            }
            return switch_stmt;
        }

    case IStatement::SK_WHILE:
        {
            IStatement_loop *loop_stmt = cast<IStatement_loop>(stmt);
            IExpression const *cond    = local_opt(loop_stmt->get_condition());

            loop_stmt->set_condition(cond);

            if (IExpression_literal const *lit = as<IExpression_literal>(cond)) {
                IValue_bool const *v = cast<IValue_bool>(lit->get_value());

                if (v->get_value()) {
                    // endless loop
                } else {
                    // not executed
                    return NULL;
                }
            }
            IStatement const *body = local_opt(loop_stmt->get_body());
            if (body == NULL) {
                // body is empty, preserve just the condition
                Position const &pos = loop_stmt->access_position();
                IStatement *n_stmt = m_stmt_factory.create_expression(cond, POS(pos));

                return local_opt(n_stmt);
            }
            loop_stmt->set_body(body);
            return loop_stmt;
        }

    case IStatement::SK_DO_WHILE:
        {
            IStatement_loop *loop_stmt = cast<IStatement_loop>(stmt);
            IExpression const *cond    = local_opt(loop_stmt->get_condition());

            loop_stmt->set_condition(cond);

            if (IExpression_literal const *lit = as<IExpression_literal>(cond)) {
                IValue_bool const *v = cast<IValue_bool>(lit->get_value());

                if (v->get_value()) {
                    // endless loop
                } else {
                    // body executed once
                    return local_opt(loop_stmt->get_body());
                }
            }
            IStatement const *body   = loop_stmt->get_body();
            IStatement const *n_body = local_opt(body);
            if (n_body == NULL) {
                // do-while body cannot be empty
                Position const &pos = body->access_position();
                n_body = m_stmt_factory.create_expression(NULL, POS(pos));
            }
            loop_stmt->set_body(n_body);
            return loop_stmt;
        }

    case IStatement::SK_FOR:
        {
            IStatement_for    *for_stmt = cast<IStatement_for>(stmt);
            IExpression const *cond     = for_stmt->get_condition();

            if (cond != NULL) {
                cond = local_opt(cond);
                for_stmt->set_condition(cond);
                if (IExpression_literal const *lit = as<IExpression_literal>(cond)) {
                    IValue_bool const *v = cast<IValue_bool>(lit->get_value());

                    if (v->get_value()) {
                        // endless loop, remove the condition at all
                        for_stmt->set_condition(NULL);
                    } else {
                        // loop is not executed, only the init statement will prevail
                        IStatement const *init = for_stmt->get_init();

                        if (is<IStatement_declaration>(init)) {
                            // the for statement created a scope around the if, so we need a
                            // block now
                            Position const &pos = for_stmt->access_position();
                            IStatement_compound *n_init = m_stmt_factory.create_compound(POS(pos));
                            n_init->add_statement(init);
                            init = n_init;
                        }

                        return init != NULL ? local_opt(init) : NULL;
                    }
                }
            }

            IStatement const *body   = for_stmt->get_body();
            IStatement const *n_body = local_opt(body);
            if (n_body == NULL) {
                // for body cannot be empty
                Position const &pos = body->access_position();
                n_body = m_stmt_factory.create_expression(NULL, POS(pos));
            }
            for_stmt->set_body(body);

            IExpression const *next = for_stmt->get_update();
            if (next != NULL) {
                next = local_opt(next);
                for_stmt->set_update(next);
            }
            return for_stmt;
        }

    case IStatement::SK_EXPRESSION:
        {
            IStatement_expression *e_stmt = cast<IStatement_expression>(stmt);
            IExpression const *expr = e_stmt->get_expression();

            if (expr != NULL) {
                Expr_info const &expr_info = m_stmt_info_data.get_expr_info(expr);

                if (!expr_info.m_has_effect) {
                    // useless expression
                    return NULL;
                }

                IExpression const *n_expr = local_opt(expr);
                if (n_expr != expr)
                    e_stmt->set_expression(n_expr);
            } else {
                // useless
                return NULL;
            }
            return e_stmt;
        }

    case IStatement::SK_DECLARATION:
        {
            IStatement_declaration *decl_stmt = cast<IStatement_declaration>(stmt);
            IDeclaration const     *decl      = decl_stmt->get_declaration();
            IDeclaration const     *n_decl    = local_opt(decl);

            if (n_decl == NULL)
                return NULL;
            if (n_decl != decl)
                decl_stmt->set_declaration(n_decl);
            return decl_stmt;
        }

    case IStatement::SK_BREAK:
    case IStatement::SK_CONTINUE:
        return stmt;

    case IStatement::SK_RETURN:
        {
            IStatement_return *r_stmt = cast<IStatement_return>(stmt);
            IExpression const *expr = r_stmt->get_expression();

            IExpression const *n_expr = local_opt(expr);
            if (n_expr != expr)
                r_stmt->set_expression(n_expr);

            return r_stmt;
        }
    }
    MDL_ASSERT(!"invalid statement found");
    return NULL;
}

/// Check if all necessary sub expression are literals, so this expression should be folded.
static bool should_be_folded(IExpression const *expr)
{
    if (is<IExpression_binary>(expr)) {
        IExpression_binary const *bin_expr = cast<IExpression_binary>(expr);
        if (bin_expr->get_operator() == IExpression_binary::OK_SELECT) {
            // need special handling for the SELECT operator, which requires only the left
            // to be a literal
            return is<IExpression_literal>(bin_expr->get_left_argument());
        }
    }
    for (int i = 0, n = expr->get_sub_expression_count(); i < n; ++i) {
        if (!is<IExpression_literal>(expr->get_sub_expression(i)))
            return false;
    }
    return true;
}

/// Check if the given operator contains an assignment.
static bool is_assign_operator(IExpression_binary::Operator op)
{
    switch (op) {
    case IExpression_binary::OK_ASSIGN:
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
        return true;
    default:
        return false;
    }
}

/// Check if the given expression has a side-effect (and cannot just be removed).
static bool has_side_effect(IExpression const *expr)
{
    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        // should not happen, but if it does, do not optimize away
        return true;
    case IExpression::EK_LITERAL:
    case IExpression::EK_REFERENCE:
        return false;
    case IExpression::EK_UNARY:
        {
            IExpression_unary const *unary = cast<IExpression_unary>(expr);
            switch (unary->get_operator()) {
            case IExpression_unary::OK_BITWISE_COMPLEMENT:
            case IExpression_unary::OK_LOGICAL_NOT:
            case IExpression_unary::OK_POSITIVE:
            case IExpression_unary::OK_NEGATIVE:
            case IExpression_unary::OK_CAST:
                return has_side_effect(unary->get_argument());
            case IExpression_unary::OK_PRE_INCREMENT:
            case IExpression_unary::OK_PRE_DECREMENT:
            case IExpression_unary::OK_POST_INCREMENT:
            case IExpression_unary::OK_POST_DECREMENT:
                return true;
            }
            MDL_ASSERT(!"unsupported unary expression kind");
            return true;
        }
    case IExpression::EK_BINARY:
        {
            IExpression_binary const *binary = cast<IExpression_binary>(expr);
            IExpression_binary::Operator op = binary->get_operator();

            if (is_assign_operator(op))
                return true;

            IExpression const *lhs = binary->get_left_argument();
            if (has_side_effect(lhs))
                return true;
            IExpression const *rhs = binary->get_right_argument();
            if (has_side_effect(rhs))
                return true;
            return false;
        }
    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional const *c_expr = cast<IExpression_conditional>(expr);
            IExpression const *cond = c_expr->get_condition();
            if (has_side_effect(cond))
                return true;
            IExpression const *t_ex = c_expr->get_true();
            if (has_side_effect(t_ex))
                return true;
            IExpression const *f_ex = c_expr->get_false();
            if (has_side_effect(f_ex))
                return true;
            return false;
        }

    case IExpression::EK_CALL:
        // in MDL, calls do not have a side effect
        return false;
    case IExpression::EK_LET:
        // by definition, assignments are NOT allowed in let expressions
        return false;
    }
    MDL_ASSERT(!"unsupported expression kind");
    return true;
}

/// Check if the given expression is of matrix type.
static bool is_matrix_typed(IExpression const *expr) {
    return as<IType_matrix>(expr->get_type()) != NULL;
}

// Promote a result to the given type (explicitly).
IExpression const *Optimizer::promote(IExpression const *expr, IType const *type)
{
    IType const *e_type = expr->get_type()->skip_type_alias();
    type = type->skip_type_alias();

    if (e_type == type)
        return expr;

    // otherwise add a conversion
    IExpression const *res = m_nt_ana.convert_to_type_explicit(expr, type);
    MDL_ASSERT(res != NULL && "Unexpected conversion failure");
    return local_opt(res);
}

/// Skip an array copy constructor.
///
/// \param arr_constr  an array constructor
///
/// \return the argument if its a copy constructor, NULL else
static IExpression const *skip_array_copy_constructor(
    IExpression_call *arr_constr)
{
    if (arr_constr->get_argument_count() != 1)
        return NULL;
    IArgument const   *arg  = arr_constr->get_argument(0);
    IExpression const *expr = arg->get_argument_expr();

    IType const *res_type = arr_constr->get_type()->skip_type_alias();
    IType const *arg_type = expr->get_type()->skip_type_alias();

    if (res_type == arg_type) {
        MDL_ASSERT(is<IType_array>(arg_type));
        return expr;
    }
    return NULL;
}

// Run local optimizations.
IExpression const *Optimizer::local_opt(IExpression const *cexpr)
{
    IExpression *expr = const_cast<IExpression *>(cexpr);

    switch (expr->get_kind()) {
    case IExpression::EK_INVALID:
        // should not happen
        break;

    case IExpression::EK_LITERAL:
        // cannot optimize further
        return expr;

    case IExpression::EK_REFERENCE:
        // try const folding
        break;

    case IExpression::EK_UNARY:
        {
            IExpression_unary *unary = cast<IExpression_unary>(expr);
            IExpression const *arg   = local_opt(unary->get_argument());

            unary->set_argument(arg);

            IExpression_unary::Operator op = unary->get_operator();

            switch (op) {
            case IExpression_unary::OK_BITWISE_COMPLEMENT:
            case IExpression_unary::OK_NEGATIVE:
                {
                    // idempotent operators
                    if (IExpression_unary const *uarg = as<IExpression_unary>(arg)) {
                        if (uarg->get_operator() == op)
                            return uarg->get_argument();
                    }
                }
                break;
            case IExpression_unary::OK_LOGICAL_NOT:
                {
                    // idempotent
                    if (IExpression_unary const *uarg = as<IExpression_unary>(arg)) {
                        if (uarg->get_operator() == op)
                            return uarg->get_argument();
                    }
                    if (IExpression_binary const *uarg = as<IExpression_binary>(arg)) {
                        IExpression_binary::Operator op = uarg->get_operator();
                        IExpression_binary::Operator not_opt;

                        IExpression const *lhs = uarg->get_left_argument();
                        IType const *l_type = lhs->get_type();
                        if (negate_relation(op, l_type, not_opt)) {
                            // can negate the relation: do it
                            IExpression const *rhs = uarg->get_right_argument();
                            Position const    &pos = uarg->access_position();

                            IExpression_binary *res = create_binary(not_opt, lhs, rhs, pos);
                            return local_opt(res);
                        }
                    }
                }
                break;

            case IExpression_unary::OK_POSITIVE:
                return arg;

            case IExpression_unary::OK_CAST:
                {
                    IType const *dst_type = unary->get_type()->skip_type_alias();
                    IType const *src_type = arg->get_type()->skip_type_alias();

                    if (src_type == dst_type) {
                        // cast to itself
                        return arg;
                    }
                    if (IExpression_unary const *uarg = as<IExpression_unary>(arg)) {
                        IExpression const *arg      = uarg->get_argument();
                        IType const       *src_type = arg->get_type()->skip_type_alias();

                        if (src_type == dst_type) {
                            // cast to type and back to itself
                            return arg;
                        }
                    }
                }
                break;

            default:
                break;
            }
            break;
        }

    case IExpression::EK_BINARY:
        {
            IExpression_binary *binary = cast<IExpression_binary>(expr);
            IExpression_binary::Operator op = binary->get_operator();

            IExpression const *lhs = local_opt(binary->get_left_argument());
            IExpression const *rhs = local_opt(binary->get_right_argument());

            binary->set_left_argument(lhs);
            binary->set_right_argument(rhs);

            // normalize: literal to RIGHT
            switch (op) {
            case IExpression_binary::OK_MULTIPLY:
                if (is_matrix_typed(lhs) || is_matrix_typed(rhs)) {
                    // matrix multiplication is not symmetric
                    break;
                }
                // fall through
            case IExpression_binary::OK_PLUS:
            case IExpression_binary::OK_BITWISE_AND:
            case IExpression_binary::OK_BITWISE_XOR:
            case IExpression_binary::OK_BITWISE_OR:
                if (is<IExpression_literal>(lhs) && !is<IExpression_literal>(rhs)) {
                    IExpression const *t = lhs;
                    lhs = rhs;
                    rhs = t;
                }
                break;

            case IExpression_binary::OK_LOGICAL_AND:
                {
                    if (IExpression_literal const *l_c = as<IExpression_literal>(lhs)) {
                        IValue const *l_v = l_c->get_value();

                        if (l_v->is_one()) {
                            // true && x ==> RESULT_TYPE(x)
                            return promote(rhs, binary->get_type());
                        } else if (l_v->is_zero()) {
                            // false && x ==> RESULT_TYPE(false)
                            return promote(l_c, binary->get_type());
                        }
                    }

                    if (IExpression_literal const *r_c = as<IExpression_literal>(rhs)) {
                        IValue const *r_v = r_c->get_value();

                        if (r_v->is_one()) {
                            // x && true ==> RESULT_TYPE(x)
                            return promote(lhs, binary->get_type());
                        } else if (r_v->is_zero()) {
                            // x && false ==> x,RESULT_TYPE(false)
                            if (!has_side_effect(lhs)) {
                                return promote(r_c, binary->get_type());
                            }
                            return create_binary(
                                IExpression_binary::OK_SEQUENCE,
                                lhs, rhs,
                                expr->access_position());
                        }
                    }
                }
                break;

            case IExpression_binary::OK_LOGICAL_OR:
                {
                    if (IExpression_literal const *l_c = as<IExpression_literal>(lhs)) {
                        IValue const *l_v = l_c->get_value();

                        if (l_v->is_one()) {
                            // true || x ==> RESULT_TYPE(true)
                            return promote(l_c, binary->get_type());
                        } else if (l_v->is_zero()) {
                            // false || x ==> RESULT_TYPE(x)
                            return promote(rhs, binary->get_type());
                        }
                    }

                    if (IExpression_literal const *r_c = as<IExpression_literal>(rhs)) {
                        IValue const *r_v = r_c->get_value();

                        if (r_v->is_one()) {
                            // x || true ==> x,RESULT_TYPE(true)
                            if (!has_side_effect(lhs)) {
                                return promote(r_c, binary->get_type());
                            }
                            return create_binary(
                                IExpression_binary::OK_SEQUENCE,
                                lhs, rhs,
                                expr->access_position());
                        } else if (r_v->is_zero()) {
                            // x || false ==> RESULT_TYPE(x)
                            return promote(lhs, binary->get_type());
                        }
                    }
                }
                break;

            // handle relations
            case IExpression_binary::OK_LESS:
            case IExpression_binary::OK_LESS_OR_EQUAL:
            case IExpression_binary::OK_GREATER_OR_EQUAL:
            case IExpression_binary::OK_GREATER:
            case IExpression_binary::OK_EQUAL:
            case IExpression_binary::OK_NOT_EQUAL:
                if (is<IExpression_literal>(lhs) && !is<IExpression_literal>(rhs)) {
                    IExpression_binary::Operator inv_op;
                    if (inverse_relation(op, inv_op)) {
                        // can invert it, put literal to right
                        IExpression const *t = lhs;
                        lhs = rhs;
                        rhs = t;

                        Position const &pos = binary->access_position();
                        binary = create_binary(inv_op, lhs, rhs, pos);
                    }
                }
                break;
            case IExpression_binary::OK_SEQUENCE:
                if (!has_side_effect(lhs))
                    return rhs;
                break;
            default:
                break;
            }

            binary->set_left_argument(lhs);
            binary->set_right_argument(rhs);

            // handle constant at right side
            if (IExpression_literal const *lit = as<IExpression_literal>(rhs)) {
                IValue const *rv = lit->get_value();
                switch (op) {
                case IExpression_binary::OK_MULTIPLY:
                    if (rv->is_one()) {
                        /* lhs * 1 = RESULT_TYPE(lhs) */
                        if (is<IValue_vector>(rv) && is_matrix_typed(lhs)) {
                            // does not work for matrix * vector, vector(1) is NOT the
                            // neutral element
                        } else {
                            return promote(lhs, expr->get_type());
                        }
                    }
                    break;
                case IExpression_binary::OK_PLUS:
                case IExpression_binary::OK_MINUS:
                    if (rv->is_zero()) {
                        /* lhs +/- 0 = RESULT_TYPE(lhs) */
                        return promote(lhs, expr->get_type());
                    }
                    break;

                case IExpression_binary::OK_BITWISE_AND:
                    if (rv->is_zero()) {
                        /* lhs & 0 = 0 */
                        return rhs;
                    }
                    if (rv->is_all_one()) {
                        // lhs & 1...1 = RESULT_TYPE(lhs)
                        return promote(lhs, expr->get_type());
                    }
                    break;
                case IExpression_binary::OK_BITWISE_XOR:
                    if (rv->is_zero()) {
                        /* lhs ^ 0 = RESULT_TYPE(lhs) */
                        return promote(lhs, expr->get_type());
                    }
                    break;
                case IExpression_binary::OK_BITWISE_OR:
                    if (rv->is_zero()) {
                        /* lhs | 0 = RESULT_TYPE(lhs) */
                        return promote(lhs, expr->get_type());
                    }
                    if (rv->is_all_one()) {
                        /* lhs | 1...1 = RESULT_TYPE(1...1) */
                        return promote(rhs, expr->get_type());
                    }
                    break;
                case IExpression_binary::OK_LOGICAL_AND:
                    if (rv->is_one()) {
                        // x && true = RESULT_TYPE(x)
                        return promote(lhs, expr->get_type());
                    }
                    if (rv->is_zero() && !has_side_effect(lhs)) {
                        // x && false = RESULT_TYPE(false)  if x side effect free
                        return promote(rhs, expr->get_type());
                    }
                    break;
                case IExpression_binary::OK_LOGICAL_OR:
                    if (rv->is_zero()) {
                        // x || false = RESULT_TYPE(x)
                        return promote(lhs, expr->get_type());
                    }
                    if (rv->is_one() && !has_side_effect(lhs)) {
                        // x || true = RESULT_TYPE(true)  if x side effect free
                        return promote(rhs, expr->get_type());
                    }
                    break;
                case IExpression_binary::OK_SHIFT_LEFT:
                case IExpression_binary::OK_SHIFT_RIGHT:
                case IExpression_binary::OK_UNSIGNED_SHIFT_RIGHT:
                    if (rv->is_zero()) {
                        // x SHIFT 0 = x
                        return lhs;
                    }
                    break;
                default:
                    break;
                }
            }
            // handle constant at left side
            if (IExpression_literal const *lit = as<IExpression_literal>(lhs)) {
                IValue const *lv = lit->get_value();
                switch (op) {
                case IExpression_binary::OK_MULTIPLY:
                    if (lv->is_one()) {
                        /* 1 * rhs = RESULT_TYPE(rhs) (matrix mult only) */
                        if (is<IValue_vector>(lv) && is_matrix_typed(rhs)) {
                            // does not work for vector * matrix, vector(1) is NOT the
                            // neutral element
                        } else {
                            return promote(rhs, expr->get_type());
                        }
                    }
                    break;
                case IExpression_binary::OK_LOGICAL_AND:
                    if (lv->is_one()) {
                        // true && x = RESULT_TYPE(x)
                        return promote(rhs, expr->get_type());
                    }
                    if (lv->is_zero()) {
                        // false && x = RESULT_TYPE(false)
                        return promote(lhs, expr->get_type());
                    }
                    break;
                case IExpression_binary::OK_LOGICAL_OR:
                    if (lv->is_zero()) {
                        // false || x = RESULT_TYPE(x)
                        return promote(rhs, expr->get_type());
                    }
                    if (lv->is_one()) {
                        // true || x = RESULT_TYPE(true)
                        return promote(lhs, expr->get_type());
                    }
                    break;
                default:
                    break;
                }
            }
            break;
        }

    case IExpression::EK_CONDITIONAL:
        {
            IExpression_conditional *c_expr = cast<IExpression_conditional>(expr);
            IExpression const *cond = local_opt(c_expr->get_condition());

            if (IExpression_literal const *lit = as<IExpression_literal>(cond)) {
                IValue_bool const *b = cast<IValue_bool>(lit->get_value());

                if (b->get_value())
                    return local_opt(c_expr->get_true());
                else
                    return local_opt(c_expr->get_false());
            }

            IExpression const *t_ex = local_opt(c_expr->get_true());
            IExpression const *f_ex = local_opt(c_expr->get_false());

            if (is<IType_bool>(c_expr->get_type()->skip_type_alias())) {
                if (IExpression_literal const *lit_t = as<IExpression_literal>(t_ex)) {
                    if (IExpression_literal const *lit_f = as<IExpression_literal>(f_ex)) {
                        IValue_bool const *v_t = cast<IValue_bool>(lit_t->get_value());
                        IValue_bool const *v_f = cast<IValue_bool>(lit_f->get_value());

                        if (v_t == v_f) {
                            // cond ? c : c => c
                            return lit_t;
                        }
                        if (v_t->get_value()) {
                            // cond ? true : false => cond
                            return cond;
                        }
                        if (v_f->get_value()) {
                            // cond ? false : true => !cond
                            return create_unary(
                                IExpression_unary::OK_LOGICAL_NOT,
                                cond,
                                c_expr->access_position());
                        }
                    }
                }
            }

            c_expr->set_condition(cond);
            c_expr->set_true(t_ex);
            c_expr->set_false(f_ex);
            return c_expr;
        }

    case IExpression::EK_CALL:
        {
            IExpression_call *call = cast<IExpression_call>(expr);
            int n = call->get_argument_count();
            bool all_const = true;

            VLA<IValue const *> c_args(m_module.get_allocator(), n);

            for (int i = 0; i < n; ++i) {
                IArgument         *arg  = const_cast<IArgument *>(call->get_argument(i));
                IExpression const *expr = local_opt(arg->get_argument_expr());

                arg->set_argument_expr(expr);
                if (IExpression_literal const *lit = as<IExpression_literal>(expr)) {
                    c_args[i] = lit->get_value();
                } else {
                    all_const = false;
                }
            }

            IExpression_reference const *ref = as<IExpression_reference>(call->get_reference());
            if (ref != NULL) {
                if (ref->is_array_constructor()) {
                    IType_array const *a_type =
                        cast<IType_array>(call->get_type()->skip_type_alias());

                    if (IExpression const *arg = skip_array_copy_constructor(call))
                        return arg;

                    if (all_const) {
                        // an array constructor with all elements constant.
                        if (!a_type->is_immediate_sized()) {
                            // cannot fold array constructors of deferred size
                            return expr;
                        }

                        IValue const *v = NULL;
                        if (n == 0) {
                            // default constructor
                            n = a_type->get_size();

                            VLA<IValue const *> c_args(m_module.get_allocator(), n);
                            IValue const *def_val = m_module.create_default_value(
                                m_module.get_value_factory(), a_type->get_element_type());
                            for (int i = 0; i < n; ++i)
                                c_args[i] = def_val;
                            v = m_value_factory.create_compound(a_type, c_args.data(), n);
                        } else {
                            v = m_value_factory.create_compound(a_type, c_args.data(), n);
                        }
                        if (!is<IValue_bad>(v)) {
                            // can evaluate it!
                            Position const *pos = &call->access_position();
                            return m_module.create_literal(v, pos);
                        }
                    }
                } else {
                    // regular call
                    if (all_const) {
                        IDefinition const *def = ref->get_definition();
                        IDefinition::Semantics sema = def->get_semantics();

                        if (sema != IDefinition::DS_UNKNOWN) {
                            IValue const *v = expr->fold(
                                &m_module, m_module.get_value_factory(), NULL);
                            if (!is<IValue_bad>(v)) {
                                Position const *pos = &expr->access_position();

                                return m_module.create_literal(v, pos);
                            }

                            v = m_compiler->evaluate_intrinsic_function(
                                &m_value_factory,
                                sema,
                                c_args.data(),
                                n);

                            if (!is<IValue_bad>(v)) {
                                // can evaluate it!
                                Position const *pos = &call->access_position();
                                return m_module.create_literal(v, pos);
                            }
                        }
                    }
                    // try inlining
                    if (IExpression const *inlined_expr = do_inline(call))
                        return inlined_expr;
                }
            }
            break;
        }

    case IExpression::EK_LET:
        {
            IExpression_let *let = cast<IExpression_let>(expr);

            for (int i = 0, n = let->get_declaration_count(); i < n; ++i) {
                IDeclaration const *decl = let->get_declaration(i);

                // it is not expected that the declaration is exchange or will disappear
                local_opt(decl);
            }
            IExpression const *expr = local_opt(let->get_expression());
            let->set_expression(expr);
        }
        break;
    }

    // try const folding
    if (should_be_folded(expr)) {
        // Note: fold is a deep recursive function used to fold constant expressions.
        // We don't need the deep recursion here because the optimizer walks the tree
        // in DFS order, hence we limit the fold call to places where all sub-expressions
        // are constant.
        IValue const *v = expr->fold(&m_module, m_module.get_value_factory(), NULL);
        if (!is<IValue_bad>(v)) {
            Position const *pos = &expr->access_position();

            return m_module.create_literal(v, pos);
        }
    }
    return expr;
}

// Inline the given call.
IExpression const *Optimizer::do_inline(IExpression_call *call)
{
    // FIXME: set the inline level to 3 (one above the default 2) to
    // disabled it in the default case because of bug 11154 ...
    if (m_opt_level < 3)
        return NULL;

    // Very simple inliner yet.
    IExpression_reference const *ref = as<IExpression_reference>(call->get_reference());
    if (ref == NULL)
        return NULL;

    if (ref->is_array_constructor())
        return NULL;

    Definition const *def = impl_cast<Definition>(ref->get_definition());
    if (def->has_flag(Definition::DEF_NO_INLINE)) {
        // inlining forbidden
        return NULL;
    }

    IDeclaration const *idecl = m_module.get_original_definition(def)->get_declaration();
    if (idecl == NULL)
        return NULL;
    IDeclaration_function const *fdecl = as<IDeclaration_function>(idecl);
    if (fdecl == NULL)
        return NULL;

    IStatement const *body = fdecl->get_body();
    if (body == NULL)
        return NULL;

    IStatement_compound const *block = as<IStatement_compound>(body);

    if (block == NULL || block->get_statement_count() != 1)
        return NULL;

    IStatement_return const *ret_stmt = as<IStatement_return>(block->get_statement(0));
    if (ret_stmt == NULL)
        return NULL;

    IExpression const *expr = ret_stmt->get_expression();
    switch (expr->get_kind()) {
    case IExpression::EK_LITERAL:
        // simple case: just a literal
        return m_module.clone_expr(expr, NULL);

    case IExpression::EK_REFERENCE:
        {
            IExpression_reference const *ref = cast<IExpression_reference>(expr);
            if (ref->is_array_constructor())
                return NULL;

            IDefinition const *def = ref->get_definition();

            switch (def->get_kind()) {
            case IDefinition::DK_PARAMETER:
                {
                    // return of an unmodified parameter
                    for (int i = 0, n = fdecl->get_parameter_count(); i < n; ++i) {
                        IParameter const   *param = fdecl->get_parameter(i);
                        ISimple_name const *p_name = param->get_name();
                        if (p_name->get_definition() == def)
                            return call->get_argument(i)->get_argument_expr();
                    }
                }
                return NULL;
            case IDefinition::DK_ARRAY_SIZE:
                {
                    // return an array size
                    ISymbol const *size_sym = def->get_symbol();

                    for (int i = 0, n = fdecl->get_parameter_count(); i < n; ++i) {
                        IParameter const  *param  = fdecl->get_parameter(i);
                        IType_name const  *t_name = param->get_type_name();
                        IType_array const *a_type = as<IType_array>(t_name->get_type());

                        if (a_type != NULL && !a_type->is_immediate_sized()) {
                            IType_array_size const *size = a_type->get_deferred_size();

                            if (size->get_size_symbol() == size_sym) {
                                IExpression const *arg = call->get_argument(i)->get_argument_expr();
                                IType_array const *a_arg = as<IType_array>(arg->get_type());

                                if (a_arg != NULL && a_arg->is_immediate_sized()) {
                                    int size = a_arg->get_size();
                                    IValue const *v = m_value_factory.create_int(size);

                                    Position const *pos = &call->access_position();
                                    return m_module.create_literal(v, pos);
                                }
                            }
                        }
                    }
                }
                return NULL;
            default:
                break;
            }
        }
    default:
        break;
    }

    return NULL;
}

// Constructor.
Optimizer::Optimizer(
    MDL                  *compiler,
    Module               &module,
    NT_analysis          &nt_ana,
    Stmt_info_data const &stmt_info_data,
    int                  opt_level)
: m_compiler(compiler)
, m_module(module)
, m_nt_ana(nt_ana)
, m_stmt_factory(*module.get_statement_factory())
, m_expr_factory(*module.get_expression_factory())
, m_value_factory(*module.get_value_factory())
, m_stmt_info_data(stmt_info_data)
, m_opt_level(opt_level)
{
}

}  // mdl
}  // mi
